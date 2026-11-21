import requests
import pandas as pd
import joblib
from pathlib import Path

# Load model bundle once at startup
MODEL_PATH = Path(__file__).resolve().parent / "crop_model_bundle.joblib"
try:
    bundle = joblib.load(MODEL_PATH)
except Exception as e:
    bundle = None
    print(f"Warning: Could not load crop_model_bundle.joblib: {e}")

def recommend_crop(state, district, season, ph, nitrogen, soc, clay, sand):
    global bundle
    if bundle is None:
        try:
            bundle = joblib.load(MODEL_PATH)
        except Exception as e:
            return {"error": f"Model bundle could not be loaded: {str(e)}"}

    # 1. Geocoding lookup via Open-Meteo
    try:
        geo_res = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={
                "name": district.strip(),
                "countryCode": "IN",
                "count": 10
            },
            timeout=10
        )
        if geo_res.status_code != 200:
            return {"error": f"Open-Meteo Geocoding API returned status code {geo_res.status_code}"}
        geo = geo_res.json()
    except Exception as e:
        return {"error": f"Geocoding request failed: {str(e)}"}

    places = geo.get("results", [])
    if not places:
        return {"error": f"District '{district}' not found in India"}

    place = next(
        (p for p in places if (p.get("admin1") or "").strip().lower() == state.strip().lower()),
        None
    )

    if place is None:
        available_states = [p.get("admin1") for p in places if p.get("admin1")]
        hint = f" (Found in: {', '.join(set(available_states))})" if available_states else ""
        return {"error": f"District '{district}' not found in the state '{state}'.{hint}"}

    lat = place["latitude"]
    lon = place["longitude"]

    # 2. Historical weather data from NASA POWER API
    try:
        weather_res = requests.get(
            "https://power.larc.nasa.gov/api/temporal/daily/point",
            params={
                "parameters": "T2M,RH2M,PRECTOTCORR",
                "community": "AG",
                "longitude": lon,
                "latitude": lat,
                "start": "19970101",
                "end": "20151231",
                "format": "JSON"
            },
            timeout=30
        )
        if weather_res.status_code != 200:
            return {"error": f"NASA POWER API returned status code {weather_res.status_code}"}
        weather_json = weather_res.json()
        if "properties" not in weather_json or "parameter" not in weather_json["properties"]:
            return {"error": "NASA POWER API response did not contain expected weather parameters"}
        weather = weather_json["properties"]["parameter"]
    except Exception as e:
        return {"error": f"NASA POWER API request failed: {str(e)}"}

    data = pd.DataFrame(weather)
    data.index = pd.to_datetime(data.index)
    data = data.replace(-999, float("nan")).dropna()

    months = {
        "Kharif": [6, 7, 8, 9, 10],
        "Rabi": [11, 12, 1, 2, 3],
        "Summer": [3, 4, 5],
        "Autumn": [9, 10, 11],
        "Winter": [12, 1, 2],
        "Whole Year": list(range(1, 13))
    }

    normalized_season = season.strip().title()
    matched_season = next((k for k in months if k.lower() == season.strip().lower()), None)

    if matched_season is None:
        return {"error": f"Invalid season name '{season}'. Valid seasons are: {', '.join(months.keys())}"}

    data = data[data.index.month.isin(months[matched_season])]

    if data.empty:
        return {"error": "Weather data not available for the selected season"}

    temp = data["T2M"].mean()
    humidity = data["RH2M"].mean()
    rainfall = data["PRECTOTCORR"].sum() / 19

    # 3. Elevation lookup
    try:
        elevation_res = requests.get(
            "https://api.open-meteo.com/v1/elevation",
            params={"latitude": lat, "longitude": lon},
            timeout=10
        )
        if elevation_res.status_code != 200:
            return {"error": f"Elevation API returned status code {elevation_res.status_code}"}
        elevation_json = elevation_res.json()
        if "elevation" not in elevation_json or not elevation_json["elevation"]:
            return {"error": "Elevation data not available"}
        elevation = elevation_json["elevation"][0]
    except Exception as e:
        return {"error": f"Elevation API request failed: {str(e)}"}

    # 4. Model Prediction
    sample = pd.DataFrame([{
        "soil_ph_h2o_0_30cm": ph,
        "soil_total_n_g_kg_0_30cm": nitrogen,
        "soil_soc_g_kg_0_30cm": soc,
        "soil_clay_pct_0_30cm": clay,
        "soil_sand_pct_0_30cm": sand,
        "temp_mean_c": temp,
        "humidity_pct": humidity,
        "rainfall_mm": rainfall,
        "elevation_m": elevation,
        "season": matched_season
    }])

    sample = pd.get_dummies(sample, columns=["season"])
    sample = sample.reindex(columns=bundle["features"], fill_value=0)

    sample_scaled = bundle["scaler"].transform(sample)

    probabilities = bundle["model"].predict_proba(sample_scaled)[0]
    crops = bundle["model"].classes_

    top = probabilities.argsort()[-3:][::-1]

    recommendations = []
    for i in top:
        recommendations.append({
            "crop": str(crops[i]),
            "confidence": round(float(probabilities[i] * 100), 2)
        })

    return {
        "state": state,
        "district": district,
        "season": matched_season,
        "weather": {
            "temperature": round(float(temp), 2),
            "humidity": round(float(humidity), 2),
            "rainfall": round(float(rainfall), 2)
        },
        "top_3_crops": recommendations
    }