from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from weather import recommend_crop

app = FastAPI(title="Crop Recommendation API")

class CropInput(BaseModel):
    state: str = Field(..., example="Tamil Nadu", description="State name (e.g. Tamil Nadu, Maharashtra, Punjab)")
    district: str = Field(..., example="Coimbatore", description="District name")
    season: str = Field(..., example="Kharif", description="Season: Kharif, Rabi, Summer, Autumn, Winter, or Whole Year")
    ph: float = Field(..., example=6.5, description="Soil pH (0-30cm)")
    nitrogen: float = Field(..., example=0.5, description="Soil total nitrogen g/kg (0-30cm)")
    soc: float = Field(..., example=5.0, description="Soil organic carbon g/kg (0-30cm)")
    clay: float = Field(..., example=20.0, description="Soil clay percentage (0-30cm)")
    sand: float = Field(..., example=40.0, description="Soil sand percentage (0-30cm)")

    class Config:
        json_schema_extra = {
            "example": {
                "state": "Tamil Nadu",
                "district": "Coimbatore",
                "season": "Kharif",
                "ph": 6.5,
                "nitrogen": 0.5,
                "soc": 5.0,
                "clay": 20.0,
                "sand": 40.0
            }
        }

@app.get("/")
def home():
    return {"message": "Crop Recommendation API is running"}

@app.post("/predict")
def predict(data: CropInput):
    try:
        result = recommend_crop(
            data.state,
            data.district,
            data.season,
            data.ph,
            data.nitrogen,
            data.soc,
            data.clay,
            data.sand
        )
        if isinstance(result, dict) and "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")