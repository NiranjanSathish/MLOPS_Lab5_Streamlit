from fastapi import FastAPI, status, HTTPException
from pydantic import BaseModel
from typing import Dict
from predict import predict_data, predict_with_prob
from data import get_feature_names, get_target_names

app = FastAPI(
    title="Wine Classification API",
    description="A FastAPI application for wine classification using Random Forest",
    version="2.0.0"
)

class WineData(BaseModel):
    alcohol: float
    malic_acid: float
    ash: float
    alcalinity_of_ash: float
    magnesium: float
    total_phenols: float
    flavanoids: float
    nonflavanoid_phenols: float
    proanthocyanins: float
    color_intensity: float
    hue: float
    od280_od315_of_diluted_wines: float
    proline: float

class WineResponse(BaseModel):
    prediction: int
    class_name: str

class WineResponseWithProba(BaseModel):
    prediction: int
    class_name: str
    probabilities: Dict[str, float]

@app.get("/", status_code=status.HTTP_200_OK)
async def health_ping():
    return {
        "status": "healthy",
        "model": "Random Forest Wine Classifier",
        "version": "2.0.0"
    }

@app.post("/predict", response_model=WineResponse)
async def predict_wine(wine_features: WineData):
    """Predict wine class based on chemical features."""
    try:
        features = [[
            wine_features.alcohol,
            wine_features.malic_acid,
            wine_features.ash,
            wine_features.alcalinity_of_ash,
            wine_features.magnesium,
            wine_features.total_phenols,
            wine_features.flavanoids,
            wine_features.nonflavanoid_phenols,
            wine_features.proanthocyanins,
            wine_features.color_intensity,
            wine_features.hue,
            wine_features.od280_od315_of_diluted_wines,
            wine_features.proline
        ]]

        prediction = predict_data(features)
        target_names = get_target_names()
        
        return WineResponse(
            prediction=int(prediction[0]),
            class_name=target_names[prediction[0]]
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict-with-probability", response_model=WineResponseWithProba)
async def predict_wine_with_probability(wine_features: WineData):
    """Predict wine class with prediction probabilities."""
    try:
        features = [[
            wine_features.alcohol,
            wine_features.malic_acid,
            wine_features.ash,
            wine_features.alcalinity_of_ash,
            wine_features.magnesium,
            wine_features.total_phenols,
            wine_features.flavanoids,
            wine_features.nonflavanoid_phenols,
            wine_features.proanthocyanins,
            wine_features.color_intensity,
            wine_features.hue,
            wine_features.od280_od315_of_diluted_wines,
            wine_features.proline
        ]]

        result = predict_with_prob(features)
        
        return WineResponseWithProba(**result)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))