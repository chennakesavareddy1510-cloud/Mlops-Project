import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# Load trained model
model = joblib.load("model.pkl")


class PredictRequest(BaseModel):
    features: list[float]

    model_config = {
        "json_schema_extra": {
            "example": {"features": [5.1, 3.5, 1.4, 0.2]}
        }
    }


@app.get("/")
def home():
    return {"message": "ML Model API is running"}


@app.post("/predict")
def predict(request: PredictRequest):
    """
    Send a JSON body: {"features": [5.1, 3.5, 1.4, 0.2]}
    """
    prediction = model.predict([request.features])
    return {"prediction": prediction.tolist()}