from fastapi import FastAPI, HTTPException
import numpy as np
from pydantic import BaseModel
from keras.models import load_model


class HouseFeatures(BaseModel):
    feature1: float
    feature2: float
    feature3: float
    feature4: float
    feature5: float


app = FastAPI()

@app.on_event('startup')
def load_model_on_startup():
    global model
    try:
        model = load_model('house_price_model.h5')
    except Exception as e:
        print(f"Error loading model: {e}")

@app.get("/")
def home():
    return {"message": "Welcome to the House Price Prediction API"}

@app.post("/predict")
def predict_price(features: HouseFeatures):
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded")
    input_data = np.array([[features.feature1, features.feature2, features.feature3, features.feature4, features.feature5]])
    prediction = model.predict(input_data)
    predicted_price =  float(prediction[0][0])
    return {
        "predicted_price_usd": predicted_price
        }
