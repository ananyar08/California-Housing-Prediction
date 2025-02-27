from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
import numpy as np
import pickle
import requests

app = FastAPI(title="California Housing Prediction API")

# Serve the HTML UI when visiting the root URL
@app.get("/")
def read_index():
    return FileResponse("index.html")

# Load your saved models (ensure these files are in the same directory)
CLASSIFIER_URL = "https://drive.google.com/file/d/1GBPmrVN4dm0Ut2DkvWXq0gYXFYoeiFXn/view?usp=sharing"
REGRESSOR_URL = "https://drive.google.com/file/d/17B_Vc1Q_RriECHsV61aucoAbGGcffc2v/view?usp=sharing"

# Define input data model
class DataInput(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float

@app.post("/predict/regression")
def predict_regression(data: DataInput):
    input_data = np.array([[data.MedInc, data.HouseAge, data.AveRooms, data.AveBedrms,
                             data.Population, data.AveOccup, data.Latitude, data.Longitude]])
    prediction = regression_model.predict(input_data)
    return {"predicted_median_house_value": prediction[0]}

@app.post("/predict/classification")
def predict_classification(data: DataInput):
    input_data = np.array([[data.MedInc, data.HouseAge, data.AveRooms, data.AveBedrms,
                             data.Population, data.AveOccup, data.Latitude, data.Longitude]])
    prediction = classification_model.predict(input_data)
    return {"predicted_category": prediction[0]}


