from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
import numpy as np
import pickle
import requests
import os

app = FastAPI(title="California Housing Prediction API")

# Serve the HTML UI when visiting the root URL
@app.get("/")
def read_index():
    return FileResponse("index.html")

# Google Drive direct download links (modify the IDs)
CLASSIFIER_URL = "https://drive.google.com/uc?id=1GBPmrVN4dm0Ut2DkvWXq0gYXFYoeiFXn"
REGRESSOR_URL = "https://drive.google.com/uc?id=17B_Vc1Q_RriECHsV61aucoAbGGcffc2v"

# File paths
CLASSIFIER_FILE = "tuned_rf_classifier.pkl"
REGRESSOR_FILE = "tuned_rf_regressor.pkl"

# Function to download models
def download_file(url, filename):
    if not os.path.exists(filename):  # Avoid redownloading
        response = requests.get(url, allow_redirects=True)
        with open(filename, "wb") as file:
            file.write(response.content)

# Download models
download_file(CLASSIFIER_URL, CLASSIFIER_FILE)
download_file(REGRESSOR_URL, REGRESSOR_FILE)

# Load models
with open(CLASSIFIER_FILE, "rb") as f:
    classification_model = pickle.load(f)

with open(REGRESSOR_FILE, "rb") as f:
    regression_model = pickle.load(f)

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



