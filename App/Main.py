from fastapi import FastAPI
import pandas as pd
from App.feature_engineering import create_features
from prometheus_fastapi_instrumentator import Instrumentator

from App.model_loader import (
    load_model,
    load_threshold
)

from App.schemas import (
    PatientData,
    PredictionResponse
)

app = FastAPI(
    title="HealthSense AI API",
    description="Healthcare Readmission Prediction API",
    version="1.0"
)

Instrumentator().instrument(app).expose(app)



# Load ML artifacts when API starts
model = load_model()

threshold = load_threshold()



@app.get("/")
def home():

    return {
        "message":
        "HealthSense AI API is running"
    }



@app.get("/model-info")
def model_info():

    return {

        "model":
        "XGBoost Readmission Predictor",

        "threshold":
        threshold
    }



@app.post(
    "/predict",
    response_model=PredictionResponse
)
def predict(patient: PatientData):

    # Convert input data to dictionary
    patient_data = patient.model_dump()


    input_df = create_features(
    patient_data
)


    # Get prediction probability
    probability = model.predict_proba(
        input_df
    )[0][1]


    # Apply threshold
    prediction = (
        probability >= threshold
    )


    # Convert boolean to integer
    prediction = int(prediction)


    # Determine risk level
    if prediction == 1:
        risk_level = "High"
    else:
        risk_level = "Low"


    return {

        "readmission_probability":
            round(float(probability),4),

        "prediction":
            prediction,

        "risk_level":
            risk_level
    }