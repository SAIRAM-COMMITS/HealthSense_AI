import joblib
import json

from pathlib import Path


# Project root directory
BASE_DIR = Path(__file__).resolve().parent.parent


# Model paths
MODEL_PATH = (
    BASE_DIR
    / "Models"
    / "xgb_readmission_model.pkl"
)


THRESHOLD_PATH = (
    BASE_DIR
    / "Models"
    / "threshold.json"
)



def load_model():

    """
    Load trained XGBoost pipeline.
    """

    model = joblib.load(
        MODEL_PATH
    )

    return model



def load_threshold():

    """
    Load classification threshold.
    """

    with open(
        THRESHOLD_PATH,
        "r"
    ) as file:

        config = json.load(file)


    return config[
        "classification_threshold"
    ]