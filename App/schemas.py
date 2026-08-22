from pydantic import BaseModel


class PatientData(BaseModel):

    time_in_hospital: int

    number_inpatient: int

    number_emergency: int

    number_outpatient: int

    num_medications: int

    number_diagnoses: int


    # Additional model features

    age: str = "[50-60)"

    gender: str = "Female"

    race: str = "Caucasian"


    admission_type_id: int = 1

    admission_source_id: int = 7

    discharge_disposition_id: int = 1


    num_lab_procedures: int = 40

    num_procedures: int = 1


    insulin: str = "No"

    diabetesMed: str = "Yes"


    A1Cresult_available: int = 0

    max_glu_serum_available: int = 0


class PredictionResponse(BaseModel):

    readmission_probability: float

    prediction: int

    risk_level: str