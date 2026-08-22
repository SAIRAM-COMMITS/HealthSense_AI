import pandas as pd


def create_features(patient):

    df = pd.DataFrame(
        [patient]
    )


    # Previous visits
    df["total_previous_visits"] = (
        df["number_inpatient"]
        +
        df["number_emergency"]
        +
        df["number_outpatient"]
    )


    # High utilization patient

    df["high_utilization_patient"] = (
        df["total_previous_visits"] >= 3
    ).astype(int)



    # High risk utilization

    df["high_risk_utilization"] = (
        (df["number_inpatient"] >= 2)
        |
        (df["number_emergency"] >= 2)
    ).astype(int)



    # Medication burden

    df["high_medication_burden"] = (
        df["num_medications"] >= 20
    ).astype(int)



    # Diagnosis count

    df["diagnosis_count"] = (
        df["number_diagnoses"]
    )


    # Hospital stay category

    df["hospital_stay_category"] = pd.cut(
        df["time_in_hospital"],
        bins=[0,3,7,100],
        labels=[
            "Short",
            "Medium",
            "Long"
        ]
    )



    # Diabetes diagnosis placeholder

    df["has_diabetes_diagnosis"] = 1



    # Diagnosis categories

    df["primary_diagnosis_category"] = "Diabetes"

    df["diag_2_category"] = "Diabetes"

    df["diag_3_category"] = "Diabetes"



    return df
