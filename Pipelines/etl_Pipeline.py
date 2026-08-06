import pandas as pd
from pathlib import Path


# Project directories
BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DATA_PATH = BASE_DIR / "Data" / "Raw" / "diabetic_data.csv"
PROCESSED_DATA_PATH = BASE_DIR / "Data" / "Processed" / "diabetic_data_processed.csv"



def extract_data():
    """
    Extract data from the raw CSV file.
    """
    print("Loading raw healthcare dataset...")

    df = pd.read_csv(RAW_DATA_PATH)

    print(f"Dataset loaded successfully.")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")

    return df

def profile_data(df):
    """
    Analyze the basic structure and quality
    of the healthcare dataset.
    """

    print("\n" + "=" * 60)
    print("DATASET PROFILE")
    print("=" * 60)

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nDataset information:")
    print(df.info())

    # Dataset dimensions
    print("\nDataset Shape:")
    print(df.shape)

    # Column names
    print("\nColumn Names:")
    print(df.columns.tolist())

    # Data types
    print("\nData Types:")
    print(df.dtypes)

    # Missing values
    print("\nMissing Values:")
    print(df.isnull().sum())

    # Duplicate rows
    print("\nDuplicate Rows:")
    print(df.duplicated().sum())

    # Target distribution
    print("\nReadmission Distribution:")
    print(df["readmitted"].value_counts())

    # Statistical summary
    print("\nStatistical Summary:")
    print(df.describe())


def transform_data(df):
    """
    Clean and transform the raw healthcare dataset.
    """

    print("\nStarting data transformation...")

    # Create a copy
    df = df.copy()

    # Replace '?' with missing values
    df = df.replace("?", pd.NA)

    print("Replaced '?' values with missing values.")

    # Finding the Missing Values count
    print("\nMissing Value Analysis:")
    print("-" * 40)

    missing_count = df.isna().sum()

    missing_percentage = (
        df.isna().mean() * 100
    )

    missing_summary = pd.DataFrame({
        "Missing_Count": missing_count,
        "Missing_Percentage": missing_percentage
    })

    missing_summary = missing_summary.sort_values(
        by="Missing_Percentage",
        ascending=False
    )

    print(missing_summary)


    # Create missingness indicators
    df["max_glu_serum_available"] = (
        df["max_glu_serum"].notna().astype(int)
    )

    df["A1Cresult_available"] = (
        df["A1Cresult"].notna().astype(int)
    )

    print("\nCreated missingness indicator features.")

    print(
        "max_glu_serum_available distribution:"
    )

    print(
        df["max_glu_serum_available"].value_counts()
    )

    print(
        "\nA1Cresult_available distribution:"
    )

    print(
        df["A1Cresult_available"].value_counts()
    )

    # Drop columns that are not useful for our first model
    columns_to_drop = [
        "weight",
        "payer_code"
    ]

    df = df.drop(
        columns=columns_to_drop,
        errors="ignore"
    )

    print(
        "\nDropped columns:"
    )

    print(columns_to_drop)


    # Fill missing categorical values
    df["medical_specialty"] = (
        df["medical_specialty"]
        .fillna("Unknown")
    )

    df["race"] = (
        df["race"]
        .fillna("Unknown")
    )

    print(
        "\nFilled missing categorical values "
        "with 'Unknown'."
    )
    
    # just adding unknown for missing values
    diagnosis_columns = [
        "diag_1",
        "diag_2",
        "diag_3"
    ]

    for column in diagnosis_columns:

        df[column] = (
            df[column]
            .fillna("Unknown")
        )

    print(
        "\nFilled missing diagnosis codes "
        "with 'Unknown'."
    )

    # So without removing just filling with not tested
    df["max_glu_serum"] = (
        df["max_glu_serum"]
        .fillna("Not Tested")
    )

    df["A1Cresult"] = (
        df["A1Cresult"]
        .fillna("Not Tested")
    )

    print(
        "\nFilled missing glucose and A1C results "
        "with 'Not Tested'."
    )


    # Convert readmission target into binary values
    df["readmitted"] = df["readmitted"].replace({
        "<30": 1,
        ">30": 1,
        "NO": 0
    })

    print(
        "\nConverted readmission target."
    )

    print(
        df["readmitted"].value_counts()
    )


    return df


def validate_data(df):
    """
    Validate the transformed healthcare dataset.
    """

    print("\nStarting data validation...")
    print("=" * 60)


    if df.empty:
        raise ValueError(
            "Validation failed: Dataset is empty."
        )

    print("✓ Dataset is not empty.")

    # Checking that we have the required columns
    required_columns = [
        "encounter_id",
        "patient_nbr",
        "age",
        "gender",
        "readmitted"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Validation failed. Missing columns: "
            f"{missing_columns}"
        )

    print("✓ Required columns are present.")

    # Check target Values
    valid_target_values = {0, 1}

    actual_target_values = set(
        df["readmitted"].dropna().unique()
    )

    if not actual_target_values.issubset(
        valid_target_values
    ):
        raise ValueError(
            "Validation failed: "
            "Invalid values found in readmitted column."
        )

    print(
        "✓ Readmission target contains "
        "only valid values: 0 and 1."
    )

 
    #Check missing target values
    missing_target = df["readmitted"].isna().sum()

    if missing_target > 0:
        raise ValueError(
            f"Validation failed: "
            f"{missing_target} missing target values."
        )

    print(
        "✓ No missing values in readmission target."
    )

   
    #Check duplicate encounter IDs
    duplicate_encounters = (
        df["encounter_id"].duplicated().sum()
    )

    print(
        f"Duplicate encounter IDs: "
        f"{duplicate_encounters}"
    )

    
    # Check duplicate patient IDs
    duplicate_patients = (
        df["patient_nbr"].duplicated().sum()
    )

    print(
        f"Repeated patient IDs: "
        f"{duplicate_patients}"
    )

    # Checking the missing values
    total_missing = (
        df.isna().sum().sum()
    )

    print(
        f"Total remaining missing values: "
        f"{total_missing}"
    )

    print(
        f"Final rows: "
        f"{df.shape[0]}"
    )

    print(
        f"Final columns: "
        f"{df.shape[1]}"
    )


def load_data(df):
    """
    Save the processed healthcare dataset.
    """

    print("\nStarting data loading...")

    # Create processed data directory
    PROCESSED_DATA_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save processed dataset
    df.to_csv(
        PROCESSED_DATA_PATH,
        index=False
    )

    print(
        "\n✓ Processed dataset saved successfully."
    )

    print(
        f"Location:\n"
        f"{PROCESSED_DATA_PATH}"
    )


def main():
    print("\nStarting HealthSense AI ETL Pipeline...")

    # Extract
    df = extract_data()

    # Profile
    profile_data(df)

    # Transform
    df_clean = transform_data(df)

    #Validate
    validate_data(df_clean)

    #Load
    load_data(df_clean)

    print("\nETL PIPELINE COMPLETED SUCCESSFULLY")

if __name__ == "__main__":
    main()