import os
import sys
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


DATA_PATH = "data/student_placement.csv"
MODEL_PATH = "placement_model.pkl"


def validate_data(data):
    required_columns = [
        "CGPA",
        "Attendance",
        "CodingScore",
        "Projects",
        "Internship",
        "Placement"
    ]

    if data.empty:
        print("ERROR: Dataset is empty")
        sys.exit(1)

    if list(data.columns) != required_columns:
        print("ERROR: Invalid dataset columns")
        sys.exit(1)

    if data.isnull().values.any():
        print("ERROR: Missing values detected")
        sys.exit(1)

    print("Data validation successful")


def train_model():
    data = pd.read_csv(DATA_PATH)

    validate_data(data)

    X = data.drop("Placement", axis=1)
    y = data["Placement"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print("Model Accuracy:", accuracy)

    if accuracy < 0.80:
        print("ERROR: Accuracy is below 80%")
        sys.exit(1)

    joblib.dump(model, MODEL_PATH)

    if not os.path.exists(MODEL_PATH):
        print("ERROR: Model file was not generated")
        sys.exit(1)

    print("Model trained successfully")
    print("Model saved as:", MODEL_PATH)

    return model, accuracy


if __name__ == "__main__":
    train_model()
