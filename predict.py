import joblib


MODEL_PATH = "placement_model.pkl"


def predict_placement(
    cgpa,
    attendance,
    coding_score,
    projects,
    internship
):
    model = joblib.load(MODEL_PATH)

    student_data = [[
        cgpa,
        attendance,
        coding_score,
        projects,
        internship
    ]]

    prediction = model.predict(student_data)[0]

    if prediction == 1:
        return "PLACED"
    else:
        return "NOT PLACED"


if __name__ == "__main__":

    result = predict_placement(
        cgpa=8.5,
        attendance=92,
        coding_score=85,
        projects=3,
        internship=1
    )

    print("Prediction:", result)

    