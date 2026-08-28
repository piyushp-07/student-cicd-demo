import os
from predict import predict_placement


def test_model_file_exists():
    assert os.path.exists("placement_model.pkl")


def test_high_performing_student():
    result = predict_placement(
        9.0,
        95,
        90,
        4,
        1
    )

    assert result == "PLACED"


def test_low_performing_student():
    result = predict_placement(
        5.8,
        60,
        40,
        1,
        0
    )

    assert result == "NOT PLACED"

    