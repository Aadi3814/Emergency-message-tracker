import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pytest
from src.priority import get_priority_level, evaluate_review_status, format_prediction_result

def test_priority_mapping():
    assert get_priority_level("Rescue Request") == "CRITICAL"
    assert get_priority_level("Casualties / Injuries") == "CRITICAL"
    assert get_priority_level("Infrastructure Damage") == "HIGH"
    assert get_priority_level("Not Relevant") == "LOW"

def test_review_status():
    assert evaluate_review_status(0.85, threshold=0.60) == "Model Prediction"
    assert evaluate_review_status(0.45, threshold=0.60) == "Human Review Recommended"

def test_format_prediction_result():
    probs = {
        "Rescue Request": 0.90,
        "Infrastructure Damage": 0.05,
        "Not Relevant": 0.03,
        "Casualties / Injuries": 0.02
    }
    res = format_prediction_result("Rescue Request", probs, threshold=0.60)
    assert res["predicted_category"] == "Rescue Request"
    assert res["priority_level"] == "CRITICAL"
    assert res["confidence_score"] == 0.90
    assert res["review_status"] == "Model Prediction"
    assert len(res["top_predictions"]) == 4
