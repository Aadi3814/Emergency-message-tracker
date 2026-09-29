import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pytest
import pandas as pd
from src.inference import EmergencyInferenceEngine

@pytest.fixture
def engine():
    return EmergencyInferenceEngine()

def test_single_prediction_valid(engine):
    text = "People trapped inside collapsed building after earthquake, need immediate rescue help."
    res = engine.predict_single(text, model_type="baseline")
    assert "predicted_category" in res
    assert "priority_level" in res
    assert "confidence_score" in res
    assert "review_status" in res
    assert "explanation" in res
    assert res["predicted_category"] in ["Rescue Request", "Infrastructure Damage", "Casualties / Injuries", "Not Relevant"]

def test_single_prediction_empty(engine):
    res = engine.predict_single("   ", model_type="baseline")
    assert res["predicted_category"] == "Not Relevant"
    assert res["priority_level"] == "LOW"
    assert res["review_status"] == "Human Review Recommended"

def test_batch_prediction(engine):
    df = pd.DataFrame({"text": [
        "Need urgent medical help for injured victims",
        "Road blocked due to fallen trees and power lines",
        "Just watching news about the earthquake"
    ]})
    res_df = engine.predict_batch(df, text_col="text", model_type="baseline")
    assert "Predicted Category" in res_df.columns
    assert "Priority Level" in res_df.columns
    assert "Confidence" in res_df.columns
    assert "Review Status" in res_df.columns
    assert len(res_df) == 3
