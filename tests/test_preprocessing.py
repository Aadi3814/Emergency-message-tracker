import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pytest
import pandas as pd
from src.preprocessing import clean_text_for_tfidf, clean_text_for_transformer, preprocess_dataset

def test_clean_text_for_tfidf():
    raw = "PLEASE HELP us! 5 people trapped near http://test.com @rescue_team #Emergency"
    cleaned = clean_text_for_tfidf(raw)
    assert "http" not in cleaned
    assert "@rescue_team" not in cleaned
    assert "emergency" in cleaned
    assert "please help" in cleaned

def test_clean_text_for_transformer():
    raw = "URGENT: Bridge collapsed in #Portoviejo http://t.co/123 @news"
    cleaned = clean_text_for_transformer(raw)
    assert "URGENT:" in cleaned
    assert "Portoviejo" in cleaned
    assert "http" not in cleaned

def test_preprocess_dataset():
    df = pd.DataFrame({"text": ["Sample tweet 1", None, "   "]})
    processed = preprocess_dataset(df, mode="tfidf")
    assert len(processed) == 3
    assert "clean_text" in processed.columns
    assert processed["clean_text"].iloc[1] == "no text content"
