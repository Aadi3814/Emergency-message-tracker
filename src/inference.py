import os
import joblib
import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.data_loader import load_config
from src.preprocessing import clean_text_for_tfidf, clean_text_for_transformer
from src.priority import format_prediction_result
from src.explainability import explain_tfidf_prediction, explain_transformer_prediction

class EmergencyInferenceEngine:
    def __init__(self, config_path="config.yaml"):
        self.config = load_config(config_path)
        self.b_cfg = self.config["baseline_model"]
        self.t_cfg = self.config["transformer_model"]
        
        # Load label encoder
        if os.path.exists(self.b_cfg["label_encoder_path"]):
            self.le = joblib.load(self.b_cfg["label_encoder_path"])
        else:
            self.le = None
            
        # Load baseline model
        if os.path.exists(self.b_cfg["model_path"]):
            self.baseline_pipeline = joblib.load(self.b_cfg["model_path"])
        else:
            self.baseline_pipeline = None
            
        # Load transformer model if available
        self.transformer_save_dir = self.t_cfg["save_dir"]
        if os.path.exists(self.transformer_save_dir):
            try:
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                try:
                    self.t_tokenizer = AutoTokenizer.from_pretrained(self.transformer_save_dir)
                except Exception:
                    self.t_tokenizer = AutoTokenizer.from_pretrained(self.transformer_save_dir, use_fast=False)
                self.t_model = AutoModelForSequenceClassification.from_pretrained(self.transformer_save_dir).to(self.device)
                self.t_model.eval()
            except Exception as e:
                print(f"Warning: Could not load transformer model: {e}")
                self.t_model = None
        else:
            self.t_model = None

    def predict_single(self, text: str, model_type: str = "baseline", threshold: float = 0.60) -> dict:
        """
        Predicts single message category, confidence, priority, and explanation.
        """
        if not isinstance(text, str) or not text.strip():
            return {
                "error": "Empty or invalid message text.",
                "predicted_category": "Not Relevant",
                "priority_level": "LOW",
                "confidence_score": 0.0,
                "confidence_percentage": "0.0%",
                "review_status": "Human Review Recommended",
                "top_predictions": [],
                "explanation": "No text provided for analysis."
            }
            
        if model_type == "transformer" and self.t_model is not None:
            clean_text = clean_text_for_transformer(text)
            inputs = self.t_tokenizer(clean_text, padding=True, truncation=True, max_length=128, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.t_model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
                
            pred_idx = np.argmax(probs)
            pred_category = self.le.inverse_transform([pred_idx])[0]
            prob_dict = {cat: float(probs[i]) for i, cat in enumerate(self.le.classes_)}
            
            result = format_prediction_result(pred_category, prob_dict, threshold=threshold)
            exp_info = explain_transformer_prediction(self.t_model, self.t_tokenizer, self.le, text)
            result["explanation"] = exp_info["explanation"]
            result["top_features"] = exp_info["top_features"]
            result["model_used"] = "DistilBERT Fine-Tuned"
            return result
            
        else:
            # Baseline TF-IDF + Logistic Regression fallback
            if self.baseline_pipeline is None:
                raise RuntimeError("Baseline model is not loaded. Train the baseline model first.")
                
            clean_text = clean_text_for_tfidf(text)
            probs = self.baseline_pipeline.predict_proba([clean_text])[0]
            pred_idx = np.argmax(probs)
            pred_category = self.le.inverse_transform([pred_idx])[0]
            prob_dict = {cat: float(probs[i]) for i, cat in enumerate(self.le.classes_)}
            
            result = format_prediction_result(pred_category, prob_dict, threshold=threshold)
            exp_info = explain_tfidf_prediction(self.baseline_pipeline, self.le, text)
            result["explanation"] = exp_info["explanation"]
            result["top_features"] = exp_info["top_features"]
            result["model_used"] = "TF-IDF + Logistic Regression"
            return result

    def predict_batch(self, df: pd.DataFrame, text_col: str = "text", model_type: str = "baseline", threshold: float = 0.60) -> pd.DataFrame:
        """
        Safely processes batch dataframe for CSV upload in Streamlit dashboard.
        """
        df_res = df.copy()
        if text_col not in df_res.columns:
            raise KeyError(f"Missing required text column '{text_col}' in uploaded file.")
            
        categories = []
        priorities = []
        confidences = []
        statuses = []
        
        for idx, row in df_res.iterrows():
            msg_text = row[text_col]
            try:
                res = self.predict_single(str(msg_text), model_type=model_type, threshold=threshold)
                categories.append(res["predicted_category"])
                priorities.append(res["priority_level"])
                confidences.append(res["confidence_percentage"])
                statuses.append(res["review_status"])
            except Exception as e:
                categories.append("Not Relevant")
                priorities.append("LOW")
                confidences.append("0.0%")
                statuses.append("Error Processing Row")
                
        df_res["Predicted Category"] = categories
        df_res["Priority Level"] = priorities
        df_res["Confidence"] = confidences
        df_res["Review Status"] = statuses
        
        return df_res

if __name__ == "__main__":
    engine = EmergencyInferenceEngine()
    test_msg = "PLEASE HELP US! 5 people trapped under collapsed building in Portoviejo, need rescue urgently!!!"
    print("Test Message:", test_msg)
    res = engine.predict_single(test_msg, model_type="baseline")
    print("\nInference Output:")
    print(res)
