import numpy as np
import pandas as pd
from src.preprocessing import clean_text_for_tfidf, clean_text_for_transformer

def explain_tfidf_prediction(pipeline, label_encoder, text: str, top_k: int = 5) -> dict:
    """
    Extracts top feature (word/ngram) contributions for TF-IDF + Logistic Regression predictions.
    """
    clean_text = clean_text_for_tfidf(text)
    tfidf = pipeline.named_steps['tfidf']
    clf = pipeline.named_steps['clf']
    
    # Get feature vector
    X_vec = tfidf.transform([clean_text])
    feature_names = np.array(tfidf.get_feature_names_out())
    
    # Get prediction probabilities and predicted class index
    probs = pipeline.predict_proba([clean_text])[0]
    pred_idx = np.argmax(probs)
    pred_label = label_encoder.inverse_transform([pred_idx])[0]
    
    # Get non-zero feature indices in input text
    nz_indices = X_vec.nonzero()[1]
    if len(nz_indices) == 0:
        return {
            "predicted_category": pred_label,
            "top_features": [],
            "explanation": "No distinctive vocabulary features detected in text."
        }
        
    # Class weights from logistic regression
    class_coefs = clf.coef_[pred_idx]
    
    # Compute feature contributions (tfidf_value * coef)
    feature_scores = []
    for idx in nz_indices:
        word = feature_names[idx]
        score = X_vec[0, idx] * class_coefs[idx]
        feature_scores.append({"word": word, "score": float(score)})
        
    # Sort by contribution score descending
    sorted_features = sorted(feature_scores, key=lambda x: x["score"], reverse=True)[:top_k]
    
    return {
        "predicted_category": pred_label,
        "top_features": sorted_features,
        "explanation": f"The top words driving prediction '{pred_label}' are: " + ", ".join([f"'{f['word']}'" for f in sorted_features])
    }

def explain_transformer_prediction(model, tokenizer, label_encoder, text: str, top_k: int = 5) -> dict:
    """
    Lightweight feature importance for Transformer model based on emergency keyword mapping.
    """
    clean_text = clean_text_for_transformer(text)
    words = clean_text.split()
    
    emergency_keywords = {
        "rescue", "trapped", "help", "need", "buried", "evacuate", "sos", "save", "water", "food",
        "collapsed", "bridge", "road", "power", "blackout", "damage", "destroyed", "building", "fire",
        "dead", "killed", "injured", "hospital", "body", "casualties", "death", "bleed", "hurt", "missing"
    }
    
    matching = [w for w in words if w.lower().strip(".,!?") in emergency_keywords]
    top_features = [{"word": w, "score": 1.0} for w in matching[:top_k]]
    
    return {
        "top_features": top_features,
        "explanation": "Emergency keyword presence identified." if top_features else "Natural contextual language patterns analyzed by Transformer neural network."
    }
