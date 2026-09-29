import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, f1_score

from src.data_loader import load_processed_data, load_config
from src.preprocessing import preprocess_dataset

def train_baseline_model(config_path="config.yaml"):
    config = load_config(config_path)
    b_cfg = config["baseline_model"]
    
    models_dir = os.path.dirname(b_cfg["model_path"])
    os.makedirs(models_dir, exist_ok=True)
    
    data = load_processed_data(config_path)
    train_df = preprocess_dataset(data["train"], mode="tfidf")
    val_df = preprocess_dataset(data["validation"], mode="tfidf") if data["validation"] is not None else None
    
    # Encode target labels
    le = LabelEncoder()
    y_train = le.fit_transform(train_df["label"])
    
    # Save label encoder
    joblib.dump(le, b_cfg["label_encoder_path"])
    print(f"Saved LabelEncoder to {b_cfg['label_encoder_path']}")
    print("Classes mapped:", dict(zip(le.classes_, range(len(le.classes_)))))
    
    X_train = train_df["clean_text"]
    
    # Build TF-IDF + Logistic Regression pipeline
    print("Building TF-IDF + Logistic Regression pipeline...")
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=b_cfg.get("max_features", 10000),
            ngram_range=tuple(b_cfg.get("ngram_range", [1, 2])),
            sublinear_tf=b_cfg.get("sublinear_tf", True),
            stop_words='english'
        )),
        ('clf', LogisticRegression(
            C=b_cfg.get("c_param", 1.0),
            max_iter=b_cfg.get("max_iter", 1000),
            class_weight='balanced',
            random_state=config["project"]["random_seed"]
        ))
    ])
    
    print("Training baseline model on train dataset...")
    pipeline.fit(X_train, y_train)
    
    train_preds = pipeline.predict(X_train)
    train_acc = accuracy_score(y_train, train_preds)
    train_f1 = f1_score(y_train, train_preds, average='macro')
    print(f"Train Accuracy: {train_acc:.4f} | Train Macro F1: {train_f1:.4f}")
    
    if val_df is not None:
        y_val = le.transform(val_df["label"])
        X_val = val_df["clean_text"]
        val_preds = pipeline.predict(X_val)
        val_acc = accuracy_score(y_val, val_preds)
        val_f1 = f1_score(y_val, val_preds, average='macro')
        print(f"Validation Accuracy: {val_acc:.4f} | Validation Macro F1: {val_f1:.4f}")
        print("\n--- Validation Classification Report ---")
        print(classification_report(y_val, val_preds, target_names=le.classes_))
        
    # Save pipeline
    joblib.dump(pipeline, b_cfg["model_path"])
    print(f"Saved baseline model to {b_cfg['model_path']}")
    
    return pipeline, le

if __name__ == "__main__":
    train_baseline_model()
