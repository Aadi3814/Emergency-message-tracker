import os
import json
import joblib
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support, confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.data_loader import load_processed_data, load_config
from src.preprocessing import preprocess_dataset

def evaluate_baseline(config_path="config.yaml"):
    config = load_config(config_path)
    b_cfg = config["baseline_model"]
    figures_dir = config["reports"]["figures_dir"]
    os.makedirs(figures_dir, exist_ok=True)
    
    if not os.path.exists(b_cfg["model_path"]) or not os.path.exists(b_cfg["label_encoder_path"]):
        raise FileNotFoundError("Baseline model artifacts not found. Run train_baseline.py first.")
        
    pipeline = joblib.load(b_cfg["model_path"])
    le = joblib.load(b_cfg["label_encoder_path"])
    
    data = load_processed_data(config_path)
    test_df = preprocess_dataset(data["test"], mode="tfidf")
    
    y_test = le.transform(test_df["label"])
    X_test = test_df["clean_text"]
    
    y_pred = pipeline.predict(X_test)
    y_probs = pipeline.predict_proba(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_test, y_pred, average="macro")
    p_w, r_w, f1_w, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted")
    
    # Per-class metrics
    p_cls, r_cls, f1_cls, supp_cls = precision_recall_fscore_support(y_test, y_pred, average=None, labels=range(len(le.classes_)))
    
    per_class_results = {}
    for idx, name in enumerate(le.classes_):
        per_class_results[name] = {
            "precision": float(p_cls[idx]),
            "recall": float(r_cls[idx]),
            "f1_score": float(f1_cls[idx]),
            "support": int(supp_cls[idx])
        }
        
    # Confusion Matrix Plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title("Baseline Model Confusion Matrix (TF-IDF + Logistic Regression)", fontsize=13, fontweight="bold")
    plt.xlabel("Predicted Label", fontsize=11)
    plt.ylabel("Actual Label", fontsize=11)
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    cm_path = os.path.join(figures_dir, "confusion_matrix_baseline.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    
    # Error analysis: collect sample misclassifications
    misclassified = []
    test_df["predicted"] = le.inverse_transform(y_pred)
    test_df["confidence"] = np.max(y_probs, axis=1)
    
    errors = test_df[test_df["label"] != test_df["predicted"]]
    for _, row in errors.head(15).iterrows():
        misclassified.append({
            "text": row["text"],
            "actual": row["label"],
            "predicted": row["predicted"],
            "confidence": float(row["confidence"])
        })
        
    results = {
        "model_name": "Baseline (TF-IDF + Logistic Regression)",
        "accuracy": float(acc),
        "macro_precision": float(p_macro),
        "macro_recall": float(r_macro),
        "macro_f1": float(f1_macro),
        "weighted_f1": float(f1_w),
        "per_class": per_class_results,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_path": cm_path,
        "misclassified_samples": misclassified
    }
    
    return results

def evaluate_transformer(config_path="config.yaml"):
    config = load_config(config_path)
    t_cfg = config["transformer_model"]
    b_cfg = config["baseline_model"]
    figures_dir = config["reports"]["figures_dir"]
    
    save_dir = t_cfg["save_dir"]
    if not os.path.exists(save_dir):
        print(f"Transformer model directory '{save_dir}' does not exist yet. Skipping transformer evaluation.")
        return None
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    le = joblib.load(b_cfg["label_encoder_path"])
    
    data = load_processed_data(config_path)
    test_df = preprocess_dataset(data["test"], mode="transformer")
    
    # On CPU, sample test set for evaluation speed if needed
    if not torch.cuda.is_available() and len(test_df) > 3000:
        print("CPU detected: evaluating transformer on 3000 stratified test samples for speed...")
        sampled_test = []
        for label_val, group in test_df.groupby("label"):
            n_samples = max(1, int(3000 * len(group) / len(test_df)))
            sampled_test.append(group.sample(n=min(len(group), n_samples), random_state=42))
        test_df = pd.concat(sampled_test, ignore_index=True).reset_index(drop=True)
        
    y_test = le.transform(test_df["label"])
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(save_dir)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(save_dir, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(save_dir).to(device)
    model.eval()
    
    preds, probs = [], []
    batch_size = 32
    
    with torch.no_grad():
        for i in range(0, len(test_df), batch_size):
            batch_texts = test_df["clean_text"].iloc[i:i+batch_size].tolist()
            inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
            outputs = model(**inputs)
            logits = outputs.logits
            batch_probs = torch.softmax(logits, dim=-1).cpu().numpy()
            batch_preds = np.argmax(batch_probs, axis=-1)
            
            preds.extend(batch_preds)
            probs.extend(batch_probs)
            
    preds = np.array(preds)
    probs = np.array(probs)
    
    acc = accuracy_score(y_test, preds)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_test, preds, average="macro")
    p_w, r_w, f1_w, _ = precision_recall_fscore_support(y_test, preds, average="weighted")
    
    p_cls, r_cls, f1_cls, supp_cls = precision_recall_fscore_support(y_test, preds, average=None, labels=range(len(le.classes_)))
    
    per_class_results = {}
    for idx, name in enumerate(le.classes_):
        per_class_results[name] = {
            "precision": float(p_cls[idx]),
            "recall": float(r_cls[idx]),
            "f1_score": float(f1_cls[idx]),
            "support": int(supp_cls[idx])
        }
        
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", 
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title("Advanced Model Confusion Matrix (DistilBERT Fine-Tuned)", fontsize=13, fontweight="bold")
    plt.xlabel("Predicted Label", fontsize=11)
    plt.ylabel("Actual Label", fontsize=11)
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    cm_path = os.path.join(figures_dir, "confusion_matrix_transformer.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    
    misclassified = []
    test_df["predicted"] = le.inverse_transform(preds)
    test_df["confidence"] = np.max(probs, axis=1)
    errors = test_df[test_df["label"] != test_df["predicted"]]
    for _, row in errors.head(15).iterrows():
        misclassified.append({
            "text": row["text"],
            "actual": row["label"],
            "predicted": row["predicted"],
            "confidence": float(row["confidence"])
        })
        
    results = {
        "model_name": "Advanced (DistilBERT Fine-Tuned)",
        "accuracy": float(acc),
        "macro_precision": float(p_macro),
        "macro_recall": float(r_macro),
        "macro_f1": float(f1_macro),
        "weighted_f1": float(f1_w),
        "per_class": per_class_results,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_path": cm_path,
        "misclassified_samples": misclassified
    }
    
    return results

def run_evaluation(config_path="config.yaml"):
    config = load_config(config_path)
    res_path = config["reports"]["results_json"]
    figures_dir = config["reports"]["figures_dir"]
    
    print("Evaluating Baseline Model...")
    baseline_res = evaluate_baseline(config_path)
    
    print("Checking/Evaluating Transformer Model...")
    transformer_res = evaluate_transformer(config_path)
    
    combined = {
        "baseline": baseline_res,
        "transformer": transformer_res
    }
    
    with open(res_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2)
    print(f"Saved evaluation results to {res_path}")
    
    # Generate Comparison Plot
    models_to_compare = [baseline_res]
    if transformer_res is not None:
        models_to_compare.append(transformer_res)
        
    comp_df = pd.DataFrame([
        {
            "Model": m["model_name"],
            "Accuracy": m["accuracy"],
            "Macro Precision": m["macro_precision"],
            "Macro Recall": m["macro_recall"],
            "Macro F1": m["macro_f1"],
            "Weighted F1": m["weighted_f1"]
        } for m in models_to_compare
    ])
    
    plt.figure(figsize=(10, 5))
    df_melted = comp_df.melt(id_vars=["Model"], var_name="Metric", value_name="Score")
    ax = sns.barplot(data=df_melted, x="Metric", y="Score", hue="Model", palette="muted")
    plt.title("Model Performance Comparison (Baseline vs Advanced)", fontsize=14, fontweight="bold")
    plt.ylim(0, 1.05)
    
    for p in ax.patches:
        h = p.get_height()
        if h > 0:
            ax.annotate(f"{h:.3f}", (p.get_x() + p.get_width() / 2., h + 0.01),
                        ha='center', va='bottom', fontsize=8)
                        
    plt.tight_layout()
    comp_path = os.path.join(figures_dir, "model_comparison.png")
    plt.savefig(comp_path, dpi=300)
    plt.close()
    print(f"Saved comparison plot to {comp_path}")
    
    return combined

if __name__ == "__main__":
    run_evaluation()
