import os
import torch
import joblib
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, f1_score

from src.data_loader import load_processed_data, load_config
from src.preprocessing import preprocess_dataset

class EmergencyDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.texts)
        
    def __getitem__(self, item):
        text = str(self.texts[item])
        inputs = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        item_dict = {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0)
        }
        if self.labels is not None:
            item_dict["labels"] = torch.tensor(self.labels[item], dtype=torch.long)
        return item_dict

def train_transformer_model(config_path="config.yaml", sample_size=8000 if not torch.cuda.is_available() else None):
    config = load_config(config_path)
    t_cfg = config["transformer_model"]
    b_cfg = config["baseline_model"]
    
    save_dir = t_cfg["save_dir"]
    os.makedirs(save_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device for Transformer training: {device}")
    
    data = load_processed_data(config_path)
    train_df = preprocess_dataset(data["train"], mode="transformer")
    val_df = preprocess_dataset(data["validation"], mode="transformer")
    
    # Load LabelEncoder
    le = joblib.load(b_cfg["label_encoder_path"])
    num_labels = len(le.classes_)
    
    if sample_size and len(train_df) > sample_size:
        print(f"CPU detected: Stratified sampling {sample_size} training examples for efficient fine-tuning...")
        sampled_dfs = []
        for label_val, group in train_df.groupby("label"):
            n_samples = max(1, int(sample_size * len(group) / len(train_df)))
            sampled_dfs.append(group.sample(n=min(len(group), n_samples), random_state=42))
        train_df = pd.concat(sampled_dfs, ignore_index=True).sample(frac=1.0, random_state=42).reset_index(drop=True)
        sampled_val = []
        for label_val, group in val_df.groupby("label"):
            n_samples = max(1, int(2000 * len(group) / len(val_df)))
            sampled_val.append(group.sample(n=min(len(group), n_samples), random_state=42))
        val_df = pd.concat(sampled_val, ignore_index=True).sample(frac=1.0, random_state=42).reset_index(drop=True)
        
    # Prepare labels
    y_train = le.transform(train_df["label"])
    y_val = le.transform(val_df["label"])
    
    model_name = t_cfg["model_name"]
    print(f"Loading pretrained tokenizer & model: {model_name}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=num_labels,
        id2label={i: name for i, name in enumerate(le.classes_)},
        label2id={name: i for i, name in enumerate(le.classes_)}
    ).to(device)
    
    # Datasets & Loaders
    batch_size = t_cfg.get("batch_size", 16)
    max_len = t_cfg.get("max_length", 128)
    
    train_dataset = EmergencyDataset(train_df["clean_text"].values, y_train, tokenizer, max_len=max_len)
    val_dataset = EmergencyDataset(val_df["clean_text"].values, y_val, tokenizer, max_len=max_len)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    epochs = t_cfg.get("epochs", 2)
    optimizer = AdamW(model.parameters(), lr=float(t_cfg.get("learning_rate", 2e-5)))
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    
    best_val_f1 = 0.0
    
    for epoch in range(1, epochs + 1):
        print(f"\n--- Epoch {epoch}/{epochs} ---")
        model.train()
        total_loss = 0.0
        
        for step, batch in enumerate(train_loader, 1):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            if step % 100 == 0 or step == len(train_loader):
                print(f"Batch {step}/{len(train_loader)} - Loss: {loss.item():.4f}")
                
        avg_train_loss = total_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                preds = torch.argmax(logits, dim=-1).cpu().numpy()
                
                val_preds.extend(preds)
                val_targets.extend(labels.cpu().numpy())
                
        val_acc = accuracy_score(val_targets, val_preds)
        val_f1 = f1_score(val_targets, val_preds, average="macro")
        
        print(f"Epoch {epoch} - Avg Train Loss: {avg_train_loss:.4f} | Val Acc: {val_acc:.4f} | Val Macro F1: {val_f1:.4f}")
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
            print(f"Saved best DistilBERT model to {save_dir}")
            
    return model, tokenizer

if __name__ == "__main__":
    train_transformer_model()
