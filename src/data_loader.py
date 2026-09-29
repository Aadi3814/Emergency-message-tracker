import os
import yaml
import pandas as pd
from datasets import load_dataset, VerificationMode

def load_config(config_path="config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def download_and_process_humaid(config_path="config.yaml"):
    config = load_config(config_path)
    dataset_cfg = config["dataset"]
    
    raw_dir = dataset_cfg["raw_dir"]
    processed_dir = dataset_cfg["processed_dir"]
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    
    print(f"Loading HF dataset '{dataset_cfg['hf_dataset_name']}'...")
    ds = load_dataset(dataset_cfg["hf_dataset_name"], verification_mode=VerificationMode.NO_CHECKS)
    
    label_map = dataset_cfg["label_mapping"]
    
    processed_data = {}
    
    for split_name in ds.keys():
        df = pd.DataFrame(ds[split_name])
        
        # Save raw data
        raw_path = os.path.join(raw_dir, f"{split_name}_raw.csv")
        df.to_csv(raw_path, index=False)
        print(f"Saved raw split '{split_name}' ({len(df)} rows) to {raw_path}")
        
        # Standardize columns
        text_col = dataset_cfg["text_column"]
        label_col = dataset_cfg["label_column"]
        
        df_proc = pd.DataFrame()
        df_proc["text"] = df[text_col].astype(str)
        df_proc["original_label"] = df[label_col].astype(str)
        
        # Map label
        df_proc["label"] = df_proc["original_label"].map(label_map)
        
        # Handle any unmapped labels gracefully
        unmapped = df_proc["label"].isna()
        if unmapped.any():
            print(f"Warning: {unmapped.sum()} unmapped labels found in split '{split_name}'. Setting to 'Not Relevant'.")
            df_proc["label"] = df_proc["label"].fillna("Not Relevant")
            
        processed_path = os.path.join(processed_dir, f"{split_name}.csv")
        df_proc.to_csv(processed_path, index=False)
        print(f"Saved processed split '{split_name}' ({len(df_proc)} rows) to {processed_path}")
        
        processed_data[split_name] = df_proc
        
    return processed_data

def load_processed_data(config_path="config.yaml"):
    config = load_config(config_path)
    proc_dir = config["dataset"]["processed_dir"]
    
    train_path = os.path.join(proc_dir, "train.csv")
    val_path = os.path.join(proc_dir, "validation.csv")
    test_path = os.path.join(proc_dir, "test.csv")
    
    if not (os.path.exists(train_path) and os.path.exists(test_path)):
        print("Processed data not found locally. Running download and processing pipeline...")
        return download_and_process_humaid(config_path)
        
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path) if os.path.exists(val_path) else None
    test_df = pd.read_csv(test_path)
    
    return {"train": train_df, "validation": val_df, "test": test_df}

def get_dataset_summary(data_dict):
    summary = {}
    for split_name, df in data_dict.items():
        if df is not None:
            summary[split_name] = {
                "total_rows": len(df),
                "label_counts": df["label"].value_counts().to_dict(),
                "missing_text_count": df["text"].isna().sum(),
                "duplicate_text_count": df["text"].duplicated().sum()
            }
    return summary

if __name__ == "__main__":
    data = download_and_process_humaid()
    summary = get_dataset_summary(data)
    print("\n--- DATASET SUMMARY ---")
    for split, info in summary.items():
        print(f"\n[{split.upper()}] Total: {info['total_rows']}")
        print(" Label Distribution:")
        for lbl, cnt in info['label_counts'].items():
            print(f"   - {lbl}: {cnt}")
