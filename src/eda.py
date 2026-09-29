import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer

from src.data_loader import load_processed_data, load_config
from src.preprocessing import clean_text_for_tfidf

def run_eda(config_path="config.yaml"):
    config = load_config(config_path)
    figures_dir = config["reports"]["figures_dir"]
    os.makedirs(figures_dir, exist_ok=True)
    
    data = load_processed_data(config_path)
    train_df = data["train"]
    val_df = data["validation"]
    test_df = data["test"]
    
    combined_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    
    # Preprocess for length analysis
    combined_df["clean_text"] = combined_df["text"].apply(clean_text_for_tfidf)
    combined_df["word_count"] = combined_df["clean_text"].apply(lambda x: len(x.split()))
    combined_df["char_count"] = combined_df["clean_text"].apply(len)
    
    # Statistics dict
    eda_stats = {
        "total_messages": len(combined_df),
        "split_counts": {
            "train": len(train_df),
            "validation": len(val_df) if val_df is not None else 0,
            "test": len(test_df)
        },
        "num_classes": combined_df["label"].nunique(),
        "class_distribution": combined_df["label"].value_counts().to_dict(),
        "missing_text": int(combined_df["text"].isna().sum()),
        "duplicate_messages": int(combined_df["text"].duplicated().sum()),
        "message_length": {
            "avg_words": float(combined_df["word_count"].mean()),
            "median_words": float(combined_df["word_count"].median()),
            "max_words": int(combined_df["word_count"].max()),
            "avg_chars": float(combined_df["char_count"].mean())
        }
    }
    
    # 1. Class Distribution Plot
    plt.figure(figsize=(10, 6))
    palette = sns.color_palette("viridis", n_colors=combined_df["label"].nunique())
    ax = sns.countplot(
        data=combined_df,
        y="label",
        hue="label",
        legend=False,
        order=combined_df["label"].value_counts().index,
        palette=palette
    )
    plt.title("Class Distribution in HumAID Normalized Emergency Dataset", fontsize=14, fontweight="bold")
    plt.xlabel("Count", fontsize=12)
    plt.ylabel("Category", fontsize=12)
    
    # Annotate bars
    total = len(combined_df)
    for p in ax.patches:
        percentage = f"{100 * p.get_width() / total:.1f}%"
        x = p.get_x() + p.get_width() + total * 0.005
        y = p.get_y() + p.get_height() / 2
        ax.annotate(f"{int(p.get_width()):,} ({percentage})", (x, y), ha="left", va="center", fontsize=10)
        
    plt.tight_layout()
    class_dist_path = os.path.join(figures_dir, "class_distribution.png")
    plt.savefig(class_dist_path, dpi=300)
    plt.close()
    print(f"Saved: {class_dist_path}")
    
    # 2. Message Length Distribution Plot
    plt.figure(figsize=(10, 5))
    sns.histplot(data=combined_df, x="word_count", hue="label", kde=True, bins=30, element="step")
    plt.title("Message Word Count Distribution by Category", fontsize=14, fontweight="bold")
    plt.xlabel("Word Count", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.tight_layout()
    len_dist_path = os.path.join(figures_dir, "message_length_dist.png")
    plt.savefig(len_dist_path, dpi=300)
    plt.close()
    print(f"Saved: {len_dist_path}")
    
    # 3. Top Words by Category
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    categories = sorted(combined_df["label"].unique())
    
    cv = CountVectorizer(stop_words="english", max_features=10)
    
    for idx, cat in enumerate(categories):
        cat_texts = combined_df[combined_df["label"] == cat]["clean_text"]
        dtm = cv.fit_transform(cat_texts)
        words = cv.get_feature_names_out()
        counts = dtm.toarray().sum(axis=0)
        
        word_df = pd.DataFrame({"word": words, "count": counts}).sort_values("count", ascending=True)
        
        ax = axes[idx]
        ax.barh(word_df["word"], word_df["count"], color=sns.color_palette("muted")[idx])
        ax.set_title(f"Top 10 Terms: {cat}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Frequency")
        
    plt.suptitle("Most Frequent Words per Category", fontsize=16, fontweight="bold")
    plt.tight_layout()
    words_path = os.path.join(figures_dir, "most_frequent_words.png")
    plt.savefig(words_path, dpi=300)
    plt.close()
    print(f"Saved: {words_path}")
    
    # Save eda stats
    eda_json_path = os.path.join(figures_dir, "eda_summary.json")
    with open(eda_json_path, "w", encoding="utf-8") as f:
        json.dump(eda_stats, f, indent=2)
        
    print(f"Saved EDA stats to {eda_json_path}")
    return eda_stats

if __name__ == "__main__":
    stats = run_eda()
    print(json.dumps(stats, indent=2))
