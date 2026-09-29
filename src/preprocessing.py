import re
import html
import pandas as pd

def clean_text_for_tfidf(text: str) -> str:
    """
    NLP preprocessing tailored for classical TF-IDF models.
    Cleans noise while preserving emergency sentiment indicators.
    """
    if not isinstance(text, str):
        return ""
        
    # Unescape HTML entities (e.g. &amp; -> &, &lt; -> <)
    text = html.unescape(text)
    
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Remove twitter handles / mentions (@user)
    text = re.sub(r'@\w+', '', text)
    
    # Standardize hashtags (convert #EcuadorEarthquake -> EcuadorEarthquake)
    text = re.sub(r'#(\w+)', r'\1', text)
    
    # Normalize repeated punctuation while keeping urgency (!!! -> !)
    text = re.sub(r'!{2,}', ' ! ', text)
    text = re.sub(r'\?{2,}', ' ? ', text)
    
    # Lowercase
    text = text.lower()
    
    # Remove special non-alphanumeric characters except basic punctuation
    text = re.sub(r'[^a-z0-9\s!\?\.\,]', ' ', text)
    
    # Collapse multiple whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def clean_text_for_transformer(text: str) -> str:
    """
    Minimal NLP cleaning for Transformer models (e.g. DistilBERT),
    preserving natural language capitalization, punctuation, and structure.
    """
    if not isinstance(text, str):
        return ""
        
    # Unescape HTML entities
    text = html.unescape(text)
    
    # Remove URLs as they don't carry semantic disaster urgency
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Remove mentions
    text = re.sub(r'@\w+', '', text)
    
    # Strip hashtag symbols
    text = re.sub(r'#(\w+)', r'\1', text)
    
    # Collapse extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def preprocess_dataset(df: pd.DataFrame, text_col: str = "text", mode: str = "tfidf") -> pd.DataFrame:
    """
    Applies preprocessing to a dataframe. Handles missing values and empty strings.
    """
    df = df.copy()
    
    # Fill missing text
    df[text_col] = df[text_col].fillna("")
    
    clean_fn = clean_text_for_tfidf if mode == "tfidf" else clean_text_for_transformer
    
    df["clean_text"] = df[text_col].apply(clean_fn)
    
    # Replace empty clean_text with placeholder if necessary
    empty_mask = df["clean_text"].str.strip() == ""
    if empty_mask.any():
        df.loc[empty_mask, "clean_text"] = "no text content"
        
    return df

if __name__ == "__main__":
    sample = "RT @emergency_help: PLEASE HELP!!! 3 people trapped in collapsed building near http://example.com #EcuadorEarthquake"
    print("Original:", sample)
    print("TF-IDF Cleaned:", clean_text_for_tfidf(sample))
    print("Transformer Cleaned:", clean_text_for_transformer(sample))
