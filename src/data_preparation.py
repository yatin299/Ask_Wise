"""
data_preparation.py
-------------------
Phase 1: Dataset Preparation, Exploratory Data Analysis (EDA), and Stratified Splitting
Project: Intelligent Question Quality Assessment and Expert Routing System for Q&A Platforms

This script:
1. Loads the 60,000 Stack Overflow Questions with Quality Rating dataset (train.csv + valid.csv).
2. Performs Exploratory Data Analysis (sample counts, class distribution, missing values, text lengths).
3. Preprocesses and cleans text (stripping HTML markup, unescaping HTML entities, whitespace normalization)
   while preserving natural phrasing and technical tokens (C++, C#, .NET, etc.).
4. Combines Title and Body into a unified 'cleaned_text' representation.
5. Performs stratified train (80%), validation (10%), and test (10%) splits with a fixed seed (42).
6. Saves the splits to data/processed/ for downstream model training and evaluation.
"""

import os
import re
import html
import logging
from typing import Tuple, Dict
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Regex pattern for stripping HTML tags cleanly
HTML_TAG_REGEX = re.compile(r"<[^>]+>")
# Whitespace normalization regex
WHITESPACE_REGEX = re.compile(r"\s+")


def clean_html_and_text(text: str) -> str:
    """
    Cleans raw HTML markup from text while preserving technical vocabulary and code tokens.
    
    Academic Rationale:
    Sentence-BERT relies on bidirectional self-attention trained on natural sentence structure.
    Aggressive text reduction (such as removing stopwords or aggressive stemming/lemmatization)
    harms semantic context and breaks technical identifiers (e.g., 'C++', 'in', 'is').
    Therefore, we:
    - Replace HTML entities (e.g., &lt; -> <, &amp; -> &).
    - Strip HTML tags while adding a separating space to prevent word concatenation.
    - Normalize whitespace.
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    
    # 1. Unescape HTML entities (e.g., &lt;pre&gt; -> <pre>, &quot; -> ")
    unescaped = html.unescape(text)
    
    # 2. Replace HTML tags with a single space to avoid merging adjacent tags/tokens
    tag_stripped = HTML_TAG_REGEX.sub(" ", unescaped)
    
    # 3. Normalize multiple whitespace characters (tabs, newlines, multiple spaces) to a single space
    normalized = WHITESPACE_REGEX.sub(" ", tag_stripped).strip()
    
    return normalized


def load_raw_data(raw_data_dir: str) -> pd.DataFrame:
    """
    Loads raw CSV files (train.csv and valid.csv) from the raw data directory
    and concatenates them into a single comprehensive DataFrame.
    """
    train_path = os.path.join(raw_data_dir, "train.csv")
    valid_path = os.path.join(raw_data_dir, "valid.csv")
    
    if not os.path.exists(train_path) or not os.path.exists(valid_path):
        raise FileNotFoundError(
            f"Expected raw files not found in {raw_data_dir}. "
            f"Ensure both 'train.csv' and 'valid.csv' are present."
        )
    
    logger.info(f"Loading raw train set from: {train_path}")
    df_train_raw = pd.read_csv(train_path)
    logger.info(f"Train raw shape: {df_train_raw.shape}")
    
    logger.info(f"Loading raw validation set from: {valid_path}")
    df_valid_raw = pd.read_csv(valid_path)
    logger.info(f"Valid raw shape: {df_valid_raw.shape}")
    
    # Concatenate to reconstruct original 60,000 question corpus
    df_full = pd.concat([df_train_raw, df_valid_raw], ignore_index=True)
    logger.info(f"Total concatenated dataset shape: {df_full.shape}")
    
    return df_full


def perform_eda(df: pd.DataFrame) -> Dict:
    """
    Performs comprehensive Exploratory Data Analysis (EDA) on the raw/intermediate dataset:
    - Total sample count
    - Class distribution and balance
    - Missing values analysis
    - Word count and character length statistics
    """
    print("\n" + "=" * 70)
    print("                  EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 70)
    
    total_samples = len(df)
    print(f"\n1. TOTAL SAMPLE COUNT: {total_samples:,} records")
    print(f"   Columns present: {list(df.columns)}")
    
    # Check for missing values
    print("\n2. MISSING VALUES PER COLUMN:")
    null_counts = df.isnull().sum()
    for col, count in null_counts.items():
        pct = (count / total_samples) * 100
        print(f"   - {col:15s}: {count:,} missing ({pct:.2f}%)")
    
    # Class distribution analysis
    print("\n3. TARGET CLASS DISTRIBUTION (Y):")
    class_counts = df["Y"].value_counts()
    class_percentages = df["Y"].value_counts(normalize=True) * 100
    for cls in class_counts.index:
        print(f"   - {cls:10s}: {class_counts[cls]:,d} samples ({class_percentages[cls]:.2f}%)")
        
    # Text lengths
    title_char_lens = df["Title"].astype(str).str.len()
    title_word_lens = df["Title"].astype(str).str.split().str.len()
    body_char_lens = df["Body"].astype(str).str.len()
    body_word_lens = df["Body"].astype(str).str.split().str.len()
    
    print("\n4. RAW TEXT LENGTH DISTRIBUTION SUMMARY:")
    print("   Title Lengths (words):")
    print(f"     Mean: {title_word_lens.mean():.1f} | Median: {title_word_lens.median():.0f} | Min: {title_word_lens.min()} | Max: {title_word_lens.max()}")
    print("   Body Lengths (words - including HTML tags):")
    print(f"     Mean: {body_word_lens.mean():.1f} | Median: {body_word_lens.median():.0f} | Min: {body_word_lens.min()} | Max: {body_word_lens.max()}")
    print("=" * 70 + "\n")
    
    return {
        "total_samples": total_samples,
        "null_counts": null_counts.to_dict(),
        "class_distribution": class_counts.to_dict(),
    }


def clean_and_prepare_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans Title and Body fields, combines them into 'cleaned_text', and filters invalid entries.
    """
    logger.info("Starting text cleaning and HTML tag removal...")
    
    # Ensure Title and Body are strings
    df["Title"] = df["Title"].fillna("").astype(str)
    df["Body"] = df["Body"].fillna("").astype(str)
    
    # Clean Title (normalize whitespace, unescape entities)
    df["cleaned_title"] = df["Title"].apply(clean_html_and_text)
    
    # Clean Body (strip HTML markup, unescape entities, normalize whitespace)
    df["cleaned_body"] = df["Body"].apply(clean_html_and_text)
    
    # Combine Title and Body: title provides high-level intent, body provides details/context
    df["cleaned_text"] = df["cleaned_title"] + " " + df["cleaned_body"]
    
    # Clean whitespace in combined text
    df["cleaned_text"] = df["cleaned_text"].str.strip()
    
    # Check for empty or malformed cleaned text
    empty_mask = df["cleaned_text"].str.len() == 0
    empty_count = empty_mask.sum()
    if empty_count > 0:
        logger.warning(f"Found {empty_count} samples with empty cleaned_text. Dropping them.")
        df = df[~empty_mask].copy()
    
    # Calculate post-cleaning word length statistics
    cleaned_word_lens = df["cleaned_text"].str.split().str.len()
    print("\n5. POST-CLEANING COMBINED TEXT LENGTH DISTRIBUTION (WORDS):")
    print(f"   Mean: {cleaned_word_lens.mean():.1f} | Median: {cleaned_word_lens.median():.0f} | "
          f"Min: {cleaned_word_lens.min()} | Max: {cleaned_word_lens.max()} | "
          f"25th Pct: {cleaned_word_lens.quantile(0.25):.0f} | 75th Pct: {cleaned_word_lens.quantile(0.75):.0f} | 95th Pct: {cleaned_word_lens.quantile(0.95):.0f}")
    
    return df


def create_stratified_splits(
    df: pd.DataFrame,
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits the dataset into Train (80%), Validation (10%), and Test (10%) sets
    using stratified sampling on label 'Y' to guarantee identical class proportions.
    
    Academic Rationale:
    - Stratified splitting ensures that each split retains the exact class distribution
      of the ground truth dataset, preventing evaluation bias.
    - The test set is isolated completely and never seen during training or tuning.
    - Fixed random_state (42) provides strict experimental reproducibility.
    """
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    logger.info(f"Performing stratified split ({int(train_ratio*100)}% Train, "
                f"{int(val_ratio*100)}% Validation, {int(test_ratio*100)}% Test) with seed {random_state}...")
    
    # Step 1: Separate Test set (10% of total)
    df_train_val, df_test = train_test_split(
        df,
        test_size=test_ratio,
        stratify=df["Y"],
        random_state=random_state
    )
    
    # Step 2: From remaining 90%, separate Validation set (10% of total => 0.10 / 0.90 ≈ 0.1111)
    relative_val_size = val_ratio / (train_ratio + val_ratio)
    df_train, df_val = train_test_split(
        df_train_val,
        test_size=relative_val_size,
        stratify=df_train_val["Y"],
        random_state=random_state
    )
    
    return df_train, df_val, df_test


def print_split_distributions(df_train: pd.DataFrame, df_val: pd.DataFrame, df_test: pd.DataFrame):
    """
    Prints the class distributions across Train, Validation, and Test sets.
    """
    print("\n" + "=" * 70)
    print("               STRATIFIED SPLIT DISTRIBUTION VERIFICATION")
    print("=" * 70)
    
    splits = [("TRAIN (80%)", df_train), ("VALIDATION (10%)", df_val), ("TEST (10%)", df_test)]
    
    for name, split_df in splits:
        total = len(split_df)
        print(f"\n{name} -- Total: {total:,d} samples")
        counts = split_df["Y"].value_counts()
        pcts = split_df["Y"].value_counts(normalize=True) * 100
        for cls in sorted(counts.index):
            print(f"   - {cls:10s}: {counts[cls]:,d} ({pcts[cls]:.2f}%)")
    
    print("=" * 70 + "\n")


def save_processed_splits(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    output_dir: str
):
    """
    Saves the cleaned splits as CSV files in the processed data directory.
    Retains all essential columns: Id, Title, Body, cleaned_text, Tags, CreationDate, Y.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    output_columns = ["Id", "Title", "Body", "cleaned_text", "Tags", "CreationDate", "Y"]
    
    # Filter columns to only those present in the dataframe
    cols_to_save = [c for c in output_columns if c in df_train.columns]
    
    train_path = os.path.join(output_dir, "train.csv")
    val_path = os.path.join(output_dir, "validation.csv")
    test_path = os.path.join(output_dir, "test.csv")
    
    logger.info(f"Saving Train split to: {train_path}")
    df_train[cols_to_save].to_csv(train_path, index=False)
    
    logger.info(f"Saving Validation split to: {val_path}")
    df_val[cols_to_save].to_csv(val_path, index=False)
    
    logger.info(f"Saving Test split to: {test_path}")
    df_test[cols_to_save].to_csv(test_path, index=False)
    
    logger.info("All processed splits saved successfully.")


def main():
    # Resolve directory paths relative to the project root
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    raw_dir = os.path.join(base_dir, "data", "raw")
    processed_dir = os.path.join(base_dir, "data", "processed")
    
    logger.info(f"Project base directory: {base_dir}")
    logger.info(f"Raw data directory: {raw_dir}")
    logger.info(f"Processed data output directory: {processed_dir}")
    
    # 1. Load raw data
    df_full = load_raw_data(raw_dir)
    
    # 2. Perform Exploratory Data Analysis (EDA) on raw data
    perform_eda(df_full)
    
    # 3. Clean and prepare combined text
    df_cleaned = clean_and_prepare_dataset(df_full)
    
    # 4. Create Stratified Splits (80/10/10)
    df_train, df_val, df_test = create_stratified_splits(
        df_cleaned,
        train_ratio=0.80,
        val_ratio=0.10,
        test_ratio=0.10,
        random_state=42
    )
    
    # 5. Print Split Distributions
    print_split_distributions(df_train, df_val, df_test)
    
    # 6. Save splits to data/processed/
    save_processed_splits(df_train, df_val, df_test, processed_dir)
    
    logger.info("Phase 1: Dataset preparation and inspection completed successfully!")


if __name__ == "__main__":
    main()
