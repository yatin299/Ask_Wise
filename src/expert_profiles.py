"""
expert_profiles.py
------------------
Phase 4: Expert Profile Construction from StackSample Historical Data
Project: Intelligent Question Quality Assessment and Expert Routing System for Q&A Platforms

This module:
1. Loads a manageable subset of StackSample Answers.csv and Questions.csv.
2. Builds per-expert records by mapping: Question -> Answer -> Answerer (OwnerUserId).
3. Cleans the question text (Title + Body) for each answered question.
4. Encodes each question using Sentence-BERT (all-MiniLM-L6-v2).
5. Aggregates each expert's question embeddings into a single mean vector (expert profile vector).
6. Filters experts with a minimum number of answers to ensure meaningful profiles.
7. Saves expert profile vectors, metadata, and question indices to data/embeddings/.

Academic Rationale:
Expert profile vectors represent the centroid of the semantic space covered by an expert's
historical Q&A activity. This is an established approximation in expertise retrieval systems
(see: Zhang et al., "Expert Finding in Community Question Answering", 2007).
For a first prototype, the simple unweighted mean is preferable for interpretability and
reproducibility. A score-weighted mean can be explored as a future extension.
"""

import os
import re
import html
import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
from tqdm import tqdm

# Silence HuggingFace symlink warning on Windows
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

HTML_TAG_REGEX = re.compile(r"<[^>]+>")
WHITESPACE_REGEX = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Strips HTML tags and normalizes whitespace; mirrors data_preparation.py cleaning."""
    if not isinstance(text, str) or not text.strip():
        return ""
    unescaped = html.unescape(text)
    tag_stripped = HTML_TAG_REGEX.sub(" ", unescaped)
    return WHITESPACE_REGEX.sub(" ", tag_stripped).strip()


def load_stacksample_subset(
    raw_dir: str,
    max_answers: int = 200_000,
    min_answer_score: int = 1,
    min_answers_per_expert: int = 5
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads a manageable subset of StackSample Answers.csv and Questions.csv.

    Args:
        raw_dir: Path to data/raw/ containing Answers.csv and Questions_SO.csv.
        max_answers: Maximum number of answer rows to load (for memory efficiency).
        min_answer_score: Minimum answer score to retain (filters out noise).
        min_answers_per_expert: Minimum answers per user to qualify as an 'expert'.

    Returns:
        (answers_df, questions_df): Filtered DataFrames.

    Academic Note:
    Loading the full 1.5 GB Answers.csv requires ~6-8 GB RAM. We instead take the
    first max_answers rows, which is a common subsampling approach in academic prototypes.
    This still yields thousands of experts with hundreds of thousands of answered questions.
    """
    answers_path = os.path.join(raw_dir, "Answers.csv")
    questions_path = os.path.join(raw_dir, "Questions_SO.csv")

    if not os.path.exists(answers_path):
        raise FileNotFoundError(f"Answers.csv not found at {answers_path}")
    if not os.path.exists(questions_path):
        raise FileNotFoundError(f"Questions_SO.csv not found at {questions_path}")

    logger.info(f"Loading up to {max_answers:,} answer rows from Answers.csv...")
    answers_df = pd.read_csv(
        answers_path,
        usecols=["Id", "OwnerUserId", "ParentId", "Score"],
        nrows=max_answers,
        encoding="latin-1"
    )
    logger.info(f"Loaded {len(answers_df):,} answer rows. Columns: {answers_df.columns.tolist()}")

    # Drop rows without a valid user ID (anonymous answers)
    answers_df = answers_df.dropna(subset=["OwnerUserId"]).copy()
    answers_df["OwnerUserId"] = answers_df["OwnerUserId"].astype(int)

    # Filter by score to retain quality signals
    logger.info(f"Filtering answers with Score >= {min_answer_score}...")
    answers_df = answers_df[answers_df["Score"] >= min_answer_score].reset_index(drop=True)
    logger.info(f"Retained {len(answers_df):,} answers after score filter.")

    # Filter users with minimum number of answers
    expert_answer_counts = answers_df["OwnerUserId"].value_counts()
    qualified_experts = expert_answer_counts[expert_answer_counts >= min_answers_per_expert].index
    answers_df = answers_df[answers_df["OwnerUserId"].isin(qualified_experts)].reset_index(drop=True)
    logger.info(f"Retained {len(answers_df):,} answers from {len(qualified_experts):,} qualified experts.")

    # Get the unique question IDs (ParentId in Answers = QuestionId)
    needed_question_ids = set(answers_df["ParentId"].unique())
    logger.info(f"Need to fetch {len(needed_question_ids):,} unique questions from Questions_SO.csv...")

    # Load Questions in chunks to avoid memory overflow
    questions_chunks = []
    chunk_size = 50_000
    for chunk in pd.read_csv(
        questions_path,
        usecols=["Id", "Title", "Body"],
        chunksize=chunk_size,
        encoding="latin-1"
    ):
        matched = chunk[chunk["Id"].isin(needed_question_ids)]
        if len(matched) > 0:
            questions_chunks.append(matched)
        if len(questions_chunks) > 0 and sum(len(c) for c in questions_chunks) >= len(needed_question_ids):
            break

    questions_df = pd.concat(questions_chunks, ignore_index=True) if questions_chunks else pd.DataFrame()
    questions_df = questions_df.drop_duplicates(subset=["Id"]).reset_index(drop=True)
    logger.info(f"Loaded {len(questions_df):,} matched unique questions.")

    return answers_df, questions_df


def build_expert_profiles(
    answers_df: pd.DataFrame,
    questions_df: pd.DataFrame,
    embeddings_dir: str,
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 128
) -> Tuple[np.ndarray, List[Dict]]:
    """
    Constructs expert profile vectors by:
    1. Joining answers to questions (to get the text for each answered question).
    2. Cleaning the combined (title + body) text.
    3. Encoding all distinct question texts via Sentence-BERT.
    4. For each expert, averaging (mean) their answered question embeddings.

    Returns:
        (expert_matrix, expert_metadata): NumPy array of shape (N_experts, 384)
                                          and a list of expert metadata dicts.
    """
    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading Sentence-BERT model '{model_name}' on {device}...")
    model = SentenceTransformer(model_name, device=device)

    # Join answers to questions
    logger.info("Joining answers to questions to build expert -> text mapping...")
    merged = answers_df.merge(
        questions_df,
        left_on="ParentId",
        right_on="Id",
        how="inner",
        suffixes=("_ans", "_q")
    )
    logger.info(f"Merged records: {len(merged):,} (answer-question pairs with text)")

    # Clean combined text
    merged["Title"] = merged["Title"].fillna("").astype(str)
    merged["Body"] = merged["Body"].fillna("").astype(str)
    merged["cleaned_text"] = (merged["Title"].apply(clean_text) + " " + merged["Body"].apply(clean_text)).str.strip()

    # Encode all unique question texts (deduplicated to save compute)
    unique_questions = merged[["ParentId", "cleaned_text"]].drop_duplicates(subset=["ParentId"]).reset_index(drop=True)
    logger.info(f"Encoding {len(unique_questions):,} unique question texts...")
    texts = unique_questions["cleaned_text"].tolist()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True
    ).astype(np.float32)

    # Map ParentId -> embedding
    qid_to_embedding = {
        int(unique_questions.iloc[i]["ParentId"]): embeddings[i]
        for i in range(len(unique_questions))
    }

    # Build per-expert mean profile vector
    logger.info("Aggregating per-expert mean profile vectors...")
    expert_ids = sorted(merged["OwnerUserId"].unique())
    expert_vectors = []
    expert_metadata = []

    for expert_id in tqdm(expert_ids, desc="Building expert profiles"):
        expert_rows = merged[merged["OwnerUserId"] == expert_id]
        q_ids = expert_rows["ParentId"].unique()
        vecs = [qid_to_embedding[qid] for qid in q_ids if qid in qid_to_embedding]
        if len(vecs) == 0:
            continue
        mean_vec = np.mean(vecs, axis=0)
        # Re-normalize the mean vector to unit L2 norm
        norm = np.linalg.norm(mean_vec)
        if norm > 0:
            mean_vec = mean_vec / norm
        expert_vectors.append(mean_vec)
        expert_metadata.append({
            "expert_id": int(expert_id),
            "num_questions_answered": int(len(q_ids)),
            "total_answer_score": int(expert_rows["Score"].sum()),
            "mean_answer_score": float(expert_rows["Score"].mean()),
        })

    expert_matrix = np.stack(expert_vectors, axis=0).astype(np.float32)
    logger.info(f"Built expert profile matrix: shape = {expert_matrix.shape}")
    return expert_matrix, expert_metadata


def save_expert_profiles(
    expert_matrix: np.ndarray,
    expert_metadata: List[Dict],
    embeddings_dir: str
):
    """Saves expert vectors and metadata to data/embeddings/."""
    os.makedirs(embeddings_dir, exist_ok=True)
    vec_path = os.path.join(embeddings_dir, "expert_vectors.npy")
    meta_path = os.path.join(embeddings_dir, "expert_metadata.json")

    np.save(vec_path, expert_matrix)
    with open(meta_path, "w") as f:
        json.dump(expert_metadata, f, indent=2)

    logger.info(f"Saved expert vectors to: {vec_path} (shape={expert_matrix.shape})")
    logger.info(f"Saved expert metadata to: {meta_path} ({len(expert_metadata)} experts)")


if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    raw_dir = os.path.join(base_dir, "data", "raw")
    embed_dir = os.path.join(base_dir, "data", "embeddings")

    logger.info("=== Phase 4: Expert Profile Construction ===")

    answers_df, questions_df = load_stacksample_subset(
        raw_dir,
        max_answers=200_000,
        min_answer_score=1,
        min_answers_per_expert=5
    )

    expert_matrix, expert_metadata = build_expert_profiles(
        answers_df, questions_df, embed_dir, batch_size=128
    )

    save_expert_profiles(expert_matrix, expert_metadata, embed_dir)

    logger.info(f"Phase 4 complete. Total qualified experts: {len(expert_metadata):,}")
    print(f"\nSample expert profiles (first 5):")
    for m in expert_metadata[:5]:
        print(f"  Expert {m['expert_id']:7d} | Answered: {m['num_questions_answered']:5d} | "
              f"Total Score: {m['total_answer_score']:6d} | Mean Score: {m['mean_answer_score']:.1f}")
