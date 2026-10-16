"""
expert_router.py
----------------
Phase 5: Expert Routing via Cosine Similarity
Project: Intelligent Question Quality Assessment and Expert Routing System for Q&A Platforms

This module:
1. Loads precomputed expert profile vectors and metadata.
2. For a given question embedding (384-d SBERT vector), computes cosine similarity
   against all expert vectors using an efficient vectorised NumPy operation.
3. Returns the Top-K ranked experts with similarity scores and metadata.

Academic Rationale:
Cosine similarity is the standard metric for comparing unit-normalised dense embeddings.
Since both the question vector and expert profile vectors are L2-normalised, the cosine
similarity equals the dot product, enabling a single matrix multiplication.
This is computationally efficient and exact (no approximation), making FAISS or HNSW
unnecessary until the expert database scales to millions of profiles.
"""

import os
import json
import logging
import numpy as np
from typing import List, Dict, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


class ExpertRouter:
    """
    Recommends the Top-K domain experts for a given question embedding.

    The expert matching pipeline operates as follows:
        1. New question text -> Sentence-BERT -> 384-d L2-normalised vector q.
        2. Expert matrix E of shape (N_experts, 384), each row L2-normalised.
        3. Cosine similarity scores s = E @ q  (matrix-vector product; shape: N_experts).
        4. Sort descending; return Top-K expert metadata with scores.
    """

    def __init__(self, embeddings_dir: str):
        """
        Loads expert vectors and metadata from disk.

        Args:
            embeddings_dir: Path to data/embeddings/ containing expert_vectors.npy and
                            expert_metadata.json.
        """
        vec_path = os.path.join(embeddings_dir, "expert_vectors.npy")
        meta_path = os.path.join(embeddings_dir, "expert_metadata.json")

        if not os.path.exists(vec_path) or not os.path.exists(meta_path):
            raise FileNotFoundError(
                f"Expert profiles not found. Run expert_profiles.py first.\n"
                f"Expected: {vec_path}\n         {meta_path}"
            )

        logger.info(f"Loading expert profile vectors from {vec_path}...")
        self.expert_matrix = np.load(vec_path).astype(np.float32)

        with open(meta_path, "r") as f:
            self.expert_metadata = json.load(f)

        assert len(self.expert_matrix) == len(self.expert_metadata), (
            "Mismatch between expert matrix rows and metadata entries!"
        )
        logger.info(
            f"Expert router loaded: {len(self.expert_metadata):,} experts, "
            f"embedding dim = {self.expert_matrix.shape[1]}"
        )

    def recommend(
        self,
        question_embedding: np.ndarray,
        top_k: int = 5
    ) -> List[Dict]:
        """
        Recommends Top-K experts for a question embedding.

        Args:
            question_embedding: 384-d L2-normalised vector (shape: (384,) or (1, 384)).
            top_k: Number of experts to recommend.

        Returns:
            List of dicts sorted by descending cosine similarity, each containing:
              - rank: int (1 = best match)
              - cosine_similarity: float
              - expert_id: int
              - num_questions_answered: int
              - total_answer_score: int
              - mean_answer_score: float
        """
        q = np.array(question_embedding).flatten().astype(np.float32)

        # Ensure unit norm (defensive normalisation in case caller forgot)
        norm = np.linalg.norm(q)
        if norm > 0:
            q = q / norm

        # Cosine similarity as dot product (both sides are unit-normalised)
        similarities = self.expert_matrix @ q  # shape: (N_experts,)

        # Get Top-K indices in descending order
        top_k = min(top_k, len(similarities))
        top_indices = np.argpartition(similarities, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]

        results = []
        for rank, idx in enumerate(top_indices, start=1):
            expert = dict(self.expert_metadata[idx])
            expert["rank"] = rank
            expert["cosine_similarity"] = float(similarities[idx])
            results.append(expert)

        return results

    def batch_recommend(
        self,
        question_embeddings: np.ndarray,
        top_k: int = 5
    ) -> List[List[Dict]]:
        """
        Recommends Top-K experts for a batch of question embeddings.

        Args:
            question_embeddings: shape (N, 384).
            top_k: Number of experts per question.

        Returns:
            List of N recommendation lists.
        """
        return [self.recommend(q, top_k=top_k) for q in question_embeddings]


def format_recommendations(recommendations: List[Dict]) -> str:
    """Pretty-prints the Top-K expert recommendations."""
    lines = ["\n" + "=" * 65]
    lines.append("                 TOP-K EXPERT RECOMMENDATIONS")
    lines.append("=" * 65)
    for rec in recommendations:
        lines.append(
            f"  Rank #{rec['rank']:1d} | Expert ID: {rec['expert_id']:8d} | "
            f"Similarity: {rec['cosine_similarity']:.4f} | "
            f"Answered: {rec['num_questions_answered']:5d} Qs | "
            f"Total Score: {rec['total_answer_score']:7d}"
        )
    lines.append("=" * 65)
    return "\n".join(lines)
