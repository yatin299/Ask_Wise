"""
inference.py
------------
End-to-End Unified Inference Pipeline
Project: Intelligent Question Quality Assessment and Expert Routing System for Q&A Platforms

This module ties together all components:
1. Text Preprocessing   – cleans raw user input (Title + Body).
2. SBERT Encoding       – encodes cleaned text via all-MiniLM-L6-v2 (384-d vector).
3. Quality Assessment   – predicts quality class (HQ / LQ_EDIT / LQ_CLOSE) via saved MLP.
4. Quality-Based Flow   – determines routing decision based on predicted quality.
5. Expert Routing       – for HQ/LQ_EDIT: computes cosine similarity against expert vectors
                          and returns Top-K expert recommendations.

TRAINING vs INFERENCE SEPARATION:
- The system NEVER retrains when a new user submits a question.
- All model weights (best_quality_classifier.pt) and expert profiles (expert_vectors.npy)
  are loaded once from disk at startup, then reused for all subsequent queries.
- This is the correct production inference paradigm.
"""

import os
import re
import html
import json
import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn

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

INT_TO_LABEL = {0: "HQ", 1: "LQ_EDIT", 2: "LQ_CLOSE"}
LABEL_TO_INT = {v: k for k, v in INT_TO_LABEL.items()}

# Quality-based platform behaviour messages
QUALITY_FEEDBACK = {
    "HQ": (
        "✅ HIGH QUALITY: Your question is well-structured and suitable for expert routing."
    ),
    "LQ_EDIT": (
        "⚠️  LOW QUALITY (Requires Improvement): Your question needs clarification.\n"
        "     Suggestions:\n"
        "       • Add a minimal reproducible example.\n"
        "       • Describe what you have already tried.\n"
        "       • Include any relevant error messages or logs.\n"
        "       • Narrow down the problem scope.\n"
        "     Your question has been conditionally routed. Improving it will increase expert engagement."
    ),
    "LQ_CLOSE": (
        "❌ LOW QUALITY (Substantial Improvement Required): This question requires major revision.\n"
        "     It may be: too broad, unclear, off-topic, or a duplicate.\n"
        "     Recommendations:\n"
        "       • Rewrite the question with a specific, focused problem statement.\n"
        "       • Search for existing answers before posting.\n"
        "       • Expert routing has been skipped for now."
    ),
}


class QuestionQualityClassifier(nn.Module):
    """MLP architecture — must match training definition in quality_model.py exactly."""
    def __init__(self, input_dim=384, hidden_1=256, hidden_2=64, num_classes=3, dropout_p=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_1),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden_1, hidden_2),
            nn.ReLU(),
            nn.Linear(hidden_2, num_classes)
        )

    def forward(self, x):
        return self.net(x)


class InferencePipeline:
    """
    Full end-to-end inference system.

    Usage:
        pipeline = InferencePipeline(models_dir="models/", embeddings_dir="data/embeddings/")
        result = pipeline.predict(title="How do I sort a list in Python?",
                                   body="I have a list of integers...")
        print(result)
    """

    def __init__(
        self,
        models_dir: str,
        embeddings_dir: str,
        model_name: str = "all-MiniLM-L6-v2",
        top_k: int = 5
    ):
        self.top_k = top_k
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 1. Load Sentence-BERT encoder
        from sentence_transformers import SentenceTransformer
        logger.info(f"Loading SBERT model '{model_name}' on {self.device}...")
        self.encoder = SentenceTransformer(model_name, device=str(self.device))

        # 2. Load quality classifier
        classifier_path = os.path.join(models_dir, "best_quality_classifier.pt")
        if not os.path.exists(classifier_path):
            raise FileNotFoundError(f"MLP checkpoint not found: {classifier_path}")
        logger.info(f"Loading quality classifier from {classifier_path}...")
        self.classifier = QuestionQualityClassifier().to(self.device)
        self.classifier.load_state_dict(
            torch.load(classifier_path, map_location=self.device)
        )
        self.classifier.eval()

        # 3. Load expert router (optional — only if expert profiles exist)
        from expert_router import ExpertRouter
        expert_vec_path = os.path.join(embeddings_dir, "expert_vectors.npy")
        if os.path.exists(expert_vec_path):
            logger.info("Loading expert router...")
            self.router = ExpertRouter(embeddings_dir)
        else:
            logger.warning(
                "Expert profiles not found. Run expert_profiles.py first. "
                "Expert routing will be skipped."
            )
            self.router = None

        logger.info("InferencePipeline initialized. System ready for inference.")

    def _clean_text(self, text: str) -> str:
        if not isinstance(text, str) or not text.strip():
            return ""
        unescaped = html.unescape(text)
        tag_stripped = HTML_TAG_REGEX.sub(" ", unescaped)
        return WHITESPACE_REGEX.sub(" ", tag_stripped).strip()

    def predict(
        self,
        title: str,
        body: str = "",
        top_k: Optional[int] = None
    ) -> Dict:
        """
        Runs the full inference pipeline for a new user question.

        Args:
            title: Question title (str).
            body: Question body text (str, may include HTML markup).
            top_k: Override default number of expert recommendations.

        Returns:
            Dict with keys:
              - cleaned_text: str
              - predicted_class: str  ("HQ" / "LQ_EDIT" / "LQ_CLOSE")
              - class_probabilities: Dict[str, float]
              - quality_feedback: str
              - should_route: bool
              - expert_recommendations: List[Dict] (empty if not routed)
        """
        k = top_k if top_k is not None else self.top_k

        # Step 1: Preprocess
        cleaned_title = self._clean_text(title)
        cleaned_body = self._clean_text(body)
        cleaned_text = (cleaned_title + " " + cleaned_body).strip()

        # Step 2: SBERT Encoding (no gradient needed at inference)
        embedding = self.encoder.encode(
            [cleaned_text],
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True
        )[0].astype(np.float32)

        # Step 3: Quality Classification
        x = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.classifier(x)
            probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

        predicted_idx = int(np.argmax(probs))
        predicted_class = INT_TO_LABEL[predicted_idx]
        class_probabilities = {
            "HQ": float(probs[0]),
            "LQ_EDIT": float(probs[1]),
            "LQ_CLOSE": float(probs[2])
        }

        # Step 4: Quality-Based Routing Decision
        should_route = predicted_class in ("HQ", "LQ_EDIT")
        quality_feedback = QUALITY_FEEDBACK[predicted_class]

        # Step 5: Expert Routing (only for routable questions)
        expert_recommendations = []
        if should_route and self.router is not None:
            expert_recommendations = self.router.recommend(embedding, top_k=k)

        return {
            "cleaned_text": cleaned_text,
            "predicted_class": predicted_class,
            "class_probabilities": class_probabilities,
            "quality_feedback": quality_feedback,
            "should_route": should_route,
            "expert_recommendations": expert_recommendations
        }

    def print_result(self, result: Dict):
        """Pretty-prints the full inference result."""
        from expert_router import format_recommendations

        print("\n" + "=" * 65)
        print("         QUESTION QUALITY ASSESSMENT & EXPERT ROUTING")
        print("=" * 65)
        print(f"\nCleaned Input (first 200 chars):\n  {result['cleaned_text'][:200]}...")
        print(f"\nPredicted Quality Class : {result['predicted_class']}")
        print(f"Class Probabilities     : HQ={result['class_probabilities']['HQ']:.3f} | "
              f"LQ_EDIT={result['class_probabilities']['LQ_EDIT']:.3f} | "
              f"LQ_CLOSE={result['class_probabilities']['LQ_CLOSE']:.3f}")
        print(f"\nPlatform Feedback:\n  {result['quality_feedback']}")

        if result["should_route"] and result["expert_recommendations"]:
            print(format_recommendations(result["expert_recommendations"]))
        elif result["should_route"] and not result["expert_recommendations"]:
            print("\n  [Expert routing skipped: expert profiles not built yet]")
        print("=" * 65 + "\n")


# ─────────────────────────────────────────────────────────
# Demo: run this file directly to test the full pipeline
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    models_dir = os.path.join(base_dir, "models")
    embeddings_dir = os.path.join(base_dir, "data", "embeddings")

    # Add src/ to path so expert_router can be imported
    sys.path.insert(0, os.path.dirname(__file__))

    pipeline = InferencePipeline(
        models_dir=models_dir,
        embeddings_dir=embeddings_dir,
        top_k=5
    )

    # ── Test Question 1: Likely High Quality ──────────────────────────────────
    q1_title = "How to implement binary search in Python with recursion?"
    q1_body = """I need to implement a recursive binary search function in Python.
    My list is sorted in ascending order. The function should return the index
    of the target if found, or -1 if not. Here is my current attempt:
    def binary_search(arr, target, low, high):
        if low > high:
            return -1
        mid = (low + high) // 2
        if arr[mid] == target:
            return mid
    But I am not sure how to recurse correctly for left and right halves."""

    result1 = pipeline.predict(title=q1_title, body=q1_body)
    pipeline.print_result(result1)

    # ── Test Question 2: Likely Low Quality (Close) ───────────────────────────
    q2_title = "help me with my code plz urgent!!!"
    q2_body = "its not working i dont know why please help asap"

    result2 = pipeline.predict(title=q2_title, body=q2_body)
    pipeline.print_result(result2)

    # ── Test Question 3: Likely Low Quality (Edit) ────────────────────────────
    q3_title = "Django REST API returns 404 on all my endpoints"
    q3_body = """I built a Django REST Framework API and all my endpoints return 404.
    I set up urls.py with router.register for my viewsets but every request fails.
    I have not changed DEBUG setting. Any ideas what could cause this?"""

    result3 = pipeline.predict(title=q3_title, body=q3_body)
    pipeline.print_result(result3)
