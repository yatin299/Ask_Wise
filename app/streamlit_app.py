"""
Streamlit demo UI for the Question Quality Assessment & Expert Routing System.
Run:  streamlit run app/streamlit_app.py
"""
import os
import sys

import pandas as pd
import streamlit as st

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

from inference import InferencePipeline  # noqa: E402

st.set_page_config(page_title="Question Quality & Expert Routing", page_icon="🧠", layout="wide")

st.markdown(
    """
    <style>
    .stApp { background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%); color: #e2e8f0; }
    h1, h2, h3 { color: #f8fafc; }
    .badge { display:inline-block; padding:8px 20px; border-radius:999px; font-weight:700;
             font-size:1.1rem; color:white; }
    .HQ { background: linear-gradient(90deg,#10b981,#059669); }
    .LQ_EDIT { background: linear-gradient(90deg,#f59e0b,#d97706); }
    .LQ_CLOSE { background: linear-gradient(90deg,#ef4444,#b91c1c); }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading SBERT, classifier and expert profiles...")
def load_pipeline():
    return InferencePipeline(
        models_dir=os.path.join(BASE_DIR, "models"),
        embeddings_dir=os.path.join(BASE_DIR, "data", "embeddings"),
    )


EXAMPLES = {
    "— choose an example —": ("", ""),
    "Good question": (
        "How do I fix a 404 in Django REST Framework routers?",
        "I registered my viewsets with router.register in urls.py, but every endpoint returns 404. "
        "DEBUG is True and I included the router urls in the project urls.py. What could cause this?",
    ),
    "Vague question": ("help me with my code plz urgent!!!", "its not working i dont know why please help asap"),
}

st.title("🧠 Question Quality Assessment & Expert Routing")
st.caption("SBERT → MLP quality classifier → cosine-similarity expert matching")

pipeline = load_pipeline()

with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Number of experts (Top-K)", 1, 10, 5)
    ex = st.selectbox("Load example", list(EXAMPLES.keys()))

ex_title, ex_body = EXAMPLES[ex]
title = st.text_input("Question title", value=ex_title, key=f"t_{ex}")
body = st.text_area("Question body", value=ex_body, height=200, key=f"b_{ex}")

if st.button("Analyze question", type="primary"):
    if not title.strip() and not body.strip():
        st.warning("Please enter a title or body.")
    else:
        with st.spinner("Analyzing..."):
            result = pipeline.predict(title, body, top_k=top_k)

        cls = result["predicted_class"]
        st.subheader("Quality prediction")
        st.markdown(f'<span class="badge {cls}">{cls}</span>', unsafe_allow_html=True)

        probs = pd.DataFrame(
            {"Probability": result["class_probabilities"]}
        )
        st.bar_chart(probs)
        st.info(result["quality_feedback"].replace("\n", "  \n"))

        st.subheader("Expert recommendations")
        if not result["should_route"]:
            st.error("Routing skipped: improve the question and resubmit.")
        elif result["expert_recommendations"]:
            df = pd.DataFrame(result["expert_recommendations"])[
                ["rank", "expert_id", "cosine_similarity", "num_questions_answered",
                 "total_answer_score", "mean_answer_score"]
            ]
            df.columns = ["Rank", "Expert ID", "Similarity", "Answers", "Total score", "Mean score"]
            st.dataframe(df, hide_index=True, use_container_width=True)
        else:
            st.warning("No expert profiles available.")
