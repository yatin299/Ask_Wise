# AskWise — Intelligent Question Quality Assessment & Expert Routing

Neural Networks capstone project for technical Q&A platforms (such as Stack Overflow). It does three things:

1. **Assesses question quality** — classifies a question as `HQ`, `LQ_EDIT` or `LQ_CLOSE`.
2. **Routes questions to experts** — finds the Top-5 experts whose past answers are semantically closest to the question.
3. **Closes the loop** — askers post routed questions, experts answer from an inbox, and askers read the answers in a "My Questions" panel.

---

## Table of Contents
1. [Features](#features)
2. [Project Structure](#project-structure)
3. [System Architecture](#system-architecture)
4. [The ML Pipeline in Detail](#the-ml-pipeline-in-detail)
5. [The Web Application](#the-web-application)
6. [API Reference](#api-reference)
7. [Database Schema](#database-schema)
8. [Running the System](#running-the-system)
9. [Using the App (Walkthrough)](#using-the-app-walkthrough)
10. [Results](#results)
11. [Reproducibility](#reproducibility)
12. [Limitations & Future Work](#limitations--future-work)

---

## Features

| Area | What it does |
|:-----|:-------------|
| Quality check | Scores a question and returns a verdict, a 0–100 meter and improvement tips |
| Expert routing | Top-5 experts by cosine similarity, with match %, answer count and helpful votes |
| Gatekeeping | Only `HQ` / `LQ_EDIT` questions can be posted; `LQ_CLOSE` is blocked until rewritten |
| Accounts | Sign up / log in as **asker** or **expert** (PBKDF2-hashed passwords, token sessions) |
| Asker dashboard | "My Questions" with status and the answers received |
| Expert dashboard | "Expert Inbox" showing only questions routed to that expert, with a reply box |

---

## Project Structure

```text
qa-expert-routing/
├── app/
│   ├── server.py             # FastAPI backend: ML endpoints + accounts + Q&A storage
│   ├── static/index.html     # Single-page front end (vanilla HTML/CSS/JS)
│   └── streamlit_app.py      # Alternative Streamlit demo UI
├── data/
│   ├── raw/                  # train.csv, valid.csv, Answers.csv, Questions_SO.csv
│   ├── processed/            # Cleaned stratified splits: train/validation/test.csv
│   ├── embeddings/           # Cached SBERT .npy embeddings + expert profiles
│   └── app.db                # SQLite database (created automatically on first run)
├── models/
│   ├── best_quality_classifier.pt     # Trained PyTorch MLP weights
│   └── test_evaluation_results.json   # Metrics on the held-out test set
├── src/
│   ├── data_preparation.py   # Phase 1: EDA, HTML cleaning, stratified 80/10/10 split
│   ├── embedding.py          # Phase 2: SBERT feature extraction & caching
│   ├── quality_model.py      # Phase 3: MLP training, early stopping, evaluation
│   ├── expert_profiles.py    # Phase 4: Expert profile construction from StackSample
│   ├── expert_router.py      # Phase 5: Cosine-similarity Top-K matching
│   └── inference.py          # End-to-end unified inference pipeline
├── notebooks/
├── requirements.txt
└── README.md
```

---

## System Architecture

### High-level view

```text
┌────────────────────────────────────────────────────────────────────────┐
│                          Browser  (index.html)                         │
│   Ask form · Quality meter · Expert list · My Questions · Expert Inbox │
└───────────────────────────────┬────────────────────────────────────────┘
                                │  JSON over HTTP  (Bearer token)
┌───────────────────────────────▼────────────────────────────────────────┐
│                       FastAPI server  (server.py)                      │
│                                                                        │
│   ┌────────────────────┐  ┌──────────────────┐  ┌──────────────────┐   │
│   │  ML endpoints      │  │  Auth            │  │  Q&A endpoints   │   │
│   │  /api/analyze      │  │  /api/signup     │  │  /api/questions  │   │
│   │                    │  │  /api/login      │  │  /api/my-...     │   │
│   │                    │  │                  │  │  /api/inbox      │   │
│   └─────────┬──────────┘  └────────┬─────────┘  └────────┬─────────┘   │
│             │                      └──────────┬──────────┘             │
│   ┌─────────▼──────────┐             ┌────────▼──────────┐             │
│   │ InferencePipeline  │             │  SQLite (app.db)  │             │
│   │ (loaded once)      │             │ users · sessions  │             │
│   │ SBERT + MLP +      │             │ questions·answers │             │
│   │ ExpertRouter       │             └───────────────────┘             │
│   └────────────────────┘                                               │
└────────────────────────────────────────────────────────────────────────┘
```

### Training vs. inference

The models are **never retrained** when a user submits a question. Training is offline; at server start the weights and expert profiles are loaded once from disk and reused for every request.

```text
                     TRAINING PHASE (offline, run once)
   60k SO questions                       StackSample (Answers + Questions)
   (train.csv)                                       │
        │                                            ▼
        ▼                                  SBERT-encode answered questions
   SBERT encode (all-MiniLM-L6-v2)                   │
        │                                            ▼
        ▼                                  Mean-aggregate per expert, L2-normalise
   384-d embeddings (.npy cache)                     │
        │                                            ▼
        ▼                                  expert_vectors.npy + expert_metadata.json
   Train MLP (384→256→64→3)
        │
        ▼
   best_quality_classifier.pt

                     INFERENCE PHASE (online, per request)
   Title + Body ─► Clean text ─► SBERT ─► 384-d vector q
                                           ├─► MLP ─► softmax ─► HQ / LQ_EDIT / LQ_CLOSE
                                           └─► E @ q (cosine) ─► Top-K experts
   HQ / LQ_EDIT  ─► may be routed to the Top-5 experts
   LQ_CLOSE      ─► no routing; show tips to rewrite
```

---

## The ML Pipeline in Detail

### Phase 1 — Data preparation (`data_preparation.py`)
- Loads the **60k Stack Overflow Questions with Quality Rating** dataset.
- Runs EDA (class balance, text lengths).
- Cleans text: `html.unescape` → strip HTML tags → collapse whitespace.
- Concatenates **Title + Body**.
- Creates a **stratified 80 / 10 / 10** train / validation / test split (seed 42).

### Phase 2 — Feature extraction (`embedding.py`)
- Encodes every question with the frozen **Sentence-BERT `all-MiniLM-L6-v2`** model.
- Output is a **384-dimensional L2-normalised** vector per question.
- Vectors are cached as `.npy`, so training never re-runs SBERT.

### Phase 3 — Quality classifier (`quality_model.py`)
- **Labels:** `HQ` (high quality), `LQ_EDIT` (low quality, needs edits), `LQ_CLOSE` (closed).
- **Architecture:** `384 → Dense(256) → ReLU → Dropout(0.3) → Dense(64) → ReLU → Dense(3)`
- **Training:** Adam (`lr=1e-3`, `weight_decay=1e-4`), cross-entropy loss, early stopping (patience 5) on validation, best checkpoint saved.
- Evaluated once on the untouched test set.

### Phase 4 — Expert profiles (`expert_profiles.py`)
1. Load up to 200,000 rows of StackSample `Answers.csv`.
2. Drop anonymous answers and answers with score < 1.
3. Keep users with **at least 5** qualifying answers (these are the "experts").
4. Join each answer to its parent question text; clean and SBERT-encode every unique question.
5. For each expert, take the **mean** of the embeddings of the questions they answered, then re-normalise to unit length. This is the expert's *profile vector*, the centroid of their topical expertise.
6. Save `expert_vectors.npy` (N × 384) and `expert_metadata.json` (`expert_id`, `num_questions_answered`, `total_answer_score`, `mean_answer_score`).

### Phase 5 — Expert routing (`expert_router.py`)
- For a new question vector `q` and expert matrix `E`, cosine similarity is a single matrix–vector product `s = E @ q` because both sides are unit-length.
- `argpartition` picks the Top-K efficiently, and they are then sorted by score.

### Unified inference (`inference.py`)
`InferencePipeline.predict(title, body, top_k)`:

1. **Preprocess** — same cleaning as training.
2. **Encode** — SBERT → 384-d vector.
3. **Classify** — MLP logits → softmax → predicted class + probabilities.
4. **Decide** — `should_route = class in {HQ, LQ_EDIT}`.
5. **Route** — if routable, return the Top-K experts with similarity scores.

---

## The Web Application

### Server start-up (`server.py`)
1. Creates `data/app.db` and its tables if missing.
2. Loads SBERT, the MLP and the expert matrix once (takes a few seconds).
3. Serves the front end at `/` and the JSON API under `/api/`.

### Friendly presentation layer
`/api/analyze` converts raw model output into something readable:
- **Verdict & summary** per class ("Great question!", "Almost there", "Needs a rewrite").
- **Score (0–100):** a base per class (HQ 90, LQ_EDIT 55, LQ_CLOSE 20), nudged by model confidence for HQ and LQ_CLOSE.
- **Tips** for `LQ_EDIT` / `LQ_CLOSE`.
- **Expert match %:** `cosine × 100 + 35`, clamped to 0–99. This is a display calibration so the numbers are readable. It is not a probability.

### Authentication
- Passwords are salted and hashed with **PBKDF2-HMAC-SHA256 (100k iterations)**.
- Login or signup returns a random 48-hex-char **session token**, stored in the `sessions` table and in the browser's `localStorage`.
- The front end sends it as `Authorization: Bearer <token>`. A 401 logs the user out automatically.

### Roles
| Role | Can do |
|:-----|:-------|
| **Asker** | Check questions, post routed questions, read answers |
| **Expert** | See questions routed to their Expert ID, reply. The ask tool is hidden for experts |

### Question lifecycle

```text
Asker writes question
      │  "Check my question"  ──►  POST /api/analyze   (nothing saved)
      ▼
Quality verdict + Top-5 experts shown
      │  (only HQ / LQ_EDIT)
      │  "Post to these experts"  ──►  POST /api/questions
      ▼
Server re-runs the model, stores the question + the 5 routed expert IDs
      ▼
Each routed expert sees it in their Inbox  ──►  POST /api/questions/{id}/answers
      ▼
Asker clicks Refresh  ──►  GET /api/my-questions  shows the answers
```

The server **re-runs the classifier on post** rather than trusting the client, so routing can't be spoofed.

---

## API Reference

| Method | Endpoint | Auth | Purpose |
|:------:|:---------|:----:|:--------|
| GET | `/` | – | Serves the web UI |
| POST | `/api/analyze` | – | Quality verdict, score, tips and Top-5 experts (not stored) |
| POST | `/api/signup` | – | Create account (`username`, `password`, `role`, `expert_id` for experts) |
| POST | `/api/login` | – | Returns a session token |
| POST | `/api/questions` | Asker | Re-analyses, stores question and routed expert IDs |
| GET | `/api/my-questions` | Any | Asker's questions with their answers |
| GET | `/api/inbox` | Expert | Questions routed to this expert, with `answered` flag |
| POST | `/api/questions/{id}/answers` | Expert | Submit an answer (only if routed to you) |

Error codes: `400` invalid input or unroutable question · `401` not logged in / bad credentials · `403` wrong role or question not routed to you.

---

## Database Schema

SQLite file: `data/app.db`

| Table | Columns |
|:------|:--------|
| `users` | `id`, `username` (unique), `salt`, `pw` (hash), `role`, `expert_id` |
| `sessions` | `token` (PK), `user_id` |
| `questions` | `id`, `user_id`, `title`, `body`, `level`, `experts` (JSON list of IDs), `created` |
| `answers` | `id`, `question_id`, `expert_user_id`, `body`, `created` |

An expert account's `expert_id` links it to the dataset-derived expert profiles. A question appears in an inbox when that ID is in the question's `experts` list.

---

## Running the System

> Activate the virtual environment first:
> ```powershell
> .\venv\Scripts\activate
> ```

### One-time ML pipeline (skip if `models/` and `data/embeddings/` already exist)

| Phase | Command | Output |
|:-----:|:--------|:-------|
| 1 | `python src/data_preparation.py` | `data/processed/*.csv` |
| 2 | `python src/embedding.py` | `data/embeddings/*_embeddings.npy`, labels |
| 3 | `python src/quality_model.py` | `models/best_quality_classifier.pt`, test metrics |
| 4 | `python src/expert_profiles.py` | `expert_vectors.npy`, `expert_metadata.json` |
| 5 | `python src/inference.py` | CLI demo on 3 sample questions |

### Start the web app
```bash
python app/server.py
```
Open **http://localhost:8000**. The first start takes a few seconds while the SBERT model loads.

### Alternative Streamlit demo
```bash
streamlit run app/streamlit_app.py
```
This is a quality check and expert list only, with no accounts.

---

## Using the App (Walkthrough)

1. **Sign up as an asker** (top-right → *Sign up* → "I want to ask questions").
2. Enter a title and body, then click **Check my question**. The quality ring and the Top-5 experts appear.
3. Click **Post to these experts**. The confirmation lists the routed Expert IDs, for example `#118495, #12138 ...`.
4. In a **private window**, sign up as an **expert** and enter one of those IDs.
5. The expert sees the question in the **Expert Inbox**, writes an answer and clicks **Send answer**.
6. Back in the asker's window, click **Refresh**. The answer appears under **My Questions**.

Tip: use *Try a good example* / *Try a vague example* to see both routable and blocked outcomes.

---

## Results

Evaluated on the **unseen 6,000-question test set** (Phase 3):

| Metric | Value |
|:-------|------:|
| **Test accuracy** | **70.20%** |
| **Macro F1** | **0.6996** |

| Class | Precision | Recall | F1 |
|:------|:---------:|:------:|:--:|
| HQ (High Quality) | 0.81 | 0.86 | **0.84** |
| LQ_EDIT (Requires Edit) | 0.67 | 0.58 | 0.62 |
| LQ_CLOSE (Closed) | 0.62 | 0.66 | 0.64 |

Expert routing has no labelled ground truth in this prototype, so it is demonstrated qualitatively through the similarity scores.

---

## Reproducibility
- **Random seed:** `42` (NumPy, PyTorch, sklearn splits)
- **Encoder:** `all-MiniLM-L6-v2` (frozen, not fine-tuned)
- **MLP:** `384 → 256 → ReLU → Dropout(0.3) → 64 → ReLU → 3`
- **Optimiser:** Adam, `lr=1e-3`, `weight_decay=1e-4`
- **Early stopping:** patience 5 epochs
- **Datasets:** Kaggle `imoore/60k-stack-overflow-questions-with-quality-rate` (quality) and `stackoverflow/stacksample` (expert profiles)

---

## Limitations & Future Work
- **Expert identity is unverified.** Anyone can claim any Expert ID at signup. A real deployment would link expert IDs to verified accounts.
- **Expert profiles come from historical data.** They are not live users, and the unweighted mean can blur multi-topic experts. Score-weighted or multi-vector profiles are a natural extension.
- **Classifier accuracy is ~70%.** `LQ_EDIT` is the weakest class. Fine-tuning SBERT or adding handcrafted features (code blocks, length, error messages) should help.
- **Match % is a display calibration**, not a calibrated probability.
- **Prototype-grade app:** SQLite, no rate limiting, no password reset, no answer notifications (the asker refreshes manually), and tokens never expire.
