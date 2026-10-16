"""
Web server for the Q&A demo.  Run:  python app/server.py   ->  http://localhost:8000
"""
import hashlib
import json
import os
import secrets
import sqlite3
import sys
import time
from typing import Optional

import uvicorn
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

from inference import InferencePipeline  # noqa: E402

app = FastAPI(title="AskWise")
pipeline = InferencePipeline(
    models_dir=os.path.join(BASE_DIR, "models"),
    embeddings_dir=os.path.join(BASE_DIR, "data", "embeddings"),
)

FRIENDLY = {
    "HQ": {
        "verdict": "Great question!",
        "summary": "Clear and well-explained. It's ready to be sent to the people best placed to answer it.",
        "score": 90,
        "tips": [],
    },
    "LQ_EDIT": {
        "verdict": "Almost there",
        "summary": "A few improvements will help you get a faster, better answer.",
        "score": 55,
        "tips": [
            "Show the code you've tried, as a short, runnable example.",
            "Explain what you expected to happen and what happened instead.",
            "Paste any error message you're seeing.",
            "Focus on one specific problem.",
        ],
    },
    "LQ_CLOSE": {
        "verdict": "Needs a rewrite",
        "summary": "This question is too vague or broad for anyone to answer well right now.",
        "score": 20,
        "tips": [
            "Write a specific title that says exactly what you're trying to do.",
            "Describe the problem in a few clear sentences.",
            "Search first. Your question may already be answered.",
        ],
    },
}


# ---------------------------------------------------------------- storage
DB_PATH = os.path.join(BASE_DIR, "data", "app.db")


def db():
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    return c


with db() as _c:
    _c.executescript("""
    CREATE TABLE IF NOT EXISTS users(
        id INTEGER PRIMARY KEY, username TEXT UNIQUE, salt TEXT, pw TEXT,
        role TEXT, expert_id INTEGER);
    CREATE TABLE IF NOT EXISTS sessions(token TEXT PRIMARY KEY, user_id INTEGER);
    CREATE TABLE IF NOT EXISTS questions(
        id INTEGER PRIMARY KEY, user_id INTEGER, title TEXT, body TEXT,
        level TEXT, experts TEXT, created REAL);
    CREATE TABLE IF NOT EXISTS answers(
        id INTEGER PRIMARY KEY, question_id INTEGER, expert_user_id INTEGER,
        body TEXT, created REAL);
    """)


def hash_pw(pw, salt):
    return hashlib.pbkdf2_hmac("sha256", pw.encode(), salt.encode(), 100_000).hex()


def current_user(authorization: Optional[str]):
    token = (authorization or "").replace("Bearer ", "")
    with db() as c:
        u = c.execute(
            "SELECT u.* FROM users u JOIN sessions s ON s.user_id=u.id WHERE s.token=?",
            (token,)).fetchone()
    if not u:
        raise HTTPException(401, "Please log in.")
    return u


def new_session(c, user_id):
    token = secrets.token_hex(24)
    c.execute("INSERT INTO sessions VALUES(?,?)", (token, user_id))
    return token


class Question(BaseModel):
    title: str = ""
    body: str = ""


class Signup(BaseModel):
    username: str
    password: str
    role: str = "asker"          # asker | expert
    expert_id: Optional[int] = None


class Login(BaseModel):
    username: str
    password: str


class Answer(BaseModel):
    body: str


@app.post("/api/analyze")
def analyze(q: Question):
    r = pipeline.predict(q.title, q.body, top_k=5)
    info = FRIENDLY[r["predicted_class"]]
    confidence = r["class_probabilities"][r["predicted_class"]]
    # Blend the class score with the model's confidence so the meter feels graded
    score = int(info["score"] + (confidence - 0.5) * 10) if r["predicted_class"] != "LQ_EDIT" else info["score"]
    experts = []
    for e in r["expert_recommendations"]:
        match = max(0, min(99, int(round(e["cosine_similarity"] * 100 + 35))))
        experts.append({
            "id": e["expert_id"],
            "match": match,
            "answers": e["num_questions_answered"],
            "reputation": e["total_answer_score"],
        })
    return {
        "level": r["predicted_class"],
        "verdict": info["verdict"],
        "summary": info["summary"],
        "score": max(5, min(98, score)),
        "tips": info["tips"],
        "routed": r["should_route"],
        "experts": experts,
    }


@app.post("/api/signup")
def signup(s: Signup):
    if s.role not in ("asker", "expert") or len(s.username) < 3 or len(s.password) < 4:
        raise HTTPException(400, "Username needs 3+ chars and password 4+ chars.")
    if s.role == "expert" and s.expert_id is None:
        raise HTTPException(400, "Experts must enter their Expert ID.")
    salt = secrets.token_hex(8)
    with db() as c:
        try:
            cur = c.execute("INSERT INTO users(username,salt,pw,role,expert_id) VALUES(?,?,?,?,?)",
                            (s.username, salt, hash_pw(s.password, salt), s.role,
                             s.expert_id if s.role == "expert" else None))
        except sqlite3.IntegrityError:
            raise HTTPException(400, "Username already taken.")
        token = new_session(c, cur.lastrowid)
    return {"token": token, "username": s.username, "role": s.role, "expert_id": s.expert_id}


@app.post("/api/login")
def login(l: Login):
    with db() as c:
        u = c.execute("SELECT * FROM users WHERE username=?", (l.username,)).fetchone()
        if not u or u["pw"] != hash_pw(l.password, u["salt"]):
            raise HTTPException(401, "Wrong username or password.")
        token = new_session(c, u["id"])
    return {"token": token, "username": u["username"], "role": u["role"], "expert_id": u["expert_id"]}


@app.post("/api/questions")
def post_question(q: Question, authorization: Optional[str] = Header(None)):
    u = current_user(authorization)
    if u["role"] != "asker":
        raise HTTPException(403, "Only askers can post questions.")
    r = pipeline.predict(q.title, q.body, top_k=5)
    if not r["should_route"]:
        raise HTTPException(400, "Improve your question before posting it.")
    ids = [e["expert_id"] for e in r["expert_recommendations"]]
    with db() as c:
        cur = c.execute(
            "INSERT INTO questions(user_id,title,body,level,experts,created) VALUES(?,?,?,?,?,?)",
            (u["id"], q.title, q.body, r["predicted_class"], json.dumps(ids), time.time()))
    return {"id": cur.lastrowid, "routed_to": ids}


@app.get("/api/my-questions")
def my_questions(authorization: Optional[str] = Header(None)):
    u = current_user(authorization)
    out = []
    with db() as c:
        for q in c.execute("SELECT * FROM questions WHERE user_id=? ORDER BY id DESC", (u["id"],)):
            answers = [{"expert_id": a["expert_id"], "body": a["body"], "created": a["created"]}
                       for a in c.execute(
                           "SELECT a.body, a.created, u.expert_id FROM answers a "
                           "JOIN users u ON u.id=a.expert_user_id WHERE a.question_id=? ORDER BY a.id",
                           (q["id"],))]
            out.append({"id": q["id"], "title": q["title"], "body": q["body"],
                        "experts": json.loads(q["experts"]), "created": q["created"],
                        "answers": answers})
    return out


@app.get("/api/inbox")
def inbox(authorization: Optional[str] = Header(None)):
    u = current_user(authorization)
    if u["role"] != "expert":
        raise HTTPException(403, "Experts only.")
    out = []
    with db() as c:
        for q in c.execute("SELECT * FROM questions ORDER BY id DESC"):
            if u["expert_id"] not in json.loads(q["experts"]):
                continue
            mine = c.execute("SELECT 1 FROM answers WHERE question_id=? AND expert_user_id=?",
                             (q["id"], u["id"])).fetchone()
            out.append({"id": q["id"], "title": q["title"], "body": q["body"],
                        "created": q["created"], "answered": bool(mine)})
    return out


@app.post("/api/questions/{qid}/answers")
def answer(qid: int, a: Answer, authorization: Optional[str] = Header(None)):
    u = current_user(authorization)
    if u["role"] != "expert" or not a.body.strip():
        raise HTTPException(403, "Experts only, and the answer can't be empty.")
    with db() as c:
        q = c.execute("SELECT * FROM questions WHERE id=?", (qid,)).fetchone()
        if not q or u["expert_id"] not in json.loads(q["experts"]):
            raise HTTPException(403, "This question wasn't routed to you.")
        c.execute("INSERT INTO answers(question_id,expert_user_id,body,created) VALUES(?,?,?,?)",
                  (qid, u["id"], a.body.strip(), time.time()))
    return {"ok": True}


@app.get("/")
def index():
    return FileResponse(os.path.join(os.path.dirname(__file__), "static", "index.html"))


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")
