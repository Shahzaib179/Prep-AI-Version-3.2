from __future__ import annotations

import hashlib
import io
import json
import os
import re
import sqlite3
import tempfile
import time
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import faiss
import gdown
import numpy as np
import pandas as pd
import streamlit as st
from docx import Document
from pypdf import PdfReader
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer
from sentence_transformers import SentenceTransformer

try:
    from ddgs import DDGS
except Exception:
    DDGS = None

try:
    from google import genai
    from google.genai import types
except Exception:
    genai = None
    types = None

try:
    from crewai import Agent, Crew, LLM, Process, Task
except Exception:
    Agent = Crew = LLM = Process = Task = None


APP_VERSION = "3.2"
DB_PATH = Path("data/prep_ai.db")
FAISS_DIR = Path("faiss_index")
ALLOWED_EXTENSIONS = {"pdf", "docx", "txt", "md"}
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_MODEL = "gemini-3.5-flash-lite"
CHUNK_SIZE = 850
CHUNK_OVERLAP = 150
MAX_UPLOAD_MB = 50

st.set_page_config(
    page_title="Prep AI V3.2",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================
# General utilities
# =========================

def now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat()


def safe_filename(name: str) -> str:
    name = Path(name).name
    name = re.sub(r"[^\w.\- ()]+", "_", name, flags=re.UNICODE)
    return name[:180] or "document"


def get_secret(name: str) -> str:
    try:
        value = str(st.secrets.get(name, "")).strip()
    except Exception:
        value = ""
    return value or os.getenv(name, "").strip()


def get_student_id() -> str:
    if "student_id" not in st.session_state:
        st.session_state.student_id = hashlib.sha256(
            f"prep-ai-{time.time_ns()}".encode()
        ).hexdigest()[:16]
    return st.session_state.student_id


def json_loads_safe(text: str, default: Any = None) -> Any:
    try:
        return json.loads(text.strip())
    except Exception:
        match = re.search(r"(\{.*\}|\[.*\])", text or "", flags=re.S)
        if match:
            try:
                return json.loads(match.group(1))
            except Exception:
                pass
    return default


# =========================
# SQLite learning memory
# =========================

def init_database(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        PRAGMA foreign_keys=ON;

        CREATE TABLE IF NOT EXISTS students (
            student_id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            last_seen TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS question_attempts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT NOT NULL,
            subject TEXT,
            chapter TEXT,
            topic TEXT,
            concept TEXT,
            difficulty TEXT,
            question_hash TEXT,
            is_correct INTEGER NOT NULL,
            student_answer TEXT,
            correct_answer TEXT,
            source_file TEXT,
            source_page TEXT,
            attempted_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS mistakes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT NOT NULL,
            question_hash TEXT,
            subject TEXT,
            chapter TEXT,
            topic TEXT,
            concept TEXT,
            mistake_count INTEGER DEFAULT 1,
            last_mistake_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS mastery (
            student_id TEXT NOT NULL,
            subject TEXT NOT NULL,
            chapter TEXT,
            topic TEXT NOT NULL,
            attempts INTEGER DEFAULT 0,
            correct INTEGER DEFAULT 0,
            accuracy REAL DEFAULT 0,
            mastery_score REAL DEFAULT 0,
            last_studied TEXT,
            next_review TEXT,
            PRIMARY KEY(student_id, subject, chapter, topic)
        );

        CREATE TABLE IF NOT EXISTS study_plans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT NOT NULL,
            exam_name TEXT,
            exam_date TEXT,
            plan_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS achievements (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT NOT NULL,
            achievement TEXT NOT NULL,
            earned_at TEXT NOT NULL,
            UNIQUE(student_id, achievement)
        );

        CREATE TABLE IF NOT EXISTS agent_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT NOT NULL,
            user_message TEXT NOT NULL,
            agent_response TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        """
    )
    conn.commit()


@st.cache_resource
def get_db() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH.as_posix(), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    init_database(conn)
    return conn


def ensure_student(conn: sqlite3.Connection, student_id: str) -> None:
    conn.execute(
        """
        INSERT INTO students(student_id, created_at, last_seen)
        VALUES (?, ?, ?)
        ON CONFLICT(student_id)
        DO UPDATE SET last_seen=excluded.last_seen
        """,
        (student_id, now_iso(), now_iso()),
    )
    conn.commit()


def record_attempt(
    conn: sqlite3.Connection,
    student_id: str,
    q: dict[str, Any],
    correct: bool,
    student_answer: str,
) -> None:
    subject = q.get("subject") or "General"
    chapter = q.get("chapter") or ""
    topic = q.get("topic") or "General"
    concept = q.get("concept") or topic
    difficulty = q.get("difficulty") or "Medium"
    qhash = hashlib.sha256(
        re.sub(r"\W+", "", q.get("question", "").lower()).encode()
    ).hexdigest()[:24]

    conn.execute(
        """
        INSERT INTO question_attempts
        (student_id, subject, chapter, topic, concept, difficulty,
         question_hash, is_correct, student_answer, correct_answer,
         source_file, source_page, attempted_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            student_id, subject, chapter, topic, concept, difficulty,
            qhash, int(correct), student_answer, q.get("answer", ""),
            q.get("source_file", ""), str(q.get("source_page", "")), now_iso(),
        ),
    )

    if not correct:
        row = conn.execute(
            """
            SELECT id FROM mistakes
            WHERE student_id=? AND question_hash=?
            """,
            (student_id, qhash),
        ).fetchone()
        if row:
            conn.execute(
                """
                UPDATE mistakes
                SET mistake_count=mistake_count+1, last_mistake_at=?
                WHERE id=?
                """,
                (now_iso(), row["id"]),
            )
        else:
            conn.execute(
                """
                INSERT INTO mistakes
                (student_id, question_hash, subject, chapter, topic, concept,
                 mistake_count, last_mistake_at)
                VALUES (?, ?, ?, ?, ?, ?, 1, ?)
                """,
                (
                    student_id, qhash, subject, chapter, topic, concept, now_iso()
                ),
            )
    conn.commit()


def recalculate_mastery(conn: sqlite3.Connection, student_id: str) -> None:
    rows = conn.execute(
        """
        SELECT subject, chapter, topic, difficulty, is_correct, attempted_at
        FROM question_attempts
        WHERE student_id=?
        ORDER BY attempted_at DESC
        """,
        (student_id,),
    ).fetchall()

    groups: dict[tuple[str, str, str], list[sqlite3.Row]] = defaultdict(list)
    for row in rows:
        groups[(row["subject"], row["chapter"] or "", row["topic"])].append(row)

    weights = {"Easy": 1.0, "Medium": 1.15, "Hard": 1.35, "Expert": 1.5}

    for (subject, chapter, topic), attempts in groups.items():
        total_w = sum(weights.get(r["difficulty"], 1.0) for r in attempts)
        correct_w = sum(
            weights.get(r["difficulty"], 1.0) * int(r["is_correct"])
            for r in attempts
        )
        weighted_accuracy = 100 * correct_w / total_w if total_w else 0

        recent = attempts[:10]
        recent_accuracy = (
            100 * sum(int(r["is_correct"]) for r in recent) / len(recent)
            if recent else 0
        )
        consistency = max(
            0.0,
            100.0 - np.std([int(r["is_correct"]) for r in recent]) * 100,
        ) if recent else 0.0

        mistakes = conn.execute(
            """
            SELECT COALESCE(SUM(mistake_count),0) AS n
            FROM mistakes
            WHERE student_id=? AND subject=? AND chapter=? AND topic=?
            """,
            (student_id, subject, chapter, topic),
        ).fetchone()["n"]

        score = (
            0.55 * weighted_accuracy
            + 0.25 * recent_accuracy
            + 0.20 * consistency
            - min(float(mistakes) * 2.0, 20.0)
        )
        score = max(0.0, min(100.0, score))

        if score < 40:
            review_days = 1
        elif score < 60:
            review_days = 2
        elif score < 75:
            review_days = 4
        elif score < 90:
            review_days = 7
        else:
            review_days = 14

        last_studied = attempts[0]["attempted_at"]
        next_review = (
            datetime.fromisoformat(last_studied)
            + timedelta(days=review_days)
        ).date().isoformat()

        conn.execute(
            """
            INSERT INTO mastery
            (student_id, subject, chapter, topic, attempts, correct,
             accuracy, mastery_score, last_studied, next_review)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(student_id, subject, chapter, topic)
            DO UPDATE SET attempts=excluded.attempts,
                          correct=excluded.correct,
                          accuracy=excluded.accuracy,
                          mastery_score=excluded.mastery_score,
                          last_studied=excluded.last_studied,
                          next_review=excluded.next_review
            """,
            (
                student_id, subject, chapter, topic, len(attempts),
                sum(int(r["is_correct"]) for r in attempts),
                round(weighted_accuracy, 2), round(score, 2),
                last_studied, next_review,
            ),
        )
    conn.commit()


def get_mastery(conn: sqlite3.Connection, student_id: str) -> list[sqlite3.Row]:
    return conn.execute(
        """
        SELECT * FROM mastery
        WHERE student_id=?
        ORDER BY mastery_score ASC
        """,
        (student_id,),
    ).fetchall()


# =========================
# Document extraction
# =========================

def extract_pdf(data: bytes, filename: str, source: str) -> list[dict[str, Any]]:
    reader = PdfReader(io.BytesIO(data))
    records = []
    for page_no, page in enumerate(reader.pages, start=1):
        text = re.sub(r"\s+", " ", page.extract_text() or "").strip()
        if text:
            records.append({
                "filename": filename,
                "source": source,
                "file_type": "PDF",
                "page": page_no,
                "text": text,
            })
    return records


def extract_docx(data: bytes, filename: str, source: str) -> list[dict[str, Any]]:
    doc = Document(io.BytesIO(data))
    text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    text = re.sub(r"\s+", " ", text).strip()
    return [{
        "filename": filename,
        "source": source,
        "file_type": "DOCX",
        "page": None,
        "text": text,
    }] if text else []


def extract_text(data: bytes, filename: str, source: str, kind: str) -> list[dict[str, Any]]:
    text = re.sub(r"\s+", " ", data.decode("utf-8", errors="ignore")).strip()
    return [{
        "filename": filename,
        "source": source,
        "file_type": kind,
        "page": None,
        "text": text,
    }] if text else []


def extract_document(data: bytes, filename: str, source: str) -> list[dict[str, Any]]:
    ext = Path(filename).suffix.lower().lstrip(".")
    if ext == "pdf":
        return extract_pdf(data, filename, source)
    if ext == "docx":
        return extract_docx(data, filename, source)
    if ext in {"txt", "md"}:
        return extract_text(data, filename, source, ext.upper())
    raise ValueError(f"Unsupported file type: {ext or 'unknown'}")


def chunk_documents(
    records: list[dict[str, Any]],
    size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[dict[str, Any]]:
    chunks = []
    for record in records:
        words = record["text"].split()
        start = 0
        local = 0
        while start < len(words):
            end = min(start + size, len(words))
            text = " ".join(words[start:end]).strip()
            if text:
                item = dict(record)
                item.update({
                    "text": text,
                    "chunk_id": len(chunks),
                    "chunk_index": local,
                })
                chunks.append(item)
                local += 1
            if end >= len(words):
                break
            start = max(end - overlap, start + 1)

    totals = Counter(c["filename"] for c in chunks)
    for c in chunks:
        c["total_source_chunks"] = totals[c["filename"]]
    return chunks


# =========================
# Embeddings / FAISS
# =========================

@st.cache_resource
def get_embedding_model() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL)


def build_index(
    chunks: list[dict[str, Any]],
) -> tuple[faiss.IndexFlatIP, list[dict[str, Any]]]:
    model = get_embedding_model()
    vectors = model.encode(
        [c["text"] for c in chunks],
        normalize_embeddings=True,
        show_progress_bar=False,
        batch_size=32,
    ).astype("float32")

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index, chunks


@st.cache_resource
def load_database_index() -> tuple[faiss.Index, list[dict[str, Any]]] | None:
    index_path = FAISS_DIR / "database.faiss"
    metadata_path = FAISS_DIR / "metadata.json"

    if not index_path.exists() or not metadata_path.exists():
        return None

    index = faiss.read_index(str(index_path))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    if isinstance(metadata, dict):
        metadata = metadata.get("chunks", metadata.get("metadata", []))

    return index, metadata


def keyword_score(query: str, text: str) -> float:
    stop = {
        "the", "and", "for", "with", "from", "that", "this", "what",
        "how", "why", "about", "explain", "into", "are", "was", "were",
    }
    terms = [
        t.lower()
        for t in re.findall(r"[A-Za-z0-9]+", query)
        if len(t) > 2 and t.lower() not in stop
    ]
    if not terms:
        return 0.0
    counts = Counter(re.findall(r"[A-Za-z0-9]+", text.lower()))
    return min(
        1.0,
        sum(min(counts[t], 3) for t in terms) / (3 * len(terms)),
    )


def hybrid_search(
    index: faiss.Index,
    metadata: list[dict[str, Any]],
    query: str,
    top_k: int = 7,
) -> list[dict[str, Any]]:
    model = get_embedding_model()
    q = model.encode([query], normalize_embeddings=True).astype("float32")
    n = min(max(top_k * 3, 12), index.ntotal)
    semantic_scores, ids = index.search(q, n)

    results = []
    for score, idx in zip(semantic_scores[0], ids[0]):
        if idx < 0 or idx >= len(metadata):
            continue
        item = dict(metadata[idx])
        item["semantic_score"] = float(score)
        item["keyword_score"] = keyword_score(query, item.get("text", ""))
        item["hybrid_score"] = (
            0.75 * item["semantic_score"] + 0.25 * item["keyword_score"]
        )
        results.append(item)

    results.sort(key=lambda x: x["hybrid_score"], reverse=True)
    return results[:top_k]


# =========================
# Google Drive
# =========================

def download_drive_source(url: str) -> list[tuple[bytes, str, str]]:
    if "drive.google.com" not in url.lower():
        raise ValueError("Please enter a valid Google Drive link.")

    temp = Path(tempfile.mkdtemp(prefix="prep_ai_drive_"))
    files: list[tuple[bytes, str, str]] = []

    if "/folders/" in url:
        folder = temp / "folder"
        folder.mkdir(parents=True, exist_ok=True)

        # gdown 6.x detects Drive folder URLs automatically.
        # Do not pass fuzzy=; that caused compatibility errors in older code.
        gdown.download_folder(
            url=url,
            output=str(folder),
            quiet=True,
        )

        paths = [
            p for p in folder.rglob("*")
            if p.is_file()
            and p.suffix.lower().lstrip(".") in ALLOWED_EXTENSIONS
        ]
        for path in paths:
            # path.name is the real downloaded filename reported by Drive.
            files.append(
                (path.read_bytes(), path.name, f"Google Drive/{path.name}")
            )
    else:
        out = temp / "file"
        out.mkdir(parents=True, exist_ok=True)

        downloaded = gdown.download(
            url=url,
            output=str(out) + os.sep,
            quiet=True,
        )

        path = Path(downloaded) if downloaded else None
        if not path or not path.is_file():
            candidates = [
                p for p in out.rglob("*")
                if p.is_file()
                and p.suffix.lower().lstrip(".") in ALLOWED_EXTENSIONS
            ]
            path = candidates[0] if candidates else None

        if not path:
            raise ValueError("Google Drive file could not be downloaded.")

        if path.suffix.lower().lstrip(".") not in ALLOWED_EXTENSIONS:
            raise ValueError(
                f"Unsupported Google Drive file type: {path.suffix}"
            )

        files.append(
            (path.read_bytes(), path.name, f"Google Drive/{path.name}")
        )

    if not files:
        raise ValueError(
            "No supported PDF, DOCX, TXT or MD files were found."
        )
    return files


# =========================
# Gemini
# =========================

@st.cache_resource
def get_gemini_client(api_key: str):
    if genai is None:
        raise RuntimeError("google-genai is not installed.")
    return genai.Client(api_key=api_key)


def current_model() -> str:
    return st.session_state.get("llm_model", DEFAULT_MODEL)


def gemini_generate(prompt: str, json_mode: bool = False) -> str:
    key = get_secret("GEMINI_API_KEY")
    if not key:
        raise RuntimeError(
            "GEMINI_API_KEY is missing. Add it to Streamlit Secrets."
        )

    client = get_gemini_client(key)
    kwargs = {
        "temperature": 0.2,
        "max_output_tokens": 8192,
    }
    if json_mode:
        kwargs["response_mime_type"] = "application/json"

    response = client.models.generate_content(
        model=current_model(),
        contents=prompt,
        config=types.GenerateContentConfig(**kwargs),
    )
    return (response.text or "").strip()


def build_context(chunks: list[dict[str, Any]], max_chars: int = 18000) -> str:
    blocks = []
    total = 0

    for i, c in enumerate(chunks, start=1):
        block = (
            f"[SOURCE {i}]\n"
            f"Filename: {c.get('filename', '')}\n"
            f"Page: {c.get('page') or 'N/A'}\n"
            f"Chunk: {c.get('chunk_index', c.get('chunk_id', 'N/A'))}\n"
            f"Text: {c.get('text', '')}\n"
        )
        if total + len(block) > max_chars:
            break
        blocks.append(block)
        total += len(block)

    return "\n".join(blocks)


# =========================
# Adaptive MCQs
# =========================

def difficulty_from_mastery(score: float) -> str:
    if score < 60:
        return "Easy"
    if score < 75:
        return "Medium"
    return "Hard"


def generate_mcqs(
    subject: str,
    chapter: str,
    topic: str,
    difficulty: str,
    count: int,
    context: str,
    instructions: str = "",
) -> list[dict[str, Any]]:
    prompt = f"""
You are Prep AI, an adaptive MDCAT learning system.

Generate exactly {count} high-quality MCQs.

Subject: {subject}
Chapter: {chapter}
Topic: {topic}
Difficulty: {difficulty}

Use ONLY the supplied context.
Never invent facts that are not supported by the context.
If the context is insufficient, return fewer questions.

Each question must have:
- one clear stem
- exactly four options A, B, C, D
- exactly one correct answer
- concise explanation
- concept
- source filename/page when identifiable

Return JSON only:
{{
  "questions": [
    {{
      "question": "...",
      "options": {{"A":"...", "B":"...", "C":"...", "D":"..."}},
      "answer": "A",
      "explanation": "...",
      "concept": "...",
      "difficulty": "{difficulty}",
      "source_file": "...",
      "source_page": "..."
    }}
  ]
}}

Optional instructions:
{instructions}

CONTEXT:
{context}
"""

    data = json_loads_safe(gemini_generate(prompt, json_mode=True), {})
    raw = data.get("questions", []) if isinstance(data, dict) else []
    cleaned = []

    for q in raw:
        if not isinstance(q, dict):
            continue

        options = q.get("options", {})
        if isinstance(options, list) and len(options) == 4:
            options = dict(zip(["A", "B", "C", "D"], options))

        if not isinstance(options, dict):
            continue

        options = {str(k).upper(): str(v) for k, v in options.items()}
        answer = str(q.get("answer", "")).upper()

        if set(options) != {"A", "B", "C", "D"}:
            continue
        if answer not in options:
            continue
        if not str(q.get("question", "")).strip():
            continue

        q["options"] = options
        q["answer"] = answer
        q["subject"] = subject
        q["chapter"] = chapter
        q["topic"] = topic
        q["difficulty"] = q.get("difficulty") or difficulty
        cleaned.append(q)

    return cleaned[:count]


def validate_mcqs(
    questions: list[dict[str, Any]],
    context: str,
) -> list[dict[str, Any]]:
    seen = set()
    valid = []
    context_lower = context.lower()

    for q in questions:
        normalized = re.sub(
            r"\W+", " ", q["question"].lower()
        ).strip()

        if normalized in seen:
            continue
        seen.add(normalized)

        terms = [
            t for t in re.findall(r"[a-zA-Z]{4,}", normalized)
            if t not in {
                "which", "what", "where", "when", "following",
                "correct", "statement",
            }
        ]

        overlap = sum(t in context_lower for t in terms)
        if terms and overlap / len(terms) < 0.15:
            continue

        valid.append(q)

    return valid


# =========================
# PDF export
# =========================

def make_pdf(title: str, lines: list[str]) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story = [Paragraph(title, styles["Title"]), Spacer(1, 12)]

    for line in lines:
        if line.strip():
            story.append(
                Paragraph(
                    line.replace("&", "&amp;"),
                    styles["BodyText"],
                )
            )
            story.append(Spacer(1, 6))

    doc.build(story)
    return buffer.getvalue()


# =========================
# Web search + CrewAI agent
# =========================

def web_search(query: str, max_results: int = 6) -> list[dict[str, str]]:
    if DDGS is None:
        return []

    try:
        with DDGS() as search:
            raw = list(search.text(query, max_results=max_results))
    except Exception:
        return []

    trusted_domains = (
        ".gov", ".edu", "who.int", "nih.gov",
        "ncbi.nlm.nih.gov", "pubmed.ncbi.nlm.nih.gov",
    )
    trusted, other = [], []

    for item in raw:
        result = {
            "title": str(item.get("title", "")),
            "url": str(item.get("href", item.get("url", ""))),
            "snippet": str(item.get("body", item.get("snippet", ""))),
        }
        if any(d in result["url"].lower() for d in trusted_domains):
            trusted.append(result)
        else:
            other.append(result)

    return (trusted + other)[:max_results]


def run_agent(
    request: str,
    level: str,
    rag_context: str,
    web_results: list[dict[str, str]],
) -> str:
    key = get_secret("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY is missing.")

    # CrewAI is used when available. The fallback keeps the educational app
    # usable if an environment has CrewAI disabled.
    if all([Agent, Crew, LLM, Process, Task]):
        try:
            llm = LLM(
                model=f"gemini/{current_model()}",
                api_key=key,
                temperature=0.2,
            )
            agent = Agent(
                role="Prep AI Research Tutor",
                goal="Produce accurate, source-aware educational answers.",
                backstory=(
                    "You are a careful educational researcher. Prefer supplied "
                    "RAG material, then web evidence, and clearly state uncertainty."
                ),
                llm=llm,
                allow_delegation=False,
                verbose=False,
            )
            task = Task(
                description=f"""
Student level: {level}
Request: {request}

RAG context:
{rag_context}

Web results:
{json.dumps(web_results, ensure_ascii=False)}

Rules:
1. Prefer RAG context.
2. Use web results only if RAG context is insufficient.
3. Never claim a source contains information it does not contain.
4. Give a clear educational answer.
""",
                expected_output="A source-aware educational response.",
                agent=agent,
            )
            crew = Crew(
                agents=[agent],
                tasks=[task],
                process=Process.sequential,
                verbose=False,
            )
            return str(crew.kickoff())
        except Exception:
            pass

    return gemini_generate(
        f"""
You are Prep AI Research Tutor.

Student level: {level}
Request: {request}

RAG context:
{rag_context}

Web results:
{json.dumps(web_results, ensure_ascii=False)}

Prefer RAG context, then web results. Never invent source claims.
"""
    )


# =========================
# UI rendering
# =========================

def render_sources(chunks: list[dict[str, Any]]) -> None:
    if not chunks:
        return

    st.subheader("Retrieved Sources")
    for i, c in enumerate(chunks, 1):
        page = c.get("page") if c.get("page") is not None else "N/A"
        with st.expander(
            f"Source {i}: {c.get('filename', 'Unknown')} — Page {page}"
        ):
            st.caption(
                f"Chunk: {c.get('chunk_index', c.get('chunk_id', 'N/A'))} | "
                f"Hybrid score: {c.get('hybrid_score', 0):.3f}"
            )
            st.write(c.get("text", ""))


def render_mcqs() -> None:
    questions = st.session_state.get("generated_mcqs", [])
    if not questions:
        return

    st.subheader("Generated MCQs")
    pdf_lines = []

    for i, q in enumerate(questions, 1):
        st.markdown(f"**{i}. {q['question']}**")
        pdf_lines.append(f"{i}. {q['question']}")

        for k, value in q["options"].items():
            st.write(f"{k}. {value}")
            pdf_lines.append(f"{k}. {value}")

        with st.expander("Answer & Explanation"):
            st.write(f"**Answer: {q['answer']}**")
            st.write(q.get("explanation", ""))

        st.divider()

    render_sources(st.session_state.get("generated_sources", []))

    st.download_button(
        "Download MCQs as PDF",
        make_pdf("Prep AI MCQs", pdf_lines),
        "prep_ai_mcqs.pdf",
        "application/pdf",
    )


def render_quiz(conn: sqlite3.Connection, student_id: str) -> None:
    questions = st.session_state.get("quiz", [])
    if not questions:
        return

    if st.session_state.get("quiz_submitted"):
        answers = st.session_state.get("quiz_answers", {})
        correct = 0

        for i, q in enumerate(questions):
            selected = answers.get(i, "Skipped")
            is_correct = selected == q["answer"]
            correct += int(is_correct)
            record_attempt(
                conn, student_id, q, is_correct, selected
            )

        recalculate_mastery(conn, student_id)

        score = 100 * correct / len(questions)
        st.success(
            f"Quiz complete: {correct}/{len(questions)} "
            f"({score:.0f}%)."
        )

        for i, q in enumerate(questions):
            selected = answers.get(i, "Skipped")
            if selected == q["answer"]:
                st.write(f"✅ {i+1}. Correct")
            else:
                st.write(
                    f"❌ {i+1}. Your answer: {selected}; "
                    f"Correct: {q['answer']}"
                )
                st.caption(q.get("explanation", ""))

        if st.button("Practice My Weak Topics"):
            st.session_state.page = "Smart Revision"
            st.session_state.pop("quiz", None)
            st.session_state.pop("quiz_submitted", None)
            st.rerun()

        if st.button("New Quiz"):
            for key in ["quiz", "quiz_answers", "quiz_submitted"]:
                st.session_state.pop(key, None)
            st.rerun()

        return

    idx = st.session_state.get("quiz_index", 0)
    q = questions[idx]

    st.progress((idx + 1) / len(questions))
    st.subheader(f"Question {idx + 1} of {len(questions)}")
    st.write(q["question"])

    choice = st.radio(
        "Choose one answer",
        list(q["options"]),
        format_func=lambda k: f"{k}. {q['options'][k]}",
        key=f"quiz_choice_{idx}",
    )

    c1, c2 = st.columns(2)

    with c1:
        if st.button("Save & Next"):
            st.session_state.quiz_answers[idx] = choice
            if idx + 1 < len(questions):
                st.session_state.quiz_index = idx + 1
            else:
                st.session_state.quiz_submitted = True
            st.rerun()

    with c2:
        if st.button("Submit Quiz"):
            st.session_state.quiz_answers[idx] = choice
            st.session_state.quiz_submitted = True
            st.rerun()


def get_active_index() -> tuple[Any, list[dict[str, Any]]] | None:
    if st.session_state.get("personalized_index") is not None:
        return (
            st.session_state.personalized_index,
            st.session_state.personalized_metadata,
        )
    return load_database_index()


# =========================
# Personalized Learning
# =========================

def page_personalized(conn: sqlite3.Connection, student_id: str) -> None:
    st.header("📚 Personalized Learning")

    uploads = st.file_uploader(
        "Upload PDF, DOCX, TXT or MD",
        type=sorted(ALLOWED_EXTENSIONS),
        accept_multiple_files=True,
    )

    drive_url = st.text_input(
        "Optional Google Drive file/folder link",
        placeholder="https://drive.google.com/drive/folders/...",
    )

    if st.button("Process Personalized Material", type="primary"):
        try:
            records = []

            for uploaded in uploads or []:
                if uploaded.size > MAX_UPLOAD_MB * 1024 * 1024:
                    raise ValueError(
                        f"{uploaded.name} exceeds {MAX_UPLOAD_MB} MB."
                    )

                records.extend(
                    extract_document(
                        uploaded.getvalue(),
                        safe_filename(uploaded.name),
                        "Local Upload",
                    )
                )

            if drive_url.strip():
                for data, filename, source in download_drive_source(
                    drive_url.strip()
                ):
                    records.extend(
                        extract_document(data, filename, source)
                    )

            if not records:
                st.warning("No documents were provided.")
                return

            chunks = chunk_documents(records)
            index, metadata = build_index(chunks)

            st.session_state.personalized_index = index
            st.session_state.personalized_metadata = metadata
            st.session_state.personalized_info = pd.DataFrame(
                [
                    {
                        "filename": r["filename"],
                        "source": r["source"],
                        "file_type": r["file_type"],
                        "characters": len(r["text"]),
                        "page": r["page"] or "N/A",
                    }
                    for r in records
                ]
            ).drop_duplicates()

            st.success(
                f"Processed {len(records)} document section(s) and "
                f"created {len(chunks)} chunks."
            )

        except Exception as exc:
            st.error(f"Processing error: {exc}")

    if "personalized_info" not in st.session_state:
        st.info("Upload material or provide a Google Drive link to begin.")
        return

    st.subheader("Extracted Document Information")
    st.dataframe(
        st.session_state.personalized_info,
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        f"Total chunks: {len(st.session_state.personalized_metadata)}"
    )

    topic = st.text_input(
        "Chapter / Topic / Concept",
        value=st.session_state.pop("practice_topic", ""),
    )

    mode = st.selectbox(
        "Mode",
        ["MCQs", "Answer explanation", "Quiz", "Flashcards"],
    )

    difficulty = st.selectbox(
        "Difficulty",
        ["Adaptive", "Easy", "Medium", "Hard"],
    )

    count = st.number_input(
        "Number of MCQs / cards",
        min_value=1,
        max_value=100,
        value=20,
    )

    instructions = st.text_area("Optional instructions")

    if st.button("Study", type="primary"):
        if not topic.strip():
            st.warning("Enter a topic.")
            return

        retrieved = hybrid_search(
            st.session_state.personalized_index,
            st.session_state.personalized_metadata,
            topic,
        )

        if not retrieved:
            st.warning("No relevant chunks were retrieved.")
            return

        context = build_context(retrieved)

        if mode == "Answer explanation":
            answer = gemini_generate(
                f"""
Explain the following topic using ONLY the supplied context.
If the information is unavailable, say that it is not available.

Topic: {topic}

CONTEXT:
{context}
"""
            )
            st.markdown(answer)
            render_sources(retrieved)
            return

        if mode == "Flashcards":
            data = json_loads_safe(
                gemini_generate(
                    f"""
Create {min(int(count), 30)} concise flashcards about {topic}.
Use ONLY the context.

JSON:
{{"cards":[{{"front":"...","back":"..."}}]}}

CONTEXT:
{context}
""",
                    json_mode=True,
                ),
                {},
            )

            cards = data.get("cards", []) if isinstance(data, dict) else []
            for i, card in enumerate(cards, 1):
                with st.expander(
                    f"Card {i}: {card.get('front', '')}"
                ):
                    st.write(card.get("back", ""))
            render_sources(retrieved)
            return

        mastery_score = 0
        for row in get_mastery(conn, student_id):
            if row["topic"].lower() == topic.lower():
                mastery_score = row["mastery_score"]
                break

        selected_difficulty = (
            difficulty_from_mastery(mastery_score)
            if difficulty == "Adaptive"
            else difficulty
        )

        questions = generate_mcqs(
            "General",
            "",
            topic,
            selected_difficulty,
            int(count),
            context,
            instructions,
        )
        questions = validate_mcqs(questions, context)

        if not questions:
            st.warning(
                "The retrieved material was not sufficient to create "
                "validated questions."
            )
            render_sources(retrieved)
            return

        if mode == "Quiz":
            st.session_state.quiz = questions
            st.session_state.quiz_index = 0
            st.session_state.quiz_answers = {}
            st.session_state.quiz_submitted = False
        else:
            st.session_state.generated_mcqs = questions
            st.session_state.generated_sources = retrieved

        st.rerun()

    render_mcqs()
    render_quiz(conn, student_id)


# =========================
# Database Learning
# =========================

def page_database(conn: sqlite3.Connection, student_id: str) -> None:
    st.header("🧬 Database Learning")

    loaded = load_database_index()
    if loaded is None:
        st.error(
            "Missing database artifacts. Add database.faiss and metadata.json "
            "to faiss_index/."
        )
        return

    index, metadata = loaded

    subject = st.selectbox(
        "Subject",
        ["Biology", "Chemistry", "Physics", "English"],
    )
    chapter = st.text_input("Chapter (optional)")
    topic = st.text_input("Topic")
    mode = st.selectbox(
        "Mode",
        ["MCQs", "Answer explanation", "Quiz", "Flashcards"],
    )
    difficulty = st.selectbox(
        "Difficulty",
        ["Adaptive", "Easy", "Medium", "Hard"],
    )
    count = st.number_input(
        "Number of MCQs / cards",
        min_value=1,
        max_value=100,
        value=20,
    )
    instructions = st.text_area("Optional instructions")

    if st.button("Start Database Learning", type="primary"):
        if not topic.strip():
            st.warning("Enter a topic.")
            return

        query = f"{subject} {chapter} {topic}".strip()

        filtered = [
            m for m in metadata
            if not m.get("subject")
            or str(m.get("subject")).lower() == subject.lower()
        ]
        filtered = filtered or metadata

        retrieved = hybrid_search(
            index, filtered, query, top_k=8
        )

        if not retrieved:
            st.warning("No relevant database material was found.")
            return

        context = build_context(retrieved)

        mastery_score = 0
        for row in get_mastery(conn, student_id):
            if (
                row["subject"].lower() == subject.lower()
                and row["topic"].lower() == topic.lower()
            ):
                mastery_score = row["mastery_score"]
                break

        selected_difficulty = (
            difficulty_from_mastery(mastery_score)
            if difficulty == "Adaptive"
            else difficulty
        )

        if mode == "Answer explanation":
            answer = gemini_generate(
                f"""
Explain {topic} for an MDCAT student using ONLY this context.
If unavailable, say so.

CONTEXT:
{context}
"""
            )
            st.markdown(answer)
            render_sources(retrieved)

        elif mode == "Flashcards":
            data = json_loads_safe(
                gemini_generate(
                    f"""
Create {min(int(count), 30)} flashcards about {topic}.
Use only the context.

JSON:
{{"cards":[{{"front":"...","back":"..."}}]}}

CONTEXT:
{context}
""",
                    json_mode=True,
                ),
                {},
            )

            for i, card in enumerate(data.get("cards", []), 1):
                with st.expander(
                    f"Card {i}: {card.get('front', '')}"
                ):
                    st.write(card.get("back", ""))

            render_sources(retrieved)

        else:
            questions = generate_mcqs(
                subject,
                chapter,
                topic,
                selected_difficulty,
                int(count),
                context,
                instructions,
            )
            questions = validate_mcqs(questions, context)

            if mode == "Quiz":
                st.session_state.quiz = questions
                st.session_state.quiz_index = 0
                st.session_state.quiz_answers = {}
                st.session_state.quiz_submitted = False
            else:
                st.session_state.generated_mcqs = questions
                st.session_state.generated_sources = retrieved

            st.rerun()

    render_mcqs()
    render_quiz(conn, student_id)


# =========================
# Dashboard
# =========================

def page_dashboard(conn: sqlite3.Connection, student_id: str) -> None:
    st.title("🎓 Prep AI V3.2")
    st.caption("Adaptive AI Learning & Exam Preparation Platform")

    attempts = conn.execute(
        """
        SELECT COUNT(*) AS n, COALESCE(SUM(is_correct),0) AS correct
        FROM question_attempts WHERE student_id=?
        """,
        (student_id,),
    ).fetchone()

    mastery = get_mastery(conn, student_id)
    overall = (
        float(np.mean([r["mastery_score"] for r in mastery]))
        if mastery else 0
    )
    accuracy = (
        100 * attempts["correct"] / attempts["n"]
        if attempts["n"] else 0
    )

    cols = st.columns(4)
    cols[0].metric("Overall Mastery", f"{overall:.0f}%")
    cols[1].metric("Questions", attempts["n"])
    cols[2].metric("Accuracy", f"{accuracy:.0f}%")
    cols[3].metric(
        "Weakest Topic",
        mastery[0]["topic"] if mastery else "—",
    )

    if mastery:
        weak = mastery[0]
        strong = mastery[-1]
        st.warning(
            f"Weak area: **{weak['topic']}** — "
            f"{weak['mastery_score']:.0f}% mastery"
        )
        st.success(
            f"Strong area: **{strong['topic']}** — "
            f"{strong['mastery_score']:.0f}% mastery"
        )

        df = pd.DataFrame(
            [
                {
                    "Subject": r["subject"],
                    "Chapter": r["chapter"],
                    "Topic": r["topic"],
                    "Mastery": round(r["mastery_score"], 1),
                    "Accuracy": round(r["accuracy"], 1),
                    "Next Review": r["next_review"],
                }
                for r in mastery
            ]
        )
        st.subheader("Mastery by Topic")
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info(
            "Your learning profile will appear here after you complete quizzes."
        )


# =========================
# Agent
# =========================

def page_agent(conn: sqlite3.Connection, student_id: str) -> None:
    st.header("🤖 Prep AI Agent")

    level = st.selectbox(
        "Student level",
        ["Beginner", "Intermediate", "MDCAT", "University", "Advanced"],
    )
    request = st.text_area(
        "Research topic / request",
        placeholder="Example: Explain the mechanism of oxidative phosphorylation.",
    )
    use_rag = st.checkbox("Use available RAG context", True)

    if st.button("Run Prep AI Agent", type="primary"):
        if not request.strip():
            st.warning("Enter a research topic.")
            return

        rag_chunks = []
        if use_rag:
            active = get_active_index()
            if active:
                rag_chunks = hybrid_search(
                    active[0], active[1], request, top_k=6
                )

        context = build_context(rag_chunks)
        results = web_search(request)

        try:
            answer = run_agent(
                request,
                level,
                context,
                results,
            )
            st.markdown(answer)

            if rag_chunks:
                render_sources(rag_chunks)

            if results:
                st.subheader("Web Sources")
                for result in results:
                    st.markdown(
                        f"- [{result['title']}]({result['url']}) — "
                        f"{result['snippet']}"
                    )

            conn.execute(
                """
                INSERT INTO agent_sessions
                (student_id, user_message, agent_response, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (student_id, request, answer, now_iso()),
            )
            conn.commit()

        except Exception as exc:
            st.error(f"Agent error: {exc}")


# =========================
# AI Tutor
# =========================

def page_tutor(conn: sqlite3.Connection, student_id: str) -> None:
    st.header("🧑‍🏫 AI Tutor")

    level = st.selectbox(
        "Tutor level",
        ["Beginner", "Intermediate", "MDCAT", "University", "Advanced"],
    )

    if "tutor_messages" not in st.session_state:
        st.session_state.tutor_messages = []

    for msg in st.session_state.tutor_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    prompt = st.chat_input("Ask your tutor...")

    if prompt:
        st.session_state.tutor_messages.append(
            {"role": "user", "content": prompt}
        )

        profile = "\n".join(
            f"{r['subject']} / {r['topic']}: "
            f"{r['mastery_score']:.0f}%"
            for r in get_mastery(conn, student_id)[:12]
        )

        response = gemini_generate(
            f"""
You are a Socratic AI tutor for a {level} student.

Student question:
{prompt}

Learning profile:
{profile or "No previous performance data."}

Teach interactively. Diagnose the student's understanding,
explain misconceptions, and adapt the explanation to the level.
"""
        )

        st.session_state.tutor_messages.append(
            {"role": "assistant", "content": response}
        )
        st.rerun()


# =========================
# Smart Revision
# =========================

def page_revision(conn: sqlite3.Connection, student_id: str) -> None:
    st.header("🔄 Smart Revision")

    rows = get_mastery(conn, student_id)
    if not rows:
        st.info("Complete some quizzes first.")
        return

    today = date.today().isoformat()
    due = [
        r for r in rows
        if (r["next_review"] or today) <= today
    ]

    if due:
        st.subheader("Topics due today")
        for row in due:
            st.warning(
                f"{row['subject']} → {row['topic']} — "
                f"Mastery {row['mastery_score']:.0f}%"
            )
    else:
        st.success("No topics are due for revision today.")

    st.subheader("All Topics")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Subject": r["subject"],
                    "Topic": r["topic"],
                    "Mastery": r["mastery_score"],
                    "Next Review": r["next_review"],
                }
                for r in rows
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )


# =========================
# Study plan
# =========================

def page_study_plan(conn: sqlite3.Connection, student_id: str) -> None:
    st.header("🗓️ Personalized Study Plan")

    exam_name = st.text_input("Exam name", "MDCAT")
    exam_date = st.date_input(
        "Exam date",
        date.today() + timedelta(days=30),
    )
    level = st.selectbox(
        "Current level",
        ["Beginner", "Intermediate", "MDCAT", "University"],
    )
    hours = st.number_input(
        "Hours per day", 0.5, 12.0, 2.0, 0.5
    )
    days = st.slider("Days per week", 1, 7, 6)
    subjects = st.multiselect(
        "Subjects",
        ["Biology", "Chemistry", "Physics", "English"],
        default=["Biology", "Chemistry", "Physics"],
    )
    goals = st.text_area("Goals")

    if st.button("Create My Study Plan", type="primary"):
        rows = get_mastery(conn, student_id)
        days_left = max(1, (exam_date - date.today()).days)
        total_days = min(days_left, 42)
        plan = []

        for i in range(total_days):
            d = date.today() + timedelta(days=i)

            if d.weekday() >= 5 and days < 6:
                continue

            if rows:
                focus = rows[i % len(rows)]
                subject = focus["subject"]
                topic = focus["topic"]
                mastery_score = focus["mastery_score"]
            else:
                subject = subjects[i % len(subjects)] if subjects else "General"
                topic = "Core concepts"
                mastery_score = 0

            plan.append(
                {
                    "Date": d.isoformat(),
                    "Subject": subject,
                    "Topic": topic,
                    "Mastery": round(mastery_score, 1),
                    "Study Minutes": int(hours * 60),
                    "Practice MCQs": max(10, int(hours * 15)),
                    "Revision Minutes": 10,
                }
            )

        result = {
            "exam_name": exam_name,
            "exam_date": exam_date.isoformat(),
            "level": level,
            "hours": hours,
            "days_per_week": days,
            "subjects": subjects,
            "goals": goals,
            "plan": plan,
        }

        conn.execute(
            """
            INSERT INTO study_plans
            (student_id, exam_name, exam_date, plan_json, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                student_id,
                exam_name,
                exam_date.isoformat(),
                json.dumps(result),
                now_iso(),
            ),
        )
        conn.commit()

        st.session_state.study_plan = result

    if st.session_state.get("study_plan"):
        st.success("Study plan created.")
        st.dataframe(
            pd.DataFrame(st.session_state.study_plan["plan"]),
            use_container_width=True,
            hide_index=True,
        )


# =========================
# Exam mode
# =========================

def page_exam(conn: sqlite3.Connection, student_id: str) -> None:
    st.header("⏱️ Exam Mode")

    subject = st.selectbox(
        "Subject",
        ["Biology", "Chemistry", "Physics", "English"],
    )
    topic = st.text_input("Topic / Chapter")
    count = st.number_input("Questions", 5, 100, 20)
    difficulty = st.selectbox(
        "Difficulty",
        ["Easy", "Medium", "Hard", "Adaptive"],
    )
    time_limit = st.number_input(
        "Time limit (minutes)",
        5, 180, 30,
    )

    if st.button("Create Exam", type="primary"):
        loaded = load_database_index()
        if not loaded:
            st.error("Database FAISS index is missing.")
            return

        index, metadata = loaded
        retrieved = hybrid_search(
            index, metadata, f"{subject} {topic}", top_k=10
        )

        context = build_context(retrieved)
        selected = difficulty

        if selected == "Adaptive":
            selected = "Medium"
            for row in get_mastery(conn, student_id):
                if (
                    row["subject"].lower() == subject.lower()
                    and row["topic"].lower() == topic.lower()
                ):
                    selected = difficulty_from_mastery(
                        row["mastery_score"]
                    )

        questions = validate_mcqs(
            generate_mcqs(
                subject,
                "",
                topic,
                selected,
                int(count),
                context,
            ),
            context,
        )

        st.session_state.exam = questions
        st.session_state.exam_answers = {}
        st.session_state.exam_started = time.time()
        st.session_state.exam_limit = int(time_limit * 60)
        st.rerun()

    questions = st.session_state.get("exam")
    if not questions:
        return

    elapsed = int(time.time() - st.session_state.exam_started)
    remaining = max(0, st.session_state.exam_limit - elapsed)
    st.metric(
        "Time remaining",
        f"{remaining // 60:02d}:{remaining % 60:02d}",
    )

    for i, q in enumerate(questions):
        choice = st.radio(
            f"{i+1}. {q['question']}",
            list(q["options"]),
            format_func=lambda k, q=q: f"{k}. {q['options'][k]}",
            key=f"exam_{i}",
        )
        st.session_state.exam_answers[i] = choice

    if st.button("Submit Exam", type="primary"):
        correct = 0
        for i, q in enumerate(questions):
            selected = st.session_state.exam_answers.get(i, "Skipped")
            is_correct = selected == q["answer"]
            correct += int(is_correct)
            record_attempt(
                conn, student_id, q, is_correct, selected
            )

        recalculate_mastery(conn, student_id)

        st.success(
            f"Exam score: {correct}/{len(questions)} "
            f"({100*correct/len(questions):.0f}%)"
        )
        st.session_state.pop("exam", None)


# =========================
# History / settings
# =========================

def page_history(conn: sqlite3.Connection, student_id: str) -> None:
    st.header("📊 Study History")

    rows = conn.execute(
        """
        SELECT attempted_at AS Date,
               subject AS Subject,
               chapter AS Chapter,
               topic AS Topic,
               difficulty AS Difficulty,
               is_correct AS Correct
        FROM question_attempts
        WHERE student_id=?
        ORDER BY attempted_at DESC
        LIMIT 500
        """,
        (student_id,),
    ).fetchall()

    if not rows:
        st.info("No study history yet.")
        return

    st.dataframe(
        pd.DataFrame([dict(r) for r in rows]),
        use_container_width=True,
        hide_index=True,
    )


def page_settings() -> None:
    st.header("⚙️ Settings")

    models = [
        "gemini-3.5-flash-lite",
        "gemini-3.1-flash-lite",
        "gemini-3.6-flash",
        "gemini-3.8-flash",
    ]

    current = st.session_state.get("llm_model", DEFAULT_MODEL)
    st.session_state.llm_model = st.selectbox(
        "LLM Model",
        models,
        index=models.index(current) if current in models else 0,
    )

    colors = ["Blue", "Green", "Purple", "Orange"]
    st.session_state.ui_color = st.selectbox(
        "Change UI color",
        colors,
        index=colors.index(
            st.session_state.get("ui_color", "Blue")
        ),
    )

    if get_secret("GEMINI_API_KEY"):
        st.success("Gemini API key detected.")
    else:
        st.warning(
            "GEMINI_API_KEY is not configured. Add it in Streamlit Secrets."
        )


# =========================
# Main
# =========================

def main() -> None:
    conn = get_db()
    student_id = get_student_id()
    ensure_student(conn, student_id)

    st.session_state.setdefault("llm_model", DEFAULT_MODEL)
    st.session_state.setdefault("ui_color", "Blue")
    st.session_state.setdefault("page", "Dashboard")

    st.sidebar.title("Prep AI V3.2")
    st.sidebar.caption("Adaptive AI Learning Platform")

    page = st.sidebar.radio(
        "Navigation",
        [
            "Dashboard",
            "Personalized Learning",
            "Database Learning",
            "Prep AI Agent",
            "AI Tutor",
            "Exam Mode",
            "Study Plan",
            "Smart Revision",
            "History",
            "Settings",
        ],
        key="page",
    )

    if page == "Dashboard":
        page_dashboard(conn, student_id)
    elif page == "Personalized Learning":
        page_personalized(conn, student_id)
    elif page == "Database Learning":
        page_database(conn, student_id)
    elif page == "Prep AI Agent":
        page_agent(conn, student_id)
    elif page == "AI Tutor":
        page_tutor(conn, student_id)
    elif page == "Exam Mode":
        page_exam(conn, student_id)
    elif page == "Study Plan":
        page_study_plan(conn, student_id)
    elif page == "Smart Revision":
        page_revision(conn, student_id)
    elif page == "History":
        page_history(conn, student_id)
    elif page == "Settings":
        page_settings()


if __name__ == "__main__":
    main()
