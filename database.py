# -*- coding: utf-8 -*-
"""SQLite persistence for users, multi-angle embeddings, attendance logs."""

import os
import sqlite3
import time
from contextlib import contextmanager

import numpy as np

from config import DB_PATH, DATA_DIR
from utils import emb_to_bytes, bytes_to_emb


def _ensure_dir():
    os.makedirs(DATA_DIR, exist_ok=True)


@contextmanager
def get_conn():
    _ensure_dir()
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    with get_conn() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            created_at REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS embeddings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            embedding BLOB NOT NULL,
            angle TEXT,
            score REAL DEFAULT 0.0,
            created_at REAL NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
        );
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            name TEXT,
            score REAL,
            timestamp REAL NOT NULL
        );
        """)
        # index
        conn.execute("CREATE INDEX IF NOT EXISTS idx_emb_user ON embeddings(user_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_att_time ON attendance(timestamp)")


def add_user(name):
    with get_conn() as conn:
        cur = conn.execute("INSERT INTO users(name, created_at) VALUES (?, ?)", (name, time.time()))
        return cur.lastrowid


def delete_user(user_id):
    with get_conn() as conn:
        conn.execute("DELETE FROM embeddings WHERE user_id=?", (user_id,))
        conn.execute("DELETE FROM users WHERE id=?", (user_id,))


def add_embedding(user_id, embedding, angle="front", score=0.0):
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO embeddings(user_id, embedding, angle, score, created_at) VALUES (?,?,?,?,?)",
            (user_id, emb_to_bytes(embedding), angle, float(score), time.time()),
        )


def get_all_profiles():
    """Return list of (user_id, name, embedding ndarray)."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT u.id, u.name, e.embedding FROM users u JOIN embeddings e ON u.id=e.user_id"
        ).fetchall()
    out = []
    for r in rows:
        out.append((r["id"], r["name"], bytes_to_emb(r["embedding"])))
    return out


def list_users():
    with get_conn() as conn:
        rows = conn.execute("SELECT id, name, created_at FROM users ORDER BY id").fetchall()
    return [dict(r) for r in rows]


def user_embedding_count(user_id):
    with get_conn() as conn:
        r = conn.execute("SELECT COUNT(*) c FROM embeddings WHERE user_id=?", (user_id,)).fetchone()
    return int(r["c"]) if r else 0


def last_attendance(user_id):
    with get_conn() as conn:
        r = conn.execute(
            "SELECT timestamp FROM attendance WHERE user_id=? ORDER BY timestamp DESC LIMIT 1",
            (user_id,),
        ).fetchone()
    return float(r["timestamp"]) if r else None


def can_attend(user_id, cooldown=None):
    from config import ATTENDANCE_COOLDOWN_SEC
    if cooldown is None:
        cooldown = ATTENDANCE_COOLDOWN_SEC
    la = last_attendance(user_id)
    if la is None:
        return True
    return (time.time() - la) >= cooldown


def log_attendance(user_id, name, score):
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO attendance(user_id, name, score, timestamp) VALUES (?,?,?,?)",
            (user_id, name, float(score), time.time()),
        )


def recent_attendance(limit=20):
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT name, score, timestamp FROM attendance ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


if __name__ == "__main__":
    init_db()
    print("DB initialized at", DB_PATH)
