"""
Database Management Module
SQLite connection, schema creation, data seeding, and query operations.
"""

import os
import sqlite3
import pandas as pd
from datetime import datetime

# Default database path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DB_PATH = os.path.join(BASE_DIR, "database", "interview.db")
DEFAULT_CSV_PATH = os.path.join(BASE_DIR, "data", "questions.csv")


def get_db_connection(db_path=None):
    """Establishes and returns a connection to the SQLite database."""
    if db_path is None:
        db_path = DEFAULT_DB_PATH
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db(db_path=None, csv_path=None):
    """
    Initializes database tables if they do not exist
    and automatically seeds the predefined question bank.
    """
    if db_path is None:
        db_path = DEFAULT_DB_PATH
    if csv_path is None:
        csv_path = DEFAULT_CSV_PATH

    conn = get_db_connection(db_path)
    cursor = conn.cursor()

    # 1. Users Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        user_id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        email TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)

    # 2. Questions Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS questions (
        question_id INTEGER PRIMARY KEY AUTOINCREMENT,
        category TEXT NOT NULL,
        question TEXT NOT NULL,
        keywords TEXT,
        difficulty TEXT DEFAULT 'Medium',
        expected_topics TEXT
    );
    """)

    # 3. Sessions Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS sessions (
        session_id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        category TEXT NOT NULL,
        session_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        overall_score REAL DEFAULT 0.0,
        grade TEXT DEFAULT 'N/A',
        status TEXT DEFAULT 'In Progress',
        FOREIGN KEY(user_id) REFERENCES users(user_id) ON DELETE CASCADE
    );
    """)

    # 4. Responses Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS responses (
        response_id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id INTEGER NOT NULL,
        question_id INTEGER NOT NULL,
        response_type TEXT DEFAULT 'text',
        text TEXT NOT NULL,
        audio_path TEXT,
        duration REAL DEFAULT 0.0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(session_id) REFERENCES sessions(session_id) ON DELETE CASCADE,
        FOREIGN KEY(question_id) REFERENCES questions(question_id)
    );
    """)

    # 5. Analysis Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS analysis (
        analysis_id INTEGER PRIMARY KEY AUTOINCREMENT,
        response_id INTEGER UNIQUE NOT NULL,
        relevance_score REAL DEFAULT 0.0,
        filler_count INTEGER DEFAULT 0,
        filler_frequency REAL DEFAULT 0.0,
        speaking_speed_wpm REAL DEFAULT 0.0,
        vocabulary_score REAL DEFAULT 0.0,
        grammar_score REAL DEFAULT 0.0,
        fluency_score REAL DEFAULT 0.0,
        FOREIGN KEY(response_id) REFERENCES responses(response_id) ON DELETE CASCADE
    );
    """)

    # 6. Feedback Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS feedback (
        feedback_id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id INTEGER NOT NULL,
        response_id INTEGER,
        strengths TEXT,
        weaknesses TEXT,
        recommendations TEXT,
        summary TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(session_id) REFERENCES sessions(session_id) ON DELETE CASCADE,
        FOREIGN KEY(response_id) REFERENCES responses(response_id) ON DELETE SET NULL
    );
    """)

    conn.commit()

    # Seed Questions if empty
    cursor.execute("SELECT COUNT(*) FROM questions")
    count = cursor.fetchone()[0]
    if count == 0 and os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                cursor.execute("""
                INSERT INTO questions (category, question, keywords, difficulty, expected_topics)
                VALUES (?, ?, ?, ?, ?)
                """, (
                    row.get('category', 'HR'),
                    row.get('question', ''),
                    row.get('keywords', ''),
                    row.get('difficulty', 'Medium'),
                    row.get('expected_topics', '')
                ))
            conn.commit()
        except Exception as e:
            print(f"Notice: Seeding question bank encountered: {e}")

    conn.close()


def get_or_create_user(name, email=None, db_path=None):
    """Retrieves an existing user by name or registers a new user."""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE name = ?", (name.strip(),))
    user = cursor.fetchone()

    if user:
        user_dict = dict(user)
        conn.close()
        return user_dict
    else:
        cursor.execute("INSERT INTO users (name, email) VALUES (?, ?)", (name.strip(), email))
        conn.commit()
        new_id = cursor.lastrowid
        cursor.execute("SELECT * FROM users WHERE user_id = ?", (new_id,))
        new_user = dict(cursor.fetchone())
        conn.close()
        return new_user


def get_all_questions(category=None, db_path=None):
    """Fetches questions optionally filtered by category."""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    if category and category.lower() != 'all':
        cursor.execute("SELECT * FROM questions WHERE category = ? ORDER BY question_id", (category,))
    else:
        cursor.execute("SELECT * FROM questions ORDER BY question_id")
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_question_by_id(question_id, db_path=None):
    """Retrieves a single question by its primary key ID."""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM questions WHERE question_id = ?", (question_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def create_session(user_id, category, db_path=None):
    """Creates a new interview session and returns the session_id."""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("""
    INSERT INTO sessions (user_id, category, status)
    VALUES (?, ?, 'In Progress')
    """, (user_id, category))
    conn.commit()
    session_id = cursor.lastrowid
    conn.close()
    return session_id


def update_session_score(session_id, overall_score, grade="Good", status="Completed", db_path=None):
    """Updates the final score, grade, and status for a session."""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("""
    UPDATE sessions
    SET overall_score = ?, grade = ?, status = ?, session_date = CURRENT_TIMESTAMP
    WHERE session_id = ?
    """, (overall_score, grade, status, session_id))
    conn.commit()
    conn.close()


def save_response(session_id, question_id, response_type, text, audio_path=None, duration=0.0, db_path=None):
    """Saves candidate response and returns the response_id."""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("""
    INSERT INTO responses (session_id, question_id, response_type, text, audio_path, duration)
    VALUES (?, ?, ?, ?, ?, ?)
    """, (session_id, question_id, response_type, text, audio_path, duration))
    conn.commit()
    response_id = cursor.lastrowid
    conn.close()
    return response_id


def save_analysis(response_id, metrics, db_path=None):
    """Saves calculated NLP and communication metrics for a response."""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("""
    INSERT OR REPLACE INTO analysis (
        response_id, relevance_score, filler_count, filler_frequency,
        speaking_speed_wpm, vocabulary_score, grammar_score, fluency_score
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        response_id,
        metrics.get('relevance_score', 0.0),
        metrics.get('filler_count', 0),
        metrics.get('filler_frequency', 0.0),
        metrics.get('speaking_speed_wpm', 0.0),
        metrics.get('vocabulary_score', 0.0),
        metrics.get('grammar_score', 0.0),
        metrics.get('fluency_score', 0.0)
    ))
    conn.commit()
    analysis_id = cursor.lastrowid
    conn.close()
    return analysis_id


def save_feedback(session_id, response_id, strengths, weaknesses, recommendations, summary, db_path=None):
    """Stores AI-generated feedback."""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("""
    INSERT INTO feedback (session_id, response_id, strengths, weaknesses, recommendations, summary)
    VALUES (?, ?, ?, ?, ?, ?)
    """, (session_id, response_id, strengths, weaknesses, recommendations, summary))
    conn.commit()
    feedback_id = cursor.lastrowid
    conn.close()
    return feedback_id


def get_user_sessions(user_id, db_path=None):
    """Fetches all completed or active sessions for a user."""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("""
    SELECT session_id, category, session_date, overall_score, grade, status
    FROM sessions
    WHERE user_id = ?
    ORDER BY session_id DESC
    """, (user_id,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_all_sessions(db_path=None):
    """Fetches all sessions with username."""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("""
    SELECT s.session_id, u.name, s.category, s.session_date, s.overall_score, s.grade, s.status
    FROM sessions s
    LEFT JOIN users u ON s.user_id = u.user_id
    ORDER BY s.session_id DESC
    """)
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_session_details(session_id, db_path=None):
    """Fetches detailed questions, responses, analysis, and feedback for a session."""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("""
    SELECT 
        r.response_id,
        q.question_id,
        q.category,
        q.question,
        q.keywords,
        r.response_type,
        r.text AS candidate_answer,
        r.duration,
        a.relevance_score,
        a.filler_count,
        a.filler_frequency,
        a.speaking_speed_wpm,
        a.vocabulary_score,
        a.grammar_score,
        a.fluency_score,
        f.strengths,
        f.weaknesses,
        f.recommendations,
        f.summary
    FROM responses r
    JOIN questions q ON r.question_id = q.question_id
    LEFT JOIN analysis a ON r.response_id = a.response_id
    LEFT JOIN feedback f ON r.response_id = f.response_id
    WHERE r.session_id = ?
    ORDER BY r.response_id ASC
    """, (session_id,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_user_progress_data(user_id, db_path=None):
    """Retrieves chronological score history for progress plotting."""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("""
    SELECT session_id, session_date, category, overall_score, grade
    FROM sessions
    WHERE user_id = ? AND status = 'Completed'
    ORDER BY session_id ASC
    """, (user_id,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]
