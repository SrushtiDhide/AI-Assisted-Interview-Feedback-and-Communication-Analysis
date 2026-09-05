"""
Unit Tests for Database Module
"""

import os
import gc
import shutil
import tempfile
import pytest
from modules import database


@pytest.fixture
def temp_db():
    temp_dir = tempfile.mkdtemp()
    db_file = os.path.join(temp_dir, "test_interview.db")
    database.init_db(db_path=db_file)
    yield db_file
    gc.collect()
    try:
        shutil.rmtree(temp_dir, ignore_errors=True)
    except Exception:
        pass


def test_user_creation(temp_db):
    user = database.get_or_create_user("Test Student", "student@college.edu", db_path=temp_db)
    assert user["user_id"] > 0
    assert user["name"] == "Test Student"

    # Fetching again should return existing user
    user2 = database.get_or_create_user("Test Student", db_path=temp_db)
    assert user2["user_id"] == user["user_id"]


def test_session_lifecycle(temp_db):
    user = database.get_or_create_user("Candidate One", db_path=temp_db)
    session_id = database.create_session(user["user_id"], "HR", db_path=temp_db)
    assert session_id > 0

    # Save dummy response
    resp_id = database.save_response(
        session_id=session_id,
        question_id=1,
        response_type="text",
        text="Sample candidate response text",
        duration=15.0,
        db_path=temp_db
    )
    assert resp_id > 0

    # Save analysis
    analysis_id = database.save_analysis(resp_id, {
        "relevance_score": 85.0,
        "filler_count": 1,
        "filler_frequency": 2.0,
        "speaking_speed_wpm": 130.0,
        "vocabulary_score": 80.0,
        "grammar_score": 90.0,
        "fluency_score": 88.0
    }, db_path=temp_db)
    assert analysis_id > 0

    # Update session score
    database.update_session_score(session_id, 88.5, grade="A+", status="Completed", db_path=temp_db)

    # Verify session retrieval
    sessions = database.get_user_sessions(user["user_id"], db_path=temp_db)
    assert len(sessions) == 1
    assert sessions[0]["overall_score"] == 88.5
    assert sessions[0]["status"] == "Completed"
