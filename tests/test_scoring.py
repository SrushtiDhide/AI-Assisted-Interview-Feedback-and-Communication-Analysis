"""
Unit Tests for Scoring Module
"""

import pytest
from modules import scoring


def test_filler_control_score():
    score_low_filler = scoring.calculate_filler_control_score(0.2)
    score_med_filler = scoring.calculate_filler_control_score(3.5)
    score_high_filler = scoring.calculate_filler_control_score(10.0)

    assert score_low_filler == 100.0
    assert score_med_filler < score_low_filler
    assert score_high_filler < score_med_filler


def test_calculate_response_score_bounds():
    # Maximum possible scores
    perfect = scoring.calculate_response_score(
        relevance_score=100.0,
        fluency_score=100.0,
        vocabulary_score=100.0,
        grammar_score=100.0,
        filler_frequency=0.0
    )
    assert perfect["total_score"] == 100.0
    assert "A+" in perfect["grade"]

    # Poor scores
    poor = scoring.calculate_response_score(
        relevance_score=20.0,
        fluency_score=30.0,
        vocabulary_score=20.0,
        grammar_score=25.0,
        filler_frequency=12.0
    )
    assert poor["total_score"] < 40.0
    assert "D" in poor["grade"]


def test_calculate_overall_session_score():
    r1 = scoring.calculate_response_score(80, 80, 80, 80, 1.0)
    r2 = scoring.calculate_response_score(90, 90, 90, 90, 0.5)

    summary = scoring.calculate_overall_session_score([r1, r2])
    assert summary["total_questions_answered"] == 2
    assert 80.0 <= summary["overall_score"] <= 95.0
    assert summary["avg_relevance"] == 85.0
