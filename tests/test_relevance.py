"""
Unit Tests for Relevance Analysis Module
"""

import pytest
from modules import relevance


def test_cosine_similarity_identical():
    text = "Supervised learning requires labeled data for training algorithms."
    sim = relevance.calculate_cosine_similarity(text, text)
    assert sim >= 0.99


def test_cosine_similarity_unrelated():
    text1 = "Deep neural networks optimize loss functions using gradient descent backpropagation."
    text2 = "Italian pizza dough is made with flour yeast water salt and baked in an oven."
    sim = relevance.calculate_cosine_similarity(text1, text2)
    assert sim < 0.20


def test_check_keyword_coverage():
    candidate = "In supervised machine learning, models are trained on labeled datasets."
    keywords = "supervised, machine learning, labeled, regression"
    result = relevance.check_keyword_coverage(candidate, keywords)

    assert result["coverage_ratio"] >= 0.75
    assert "supervised" in result["matched_keywords"]
    assert "labeled" in result["matched_keywords"]
    assert "regression" in result["missing_keywords"]


def test_evaluate_relevance():
    question_data = {
        "question_id": 21,
        "category": "Technical",
        "question": "Explain the difference between supervised and unsupervised machine learning.",
        "keywords": "supervised, unsupervised, labeled, unlabeled, classification, clustering",
        "expected_topics": "Machine Learning, Supervision, Labels"
    }

    # Good answer
    good_answer = (
        "Supervised learning trains on labeled datasets for classification and regression tasks. "
        "Unsupervised learning deals with unlabeled data to discover hidden patterns and clustering."
    )
    res_good = relevance.evaluate_relevance(good_answer, question_data)
    assert res_good["relevance_score"] >= 65.0
    assert "supervised" in res_good["matched_keywords"]

    # Irrelevant answer
    bad_answer = "I enjoy traveling to mountains and drinking cold coffee in the morning."
    res_bad = relevance.evaluate_relevance(bad_answer, question_data)
    assert res_bad["relevance_score"] < 40.0
