"""
Unit Tests for NLP Analysis Module
"""

import pytest
from modules import nlp_analysis


def test_clean_text():
    raw = "  Hello    world!  This   is a test. \n\n"
    cleaned = nlp_analysis.clean_text(raw)
    assert cleaned == "Hello world! This is a test."


def test_tokenize_words():
    text = "Data Science and AI are transforming industries!"
    words = nlp_analysis.tokenize_words(text)
    assert "data" in words
    assert "science" in words
    assert "ai" in words
    assert len(words) == 7


def test_tokenize_sentences():
    text = "First sentence. Second sentence! Third sentence? Yes."
    sentences = nlp_analysis.tokenize_sentences(text)
    assert len(sentences) == 4
    assert sentences[0] == "First sentence."


def test_detect_filler_words():
    text = "Um, basically, I think that, you know, machine learning is sort of like statistics."
    result = nlp_analysis.detect_filler_words(text)

    assert result["filler_count"] >= 4
    assert "um" in result["fillers_found"]
    assert "basically" in result["fillers_found"]
    assert "you know" in result["fillers_found"]
    assert "sort of" in result["fillers_found"]
    assert result["filler_frequency"] > 0


def test_analyze_vocabulary():
    text = (
        "Supervised learning algorithms build mathematical representations "
        "of sample data to make predictions or decisions without being explicitly programmed."
    )
    vocab = nlp_analysis.analyze_vocabulary(text)

    assert vocab["total_words"] > 10
    assert vocab["unique_words"] > 5
    assert 0.0 < vocab["ttr"] <= 1.0
    assert vocab["vocabulary_score"] > 0


def test_analyze_grammar_and_structure():
    text = "I have developed machine learning models using Python and SQL. These models achieved high precision."
    grammar = nlp_analysis.analyze_grammar_and_structure(text)

    assert grammar["sentence_count"] == 2
    assert grammar["grammar_score"] >= 70.0
    assert grammar["readability_index"] > 0
