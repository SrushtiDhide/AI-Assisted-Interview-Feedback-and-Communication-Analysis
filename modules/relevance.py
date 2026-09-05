"""
Answer Relevance Analysis Module
TF-IDF vectorization, Cosine Similarity, and Keyword Coverage calculation.
"""

import os
import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_REF_PATH = os.path.join(BASE_DIR, "data", "reference_answers.csv")

# Cache reference answers dictionary
_REFERENCE_CACHE = None


def load_reference_answers(file_path=None):
    """Loads reference answers into an in-memory dictionary keyed by question_id."""
    global _REFERENCE_CACHE
    if _REFERENCE_CACHE is not None:
        return _REFERENCE_CACHE

    if file_path is None:
        file_path = DEFAULT_REF_PATH

    ref_dict = {}
    if os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path)
            for _, row in df.iterrows():
                qid = int(row['question_id'])
                ref_dict[qid] = {
                    "reference_answer": str(row.get('reference_answer', '')),
                    "ideal_points": str(row.get('ideal_points', '')),
                    "sample_good_answer": str(row.get('sample_good_answer', '')),
                    "sample_weak_answer": str(row.get('sample_weak_answer', ''))
                }
            _REFERENCE_CACHE = ref_dict
            return ref_dict
        except Exception as e:
            print(f"Error loading reference answers: {e}")

    _REFERENCE_CACHE = {}
    return _REFERENCE_CACHE


def calculate_cosine_similarity(text1, text2):
    """
    Computes cosine similarity between two text strings using TF-IDF representation.
    Returns float value between 0.0 and 1.0.
    """
    if not text1 or not text2:
        return 0.0

    t1 = text1.strip()
    t2 = text2.strip()
    if not t1 or not t2:
        return 0.0

    try:
        vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),
            token_pattern=r"(?u)\b[a-zA-Z0-9_-]+\b"
        )
        tfidf_matrix = vectorizer.fit_transform([t1, t2])
        sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return float(np.clip(sim, 0.0, 1.0))
    except Exception:
        # Fallback to token Jaccard similarity if TF-IDF fails (e.g., all stop words)
        set1 = set(re.findall(r'\b\w+\b', t1.lower()))
        set2 = set(re.findall(r'\b\w+\b', t2.lower()))
        union = set1.union(set2)
        if not union:
            return 0.0
        return float(len(set1.intersection(set2)) / len(union))


def calculate_embedding_similarity(text1, text2, model_name="all-MiniLM-L6-v2"):
    """
    Extension Hook: Semantic embedding similarity using Sentence-Transformers.
    Can be seamlessly swapped with calculate_cosine_similarity for dense embedding evaluation.
    Left as modular extension point for future deep learning enhancements.
    """
    try:
        from sentence_transformers import SentenceTransformer
        embedder = SentenceTransformer(model_name)
        embeddings = embedder.encode([text1, text2])
        sim = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        return float(np.clip(sim, 0.0, 1.0))
    except ImportError:
        # Gracefully defaults to TF-IDF cosine similarity when sentence-transformers is not installed
        return calculate_cosine_similarity(text1, text2)


def check_keyword_coverage(candidate_text, keywords_str):
    """
    Checks what proportion of required keywords are present in the candidate answer.
    """
    if not keywords_str or not candidate_text:
        return {
            "coverage_ratio": 0.0,
            "matched_keywords": [],
            "missing_keywords": []
        }

    raw_keywords = [k.strip().lower() for k in keywords_str.split(',') if k.strip()]
    if not raw_keywords:
        return {
            "coverage_ratio": 1.0,
            "matched_keywords": [],
            "missing_keywords": []
        }

    candidate_lower = candidate_text.lower()
    matched = []
    missing = []

    for kw in raw_keywords:
        # Check boundary match or substring for multi-word
        pattern = r'\b' + re.escape(kw) + r'\b'
        if re.search(pattern, candidate_lower):
            matched.append(kw)
        else:
            missing.append(kw)

    ratio = len(matched) / len(raw_keywords)
    return {
        "coverage_ratio": round(ratio, 2),
        "matched_keywords": matched,
        "missing_keywords": missing
    }


def evaluate_relevance(candidate_answer, question_data, ref_answers_dict=None):
    """
    Evaluates comprehensive answer relevance against question text,
    keywords, and official reference answers.

    Args:
        candidate_answer (str): The candidate's spoken or typed answer.
        question_data (dict): Question row containing 'question_id', 'question', 'keywords', 'expected_topics'.
        ref_answers_dict (dict): Optional pre-loaded reference answers.

    Returns:
        dict: Detailed relevance breakdown and normalized score (0 to 100).
    """
    if not candidate_answer or len(candidate_answer.strip().split()) < 3:
        return {
            "relevance_score": 0.0,
            "cosine_similarity_ref": 0.0,
            "cosine_similarity_question": 0.0,
            "keyword_coverage_ratio": 0.0,
            "matched_keywords": [],
            "missing_keywords": [],
            "relevance_rating": "Incomplete / Blank Answer"
        }

    if ref_answers_dict is None:
        ref_answers_dict = load_reference_answers()

    qid = question_data.get("question_id")
    question_text = question_data.get("question", "")
    keywords_str = question_data.get("keywords", "")
    topics_str = question_data.get("expected_topics", "")

    # Retrieve reference answer if available
    ref_info = ref_answers_dict.get(qid, {})
    reference_answer = ref_info.get("reference_answer", "")
    ideal_points = ref_info.get("ideal_points", "")

    # 1. Cosine similarity against question + topics
    question_context = f"{question_text} {topics_str}".strip()
    sim_question = calculate_cosine_similarity(candidate_answer, question_context)

    # 2. Cosine similarity against reference answer + ideal points
    if reference_answer:
        ref_target = f"{reference_answer} {ideal_points}".strip()
        sim_ref = calculate_cosine_similarity(candidate_answer, ref_target)
    else:
        sim_ref = sim_question

    # 3. Keyword coverage
    kw_result = check_keyword_coverage(candidate_answer, keywords_str)
    kw_ratio = kw_result["coverage_ratio"]

    # Modular Similarity Scaling:
    # In bag-of-words/TF-IDF models, two human answers expressing the same concept rarely exceed 0.50 cosine similarity.
    # We normalize TF-IDF similarity into a calibrated 0.0 - 1.0 confidence range:
    sim_ref_norm = min(1.0, sim_ref / 0.50) if sim_ref > 0 else 0.0
    sim_q_norm = min(1.0, sim_question / 0.40) if sim_question > 0 else 0.0

    # Composite weighted relevance (0 to 100):
    # - 45% Reference Answer Semantic Alignment (TF-IDF Cosine Similarity)
    # - 15% Direct Question & Expected Topics Relevance
    # - 40% Crucial Technical/Behavioral Keyword Coverage
    composite = (sim_ref_norm * 0.45) + (sim_q_norm * 0.15) + (kw_ratio * 0.40)
    relevance_score = round(min(100.0, max(0.0, composite * 100.0)), 1)

    # Categorization
    if relevance_score >= 80:
        rating = "Highly Relevant & Comprehensive"
    elif relevance_score >= 60:
        rating = "Good Relevance & Adequate"
    elif relevance_score >= 40:
        rating = "Partially Relevant / Missing Key Concepts"
    else:
        rating = "Low Relevance / Off-Topic"

    return {
        "relevance_score": relevance_score,
        "cosine_similarity_ref": round(sim_ref, 3),
        "cosine_similarity_question": round(sim_question, 3),
        "keyword_coverage_ratio": kw_ratio,
        "matched_keywords": kw_result["matched_keywords"],
        "missing_keywords": kw_result["missing_keywords"],
        "relevance_rating": rating
    }
