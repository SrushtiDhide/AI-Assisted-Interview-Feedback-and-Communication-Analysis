"""
Performance Scoring Module
Deterministic, explainable multi-component interview scoring logic.
"""


# Configurable component weights for deterministic interview scoring
SCORE_WEIGHTS = {
    "relevance": 0.35,        # 35%: Question relevance and semantic coverage
    "fluency": 0.25,          # 25%: Speaking pace (WPM) and smooth transitions
    "vocabulary": 0.20,       # 20%: Lexical richness and technical vocabulary
    "grammar": 0.10,          # 10%: Grammatical correctness and sentence balance
    "filler_control": 0.10    # 10%: Control over filler words and verbal crutches
}


def calculate_filler_control_score(filler_frequency):
    """
    Computes a score (0 - 100) reflecting how well the candidate controlled filler words.
    filler_frequency is fillers per 100 words.
    """
    if filler_frequency <= 0.5:
        return 100.0
    elif filler_frequency <= 1.5:
        return 90.0
    elif filler_frequency <= 3.0:
        return 75.0
    elif filler_frequency <= 5.0:
        return 60.0
    elif filler_frequency <= 8.0:
        return 40.0
    else:
        return max(15.0, 40.0 - (filler_frequency - 8.0) * 3.0)


def calculate_response_score(relevance_score, fluency_score, vocabulary_score, grammar_score, filler_frequency, weights=None):
    """
    Computes a transparent weighted composite score (0 - 100) for an individual answer.

    Weights (configurable via SCORE_WEIGHTS):
        - Relevance (35%): Did the candidate answer the question accurately?
        - Fluency & Speaking Speed (25%): Was speech delivery smooth and appropriately paced?
        - Vocabulary & Lexical Richness (20%): Was vocabulary varied and professional?
        - Grammar & Structure (10%): Were sentences grammatically sound and well-formed?
        - Filler Control (10%): Did the candidate minimize crutch/filler words?

    Returns:
        dict: Detailed breakdown of component scores and final weighted score.
    """
    if weights is None:
        weights = SCORE_WEIGHTS

    filler_score = calculate_filler_control_score(filler_frequency)

    weighted_score = (
        (relevance_score * weights["relevance"]) +
        (fluency_score * weights["fluency"]) +
        (vocabulary_score * weights["vocabulary"]) +
        (grammar_score * weights["grammar"]) +
        (filler_score * weights["filler_control"])
    )

    final_score = round(min(100.0, max(0.0, weighted_score)), 1)

    return {
        "total_score": final_score,
        "grade": get_performance_grade(final_score),
        "components": {
            "relevance_score": round(relevance_score, 1),
            "fluency_score": round(fluency_score, 1),
            "vocabulary_score": round(vocabulary_score, 1),
            "grammar_score": round(grammar_score, 1),
            "filler_control_score": round(filler_score, 1)
        },
        "weights": weights
    }


def calculate_overall_session_score(responses_analysis_list):
    """
    Aggregates scores across all answered questions in an interview session.
    """
    if not responses_analysis_list:
        return {
            "overall_score": 0.0,
            "grade": "N/A",
            "avg_relevance": 0.0,
            "avg_fluency": 0.0,
            "avg_vocabulary": 0.0,
            "avg_grammar": 0.0,
            "avg_filler_control": 0.0,
            "total_questions_answered": 0
        }

    n = len(responses_analysis_list)
    total_scores = [r.get("total_score", 0.0) for r in responses_analysis_list]
    overall_score = round(sum(total_scores) / n, 1)

    avg_relevance = round(sum(r["components"]["relevance_score"] for r in responses_analysis_list) / n, 1)
    avg_fluency = round(sum(r["components"]["fluency_score"] for r in responses_analysis_list) / n, 1)
    avg_vocab = round(sum(r["components"]["vocabulary_score"] for r in responses_analysis_list) / n, 1)
    avg_grammar = round(sum(r["components"]["grammar_score"] for r in responses_analysis_list) / n, 1)
    avg_filler = round(sum(r["components"]["filler_control_score"] for r in responses_analysis_list) / n, 1)

    return {
        "overall_score": overall_score,
        "grade": get_performance_grade(overall_score),
        "avg_relevance": avg_relevance,
        "avg_fluency": avg_fluency,
        "avg_vocabulary": avg_vocab,
        "avg_grammar": avg_grammar,
        "avg_filler_control": avg_filler,
        "total_questions_answered": n
    }


def get_performance_grade(score):
    """Maps numerical score to standard academic and industry performance bands."""
    if score >= 88.0:
        return "A+ (Outstanding / Placement Ready)"
    elif score >= 75.0:
        return "A (Good / Strong Competence)"
    elif score >= 60.0:
        return "B (Average / Needs Refinement)"
    elif score >= 45.0:
        return "C (Below Average / Requires Practice)"
    else:
        return "D (Needs Significant Improvement)"
