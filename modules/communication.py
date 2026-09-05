"""
Communication Analysis Module
Speaking speed (WPM), filler-word frequency, and fluency indicators.
"""


def calculate_speaking_speed(word_count, duration_seconds):
    """
    Calculates Words Per Minute (WPM) based on transcribed words and audio duration.

    Args:
        word_count (int): Number of words spoken.
        duration_seconds (float): Audio recording duration in seconds.

    Returns:
        dict: {
            'wpm': float,
            'speed_category': str,
            'speed_feedback': str,
            'speed_score': float (0-100)
        }
    """
    if duration_seconds <= 0 or word_count <= 0:
        # Default fallback for text input or zero duration
        return {
            "wpm": 135.0,  # Standard neutral conversational pace
            "speed_category": "Normal / Not Measured (Text Input)",
            "speed_feedback": "Speaking speed was not measured directly because this was a text-based response.",
            "speed_score": 85.0
        }

    duration_minutes = duration_seconds / 60.0
    wpm = round(word_count / duration_minutes, 1)

    # Standard conversational benchmarks:
    # 120 - 160 WPM: Optimal professional interview pace
    # 100 - 119 WPM: Slightly slow, deliberate
    # < 100 WPM: Very slow, hesitant
    # 161 - 180 WPM: Slightly fast
    # > 180 WPM: Very fast, difficult to follow

    if 120 <= wpm <= 160:
        category = "Optimal / Confident"
        feedback = f"Excellent speaking pace at {wpm} WPM. Your speech is clear and easy to follow."
        score = 95.0
    elif 100 <= wpm < 120:
        category = "Slightly Slow / Deliberate"
        feedback = f"Your pace is {wpm} WPM, which is slightly deliberate. Consider increasing pace moderately."
        score = 80.0
    elif wpm < 100:
        category = "Slow / Hesitant"
        feedback = f"Your pace is {wpm} WPM. Practice speaking with more continuity and reducing long pauses."
        score = max(40.0, 50.0 + (wpm / 100.0) * 30.0)
    elif 160 < wpm <= 185:
        category = "Slightly Fast / Energetic"
        feedback = f"Your pace is {wpm} WPM. You sound energetic, but slowing down slightly will improve clarity."
        score = 80.0
    else:  # > 185
        category = "Fast / Rushed"
        feedback = f"Your pace is {wpm} WPM. This is too fast for an interview; pause between key thoughts."
        score = max(40.0, 85.0 - ((wpm - 185) * 0.8))

    return {
        "wpm": wpm,
        "speed_category": category,
        "speed_feedback": feedback,
        "speed_score": round(score, 1)
    }


def calculate_fluency_indicator(word_count, filler_count, speed_score, avg_sentence_length):
    """
    Computes an integrated fluency score (0 - 100) combining speaking speed,
    filler frequency, and sentence rhythm.
    """
    if word_count == 0:
        return {
            "fluency_score": 0.0,
            "fluency_rating": "No Response",
            "fluency_feedback": "No answer provided to analyze."
        }

    # Filler density penalty (fillers per 100 words)
    filler_ratio = (filler_count / word_count) * 100
    if filler_ratio <= 1.5:
        filler_penalty = 0.0
    elif filler_ratio <= 4.0:
        filler_penalty = (filler_ratio - 1.5) * 4.0
    else:
        filler_penalty = min(35.0, 10.0 + (filler_ratio - 4.0) * 5.0)

    # Sentence length smoothness bonus/penalty
    if 10 <= avg_sentence_length <= 25:
        flow_modifier = 5.0
    elif avg_sentence_length < 5 or avg_sentence_length > 35:
        flow_modifier = -10.0
    else:
        flow_modifier = 0.0

    raw_fluency = (speed_score * 0.6) + (40.0 - filler_penalty) + flow_modifier
    fluency_score = round(min(100.0, max(20.0, raw_fluency)), 1)

    if fluency_score >= 85:
        rating = "Highly Fluent & Articulate"
        feedback = "Outstanding delivery with smooth transitions, balanced pacing, and negligible filler words."
    elif fluency_score >= 70:
        rating = "Good Fluency / Natural Flow"
        feedback = "Clear and articulate communication with minor hesitations that do not distract from the content."
    elif fluency_score >= 50:
        rating = "Moderate Fluency / Needs Polish"
        feedback = "Noticeable pauses or filler words present. Practicing structured thought delivery will help."
    else:
        rating = "Low Fluency / Frequent Disruption"
        feedback = "Speech flow is significantly hindered by frequent pauses or filler words. Focus on steady pacing."

    return {
        "fluency_score": fluency_score,
        "fluency_rating": rating,
        "fluency_feedback": feedback
    }


def analyze_communication(word_count, duration_seconds, filler_count, avg_sentence_length=15.0):
    """
    Aggregates communication analysis across speaking speed, filler density,
    and overall fluency.
    """
    speed_data = calculate_speaking_speed(word_count, duration_seconds)
    fluency_data = calculate_fluency_indicator(
        word_count=word_count,
        filler_count=filler_count,
        speed_score=speed_data["speed_score"],
        avg_sentence_length=avg_sentence_length
    )

    return {
        "speaking_speed_wpm": speed_data["wpm"],
        "speed_category": speed_data["speed_category"],
        "speed_feedback": speed_data["speed_feedback"],
        "speed_score": speed_data["speed_score"],
        "fluency_score": fluency_data["fluency_score"],
        "fluency_rating": fluency_data["fluency_rating"],
        "fluency_feedback": fluency_data["fluency_feedback"]
    }
