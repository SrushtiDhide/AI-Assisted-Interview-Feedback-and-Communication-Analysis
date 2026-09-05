"""
LLM Feedback Module
Ollama (Qwen2.5-3B-Instruct) integration for personalized feedback and question generation,
with a robust deterministic rule-based fallback when Ollama is offline.
"""

import json
import requests

OLLAMA_API_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "qwen2.5:3b"


def check_ollama_status(api_url="http://localhost:11434"):
    """Checks if the local Ollama daemon is running and reachable."""
    try:
        response = requests.get(f"{api_url}/api/tags", timeout=1.5)
        if response.status_code == 200:
            models = [m.get("name") for m in response.json().get("models", [])]
            return {
                "available": True,
                "models": models,
                "default_model_found": any(DEFAULT_MODEL in m for m in models)
            }
    except Exception:
        pass
    return {"available": False, "models": [], "default_model_found": False}


def generate_llm_response(prompt, system_prompt=None, model=DEFAULT_MODEL, timeout=30):
    """
    Sends a generation request to the local Ollama service.
    Returns generated text if successful, or None if Ollama is unreachable.
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.3,
            "top_p": 0.9,
            "num_predict": 450
        }
    }
    if system_prompt:
        payload["system"] = system_prompt

    try:
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=timeout)
        if response.status_code == 200:
            result = response.json()
            return result.get("response", "").strip()
    except Exception:
        pass
    return None


def generate_personalized_feedback(question_text, candidate_answer, metrics_summary, model=DEFAULT_MODEL):
    """
    Generates structured interview feedback: Strengths, Weaknesses, Recommendations, and Summary.
    Uses Qwen2.5-3B via Ollama if available, otherwise employs a comprehensive rule-based feedback generator.
    """
    system_prompt = (
        "You are an expert, constructive AI Interview Coach and Communication Assessor. "
        "You provide encouraging, highly specific feedback based on measured linguistic and technical metrics. "
        "Never invent fake numerical scores; strictly use the provided data. "
        "Format your answer with clear bulleted sections: "
        "### Strengths\n### Areas for Improvement\n### Actionable Recommendations\n### Summary"
    )

    user_prompt = f"""
Please analyze this candidate's interview performance based on the following measured metrics:

INTERVIEW QUESTION:
"{question_text}"

CANDIDATE ANSWER:
"{candidate_answer}"

MEASURED PERFORMANCE METRICS:
- Overall Answer Score: {metrics_summary.get('overall_score', 0)}/100 ({metrics_summary.get('grade', 'N/A')})
- Content Relevance: {metrics_summary.get('relevance_score', 0)}/100 ({metrics_summary.get('relevance_rating', '')})
- Speaking Speed: {metrics_summary.get('speaking_speed_wpm', 0)} Words Per Minute ({metrics_summary.get('speed_category', '')})
- Filler Words Count: {metrics_summary.get('filler_count', 0)} ({metrics_summary.get('filler_frequency', 0)}% of total words)
- Fluency Score: {metrics_summary.get('fluency_score', 0)}/100
- Vocabulary & Language Score: {metrics_summary.get('vocabulary_score', 0)}/100
- Grammar & Structure Score: {metrics_summary.get('grammar_score', 0)}/100

Please provide:
1. Strengths (2-3 concise bullet points)
2. Areas for Improvement (2-3 concise bullet points)
3. Actionable Recommendations (3 concrete tips to implement immediately)
4. A 2-sentence motivating Summary.
"""

    llm_output = generate_llm_response(user_prompt, system_prompt=system_prompt, model=model)

    if llm_output:
        return parse_or_format_feedback(llm_output, source="Ollama (Qwen2.5-3B-Instruct)")
    else:
        # Fallback to intelligent deterministic rule-based feedback
        return generate_rule_based_feedback(question_text, candidate_answer, metrics_summary)


def generate_rule_based_feedback(question_text, candidate_answer, metrics):
    """
    Deterministic rule-based feedback generator that evaluates metrics
    and constructs professional, actionable advice when LLM is offline.
    """
    rel_score = metrics.get("relevance_score", 0.0)
    wpm = metrics.get("speaking_speed_wpm", 0.0)
    fillers = metrics.get("filler_count", 0)
    vocab = metrics.get("vocabulary_score", 0.0)
    grammar = metrics.get("grammar_score", 0.0)
    matched_kw = metrics.get("matched_keywords", [])
    missing_kw = metrics.get("missing_keywords", [])

    strengths = []
    weaknesses = []
    recommendations = []

    # Relevance feedback
    if rel_score >= 75:
        strengths.append(f"Strong answer relevance ({rel_score}%); successfully addressed core concepts.")
        if matched_kw:
            strengths.append(f"Demonstrated domain knowledge by effectively using key terms: {', '.join(matched_kw[:4])}.")
    elif rel_score >= 50:
        strengths.append("Adequate foundational grasp of the question topic.")
        weaknesses.append("The response partially answered the prompt but missed some critical technical or situational details.")
    else:
        weaknesses.append(f"Answer relevance was low ({rel_score}%). The explanation drifted from the core question.")

    if missing_kw and len(missing_kw) <= 4:
        weaknesses.append(f"Key expected points or terms were omitted: {', '.join(missing_kw)}.")

    # Speed & Communication feedback
    if 120 <= wpm <= 160:
        strengths.append(f"Ideal speaking cadence ({wpm} WPM). Speech pacing was calm, professional, and clear.")
    elif wpm < 110 and wpm > 0:
        weaknesses.append(f"Speaking pace ({wpm} WPM) was slower than the optimal 130-150 WPM range, signaling hesitation.")
        recommendations.append("Practice speaking with continuous phrasing, rehearsing opening statements to reduce initial pauses.")
    elif wpm > 165:
        weaknesses.append(f"Speaking speed ({wpm} WPM) was rapid, which may reduce clarity for the interviewer.")
        recommendations.append("Incorporate deliberate 1-second pauses between key points to give listeners time to absorb your ideas.")

    # Filler words feedback
    if fillers == 0:
        strengths.append("Exceptional verbal discipline with zero detectable filler words.")
    elif fillers <= 2:
        strengths.append(f"Great verbal control with minimal filler usage ({fillers} filler words).")
    else:
        weaknesses.append(f"Detected {fillers} filler words/phrases, which breaks communication flow.")
        recommendations.append("Replace filler words (like 'um', 'basically', 'you know') with a brief silent breath before speaking.")

    # Vocabulary & Grammar
    if vocab >= 75:
        strengths.append("Varied and professional vocabulary with good lexical diversity.")
    else:
        recommendations.append("Enhance technical vocabulary by incorporating standard industry terminology instead of generic terms.")

    if grammar < 70:
        weaknesses.append("Several sentences were overly fragmented or had structural inconsistencies.")
        recommendations.append("Structure answers using the STAR (Situation, Task, Action, Result) method for coherent storytelling.")

    # Ensure minimum items
    if not strengths:
        strengths.append("Willingness to attempt the question and clear audio recording.")
    if not weaknesses:
        weaknesses.append("Minor polish needed to make responses even more concise and impactful.")
    if not recommendations:
        recommendations.append("Continue practicing with diverse question types to maintain this strong performance.")

    summary = (
        f"You delivered a solid response with an overall rating of {metrics.get('grade', 'Competent')}. "
        f"Focusing on the highlighted recommendations will elevate your interview performance to an exceptional standard."
    )

    return {
        "strengths": "\n".join([f"• {s}" for s in strengths[:3]]),
        "weaknesses": "\n".join([f"• {w}" for w in weaknesses[:3]]),
        "recommendations": "\n".join([f"1. {r}" if not r.startswith("1") else r for r in recommendations[:3]]),
        "summary": summary,
        "raw_text": f"### Strengths\n" + "\n".join([f"• {s}" for s in strengths]) + "\n\n### Areas for Improvement\n" + "\n".join([f"• {w}" for w in weaknesses]) + "\n\n### Actionable Recommendations\n" + "\n".join([f"• {r}" for r in recommendations]) + f"\n\n### Summary\n{summary}",
        "source": "Rule-Based Expert Engine (Offline Mode)"
    }


def parse_or_format_feedback(llm_output, source="Ollama"):
    """Extracts structured sections from the LLM response text."""
    strengths = ""
    weaknesses = ""
    recs = ""
    summary = ""

    # Parse sections if markdown headers are present
    sections = llm_output.split("###")
    for sec in sections:
        sec_lower = sec.lower().strip()
        if sec_lower.startswith("strength"):
            strengths = sec.replace("Strengths", "").strip()
        elif sec_lower.startswith("areas for improvement") or sec_lower.startswith("weakness"):
            weaknesses = sec.replace("Areas for Improvement", "").replace("Weaknesses", "").strip()
        elif sec_lower.startswith("actionable recommendation") or sec_lower.startswith("recommendation"):
            recs = sec.replace("Actionable Recommendations", "").replace("Recommendations", "").strip()
        elif sec_lower.startswith("summary"):
            summary = sec.replace("Summary", "").strip()

    if not strengths:
        strengths = llm_output[:len(llm_output)//3]
    if not weaknesses:
        weaknesses = llm_output[len(llm_output)//3:2*len(llm_output)//3]
    if not recs:
        recs = llm_output[2*len(llm_output)//3:]
    if not summary:
        summary = "Overall solid response with room for polish in structured delivery."

    return {
        "strengths": strengths,
        "weaknesses": weaknesses,
        "recommendations": recs,
        "summary": summary,
        "raw_text": llm_output,
        "source": source
    }


def generate_followup_question(previous_question, candidate_answer):
    """
    Generates an intelligent follow-up interview question based on the candidate's previous response.
    """
    prompt = (
        f"You are a professional technical and HR interviewer. "
        f"The candidate was asked: '{previous_question}'.\n"
        f"The candidate answered: '{candidate_answer}'.\n"
        f"Generate ONE short, sharp, challenging follow-up question directly probing a detail from their answer."
    )
    result = generate_llm_response(prompt)
    if result:
        # Clean quotes
        clean_q = result.replace('"', '').strip()
        # Return first sentence if multiple lines
        return clean_q.split('\n')[0]
    return f"Can you elaborate further on a specific real-world challenge you faced while implementing that?"


def generate_questions_from_resume(resume_text, count=5):
    """
    Extracts skills and projects from resume text and generates customized interview questions (FR-13).
    """
    prompt = (
        f"Based on the following candidate resume text, generate {count} specific, rigorous interview questions "
        f"(a mix of technical and project-related questions) that test the candidate's claimed skills.\n\n"
        f"RESUME TEXT:\n{resume_text[:2000]}\n\n"
        f"Return ONLY the numbered list of questions."
    )
    result = generate_llm_response(prompt)
    if result:
        lines = [line.strip() for line in result.split('\n') if line.strip() and (line[0].isdigit() or line.startswith('-'))]
        if lines:
            return lines[:count]

    # Offline fallback questions based on keyword matching
    fallback_q = [
        "1. Can you walk me through the architecture and data pipeline of the most challenging project mentioned on your resume?",
        "2. Which machine learning algorithm did you use in your projects, and how did you validate its performance?",
        "3. How did you handle data cleaning, missing values, and outliers in your data science projects?",
        "4. What was your specific individual contribution versus your teammates' contributions in your major college project?",
        "5. If you had to scale your project to handle 100x the data volume, what architectural changes would you make?"
    ]
    return fallback_q[:count]
