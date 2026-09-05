"""
Interview Management Module
Session lifecycle controller, question sequencing, answer analysis pipeline, and progress management.
"""

import os
import random
from modules import database
from modules import nlp_analysis
from modules import relevance
from modules import communication
from modules import scoring
from modules import speech_to_text
from modules import llm_feedback


class InterviewSession:
    """
    Coordinates an active interview session, managing question sequencing,
    response processing, NLP/ML feature extraction, and report compilation.
    """

    def __init__(self, user_name, category="HR", total_questions=5, user_email=None, custom_questions=None):
        self.user_name = user_name
        self.category = category
        self.total_questions = total_questions
        self.user_email = user_email
        self.custom_questions = custom_questions or []

        # Database initialization
        database.init_db()
        self.user = database.get_or_create_user(user_name, user_email)
        self.session_id = database.create_session(self.user["user_id"], self.category)

        # Questions for this session
        self.questions = self._select_questions()
        self.current_index = 0
        self.completed_responses = []

    def _select_questions(self):
        """Selects questions from question bank or custom list."""
        if self.custom_questions:
            # Custom questions (e.g. generated from resume)
            formatted = []
            for i, q in enumerate(self.custom_questions):
                formatted.append({
                    "question_id": 9000 + i,
                    "category": self.category,
                    "question": q,
                    "keywords": "project, technical, implementation, architecture, problem-solving",
                    "difficulty": "Medium",
                    "expected_topics": "Technical Competency, Project Details"
                })
            return formatted[:self.total_questions]

        # Fetch from database
        all_q = database.get_all_questions(self.category)
        if not all_q:
            # Fallback to all categories
            all_q = database.get_all_questions()

        if len(all_q) <= self.total_questions:
            selected = all_q[:]
            random.shuffle(selected)
            return selected
        else:
            return random.sample(all_q, self.total_questions)

    def get_current_question(self):
        """Returns the current question dictionary or None if completed."""
        if self.current_index < len(self.questions):
            return self.questions[self.current_index]
        return None

    def has_next_question(self):
        """Checks if more questions remain."""
        return self.current_index < len(self.questions)

    def process_answer(self, response_type="text", text_input="", audio_file_path=None, duration=0.0):
        """
        Executes the full end-to-end processing pipeline on the candidate's answer:
        1. Audio transcription (if voice response)
        2. NLP linguistic and filler analysis
        3. TF-IDF & Cosine Similarity relevance analysis
        4. Speaking speed (WPM) & fluency evaluation
        5. Composite weighted scoring
        6. Personalized AI feedback generation
        7. Persistence to SQLite database
        """
        current_q = self.get_current_question()
        if not current_q:
            return None

        transcript = text_input.strip()
        audio_path_stored = None
        calc_duration = duration

        # Step 1: Voice transcription if audio input
        if response_type == "voice" and audio_file_path:
            audio_path_stored = audio_file_path
            trans_result = speech_to_text.transcribe_audio(audio_file_path)
            if trans_result["success"] and trans_result["transcript"]:
                transcript = trans_result["transcript"]
            if trans_result["duration"] > 0:
                calc_duration = trans_result["duration"]

        if not transcript:
            transcript = "No audible or written answer was provided."

        # Step 2: NLP Analysis (Fillers, Vocabulary, Grammar, Readability)
        nlp_res = nlp_analysis.analyze_text(transcript)

        # Step 3: Relevance Analysis (TF-IDF, Cosine Similarity, Keywords)
        rel_res = relevance.evaluate_relevance(transcript, current_q)

        # Step 4: Communication Analysis (WPM, Fluency)
        comm_res = communication.analyze_communication(
            word_count=nlp_res["word_count"],
            duration_seconds=calc_duration,
            filler_count=nlp_res["filler_count"],
            avg_sentence_length=nlp_res["avg_sentence_length"]
        )

        # Step 5: Scoring
        score_res = scoring.calculate_response_score(
            relevance_score=rel_res["relevance_score"],
            fluency_score=comm_res["fluency_score"],
            vocabulary_score=nlp_res["vocabulary_score"],
            grammar_score=nlp_res["grammar_score"],
            filler_frequency=nlp_res["filler_frequency"]
        )

        # Step 6: AI Feedback Generation
        metrics_summary = {
            "overall_score": score_res["total_score"],
            "grade": score_res["grade"],
            "relevance_score": rel_res["relevance_score"],
            "relevance_rating": rel_res["relevance_rating"],
            "matched_keywords": rel_res["matched_keywords"],
            "missing_keywords": rel_res["missing_keywords"],
            "speaking_speed_wpm": comm_res["speaking_speed_wpm"],
            "speed_category": comm_res["speed_category"],
            "filler_count": nlp_res["filler_count"],
            "filler_frequency": nlp_res["filler_frequency"],
            "fluency_score": comm_res["fluency_score"],
            "vocabulary_score": nlp_res["vocabulary_score"],
            "grammar_score": nlp_res["grammar_score"]
        }

        feedback_res = llm_feedback.generate_personalized_feedback(
            question_text=current_q["question"],
            candidate_answer=transcript,
            metrics_summary=metrics_summary
        )

        # Step 7: Database Persistence
        try:
            resp_id = database.save_response(
                session_id=self.session_id,
                question_id=current_q["question_id"],
                response_type=response_type,
                text=transcript,
                audio_path=audio_path_stored,
                duration=calc_duration
            )

            database.save_analysis(resp_id, {
                "relevance_score": rel_res["relevance_score"],
                "filler_count": nlp_res["filler_count"],
                "filler_frequency": nlp_res["filler_frequency"],
                "speaking_speed_wpm": comm_res["speaking_speed_wpm"],
                "vocabulary_score": nlp_res["vocabulary_score"],
                "grammar_score": nlp_res["grammar_score"],
                "fluency_score": comm_res["fluency_score"]
            })

            database.save_feedback(
                session_id=self.session_id,
                response_id=resp_id,
                strengths=feedback_res["strengths"],
                weaknesses=feedback_res["weaknesses"],
                recommendations=feedback_res["recommendations"],
                summary=feedback_res["summary"]
            )
        except Exception as e:
            print(f"Notice: Session response persistence note: {e}")

        # Package result
        answer_result = {
            "question_index": self.current_index + 1,
            "question": current_q["question"],
            "category": current_q["category"],
            "candidate_answer": transcript,
            "response_type": response_type,
            "duration": calc_duration,
            "nlp": nlp_res,
            "relevance": rel_res,
            "communication": comm_res,
            "scoring": score_res,
            "feedback": feedback_res
        }

        self.completed_responses.append(answer_result)
        self.current_index += 1
        return answer_result

    def finish_session(self):
        """
        Finalizes interview session, computes aggregated metrics,
        and saves final score to the database.
        """
        scoring_items = [r["scoring"] for r in self.completed_responses]
        summary_score = scoring.calculate_overall_session_score(scoring_items)

        database.update_session_score(
            session_id=self.session_id,
            overall_score=summary_score["overall_score"],
            grade=summary_score["grade"],
            status="Completed"
        )

        return {
            "session_id": self.session_id,
            "user_name": self.user_name,
            "category": self.category,
            "summary_score": summary_score,
            "responses": self.completed_responses
        }
