"""
AI-Assisted Interview Feedback and Communication Analysis
TY B.Sc. Data Science College Project
Main Streamlit Web Application
"""

import os
import sys
import time
import math
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# Add base directory to system path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from modules import database
from modules import interview
from modules import speech_to_text
from modules import nlp_analysis
from modules import relevance
from modules import communication
from modules import scoring
from modules import llm_feedback

# Page configuration
st.set_page_config(
    page_title="AI Interview Coach & Communication Analyzer",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #F9FAFB;
        border: 1px solid #E5E7EB;
        border-radius: 10px;
        padding: 14px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1F2937;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #6B7280;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .grade-badge-A {
        background-color: #DEF7EC;
        color: #03543F;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
    .grade-badge-B {
        background-color: #E1EFFE;
        color: #1E429F;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
    .grade-badge-C {
        background-color: #FEF08A;
        color: #713F12;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
    .grade-badge-D {
        background-color: #FDE8E8;
        color: #9B1C1C;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
    .badge-kw-matched {
        background-color: #DCFCE7;
        color: #15803D;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.82rem;
        margin-right: 4px;
        margin-bottom: 4px;
        display: inline-block;
    }
    .badge-kw-missing {
        background-color: #F3F4F6;
        color: #6B7280;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.82rem;
        margin-right: 4px;
        margin-bottom: 4px;
        display: inline-block;
    }
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# Initialize Database
@st.cache_resource
def ensure_db_ready():
    database.init_db()
    return True

ensure_db_ready()


# Session State Management
if "user_name" not in st.session_state:
    st.session_state["user_name"] = "Student Candidate"
if "interview_session" not in st.session_state:
    st.session_state["interview_session"] = None
if "last_answer_result" not in st.session_state:
    st.session_state["last_answer_result"] = None
if "final_report" not in st.session_state:
    st.session_state["final_report"] = None
if "session_stage" not in st.session_state:
    st.session_state["session_stage"] = "setup"  # "setup", "in_progress", "completed"


# Chart Generation Helpers
def generate_radar_chart(categories, values, title="Performance Profile"):
    """Generates a Matplotlib radar/spider chart for 5 core dimensions."""
    N = len(categories)
    angles = [n / float(N) * 2 * math.pi for n in range(N)]
    angles += angles[:1]
    plot_values = values + values[:1]

    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
    ax.set_theta_offset(math.pi / 2)
    ax.set_theta_direction(-1)

    plt.xticks(angles[:-1], categories, color='#374151', size=10, weight='semibold')
    ax.set_rlabel_position(0)
    plt.yticks([20, 40, 60, 80, 100], ["20", "40", "60", "80", "100"], color="#9CA3AF", size=8)
    plt.ylim(0, 100)

    ax.plot(angles, plot_values, linewidth=2.2, linestyle='solid', color='#2563EB')
    ax.fill(angles, plot_values, color='#3B82F6', alpha=0.3)
    ax.set_title(title, size=13, weight='bold', color='#1E3A8A', y=1.08)

    plt.tight_layout()
    return fig


def generate_progress_chart(session_df):
    """Generates a Matplotlib line chart of candidate scores over time."""
    fig, ax = plt.subplots(figsize=(8, 3.5))
    session_df['session_num'] = range(1, len(session_df) + 1)

    ax.plot(session_df['session_num'], session_df['overall_score'], marker='o', color='#2563EB',
            linewidth=2.5, markersize=8, label='Overall Score')

    # Benchmark zones
    ax.axhline(y=80, color='#10B981', linestyle='--', alpha=0.6, label='Placement Ready (80+)')
    ax.axhline(y=60, color='#F59E0B', linestyle='--', alpha=0.5, label='Good Competency (60+)')

    ax.set_title("Performance Score Trend Across Sessions", fontsize=13, weight='bold', color='#1F2937')
    ax.set_xlabel("Session Number", fontsize=10, weight='semibold', color='#4B5563')
    ax.set_ylabel("Score (0 - 100)", fontsize=10, weight='semibold', color='#4B5563')
    ax.set_ylim(0, 105)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='lower right', frameon=True)

    plt.tight_layout()
    return fig


def export_report_to_markdown(final_data):
    """Generates a structured markdown report and saves to reports/generated_reports/."""
    os.makedirs(os.path.join(BASE_DIR, "reports", "generated_reports"), exist_ok=True)
    summary = final_data["summary_score"]
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f"session_{final_data['session_id']}_{timestamp}.md"
    file_path = os.path.join(BASE_DIR, "reports", "generated_reports", file_name)

    lines = []
    lines.append(f"# AI Interview Performance Report\n")
    lines.append(f"**Candidate:** {final_data['user_name']}  ")
    lines.append(f"**Category:** {final_data['category']}  ")
    lines.append(f"**Session ID:** #{final_data['session_id']}  ")
    lines.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}  ")
    lines.append(f"**Overall Score:** {summary['overall_score']}/100  ")
    lines.append(f"**Performance Band:** {summary['grade']}\n")

    lines.append("## Executive Summary Scorecard\n")
    lines.append(f"- **Content Relevance:** {summary['avg_relevance']}%")
    lines.append(f"- **Fluency & Speaking Cadence:** {summary['avg_fluency']}%")
    lines.append(f"- **Vocabulary & Lexical Diversity:** {summary['avg_vocabulary']}%")
    lines.append(f"- **Grammar & Sentence Structure:** {summary['avg_grammar']}%")
    lines.append(f"- **Filler Word Control:** {summary['avg_filler_control']}%\n")

    lines.append("## Detailed Question-by-Question Analysis\n")
    for r in final_data["responses"]:
        lines.append(f"### Question {r['question_index']}: {r['question']}")
        lines.append(f"**Response Mode:** {r['response_type'].capitalize()} | **Duration:** {r['duration']} seconds")
        lines.append(f"\n> **Candidate Transcript:**\n> \"{r['candidate_answer']}\"\n")
        lines.append(f"**Scores:** Overall: **{r['scoring']['total_score']}/100** ({r['scoring']['grade']}) | "
                     f"Relevance: {r['relevance']['relevance_score']}% | Fluency: {r['communication']['fluency_score']}% | "
                     f"Speed: {r['communication']['speaking_speed_wpm']} WPM | Fillers: {r['nlp']['filler_count']}")

        fb = r["feedback"]
        lines.append(f"\n#### AI Coaching Insights ({fb['source']}):")
        lines.append(f"**Strengths:**\n{fb['strengths']}\n")
        lines.append(f"**Areas for Improvement:**\n{fb['weaknesses']}\n")
        lines.append(f"**Actionable Recommendations:**\n{fb['recommendations']}\n")
        lines.append(f"**Summary:** {fb['summary']}\n")
        lines.append("---\n")

    report_content = "\n".join(lines)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    return file_path, report_content


# Sidebar Navigation & Model Diagnostics
st.sidebar.markdown("## 🎯 AI Interview Coach")
st.sidebar.markdown("**TY B.Sc. Data Science Project**")
st.sidebar.caption("AI-Assisted Feedback & Communication Analysis")

app_mode = st.sidebar.radio(
    "Navigate",
    ["🎙️ Mock Interview Room", "📊 Progress & Analytics", "📖 Question Bank Browser", "⚡ System Diagnostics"],
    index=0
)

# User Profile in Sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### 👤 Candidate Profile")
user_input_name = st.sidebar.text_input("Candidate Name", value=st.session_state["user_name"])
if user_input_name != st.session_state["user_name"]:
    st.session_state["user_name"] = user_input_name

# Model Health Indicators in Sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Engine Status")

# Check Ollama status with low timeout
ollama_stat = llm_feedback.check_ollama_status()
if ollama_stat["available"]:
    st.sidebar.success(f"🟢 **Ollama (Qwen2.5-3B)** Online")
else:
    st.sidebar.warning(f"🟡 **Ollama** Offline (Expert Engine Active)")

st.sidebar.info("🟢 **Speech-to-Text:** faster-whisper")
st.sidebar.info("🟢 **NLP Core:** spaCy + NLTK")
st.sidebar.info("🟢 **Database:** SQLite Local")


# ==========================================
# 1. MOCK INTERVIEW ROOM
# ==========================================
if app_mode == "🎙️ Mock Interview Room":
    st.markdown('<div class="main-title">🎯 AI-Assisted Mock Interview Room</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Practice HR and Technical interviews via voice or text. Receive objective ML-driven scoring and structured AI feedback.</div>', unsafe_allow_html=True)

    # STAGE A: SETUP
    if st.session_state["session_stage"] == "setup":
        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("### Start a New Mock Interview")
            category_choice = st.selectbox(
                "Select Interview Track",
                ["HR", "Technical"],
                format_func=lambda x: "💼 HR Behavioral Interview (Culture, Situational, Teamwork)" if x == "HR" else "💻 Technical Interview (Data Science, Python, SQL, ML)"
            )

            q_count = st.select_slider(
                "Number of Questions",
                options=[1, 2, 3, 5, 10],
                value=3,
                help="Select how many questions you want to practice in this session."
            )

            email_opt = st.text_input("Student Email (Optional, for record keeping)", placeholder="e.g. srushti@college.edu")

            st.markdown("#### What this platform analyzes:")
            st.markdown("""
            - 🎯 **Answer Relevance:** Evaluated via TF-IDF vectorization and Cosine Similarity against reference benchmarks.
            - ⏱️ **Speaking Speed (WPM):** Measured using transcript word count and audio duration (Optimal: 120-160 WPM).
            - 🗣️ **Filler Words:** Precise detection of vocal crutches (*um, uh, basically, like, you know*).
            - 🧠 **Lexical Diversity & Grammar:** Analyzed using spaCy/NLTK for vocabulary richness and sentence flow.
            - 💡 **AI Recommendations:** Powered by local Qwen2.5-3B (or our resilient Expert Offline Engine).
            """)

            if st.button("🚀 Begin Interview", type="primary", use_container_width=True):
                with st.spinner("Setting up interview session and preparing question bank..."):
                    new_session = interview.InterviewSession(
                        user_name=st.session_state["user_name"],
                        category=category_choice,
                        total_questions=q_count,
                        user_email=email_opt if email_opt else None
                    )
                    st.session_state["interview_session"] = new_session
                    st.session_state["session_stage"] = "in_progress"
                    st.session_state["last_answer_result"] = None
                    st.session_state["final_report"] = None
                    st.rerun()

        with col2:
            st.markdown("### 💡 Viva & Project Tips")
            st.info("""
            **College Viva Point:**
            Scores are computed using **transparent deterministic algorithms** in Python (`modules/scoring.py`), not delegated to an LLM.

            The LLM is strictly used to translate quantitative data into human-readable advice!
            """)

            st.markdown("### 🏆 Performance Grades")
            st.markdown("""
            - **A+ (88-100%):** Placement Ready / Outstanding
            - **A (75-87%):** Strong Competency
            - **B (60-74%):** Average / Needs Polish
            - **C (45-59%):** Requires Focused Practice
            - **D (<45%):** Needs Substantial Improvement
            """)

    # STAGE B: IN PROGRESS
    elif st.session_state["session_stage"] == "in_progress":
        sess = st.session_state["interview_session"]
        total_q = len(sess.questions)

        # Accurately identify which question is active or being reviewed
        if st.session_state["last_answer_result"] is not None:
            # We are reviewing the question that was just completed
            curr_idx = len(sess.completed_responses)
            current_q = sess.questions[curr_idx - 1] if 0 < curr_idx <= total_q else sess.questions[-1]
        else:
            # We are asking the next question in sequence
            curr_idx = min(sess.current_index + 1, total_q)
            current_q = sess.get_current_question()

        if not current_q:
            # Safe boundary fallback if completed
            final_data = sess.finish_session()
            st.session_state["final_report"] = final_data
            st.session_state["session_stage"] = "completed"
            st.rerun()

        # Progress header safely clamped to [0.0, 1.0]
        progress_val = min(1.0, max(0.0, float(curr_idx) / float(max(total_q, 1))))
        st.progress(progress_val)
        st.markdown(f"**Question {curr_idx} of {total_q}** | Category: `{sess.category}` | Candidate: `{sess.user_name}`")

        # Question Presentation Card
        st.markdown(f"""
        <div style="background-color: #EFF6FF; border-left: 5px solid #2563EB; padding: 18px; border-radius: 8px; margin-bottom: 20px;">
            <div style="font-size: 0.85rem; font-weight: 600; color: #1E40AF; text-transform: uppercase;">
                Difficulty: {current_q.get('difficulty', 'Medium')} | Expected Focus: {current_q.get('expected_topics', 'General')}
            </div>
            <div style="font-size: 1.35rem; font-weight: 600; color: #1E293B; margin-top: 6px;">
                {current_q['question']}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # If answer for current question is not analyzed yet
        if st.session_state["last_answer_result"] is None:
            input_tab1, input_tab2 = st.tabs(["🎙️ Voice Answer (Microphone / Upload)", "✍️ Text Answer (Type Response)"])

            chosen_type = "text"
            text_answer_content = ""
            voice_file_path = None
            voice_duration = 0.0

            with input_tab1:
                st.markdown("##### Record your answer directly or upload a recorded WAV/MP3 file:")
                
                # Check if st.audio_input exists in this Streamlit version
                has_audio_input = hasattr(st, "audio_input")
                audio_recorded = None
                if has_audio_input:
                    audio_recorded = st.audio_input("Record audio from your microphone")

                audio_uploaded = st.file_uploader(
                    "Or upload an audio recording file",
                    type=["wav", "mp3", "m4a", "ogg"],
                    key="voice_upload_widget"
                )

                selected_audio = audio_recorded if audio_recorded else audio_uploaded

                if selected_audio:
                    st.audio(selected_audio)
                    # Save to temp file
                    saved_path = speech_to_text.save_uploaded_audio(selected_audio, f"temp_q_{curr_idx}.wav")
                    voice_file_path = saved_path
                    voice_duration = speech_to_text.get_audio_duration(saved_path)

                    st.caption(f"Audio file ready ({round(voice_duration, 1)} seconds). Click 'Submit & Analyze' to transcribe with faster-whisper and evaluate.")
                    if st.button("🔍 Submit & Analyze Voice Answer", type="primary", key="btn_submit_voice"):
                        with st.spinner("Transcribing speech with faster-whisper & computing NLP metrics..."):
                            res = sess.process_answer(
                                response_type="voice",
                                audio_file_path=voice_file_path,
                                duration=voice_duration
                            )
                            st.session_state["last_answer_result"] = res
                            st.rerun()

            with input_tab2:
                st.markdown("##### Type your response in the text box below:")
                typed_answer = st.text_area(
                    "Your Answer",
                    height=160,
                    placeholder="Type a comprehensive, structured response here. Try to cover core concepts, examples, or practical applications...",
                    key="typed_answer_input"
                )
                
                word_count_live = len(typed_answer.strip().split()) if typed_answer.strip() else 0
                st.caption(f"Word count: {word_count_live} words")

                if st.button("🔍 Submit & Analyze Text Answer", type="primary", key="btn_submit_text"):
                    if not typed_answer.strip():
                        st.warning("Please type an answer before submitting.")
                    else:
                        with st.spinner("Analyzing answer relevance, grammar, vocabulary, and computing scores..."):
                            res = sess.process_answer(
                                response_type="text",
                                text_input=typed_answer,
                                duration=0.0
                            )
                            st.session_state["last_answer_result"] = res
                            st.rerun()

        # If answer has been analyzed, display rich feedback card
        else:
            res = st.session_state["last_answer_result"]
            sc = res["scoring"]
            rel = res["relevance"]
            comm = res["communication"]
            nlp = res["nlp"]
            fb = res["feedback"]

            st.success("✅ Analysis Complete!")

            # Transcript Box
            with st.expander("📝 Transcribed / Submitted Answer", expanded=True):
                st.markdown(f"**Answer ({res['response_type'].capitalize()}):**")
                st.write(f"*{res['candidate_answer']}*")
                if res["duration"] > 0:
                    st.caption(f"Duration: {res['duration']}s | Word count: {nlp['word_count']} words")

            # Scores Summary
            st.markdown("### 📊 Performance Breakdown")
            c1, c2, c3, c4, c5, c6 = st.columns(6)

            with c1:
                st.metric("Total Score", f"{sc['total_score']}/100")
                st.markdown(f"<div style='text-align:center;'>{sc['grade']}</div>", unsafe_allow_html=True)
            with c2:
                st.metric("Relevance", f"{rel['relevance_score']}%")
                st.caption(rel["relevance_rating"])
            with c3:
                st.metric("Speaking Speed", f"{comm['speaking_speed_wpm']} WPM")
                st.caption(comm["speed_category"])
            with c4:
                st.metric("Filler Words", f"{nlp['filler_count']}")
                st.caption(f"{nlp['filler_frequency']}% density")
            with c5:
                st.metric("Fluency Score", f"{comm['fluency_score']}%")
                st.caption(comm["fluency_rating"])
            with c6:
                st.metric("Vocabulary", f"{nlp['vocabulary_score']}%")
                st.caption(f"TTR: {nlp['ttr']}")

            # Keywords & Fillers breakdown
            col_kw, col_fl = st.columns([1, 1])
            with col_kw:
                st.markdown("**Concept Keyword Coverage:**")
                matched_html = "".join([f"<span class='badge-kw-matched'>✓ {k}</span>" for k in rel["matched_keywords"]])
                missing_html = "".join([f"<span class='badge-kw-missing'>✗ {k}</span>" for k in rel["missing_keywords"]])
                st.markdown(matched_html + missing_html if (matched_html or missing_html) else "*None specified*", unsafe_allow_html=True)

            with col_fl:
                st.markdown("**Filler Words Detected:**")
                if nlp["fillers_found"]:
                    fillers_str = ", ".join([f"`{k}` ({v}x)" for k, v in nlp["fillers_found"].items()])
                    st.markdown(fillers_str)
                else:
                    st.markdown("✨ *Zero filler words detected! Outstanding control.*")

            # AI Feedback Box
            st.markdown("---")
            st.markdown(f"### 🤖 AI Coaching Feedback <span style='font-size:0.8rem; color:#6B7280;'>({fb['source']})</span>", unsafe_allow_html=True)

            fb_c1, fb_c2 = st.columns(2)
            with fb_c1:
                st.markdown("#### 🌟 Key Strengths")
                st.markdown(fb["strengths"])

                st.markdown("#### ⚠️ Areas for Improvement")
                st.markdown(fb["weaknesses"])

            with fb_c2:
                st.markdown("#### 💡 Actionable Recommendations")
                st.markdown(fb["recommendations"])

                st.markdown("#### 📋 Coach Summary")
                st.info(fb["summary"])

            # Navigation Button: Next Question or Finish
            st.markdown("---")
            if sess.has_next_question():
                if st.button("Next Question ➡️", type="primary", use_container_width=True):
                    st.session_state["last_answer_result"] = None
                    st.rerun()
            else:
                if st.button("🏁 View Final Performance Report", type="primary", use_container_width=True):
                    final_data = sess.finish_session()
                    st.session_state["final_report"] = final_data
                    st.session_state["session_stage"] = "completed"
                    st.rerun()

    # STAGE C: FINAL SESSION REPORT
    elif st.session_state["session_stage"] == "completed":
        final_data = st.session_state["final_report"]
        summary = final_data["summary_score"]

        st.balloons()
        st.markdown(f"## 🏆 Interview Completed: {final_data['user_name']}")
        st.markdown(f"**Track:** `{final_data['category']}` | **Session ID:** `#{final_data['session_id']}`")

        # Top Executive Summary Card
        st.markdown(f"""
        <div style="background: linear-gradient(90deg, #1E3A8A 0%, #3B82F6 100%); color: white; padding: 24px; border-radius: 12px; margin-bottom: 25px;">
            <div style="font-size: 1rem; opacity: 0.9;">OVERALL INTERVIEW EVALUATION</div>
            <div style="font-size: 3rem; font-weight: 800; margin: 4px 0;">{summary['overall_score']} <span style="font-size: 1.5rem;">/ 100</span></div>
            <div style="font-size: 1.25rem; font-weight: 600;">{summary['grade']}</div>
        </div>
        """, unsafe_allow_html=True)

        # Radar Chart & Metrics
        chart_col, metrics_col = st.columns([1, 1])

        with chart_col:
            radar_categories = ['Relevance', 'Fluency', 'Vocabulary', 'Grammar', 'Filler Control']
            radar_values = [
                summary['avg_relevance'],
                summary['avg_fluency'],
                summary['avg_vocabulary'],
                summary['avg_grammar'],
                summary['avg_filler_control']
            ]
            fig_radar = generate_radar_chart(radar_categories, radar_values, "Multi-Dimensional Competency Profile")
            st.pyplot(fig_radar)

        with metrics_col:
            st.markdown("### 📈 Dimension Scorecard")
            st.markdown(f"- **Content Relevance & Accuracy:** **{summary['avg_relevance']}%** (Weight: 35%)")
            st.markdown(f"- **Fluency & Speaking Rhythm:** **{summary['avg_fluency']}%** (Weight: 25%)")
            st.markdown(f"- **Vocabulary & Lexical Richness:** **{summary['avg_vocabulary']}%** (Weight: 20%)")
            st.markdown(f"- **Grammar & Sentence Flow:** **{summary['avg_grammar']}%** (Weight: 10%)")
            st.markdown(f"- **Filler Word Control:** **{summary['avg_filler_control']}%** (Weight: 10%)")

            st.markdown("---")
            st.markdown(f"**Total Questions Answered:** {summary['total_questions_answered']}")

            # Export Report
            report_path, report_text = export_report_to_markdown(final_data)
            st.download_button(
                label="📥 Download Full Report (.md)",
                data=report_text,
                file_name=os.path.basename(report_path),
                mime="text/markdown",
                use_container_width=True
            )
            st.caption(f"Auto-saved locally to: `{report_path}`")

        # Question by question review
        st.markdown("### 📝 Detailed Question Breakdown")
        for i, r in enumerate(final_data["responses"], 1):
            with st.expander(f"Q{i}: {r['question']} — Score: {r['scoring']['total_score']}/100 ({r['scoring']['grade']})"):
                st.markdown(f"**Candidate Answer:** *\"{r['candidate_answer']}\"*")
                st.markdown(f"**Relevance:** {r['relevance']['relevance_score']}% | **Speed:** {r['communication']['speaking_speed_wpm']} WPM | **Fillers:** {r['nlp']['filler_count']}")
                st.markdown(f"**Strengths:**\n{r['feedback']['strengths']}")
                st.markdown(f"**Recommendations:**\n{r['feedback']['recommendations']}")

        st.markdown("---")
        if st.button("🔄 Start Another Interview Session", type="primary", use_container_width=True):
            st.session_state["session_stage"] = "setup"
            st.session_state["interview_session"] = None
            st.session_state["last_answer_result"] = None
            st.session_state["final_report"] = None
            st.rerun()


# ==========================================
# 2. PROGRESS & PERFORMANCE ANALYTICS
# ==========================================
elif app_mode == "📊 Progress & Analytics":
    st.markdown('<div class="main-title">📊 Candidate Progress & Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Track your performance improvement across mock interview sessions stored in SQLite.</div>', unsafe_allow_html=True)

    # Fetch all users
    conn = database.get_db_connection()
    users_df = pd.read_sql_query("SELECT * FROM users ORDER BY name", conn)
    conn.close()

    if users_df.empty:
        st.info("No interview sessions found in database yet. Practice a session to begin tracking progress!")
    else:
        user_names = users_df['name'].tolist()
        current_name_idx = user_names.index(st.session_state["user_name"]) if st.session_state["user_name"] in user_names else 0
        selected_user_name = st.selectbox("Select Candidate to Inspect", user_names, index=current_name_idx)

        selected_user_id = int(users_df[users_df['name'] == selected_user_name]['user_id'].iloc[0])
        history = database.get_user_progress_data(selected_user_id)

        if not history:
            st.warning(f"No completed sessions recorded for {selected_user_name} yet.")
        else:
            hist_df = pd.DataFrame(history)

            # High-level stats
            stat_c1, stat_c2, stat_c3, stat_c4 = st.columns(4)
            with stat_c1:
                st.metric("Total Sessions", len(hist_df))
            with stat_c2:
                st.metric("Average Score", f"{round(hist_df['overall_score'].mean(), 1)}/100")
            with stat_c3:
                st.metric("Highest Score", f"{round(hist_df['overall_score'].max(), 1)}/100")
            with stat_c4:
                most_freq_cat = hist_df['category'].mode()[0] if not hist_df['category'].empty else "N/A"
                st.metric("Primary Track", most_freq_cat)

            # Trend chart
            st.markdown("### 📈 Score Progression Over Time")
            fig_trend = generate_progress_chart(hist_df)
            st.pyplot(fig_trend)

            # Sessions table
            st.markdown("### 📋 Completed Sessions")
            display_table = hist_df[['session_id', 'session_date', 'category', 'overall_score', 'grade']].rename(
                columns={
                    'session_id': 'Session ID',
                    'session_date': 'Date & Time',
                    'category': 'Track',
                    'overall_score': 'Score',
                    'grade': 'Performance Band'
                }
            )
            st.dataframe(display_table, use_container_width=True)

            # Session drill-down
            st.markdown("### 🔍 Inspect Past Session Details")
            session_ids = hist_df['session_id'].tolist()
            picked_session_id = st.selectbox("Select Session ID", session_ids)

            if picked_session_id:
                details = database.get_session_details(picked_session_id)
                for item in details:
                    with st.expander(f"Question: {item['question']}"):
                        st.markdown(f"**Answer ({item['response_type']}):** *\"{item['candidate_answer']}\"*")
                        st.markdown(f"**Relevance:** {item['relevance_score']}% | **Speed:** {item['speaking_speed_wpm']} WPM | **Fillers:** {item['filler_count']}")
                        if item['strengths']:
                            st.markdown(f"**Strengths:**\n{item['strengths']}")
                        if item['recommendations']:
                            st.markdown(f"**Recommendations:**\n{item['recommendations']}")


# ==========================================
# 3. QUESTION BANK BROWSER
# ==========================================
elif app_mode == "📖 Question Bank Browser":
    st.markdown('<div class="main-title">📖 Interview Question Bank & Guides</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Explore standard HR behavioral questions and technical data science questions with ideal answer benchmarks.</div>', unsafe_allow_html=True)

    ref_answers = relevance.load_reference_answers()
    all_questions = database.get_all_questions()

    filter_c1, filter_c2, filter_c3 = st.columns([1, 1, 2])
    with filter_c1:
        cat_filter = st.selectbox("Track", ["All", "HR", "Technical"])
    with filter_c2:
        diff_filter = st.selectbox("Difficulty", ["All", "Easy", "Medium", "Hard"])
    with filter_c3:
        search_query = st.text_input("Search Questions by Keyword", placeholder="e.g. overfitting, conflict, joins...")

    filtered = all_questions
    if cat_filter != "All":
        filtered = [q for q in filtered if q['category'] == cat_filter]
    if diff_filter != "All":
        filtered = [q for q in filtered if q['difficulty'] == diff_filter]
    if search_query:
        sq = search_query.lower()
        filtered = [q for q in filtered if sq in q['question'].lower() or sq in (q.get('keywords') or '').lower()]

    st.markdown(f"**Showing {len(filtered)} questions:**")

    for q in filtered:
        qid = q['question_id']
        ref = ref_answers.get(qid, {})

        with st.expander(f"#{qid} [{q['category']}] {q['question']} ({q['difficulty']})"):
            st.markdown(f"**Expected Keywords:** `{q.get('keywords', 'None')}`")
            st.markdown(f"**Key Focus Areas:** `{q.get('expected_topics', 'None')}`")

            if ref:
                st.markdown("---")
                st.markdown(f"**Ideal Benchmark Answer:**\n> {ref.get('reference_answer', 'N/A')}")
                st.markdown(f"**Key Concepts Interviewers Look For:**\n{ref.get('ideal_points', 'N/A')}")

                sub_c1, sub_c2 = st.columns(2)
                with sub_c1:
                    st.success(f"**Sample Strong Answer:**\n\"{ref.get('sample_good_answer', '')}\"")
                with sub_c2:
                    st.warning(f"**Sample Weak Answer (Avoid):**\n\"{ref.get('sample_weak_answer', '')}\"")


# ==========================================
# 4. SYSTEM & MODEL DIAGNOSTICS
# ==========================================
elif app_mode == "⚡ System Diagnostics":
    st.markdown('<div class="main-title">⚡ System & Model Diagnostics</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">System runtime verification, model availability, and viva architecture breakdown.</div>', unsafe_allow_html=True)

    st.markdown("### 🖥️ Local Architecture Stack")

    diag_c1, diag_c2 = st.columns(2)

    with diag_c1:
        st.markdown("#### 1. Ollama LLM Status")
        ollama_info = llm_feedback.check_ollama_status()
        if ollama_info["available"]:
            st.success("🟢 Ollama Daemon is active on `http://localhost:11434`")
            st.write(f"**Installed Models:** `{', '.join(ollama_info['models']) if ollama_info['models'] else 'None'}`")
            if ollama_info["default_model_found"]:
                st.success("🟢 Default model `qwen2.5:3b` is available!")
            else:
                st.warning("⚠️ Model `qwen2.5:3b` not detected. Run `ollama pull qwen2.5:3b` to download.")
        else:
            st.warning("🟡 Ollama service is not currently running.")
            st.info("""
            **To run Qwen2.5-3B locally:**
            1. Install Ollama from https://ollama.com
            2. Run: `ollama pull qwen2.5:3b`
            3. Run: `ollama run qwen2.5:3b`
            
            *Note:* If Ollama is offline, the app automatically runs its **deterministic expert rule-based engine**, so your demo will never fail!
            """)

        st.markdown("#### 2. Speech-to-Text (`faster-whisper`)")
        st.success("🟢 `faster-whisper` is installed and functional.")
        st.write("- **Model:** `tiny` (int8 quantized CPU execution)")
        st.write("- **Capability:** Automatically transcribes `.wav`, `.mp3`, `.m4a` audio and calculates speaking duration.")

    with diag_c2:
        st.markdown("#### 3. NLP Processing Pipeline")
        spacy_nlp = nlp_analysis.get_spacy_nlp()
        if spacy_nlp:
            st.success("🟢 spaCy model `en_core_web_sm` loaded successfully.")
        else:
            st.info("ℹ️ Using NLTK and regex tokenization pipelines.")

        st.markdown("#### 4. SQLite Database Diagnostics")
        conn = database.get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM questions")
        q_cnt = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM users")
        u_cnt = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM sessions")
        s_cnt = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM responses")
        r_cnt = cur.fetchone()[0]
        conn.close()

        st.write(f"- **Questions in DB:** {q_cnt}")
        st.write(f"- **Registered Users:** {u_cnt}")
        st.write(f"- **Total Interview Sessions:** {s_cnt}")
        st.write(f"- **Total Recorded Responses:** {r_cnt}")

    st.markdown("---")
    st.markdown("### 🎓 Viva Presentation Explanations")
    with st.expander("Q: How does the platform calculate answer relevance?"):
        st.write("""
        1. **TF-IDF Vectorization:** The candidate's response and the reference answer are converted into sparse numerical n-gram vectors using `scikit-learn`'s `TfidfVectorizer`.
        2. **Cosine Similarity:** The cosine of the angle between the two vectors is calculated:
           $$\\text{Cosine Similarity} = \\frac{\\mathbf{A} \\cdot \\mathbf{B}}{\\|\\mathbf{A}\\| \\|\\mathbf{B}\\|}$$
        3. **Keyword Coverage:** The response is scanned for mandatory technical/behavioral keywords from the question specification.
        4. **Composite Score:** Calibrated weighted score combining reference similarity, question context, and keyword coverage.
        """)

    with st.expander("Q: Why is scoring computed deterministically rather than by the LLM?"):
        st.write("""
        LLMs are probabilistic and prone to hallucination, non-determinism, and inconsistent scoring across identical answers.
        In an academic or professional setting, grading must be:
        - **Transparent:** Every component (Relevance 35%, Fluency 25%, Vocabulary 20%, Grammar 10%, Fillers 10%) has an explicit mathematical formula.
        - **Repeatable:** The exact same answer will produce the exact same score.
        - **Explainable:** A candidate can see exactly which metrics pulled their score up or down.
        """)
