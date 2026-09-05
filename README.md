# AI-Assisted Interview Feedback and Communication Analysis
> **TY B.Sc. Data Science College Project**  
> **Candidate:** Srushti Dhide (Roll No. 54)  
> **Domain:** Natural Language Processing, Machine Learning, Speech-to-Text & Local Generative AI

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Framework-Streamlit](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![NLP-spaCy%20%2B%20NLTK](https://img.shields.io/badge/NLP-spaCy%20%7C%20NLTK-09A3D5.svg)](https://spacy.io/)
[![STT-faster--whisper](https://img.shields.io/badge/Speech--to--Text-faster--whisper-green.svg)](https://github.com/SYSTRAN/faster-whisper)
[![LLM-Qwen2.5--3B](https://img.shields.io/badge/LLM-Qwen2.5--3B%20(Ollama)-purple.svg)](https://ollama.com)
[![Database-SQLite](https://img.shields.io/badge/Database-SQLite3-003B57.svg)](https://www.sqlite.org/)

---

## 1. Project Overview

Preparing for job placements and campus interviews requires both **conceptual mastery** and **strong verbal communication skills**. Many students struggle with unstructured explanations, nervous speaking rates, and excessive filler words (*"um", "basically", "you know"*).

**AI-Assisted Interview Feedback and Communication Analysis** is a locally-runnable intelligent assessment platform designed to help students practice **HR (Behavioral)** and **Technical (Data Science & Programming)** interviews.

### Key Highlights:
- 🎙️ **Dual-Mode Answering:** Answer mock interview questions via **live voice recording**, audio file upload, or **typed text**.
- ⚡ **Local Speech-to-Text:** Transcribes spoken audio offline using `faster-whisper` (int8 quantized CPU execution).
- 📐 **Transparent ML & NLP Scoring:** Computes deterministic, explainable metrics using **pure Python, spaCy, NLTK, and scikit-learn** (TF-IDF + Cosine Similarity). **Scoring is never delegated to a black-box LLM.**
- 🤖 **Structured AI Coaching:** Uses a local LLM (**Qwen2.5-3B-Instruct via Ollama**) to translate measured metrics into structured feedback (Strengths, Areas for Improvement, Actionable Recommendations).
- 🛡️ **Resilient Offline Fallback:** If Ollama is offline, a deterministic expert rule-based engine activates automatically so evaluations never fail during live presentations or examinations.
- 📊 **Progress & Analytics:** All sessions, transcripts, and component metrics are stored in **SQLite**, with interactive Matplotlib trend charts and radar charts tracking progress over time.
- 🔒 **100% Free & Local:** Runs completely offline on standard laptops with 8–16 GB RAM and no dedicated GPU.

---

## 2. Architecture & Data Flow

```mermaid
flowchart TD
    A[Student Selects Track: HR / Technical] --> B[Question Displayed from Question Bank]
    B --> C{Answer Input Mode}
    C -->|Voice Answer| D[faster-whisper Local STT]
    C -->|Text Answer| E[Direct Text Submission]
    D --> F[Candidate Transcript + Audio Duration]
    E --> F

    F --> G1[modules/nlp_analysis.py<br/>spaCy & NLTK<br/>Word/Sent Count, Fillers, Vocab TTR, Grammar]
    F --> G2[modules/relevance.py<br/>scikit-learn TF-IDF<br/>Cosine Similarity & Keyword Coverage]
    F --> G3[modules/communication.py<br/>Speaking Speed WPM<br/>Fluency & Rhythm Evaluation]

    G1 --> H[modules/scoring.py<br/>Deterministic Weighted Formula<br/>Relevance 35% | Fluency 25% | Vocab 20% | Grammar 10% | Fillers 10%]
    G2 --> H
    G3 --> H

    H --> I[modules/llm_feedback.py<br/>Qwen2.5-3B-Instruct via Ollama<br/>or Offline Rule-Based Fallback Engine]
    
    I --> J[SQLite Database: interview.db<br/>Sessions, Responses, Analysis, Feedback]
    I --> K[Streamlit UI: Instant Metrics, Radar Chart,<br/>Downloadable Markdown Report]
```

---

## 3. Directory Structure

```
AI_Interview_Feedback/
├── app.py                      # Main Streamlit web application & user interface
├── requirements.txt            # Pinned, compatible Python dependencies
├── README.md                   # Project documentation & viva defense guide
├── .gitignore                  # Git exclusion rules for environments, databases, cache
├── data/
│   ├── questions.csv           # 40 curated questions (20 HR, 20 Technical) with keywords
│   ├── reference_answers.csv   # Ideal benchmark answers, key points, sample good/weak answers
│   └── filler_words.txt        # 24 standard single and multi-word filler phrases
├── database/
│   └── interview.db            # SQLite database (auto-created and seeded on first run)
├── modules/
│   ├── __init__.py             # Module initialization
│   ├── interview.py            # Session lifecycle controller & question sequencing
│   ├── speech_to_text.py       # faster-whisper offline transcription & audio duration
│   ├── nlp_analysis.py         # spaCy/NLTK tokenization, filler detection, vocabulary TTR
│   ├── relevance.py            # TF-IDF vectorization, Cosine Similarity, keyword coverage
│   ├── communication.py        # Speaking speed (WPM) & fluency index calculations
│   ├── scoring.py              # Deterministic weighted scoring logic (SCORE_WEIGHTS)
│   ├── llm_feedback.py         # Qwen2.5-3B Ollama integration + resilient offline fallback
│   └── database.py             # SQLite connection management & CRUD queries
├── models/
│   └── README.md               # Model specifications, cache location, and setup instructions
├── reports/
│   └── generated_reports/      # Auto-saved candidate performance reports (.md format)
├── audio/
│   └── recordings/             # Stored audio answer files for analysis & replay
├── tests/
│   ├── test_nlp.py             # Unit tests for text cleaning, tokenization, fillers, grammar
│   ├── test_relevance.py       # Unit tests for TF-IDF cosine similarity & keyword matching
│   ├── test_scoring.py         # Unit tests for scoring bounds & session aggregation
│   └── test_database.py        # Unit tests for SQLite schema, foreign keys, and queries
└── notebooks/
    └── data_analysis.ipynb     # Educational exploratory data analysis & NLP simulations
```

---

## 4. Installation & Setup Guide

### Step 1: Prerequisites
- **Operating System:** Windows 10/11, macOS, or Linux
- **Python:** Version 3.10, 3.11, 3.12, 3.13, or 3.14
- **Hardware:** 8 GB+ RAM, CPU execution (no GPU required)

### Step 2: Open Terminal & Navigate to Project
```powershell
cd "c:\Users\Srushti\Downloads\OneDrive_2026-09-02\Srushti Dhide 54(AI-Assisted Interview Feedback and Communication Analysis)\AI_Interview_Feedback"
```

### Step 3: Create & Activate Virtual Environment
```powershell
# Create virtual environment
python -m venv .venv

# Activate virtual environment (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# (On Linux/macOS: source .venv/bin/activate)
```

### Step 4: Install Dependencies
```powershell
pip install -r requirements.txt
```

### Step 5: Download NLP Language Models
```powershell
# Download spaCy English model
python -m spacy download en_core_web_sm

# Download NLTK tokenizers and stopwords
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('punkt_tab')"
```

### Step 6: (Optional) Set up Local LLM (Ollama)
To enable live AI generation with `qwen2.5:3b`:
1. Download and install Ollama from [https://ollama.com](https://ollama.com).
2. Open a terminal and pull the lightweight 3B model:
   ```bash
   ollama pull qwen2.5:3b
   ```
3. Run the Ollama service:
   ```bash
   ollama run qwen2.5:3b
   ```
> **Note:** If Ollama is not running, the application **automatically uses its built-in Rule-Based Expert Engine**. The app will never crash or hang!

---

## 5. How to Run the Application

Launch the Streamlit web application:
```powershell
streamlit run app.py
```

The app will automatically open in your default browser at:
```
http://localhost:8501
```

---

## 6. How the Scoring System Works (Explainable AI)

Unlike generic chatbot wrappers, this project implements **deterministic, explainable AI** designed for academic evaluation.

### Component Weights (`modules/scoring.py`):
| Dimension | Weight | Metric Description | Formula / Algorithm |
| :--- | :---: | :--- | :--- |
| **Content Relevance** | **35%** | Conceptual alignment with benchmark | TF-IDF Cosine Similarity + Keyword Coverage |
| **Fluency & Pacing** | **25%** | Speech delivery & conversational flow | Speaking Speed (WPM) + Sentence Rhythm |
| **Vocabulary Richness** | **20%** | Sophistication & domain terminology | Type-Token Ratio ($TTR = \frac{V}{N}$) + Long Word Ratio |
| **Grammar & Structure**| **10%** | Structural completeness & sentence balance| Automated Readability Index (ARI) + Repetition Checks |
| **Filler Control** | **10%** | Control over vocal crutches | Filler frequency per 100 words ($\%$) |

$$\text{Overall Score} = (0.35 \times S_{\text{rel}}) + (0.25 \times S_{\text{flu}}) + (0.20 \times S_{\text{voc}}) + (0.10 \times S_{\text{gram}}) + (0.10 \times S_{\text{fill}})$$

### Academic Performance Bands:
- **A+ (88 – 100%):** Outstanding / Placement Ready
- **A (75 – 87%):** Strong Competency
- **B (60 – 74%):** Good / Needs Minor Refinement
- **C (45 – 59%):** Average / Requires Focused Practice
- **D (< 45%):** Needs Substantial Improvement

---

## 7. Running Automated Unit Tests

The test suite validates NLP parsing, relevance calculation, scoring boundaries, and SQLite transactions:
```powershell
pytest tests -v
```

Expected output:
```
tests/test_database.py::test_user_creation PASSED
tests/test_database.py::test_session_lifecycle PASSED
tests/test_nlp.py::test_clean_text PASSED
tests/test_nlp.py::test_tokenize_words PASSED
tests/test_nlp.py::test_tokenize_sentences PASSED
tests/test_nlp.py::test_detect_filler_words PASSED
tests/test_nlp.py::test_analyze_vocabulary PASSED
tests/test_nlp.py::test_analyze_grammar_and_structure PASSED
tests/test_relevance.py::test_cosine_similarity_identical PASSED
tests/test_relevance.py::test_cosine_similarity_unrelated PASSED
tests/test_relevance.py::test_check_keyword_coverage PASSED
tests/test_relevance.py::test_evaluate_relevance PASSED
tests/test_scoring.py::test_filler_control_score PASSED
tests/test_scoring.py::test_calculate_response_score_bounds PASSED
tests/test_scoring.py::test_calculate_overall_session_score PASSED
============================= 15 passed in 2.3s =============================
```

---

## 8. College Viva / Project Defense Preparation

Here are key questions anticipated from external examiners and professors, along with clear scientific answers:

### Q1: How does the system compute TF-IDF and Cosine Similarity?
**Answer:**  
1. **Term Frequency (TF):** Measures the relative frequency of a term $t$ in a document $d$: $\text{TF}(t, d) = \frac{f_{t, d}}{\sum_{t'} f_{t', d}}$.
2. **Inverse Document Frequency (IDF):** Downweights universally frequent words and elevates informative terms: $\text{IDF}(t) = \ln\left(\frac{1 + N}{1 + \text{DF}(t)}\right) + 1$.
3. **Cosine Similarity:** Computes the cosine of the angle between the candidate's vector $\mathbf{A}$ and the reference answer's vector $\mathbf{B}$:
   $$\text{Cosine Similarity}(\mathbf{A}, \mathbf{B}) = \frac{\mathbf{A} \cdot \mathbf{B}}{\|\mathbf{A}\| \|\mathbf{B}\|}$$
   Unlike Euclidean distance, Cosine Similarity is length-invariant, meaning a concise answer is not unfairly penalized compared to a verbose one.

### Q2: Why did you not let the LLM generate the score directly?
**Answer:**  
Large Language Models are probabilistic next-token predictors prone to **stochastic variation** and **hallucination**. Asking an LLM to score an interview can produce inconsistent scores for the exact same text across multiple runs. In contrast, our platform uses **deterministic Python/ML equations** for the scores, reserving the LLM strictly to generate human-readable feedback based on those exact numbers.

### Q3: How is speaking speed (WPM) calculated?
**Answer:**  
The audio duration is extracted using audio headers or `faster-whisper` segment timestamps. The total word count is obtained via NLP tokenization. Speaking rate is computed as:
$$\text{WPM} = \frac{\text{Word Count}}{\text{Audio Duration in Seconds} / 60.0}$$
Conversational research identifies **120–160 WPM** as the optimal professional interview pace.

### Q4: How are multi-word filler phrases detected?
**Answer:**  
Phrases like *"you know"*, *"sort of"*, or *"to be honest"* are matched using regex word-boundary patterns (`\b`) sorted in descending order of token length. This prevents partial-word false positives (e.g. matching "like" inside "likely") and prevents overlapping double-counts.

### Q5: How is user data stored and queried?
**Answer:**  
The platform utilizes a relational SQLite schema with strict referential integrity (`PRAGMA foreign_keys = ON;`). Six tables (`users`, `questions`, `sessions`, `responses`, `analysis`, and `feedback`) ensure complete auditability of candidate practice sessions.

---

## 9. Troubleshooting Guide

| Issue | Cause | Solution |
| :--- | :--- | :--- |
| **Ollama Offline warning** | Ollama daemon is not running on port 11434. | The app will automatically use its built-in rule-based expert engine. To enable Ollama, run `ollama run qwen2.5:3b`. |
| **Microphone not working in browser** | Browser permissions blocked for localhost. | Allow microphone access in your browser's site settings or use the audio file upload widget / text mode. |
| **Missing spaCy model error** | `en_core_web_sm` not downloaded. | Run `python -m spacy download en_core_web_sm`. |
| **Database file locked on Windows** | Active SQLite connection held by an open process. | The application uses explicit `conn.close()` calls on every transaction. Restarting the Streamlit server resolves any dangling handles. |

---

## 10. Future Enhancements (Post-MVP Scope)

The following advanced capabilities are scheduled for future research iterations:
1. **Dense Semantic Embeddings:** Swapping TF-IDF with `sentence-transformers` (e.g. `all-MiniLM-L6-v2`) via the built-in modular hook in `modules/relevance.py`.
2. **Video & Visual Analysis:** Facial expression tracking, eye-contact estimation, and posture analysis via OpenCV / MediaPipe.
3. **Adaptive Dynamic Interviewing:** Multi-turn conversational interviews adapting subsequent questions based on candidate performance.
4. **Resume-Based Question Generation:** Automated extraction of skills and projects from uploaded PDF resumes.

---

## 11. Author & Acknowledgements
- **Student:** Srushti Dhide (Roll No. 54)
- **Degree:** TY B.Sc. Data Science
- **Technologies:** Streamlit, Python, scikit-learn, spaCy, NLTK, faster-whisper, Ollama, SQLite3.
