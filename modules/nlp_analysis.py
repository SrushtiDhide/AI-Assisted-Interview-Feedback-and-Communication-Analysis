"""
NLP Analysis Module
Text preprocessing, sentence structure, vocabulary richness, and filler-word detection.
"""

import os
import re
import string
from collections import Counter

# File paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_FILLER_PATH = os.path.join(BASE_DIR, "data", "filler_words.txt")

# Lazy-loaded NLP models
_SPACY_NLP = None
_NLTK_STOPWORDS = None


def get_spacy_nlp():
    """Lazily loads and caches the spaCy English pipeline."""
    global _SPACY_NLP
    if _SPACY_NLP is not None:
        return _SPACY_NLP if _SPACY_NLP is not False else None
    try:
        import spacy
        _SPACY_NLP = spacy.load("en_core_web_sm")
        return _SPACY_NLP
    except Exception:
        _SPACY_NLP = False
        return None


def get_stopwords():
    """Retrieves standard English stop words using NLTK or fallback set."""
    global _NLTK_STOPWORDS
    if _NLTK_STOPWORDS is not None:
        return _NLTK_STOPWORDS
    try:
        from nltk.corpus import stopwords
        _NLTK_STOPWORDS = set(stopwords.words("english"))
    except Exception:
        _NLTK_STOPWORDS = {
            "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
            "has", "he", "in", "is", "it", "its", "of", "on", "that", "the",
            "to", "was", "were", "will", "with"
        }
    return _NLTK_STOPWORDS


def load_filler_words(file_path=None):
    """Loads configured filler words and phrases from text file."""
    if file_path is None:
        file_path = DEFAULT_FILLER_PATH

    default_fillers = [
        "um", "uh", "er", "ah", "like", "basically", "actually", "literally",
        "you know", "i mean", "sort of", "kind of", "right", "so yeah",
        "honestly", "anyway", "well", "totally", "seriously", "you see",
        "at the end of the day", "to be honest"
    ]

    if os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                fillers = [line.strip().lower() for line in f if line.strip()]
            return list(set(fillers)) if fillers else default_fillers
        except Exception:
            return default_fillers
    return default_fillers


def clean_text(text):
    """Normalizes text by removing extra spaces and non-printable characters."""
    if not text or not isinstance(text, str):
        return ""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def tokenize_words(text):
    """
    Splits text into individual lowercase alphanumeric word tokens,
    stripping punctuation. Uses NLTK or regex boundary parsing.
    """
    cleaned = clean_text(text).lower()
    try:
        from nltk.tokenize import word_tokenize
        tokens = word_tokenize(cleaned)
        # Filter strictly to alphanumeric / contracted words
        words = [re.sub(r"^[^a-zA-Z0-9']+|[^a-zA-Z0-9']+$", "", t) for t in tokens]
        words = [w for w in words if w and any(c.isalnum() for c in w)]
        if words:
            return words
    except Exception:
        pass
    # Reliable regex fallback
    return re.findall(r"\b[a-zA-Z0-9']+\b", cleaned)


def tokenize_sentences(text):
    """
    Splits text into sentences based on punctuation.
    Leverages NLTK sent_tokenize, spaCy, or regex boundary splitting.
    """
    cleaned = clean_text(text)
    if not cleaned:
        return []

    # Try NLTK sentence tokenizer first
    try:
        from nltk.tokenize import sent_tokenize
        sents = sent_tokenize(cleaned)
        if sents:
            return [s.strip() for s in sents if s.strip()]
    except Exception:
        pass

    # Try spaCy if available
    nlp = get_spacy_nlp()
    if nlp:
        try:
            doc = nlp(cleaned)
            return [s.text.strip() for s in doc.sents if s.text.strip()]
        except Exception:
            pass

    # Fallback to regex splitting on ., !, ?
    sentences = re.split(r'(?<=[.!?])\s+', cleaned)
    return [s.strip() for s in sentences if s.strip()]


def detect_filler_words(text, filler_list=None):
    """
    Detects single-word and multi-word filler occurrences using regex word boundaries.
    Returns:
        dict: {
            'filler_count': total count,
            'fillers_found': dict of {filler: count},
            'filler_frequency': fillers per 100 words,
            'highlighted_text': text with markers around fillers
        }
    """
    if filler_list is None:
        filler_list = load_filler_words()

    text_lower = text.lower()
    words = tokenize_words(text)
    total_words = max(len(words), 1)

    fillers_found = Counter()
    # Sort filler list by length descending so multi-word phrases match before individual words
    sorted_fillers = sorted(filler_list, key=lambda x: len(x.split()), reverse=True)

    # Keep track of spans already matched to avoid duplicate counting of nested fillers
    matched_spans = []

    for filler in sorted_fillers:
        # Regex for word boundaries
        pattern = r'\b' + re.escape(filler) + r'\b'
        for match in re.finditer(pattern, text_lower):
            start, end = match.span()
            # Check overlap
            if not any(s <= start < e or s < end <= e for s, e in matched_spans):
                matched_spans.append((start, end))
                fillers_found[filler] += 1

    total_filler_count = sum(fillers_found.values())
    filler_frequency = round((total_filler_count / total_words) * 100, 2)

    return {
        "filler_count": total_filler_count,
        "fillers_found": dict(fillers_found),
        "filler_frequency": filler_frequency
    }


def analyze_vocabulary(text):
    """
    Evaluates vocabulary richness, lexical diversity (Type-Token Ratio),
    and syllable/length distribution.
    """
    words = tokenize_words(text)
    total_words = len(words)

    if total_words == 0:
        return {
            "total_words": 0,
            "unique_words": 0,
            "ttr": 0.0,
            "vocabulary_score": 0.0,
            "avg_word_length": 0.0,
            "long_words_count": 0
        }

    unique_words = len(set(words))
    # Type-Token Ratio
    ttr = unique_words / total_words

    # Words with 6+ letters (considered moderately sophisticated vocabulary)
    long_words = [w for w in words if len(w) >= 6]
    long_words_ratio = len(long_words) / total_words
    avg_word_length = sum(len(w) for w in words) / total_words

    # Score calculation (0 to 100):
    # Combines TTR (diversity) + long word ratio + volume factor
    length_multiplier = min(1.0, total_words / 40)  # full credit for answers >= 40 words
    raw_score = (ttr * 50 + long_words_ratio * 50) * length_multiplier
    vocabulary_score = round(min(100.0, max(10.0, raw_score * 1.2)), 2)

    return {
        "total_words": total_words,
        "unique_words": unique_words,
        "ttr": round(ttr, 3),
        "vocabulary_score": vocabulary_score,
        "avg_word_length": round(avg_word_length, 2),
        "long_words_count": len(long_words)
    }


def analyze_grammar_and_structure(text):
    """
    Evaluates sentence structure, sentence length variability,
    capitalization, and readability.
    """
    sentences = tokenize_sentences(text)
    words = tokenize_words(text)
    sentence_count = len(sentences)
    word_count = len(words)

    if sentence_count == 0 or word_count == 0:
        return {
            "sentence_count": 0,
            "avg_sentence_length": 0.0,
            "grammar_score": 0.0,
            "readability_index": 0.0,
            "repeated_words": []
        }

    avg_sentence_length = word_count / sentence_count

    # Check repeated adjacent words (e.g. "the the", "I I")
    repeated_words = []
    for i in range(len(words) - 1):
        if words[i].lower() == words[i + 1].lower() and words[i].isalpha():
            repeated_words.append(words[i].lower())

    # Structural penalty for runaway sentences (>35 words) or too short (<4 words)
    structure_penalties = 0
    for s in sentences:
        s_words = len(tokenize_words(s))
        if s_words > 35:
            structure_penalties += 10
        elif s_words < 4:
            structure_penalties += 5

    # Capitalization check for beginnings of sentences
    capitalization_errors = 0
    for s in sentences:
        if s and not s[0].isupper():
            capitalization_errors += 1

    # Base grammar & structure score
    base_score = 90.0
    base_score -= min(30.0, structure_penalties)
    base_score -= min(15.0, capitalization_errors * 5)
    base_score -= min(15.0, len(repeated_words) * 7)

    # Reward balanced sentence length (12-25 words)
    if 12 <= avg_sentence_length <= 25:
        base_score += 10.0

    grammar_score = round(min(100.0, max(20.0, base_score)), 2)

    # Automated Readability Index (ARI) approximation: 4.71 * (chars/words) + 0.5 * (words/sentences) - 21.43
    char_count = sum(len(w) for w in words)
    ari = 4.71 * (char_count / max(word_count, 1)) + 0.5 * avg_sentence_length - 21.43
    readability_index = round(max(1.0, min(16.0, ari)), 1)

    return {
        "sentence_count": sentence_count,
        "avg_sentence_length": round(avg_sentence_length, 2),
        "grammar_score": grammar_score,
        "readability_index": readability_index,
        "repeated_words": list(set(repeated_words))
    }


def analyze_text(text, filler_list=None):
    """
    Master NLP pipeline function aggregating all linguistic features.
    """
    cleaned = clean_text(text)
    fillers = detect_filler_words(cleaned, filler_list)
    vocab = analyze_vocabulary(cleaned)
    grammar = analyze_grammar_and_structure(cleaned)

    return {
        "text": cleaned,
        "word_count": vocab["total_words"],
        "sentence_count": grammar["sentence_count"],
        "unique_words": vocab["unique_words"],
        "ttr": vocab["ttr"],
        "vocabulary_score": vocab["vocabulary_score"],
        "avg_word_length": vocab["avg_word_length"],
        "filler_count": fillers["filler_count"],
        "fillers_found": fillers["fillers_found"],
        "filler_frequency": fillers["filler_frequency"],
        "avg_sentence_length": grammar["avg_sentence_length"],
        "grammar_score": grammar["grammar_score"],
        "readability_index": grammar["readability_index"],
        "repeated_words": grammar["repeated_words"]
    }
