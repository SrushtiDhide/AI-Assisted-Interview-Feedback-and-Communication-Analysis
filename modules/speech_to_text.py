"""
Speech-to-Text Module
Local voice-to-text transcription using faster-whisper and audio duration analysis.
"""

import os
import wave
import contextlib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIO_DIR = os.path.join(BASE_DIR, "audio", "recordings")

# Cached model instance
_WHISPER_MODEL = None


def get_audio_duration(audio_path):
    """
    Computes duration of an audio file in seconds.
    Supports standard WAV files directly via standard library `wave`,
    with fallback to soundfile or file size estimation.
    """
    if not os.path.exists(audio_path):
        return 0.0

    # Try standard wave module first (zero dependency)
    try:
        with contextlib.closing(wave.open(audio_path, 'r')) as f:
            frames = f.getnframes()
            rate = f.getframerate()
            return frames / float(rate)
    except Exception:
        pass

    # Try soundfile if installed
    try:
        import soundfile as sf
        info = sf.info(audio_path)
        return float(info.duration)
    except Exception:
        pass

    # Heuristic fallback: rough estimate assuming standard 16kHz 16-bit mono PCM (32000 bytes/sec)
    try:
        size_bytes = os.path.getsize(audio_path)
        return max(1.0, size_bytes / 32000.0)
    except Exception:
        return 0.0


def load_whisper_model(model_size="tiny", device="cpu", compute_type="int8", download_root=None):
    """
    Loads and caches the faster-whisper model.
    'tiny' or 'base' is recommended for fast inference without a dedicated GPU.
    """
    global _WHISPER_MODEL
    if _WHISPER_MODEL is not None:
        return _WHISPER_MODEL

    try:
        from faster_whisper import WhisperModel
        _WHISPER_MODEL = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
            download_root=download_root
        )
        return _WHISPER_MODEL
    except ImportError:
        raise ImportError(
            "faster-whisper is not installed. Please run 'pip install faster-whisper' to enable speech recognition."
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load faster-whisper model: {e}")


def transcribe_audio(audio_path, model_size="tiny"):
    """
    Transcribes audio file to text using faster-whisper.

    Returns:
        dict: {
            'success': bool,
            'transcript': str,
            'duration': float,
            'language': str,
            'error': str or None
        }
    """
    if not os.path.exists(audio_path):
        return {
            "success": False,
            "transcript": "",
            "duration": 0.0,
            "language": "",
            "error": f"Audio file not found at {audio_path}"
        }

    duration = get_audio_duration(audio_path)

    try:
        model = load_whisper_model(model_size=model_size)
        segments, info = model.transcribe(audio_path, beam_size=1)
        transcript_text = " ".join([seg.text.strip() for seg in segments]).strip()

        # If wave-based duration failed, use faster-whisper duration if present
        if duration <= 0.0 and hasattr(info, "duration") and info.duration:
            duration = float(info.duration)

        return {
            "success": True,
            "transcript": transcript_text,
            "duration": round(duration, 2),
            "language": info.language if hasattr(info, "language") else "en",
            "error": None
        }
    except Exception as e:
        return {
            "success": False,
            "transcript": "",
            "duration": round(duration, 2),
            "language": "",
            "error": str(e)
        }


def save_uploaded_audio(uploaded_file, filename="temp_recording.wav"):
    """
    Saves an uploaded Streamlit audio file or buffer to the recordings folder.
    """
    os.makedirs(AUDIO_DIR, exist_ok=True)
    target_path = os.path.join(AUDIO_DIR, filename)

    with open(target_path, "wb") as f:
        f.write(uploaded_file.getbuffer() if hasattr(uploaded_file, "getbuffer") else uploaded_file.read())

    return target_path
