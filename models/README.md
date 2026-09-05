# Models Directory

This directory stores configuration, cache, and documentation for local AI models used by the application.

## 1. Speech-to-Text: `faster-whisper`
- **Default Model:** `tiny` (or `base`)
- **Execution Mode:** CPU execution using `int8` quantization for fast local inference.
- **Cache Location:** Automatically downloaded to `models/whisper/` on first use.
- **Offline Setup:** Once downloaded, `faster-whisper` runs completely offline without internet connectivity.

## 2. Large Language Model: `Qwen2.5-3B-Instruct`
- **Model Runtime:** [Ollama](https://ollama.com)
- **Model Tag:** `qwen2.5:3b`
- **Command to download and start:**
  ```bash
  ollama pull qwen2.5:3b
  ollama run qwen2.5:3b
  ```
- **API Endpoint:** Runs locally on `http://localhost:11434`
- **Purpose:** 
  - Personalized interview coaching feedback (strengths, areas for improvement, actionable advice).
  - Dynamic follow-up question generation.
  - Resume-based custom question generation.
- **Resilience:** If Ollama is not running, the application automatically activates its built-in rule-based expert feedback engine so the platform never fails during evaluations or viva presentations.
