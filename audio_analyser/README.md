# Audio Analyser Module

Transcribe and analyze spoken English audio for fluency, confidence, and speaking patterns.

---

## 📁 Module Structure
```
audio_analyser/
├── config.py              # Constants (word categories, filler words, regex patterns)
├── models.py              # Whisper model caching
├── text_processing.py     # Text cleaning and word extraction
├── filler_analysis.py     # Filler word/phrase detection
├── word_analysis.py       # Word categorization and frequency
├── sentence_analysis.py   # Sentence structure analysis
├── repetition_analysis.py # Vocabulary repetition (n-grams)
├── fluency_analysis.py    # WPM calculation
├── drift_analysis.py      # Confidence drift over time
├── transcription.py       # Whisper transcription + SHA-256 hashing
├── output.py              # Console/JSON output formatting
├── main.py                # Main orchestration
└── convertor.py           # Video → Audio conversion
```

---

## 🚀 Quick Start
```python
from audio_analyser import main

# Basic usage
main(base_path="data/", audio_filename="speech.wav")

# Advanced usage
results = main(
    base_path="data/",
    audio_filename="speech.wav",
    model_size="small",
    return_json=True,
    save_json=True
)
```

---

## 📊 Key Metrics

### 1. **Fluency Metrics**
- **WPM (Words Per Minute):** `(total_words / duration_sec) × 60`
  - 90-110: Slow, 120-150: Normal, 160+: Fast

### 2. **Vocabulary Metrics**
- **Repetition Ratio:** `unique_words / total_words`
  - 0.3: Low diversity, 0.5: Normal, 0.8: High diversity
- **N-grams:** Repeated 2-word and 3-word phrases

### 3. **Sentence Analysis**
- **Avg Sentence Length:** `Σ(sentence_lengths) / num_sentences`
- **Long Sentence %:** `(sentences > threshold / total_sentences) × 100`

### 4. **Filler Analysis**
- **Filler Density:** `(filler_count / total_words) × 100`
- Tracks words: um, uh, like, so, actually, basically
- Tracks phrases: "you know", "I mean", "kind of"

### 5. **Confidence Drift**
- Splits transcript into thirds (start/middle/end)
- Tracks WPM, sentence length, filler usage over time
- Shows trends: increasing/decreasing/stable

### 6. **Word Categories**
7 categories: Self-references, Connectors (basic/advanced), Action verbs, Emotion words, Planning words, Weak language

---

## 🔒 Privacy & Security

**SHA-256 Hashing:**
- Every transcript is automatically hashed using SHA-256
- Hash stored in `.txt.sha256` file alongside transcript
- Verify integrity with `verify_transcript_integrity()`
- Hash format: 64-character hexadecimal string

**Files Created:**
```
transcript_audio2.txt        # Full transcript
transcript_audio2.txt.sha256 # SHA-256 hash for integrity
analysis_audio2.json         # Analysis results
```

---

## 📝 Function Reference

### `config.py`
- **WORD_CATEGORIES:** 7 categories, 150+ predefined words
- **FILLER_WORDS:** 15 single-word fillers
- **FILLER_PHRASES:** 11 multi-word filler phrases
- **WORD_PATTERN, SENTENCE_PATTERN:** Compiled regex

### `models.py`
- `get_device()` → Returns "cuda" or "cpu"
- `get_or_load_model(size, device)` → Loads and caches Whisper model

### `text_processing.py`
- `clean_text(text)` → Returns lowercase word list
- `extract_words_with_case(text)` → Preserves casing

### `filler_analysis.py`
- `count_filler_words(words)` → Count single-word fillers
- `count_filler_phrases(text)` → Count phrase fillers
- `analyze_filler_usage(words, transcript)` → Full filler analysis

### `word_analysis.py`
- `categorize_words(words)` → Count by category
- `get_top_words(words, k)` → Top K frequent words

### `sentence_analysis.py`
- `analyze_sentences(transcript, threshold)` → Returns 6-tuple:
  1. num_sentences
  2. avg_length
  3. longest_sentence (text)
  4. longest_length
  5. long_count
  6. long_percentage

### `repetition_analysis.py`
- `analyze_repetition(words, transcript, k)` → Returns 4-tuple:
  1. unique_words
  2. repetition_ratio
  3. top_bigrams
  4. top_trigrams

### `fluency_analysis.py`
- `calculate_fluency(words, duration)` → Returns (total_words, wpm)

### `drift_analysis.py`
- `analyze_text_segment(text, duration)` → Segment metrics
- `split_transcript_into_thirds(text)` → 3 equal text parts
- `calculate_segment_durations(seg1, seg2, seg3, total_dur, total_words)` → Duration estimates
- `analyze_confidence_drift(transcript, total_words, duration)` → Full drift analysis

### `transcription.py`
- `transcribe_audio(audio_array, model_size)` → Whisper transcription
- `hash_file(filepath)` → SHA-256 hash of file
- `save_transcript(transcript, base_path, filename)` → Save transcript + hash
- `load_transcript(base_path, filename)` → Load existing transcript
- `verify_transcript_integrity(base_path, filename)` → Verify hash matches

### `output.py`
- `print_confidence_drift(drift_data)` → Display drift analysis
- `print_analysis_results(results)` → Display all results
- `save_json(results, base_path, filename)` → Export to JSON

### `main.py`
- `analyze_transcript(...)` → Coordinates all analysis modules
- `main(...)` → Entry point, full pipeline

---

## 🎯 Analysis Pipeline
```
Audio (.wav/.mp4)
    ↓
[Convert if needed] → 16kHz mono WAV
    ↓
[Whisper] → Transcript text
    ↓
[SHA-256] → Hash file (.txt.sha256)
    ↓
[Text Processing] → Clean words
    ↓
[7 Analysis Modules] → Metrics
    ↓
[Output] → Console + JSON
```

---

## 📦 Output JSON Structure
```json
{
  "audio_info": {
    "duration_seconds": 534.21,
    "total_words": 1104,
    "words_per_minute": 124.1
  },
  "top_words": [{"word": "i", "count": 93}],
  "word_categories": {
    "Self references": {"count": 139, "percentage": 12.6}
  },
  "sentence_analysis": {
    "total_sentences": 59,
    "avg_sentence_length": 18.7,
    "longest_sentence_length": 93,
    "long_sentences": {"count": 12, "percentage": 20.3}
  },
  "vocabulary": {
    "unique_words": 304,
    "repetition_ratio": 0.28,
    "repeated_trigrams": [{"phrase": "I need to", "count": 5}]
  },
  "filler_analysis": {
    "filler_words": {"count": 69, "per_100_words": 6.2},
    "filler_phrases": {"count": 9, "per_100_words": 0.8},
    "total_fillers": {"count": 78, "per_100_words": 7.1}
  },
  "confidence_drift": {
    "start": {...},
    "middle": {...},
    "end": {...},
    "trends": {
      "speaking_speed": "stable",
      "sentence_length": "shorter",
      "filler_usage": "increasing"
    }
  }
}
```

---

## ⚙️ Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_size` | "small" | Whisper model: tiny/base/small/medium/large |
| `top_k` | 5 | Number of top words to show |
| `long_sentence_threshold` | 25 | Words to consider sentence "long" |
| `top_phrases` | 5 | Number of top repeated phrases |
| `return_json` | False | Return results as dict |
| `save_json` | False | Save results to JSON file |

---

## 🔧 Performance

- **First run:** ~10s (model load) + ~3-5s/min audio (GPU)
- **Subsequent runs:** ~2s + ~3-5s/min audio (cached model)
- **Memory:** 500MB (small), 1.5GB (medium), 3GB (large)
- **Storage:** Transcript ~1-100KB, Hash ~100 bytes, JSON ~5-20KB

---

## 📌 Notes

- Transcripts auto-hashed with SHA-256 for integrity verification
- Model cached in memory for faster repeated analysis
- GPU automatically detected and used if available
- Transcript reused if already exists (skips re-transcription)

---

**Version:** 1.0.0  
**Last Updated:** 2025-03-09