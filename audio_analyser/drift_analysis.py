"""
Confidence drift analysis over time.
"""

import re
from .filler_analysis import analyze_filler_usage
from .sentence_analysis import split_into_sentences_by_length
from .text_processing import clean_text

def analyze_text_segment(segment_text, segment_duration):
    words = clean_text(segment_text)
    word_count = len(words)

    wpm = (word_count / segment_duration) * 60 if segment_duration > 0 else 0

    # NEW: use same logic as main analysis
    sentences = split_into_sentences_by_length(words, max_len=20)
    sent_lengths = [len(s) for s in sentences]

    avg_sent_len = sum(sent_lengths) / len(sent_lengths) if sent_lengths else 0

    filler_analysis = analyze_filler_usage(words, segment_text)

    return {
        "word_count": word_count,
        "wpm": round(wpm, 1),
        "avg_sentence_length": round(avg_sent_len, 1),
        "fillers": filler_analysis
    }

from .text_processing import clean_text

def split_words_into_thirds(words):
    """
    Split word list into 3 equal parts.
    """
    n = len(words)
    return (
        words[:n//3],
        words[n//3:2*n//3],
        words[2*n//3:]
    )

def analyze_confidence_drift(transcript, total_words, audio_duration):
    if total_words < 10:
        return None

    words = clean_text(transcript)

    seg1_words, seg2_words, seg3_words = split_words_into_thirds(words)

    # Split durations equally (better than fake estimation)
    seg_duration = audio_duration / 3

    segment_1 = analyze_text_segment(" ".join(seg1_words), seg_duration)
    segment_2 = analyze_text_segment(" ".join(seg2_words), seg_duration)
    segment_3 = analyze_text_segment(" ".join(seg3_words), seg_duration)

    # Trends
    def get_trend(start, end):
        if end > start:
            return "increasing"
        elif end < start:
            return "decreasing"
        else:
            return "stable"

    wpm_trend = get_trend(segment_1['wpm'], segment_3['wpm'])
    sent_trend = get_trend(segment_1['avg_sentence_length'], segment_3['avg_sentence_length'])
    filler_trend = get_trend(
        segment_1['fillers']['total_fillers']['per_100_words'],
        segment_3['fillers']['total_fillers']['per_100_words']
    )

    return {
        "start": segment_1,
        "middle": segment_2,
        "end": segment_3,
        "trends": {
            "speaking_speed": wpm_trend,
            "sentence_length": sent_trend,
            "filler_usage": filler_trend
        }
    }