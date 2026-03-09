"""
Confidence drift analysis over time.
"""

import re
from .filler_analysis import analyze_filler_usage

def analyze_text_segment(segment_text, segment_duration):
    """
    Analyze a text segment and return speaking metrics.
    
    Args:
        segment_text (str): The text to analyze
        segment_duration (float): Duration in seconds
    
    Returns:
        dict: Contains word_count, wpm, avg_sentence_length, filler analysis
    """
    words = re.findall(r"\b[a-zA-Z']+\b", segment_text.lower())
    word_count = len(words)
    
    wpm = (word_count / segment_duration) * 60 if segment_duration > 0 else 0
    
    sents = re.split(r'[.!?]+', segment_text)
    sents = [s.strip() for s in sents if s.strip()]
    
    sent_lengths = [len(re.findall(r"\b[a-zA-Z']+\b", sent)) for sent in sents]
    avg_sent_len = sum(sent_lengths) / len(sent_lengths) if sent_lengths else 0
    
    filler_analysis = analyze_filler_usage(words, segment_text)
    
    return {
        "word_count": word_count,
        "wpm": round(wpm, 1),
        "avg_sentence_length": round(avg_sent_len, 1),
        "fillers": filler_analysis
    }

def split_transcript_into_thirds(transcript_text):
    """
    Split transcript into three equal parts by character count.
    
    Args:
        transcript_text (str): Full transcript
    
    Returns:
        tuple: (first_third, middle_third, last_third) as strings
    """
    chars_total = len(transcript_text)
    first_end = chars_total // 3
    second_end = 2 * chars_total // 3
    
    return (
        transcript_text[:first_end],
        transcript_text[first_end:second_end],
        transcript_text[second_end:]
    )

def calculate_segment_durations(segment_1, segment_2, segment_3, total_duration, total_words):
    """
    Estimate duration for each segment based on word count proportion.
    
    Args:
        segment_1, segment_2, segment_3 (str): Text segments
        total_duration (float): Total audio duration in seconds
        total_words (int): Total word count
    
    Returns:
        tuple: (duration_1, duration_2, duration_3) in seconds
    """
    words_1 = len(re.findall(r"\b[a-zA-Z']+\b", segment_1))
    words_2 = len(re.findall(r"\b[a-zA-Z']+\b", segment_2))
    words_3 = len(re.findall(r"\b[a-zA-Z']+\b", segment_3))
    
    if total_words == 0:
        return 0, 0, 0
    
    return (
        (words_1 / total_words) * total_duration,
        (words_2 / total_words) * total_duration,
        (words_3 / total_words) * total_duration
    )

def analyze_confidence_drift(transcript, total_words, audio_duration):
    """
    Analyze how speaking patterns change over the course of the session.
    
    Args:
        transcript (str): Full transcript text
        total_words (int): Total word count
        audio_duration (float): Total audio duration in seconds
    
    Returns:
        dict or None: Contains segment_1, segment_2, segment_3 metrics, or None if insufficient data
    """
    if total_words < 3:
        return None
    
    seg_1_text, seg_2_text, seg_3_text = split_transcript_into_thirds(transcript)
    dur_1, dur_2, dur_3 = calculate_segment_durations(
        seg_1_text, seg_2_text, seg_3_text, audio_duration, total_words
    )
    
    segment_1 = analyze_text_segment(seg_1_text, dur_1)
    segment_2 = analyze_text_segment(seg_2_text, dur_2)
    segment_3 = analyze_text_segment(seg_3_text, dur_3)
    
    wpm_trend = ("increasing" if segment_3['wpm'] > segment_1['wpm'] 
                 else "decreasing" if segment_3['wpm'] < segment_1['wpm'] 
                 else "stable")
    sent_trend = ("shorter" if segment_3['avg_sentence_length'] < segment_1['avg_sentence_length'] 
                  else "longer" if segment_3['avg_sentence_length'] > segment_1['avg_sentence_length'] 
                  else "stable")
    filler_trend = ("decreasing" if segment_3['fillers']['total_fillers']['per_100_words'] < segment_1['fillers']['total_fillers']['per_100_words'] 
                    else "increasing" if segment_3['fillers']['total_fillers']['per_100_words'] > segment_1['fillers']['total_fillers']['per_100_words'] 
                    else "stable")
    
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