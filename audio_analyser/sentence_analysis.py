"""
Sentence structure and length analysis.
"""

import re
from .config import SENTENCE_PATTERN, WORD_PATTERN

def analyze_sentences(transcript, long_sentence_threshold=25):
    """
    Analyze sentence structure and length.
    
    Args:
        transcript (str): Full transcript text
        long_sentence_threshold (int): Word count threshold for long sentences
    
    Returns:
        tuple: (num_sentences, avg_length, longest_sentence, longest_length, 
                long_count, long_percentage)
    """
    sentences = SENTENCE_PATTERN.split(transcript)
    sentences = [s.strip() for s in sentences if s.strip()]

    sentence_lengths = []
    longest_sentence = ""
    longest_length = 0

    for sentence in sentences:
        words_in_sentence = WORD_PATTERN.findall(sentence)
        length = len(words_in_sentence)
        sentence_lengths.append(length)
        
        if length > longest_length:
            longest_length = length
            longest_sentence = sentence

    num_sentences = len(sentence_lengths)
    avg_sentence_length = sum(sentence_lengths) / num_sentences if num_sentences > 0 else 0
    long_sentence_count = sum(1 for length in sentence_lengths if length > long_sentence_threshold)
    long_sentence_percentage = (long_sentence_count / num_sentences * 100) if num_sentences > 0 else 0

    return (num_sentences, avg_sentence_length, longest_sentence, 
            longest_length, long_sentence_count, long_sentence_percentage)