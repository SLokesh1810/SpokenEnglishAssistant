"""
Vocabulary repetition and phrase analysis.
"""

from collections import Counter
from .text_processing import extract_words_with_case

def analyze_repetition(words, transcript, top_phrases=5):
    """
    Analyze vocabulary repetition and common phrases.
    
    Args:
        words (list): List of lowercase words
        transcript (str): Original transcript
        top_phrases (int): Number of top phrases to return
    
    Returns:
        tuple: (unique_words, repetition_ratio, top_bigrams, top_trigrams)
    """
    total_words = len(words)
    unique_words = len(set(words))
    repetition_ratio = unique_words / total_words if total_words > 0 else 0

    transcript_words = extract_words_with_case(transcript)
    
    # Bigrams
    bigrams = [f"{w1} {w2}" for w1, w2 in zip(transcript_words, transcript_words[1:])]
    bigram_counts = Counter(bigrams)
    top_bigrams = [(phrase, count) for phrase, count in bigram_counts.most_common(top_phrases * 2) if count > 1][:top_phrases]

    # Trigrams
    trigrams = [f"{w1} {w2} {w3}" for w1, w2, w3 in zip(transcript_words, transcript_words[1:], transcript_words[2:])]
    trigram_counts = Counter(trigrams)
    top_trigrams = [(phrase, count) for phrase, count in trigram_counts.most_common(top_phrases * 2) if count > 1][:top_phrases]

    return unique_words, repetition_ratio, top_bigrams, top_trigrams