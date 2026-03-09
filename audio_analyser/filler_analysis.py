"""
Filler word and phrase detection.
"""

from .config import FILLER_WORDS, FILLER_PHRASES

def count_filler_words(words):
    """
    Count individual filler words.
    
    Args:
        words (list): List of lowercase words
    
    Returns:
        int: Count of filler words
    """
    return sum(1 for word in words if word in FILLER_WORDS)

def count_filler_phrases(text):
    """
    Count multi-word filler phrases in the transcript.
    
    Args:
        text (str): Original transcript text (with casing)
    
    Returns:
        tuple: (phrase_count, phrase_details)
            - phrase_count: Total number of filler phrases found
            - phrase_details: Dict with {phrase: count}
    """
    text_lower = text.lower()
    phrase_details = {}
    total_count = 0
    
    for phrase in FILLER_PHRASES:
        count = text_lower.count(phrase)
        if count > 0:
            phrase_details[phrase] = count
            total_count += count
    
    return total_count, phrase_details

def analyze_filler_usage(words, transcript):
    """
    Comprehensive filler analysis including words and phrases.
    
    Args:
        words (list): List of lowercase words
        transcript (str): Original transcript text
    
    Returns:
        dict: Complete filler analysis
    """
    total_words = len(words)
    
    filler_word_count = count_filler_words(words)
    filler_phrase_count, phrase_details = count_filler_phrases(transcript)
    total_filler_instances = filler_word_count + filler_phrase_count
    
    filler_word_per_100 = (filler_word_count / total_words * 100) if total_words > 0 else 0
    filler_phrase_per_100 = (filler_phrase_count / total_words * 100) if total_words > 0 else 0
    total_filler_per_100 = (total_filler_instances / total_words * 100) if total_words > 0 else 0
    
    return {
        "filler_words": {
            "count": filler_word_count,
            "per_100_words": round(filler_word_per_100, 1)
        },
        "filler_phrases": {
            "count": filler_phrase_count,
            "per_100_words": round(filler_phrase_per_100, 1),
            "details": phrase_details
        },
        "total_fillers": {
            "count": total_filler_instances,
            "per_100_words": round(total_filler_per_100, 1)
        }
    }