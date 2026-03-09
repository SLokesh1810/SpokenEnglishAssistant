"""
Fluency metrics (WPM, etc.).
"""

def calculate_fluency(words, audio_duration_sec):
    """
    Calculate fluency metrics.
    
    Args:
        words (list): List of words
        audio_duration_sec (float): Audio duration in seconds
    
    Returns:
        tuple: (total_words, wpm)
    """
    total_words = len(words)
    wpm = (total_words / audio_duration_sec) * 60 if audio_duration_sec > 0 else 0
    return total_words, wpm