from .scoring import fluency_score, confidence_score, clarity_score, overall_score
from .feedback import generate_feedback
from .text_processing import clean_text
from .word_analysis import categorize_words, get_top_words
from .sentence_analysis import analyze_sentences, analyze_raw_sentences
from .repetition_analysis import analyze_repetition
from .fluency_analysis import calculate_fluency
from .filler_analysis import analyze_filler_usage
from .drift_analysis import analyze_confidence_drift

def analyze_transcript(transcript, audio_duration_sec, top_k, long_sentence_threshold, top_phrases, pause_data):
    """
    Perform all analysis steps on the transcript.
    
    Args:
        transcript (str): Transcript text
        audio_duration_sec (float): Audio duration in seconds
        top_k (int): Number of top words to extract
        long_sentence_threshold (int): Threshold for long sentences
        top_phrases (int): Number of top repeated phrases
    
    Returns:
        dict: Complete analysis results
    """
    print("\n🔍 Starting analysis...")
    
    # Clean words
    print("  ➤ Cleaning text and extracting words...")
    clean_words = clean_text(transcript)

    # Word analysis
    print("  ➤ Analyzing word frequencies and categories...")
    top_words = get_top_words(clean_words, top_k)
    category_counts = categorize_words(clean_words)

    # Sentence analysis
    print("  ➤ Analyzing sentence structure...")
    (num_sentences, avg_sentence_length, longest_sentence, 
     longest_length, long_sentence_count, long_sentence_percentage) = analyze_sentences(
        clean_words, long_sentence_threshold
    )
    raw_issues = analyze_raw_sentences(transcript)

    # Repetition analysis
    print("  ➤ Analyzing vocabulary repetition...")
    unique_words, repetition_ratio, top_bigrams, top_trigrams = analyze_repetition(
        clean_words, transcript, top_phrases
    )

    # Fluency metrics
    print("  ➤ Calculating fluency metrics...")
    total_words, wpm = calculate_fluency(clean_words, audio_duration_sec)

    # Filler analysis
    print("  ➤ Analyzing filler words and phrases...")
    filler_results = analyze_filler_usage(clean_words, transcript)

    # Confidence drift
    print("  ➤ Analyzing confidence drift over time...")
    drift_data = analyze_confidence_drift(transcript, total_words, audio_duration_sec)

    print("  ✓ Analysis complete!\n")

    fluency = fluency_score(wpm)

    confidence = confidence_score(
        filler_results["total_fillers"]["per_100_words"],
        category_counts["Weak language (confidence killers)"] / total_words * 100
    )

    clarity = clarity_score(
        repetition_ratio,
        avg_sentence_length
    )

    overall = overall_score(fluency, confidence, clarity)

    # Build results
    analysis_results = {
        "audio_info": {
            "duration_seconds": round(audio_duration_sec, 2),
            "total_words": total_words,
            "words_per_minute": round(wpm, 2)
        },
        "top_words": [
            {"word": word, "count": count} for word, count in top_words
        ],
        "word_categories": {
            category: {
                "count": count,
                "percentage": round((count / total_words * 100), 1) if total_words > 0 else 0
            }
            for category, count in category_counts.items()
        },
        "sentence_analysis": {
            "total_sentences": num_sentences,
            "avg_sentence_length": round(avg_sentence_length, 1),
            "longest_sentence_length": longest_length,
            "longest_sentence_preview": longest_sentence[:150] + ("..." if len(longest_sentence) > 150 else ""),
            "long_sentences": {
                "count": long_sentence_count,
                "percentage": round(long_sentence_percentage, 1),
                "threshold": long_sentence_threshold
            },
            "raw_speech_issues": {
                "max_sentence_length": raw_issues["max_sentence_length"],
                "very_long_sentences": raw_issues["very_long_sentences"]
            }
        },
        "pause_analysis": pause_data,
        "vocabulary": {
            "total_words": total_words,
            "unique_words": unique_words,
            "repetition_ratio": round(repetition_ratio, 2),
            "repeated_trigrams": [
                {"phrase": phrase, "count": count} for phrase, count in top_trigrams
            ],
            "repeated_bigrams": [
                {"phrase": phrase, "count": count} for phrase, count in top_bigrams
            ]
        },
        "filler_analysis": filler_results,
        "confidence_drift": drift_data,
        "score": {
            "fluency": fluency,
            "confidence": confidence,
            "clarity": clarity,
            "overall": overall
        },
    }

    analysis_results["feedback"] = generate_feedback(analysis_results)

    return analysis_results