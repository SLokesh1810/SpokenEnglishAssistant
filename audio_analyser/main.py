"""
Main orchestration for audio analysis.
"""

import os
import librosa
import gc

from . import convertor
from .config import (
    DEFAULT_MODEL_SIZE,
    DEFAULT_TOP_K,
    DEFAULT_LONG_SENTENCE_THRESHOLD,
    DEFAULT_TOP_PHRASES
)
from .text_processing import clean_text
from .word_analysis import categorize_words, get_top_words
from .sentence_analysis import analyze_sentences
from .repetition_analysis import analyze_repetition
from .fluency_analysis import calculate_fluency
from .filler_analysis import analyze_filler_usage
from .drift_analysis import analyze_confidence_drift
from .transcription import transcribe_audio, save_transcript, load_transcript, verify_transcript_integrity
from .output import print_analysis_results, save_json as save_json_output

def analyze_transcript(transcript, audio_duration_sec, top_k, long_sentence_threshold, top_phrases):
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
        transcript, long_sentence_threshold
    )

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
            }
        },
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
        "confidence_drift": drift_data
    }

    return analysis_results

def main(base_path, audio_filename, 
         model_size=DEFAULT_MODEL_SIZE, 
         top_k=DEFAULT_TOP_K, 
         long_sentence_threshold=DEFAULT_LONG_SENTENCE_THRESHOLD, 
         top_phrases=DEFAULT_TOP_PHRASES, 
         return_json=False, 
         save_json=False):
    """
    Main entry point for audio analysis.
    
    Args:
        base_path (str): Directory containing audio file
        audio_filename (str): Audio filename
        model_size (str): Whisper model size
        top_k (int): Number of top words
        long_sentence_threshold (int): Long sentence threshold
        top_phrases (int): Number of top phrases
        return_json (bool): Return results as dict
        save_json (bool): Save results to JSON file
    
    Returns:
        dict or None: Analysis results if return_json=True
    """
    print("\n" + "="*50)
    print("🎙️  SPOKEN ENGLISH ASSISTANT")
    print("="*50)
    
    audio_path = os.path.join(base_path, audio_filename)
    
    # Ensure audio exists
    print(f"\n📁 Checking for audio file: {audio_filename}")
    if not os.path.exists(audio_path):
        print("  ⚠️  Audio not found, converting from video...")
        audio_path = convertor.convert(audio_path)
        print(f"  ✓ Conversion complete!")
    else:
        print(f"  ✓ Audio file found!")

    # Load audio
    print(f"\n🎵 Loading audio file...")
    audio_array, sr = librosa.load(audio_path, sr=16000)
    audio_duration_sec = librosa.get_duration(y=audio_array, sr=sr)
    print(f"  ✓ Audio loaded: {audio_duration_sec:.2f} seconds")

    # Check for existing transcript
    print(f"\n📝 Checking for existing transcript...")
    transcript = load_transcript(base_path, audio_filename)
    
    if transcript:
        print("  ✓ Transcript found, skipping transcription")
        # Verify integrity
        print("  🔒 Verifying transcript integrity...")
        if verify_transcript_integrity(base_path, audio_filename):
            print("  ✓ Integrity verified!")
        else:
            print("  ⚠️  Warning: Transcript may have been modified!")
    else:
        print("  ⚠️  Transcript not found")
        print(f"\n🎤 Starting transcription (using {model_size} model)...")
        transcript = transcribe_audio(audio_array, model_size)
        print(f"  ✓ Transcription complete!")
        print(f"\n💾 Saving transcript with hash...")
        save_transcript(transcript, base_path, audio_filename)

    # Cleanup audio array
    del audio_array
    gc.collect()

    # Analyze
    analysis_results = analyze_transcript(
        transcript, 
        audio_duration_sec, 
        top_k, 
        long_sentence_threshold, 
        top_phrases
    )

    # Output
    if save_json:
        print("\n💾 Saving JSON results...")
        save_json_output(analysis_results, base_path, audio_filename)

    if not return_json:
        print_analysis_results(analysis_results)
    
    print("\n" + "="*50)
    print("✅ ANALYSIS COMPLETE!")
    print("="*50 + "\n")

    return analysis_results if return_json else None