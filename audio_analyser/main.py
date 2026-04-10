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

from .pause_analysis import analyze_pauses
from .pipeline import analyze_transcript
from .transcription import transcribe_audio, save_transcript, load_transcript, verify_transcript_integrity
from .output import print_analysis_results, save_json as save_json_output



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
    pause_data = analyze_pauses(audio_path)
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
        top_phrases,
        pause_data
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