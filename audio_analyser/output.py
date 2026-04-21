"""
Output formatting and display.
"""

import json
import os

def print_confidence_drift(drift_data):
    """Print the session progression analysis."""
    print("\n==============================")
    print("SESSION PROGRESSION")
    print("==============================")
    
    if drift_data is None:
        print("Transcript too short for progression analysis.")
        return
    
    seg_1 = drift_data["start"]
    seg_2 = drift_data["middle"]
    seg_3 = drift_data["end"]
    
    print("Start (first third):")
    print(f"  - WPM                     : {seg_1['wpm']:.1f}")
    print(f"  - Avg sentence length     : {seg_1['avg_sentence_length']:.1f} words")
    print(f"  - Filler words per 100    : {seg_1['fillers']['filler_words']['per_100_words']:.1f}")
    print(f"  - Filler phrases per 100  : {seg_1['fillers']['filler_phrases']['per_100_words']:.1f}")
    print(f"  - Total fillers per 100   : {seg_1['fillers']['total_fillers']['per_100_words']:.1f}")
    
    print("\nMiddle:")
    print(f"  - WPM                     : {seg_2['wpm']:.1f}")
    print(f"  - Avg sentence length     : {seg_2['avg_sentence_length']:.1f} words")
    print(f"  - Filler words per 100    : {seg_2['fillers']['filler_words']['per_100_words']:.1f}")
    print(f"  - Filler phrases per 100  : {seg_2['fillers']['filler_phrases']['per_100_words']:.1f}")
    print(f"  - Total fillers per 100   : {seg_2['fillers']['total_fillers']['per_100_words']:.1f}")
    
    print("\nEnd (last third):")
    print(f"  - WPM                     : {seg_3['wpm']:.1f}")
    print(f"  - Avg sentence length     : {seg_3['avg_sentence_length']:.1f} words")
    print(f"  - Filler words per 100    : {seg_3['fillers']['filler_words']['per_100_words']:.1f}")
    print(f"  - Filler phrases per 100  : {seg_3['fillers']['filler_phrases']['per_100_words']:.1f}")
    print(f"  - Total fillers per 100   : {seg_3['fillers']['total_fillers']['per_100_words']:.1f}")
    
    trends = drift_data["trends"]
    
    # Fixed arrow mapping - handle all trend values
    def get_trend_display(trend_value):
        """Convert trend value to display string with arrow."""
        if trend_value == "increasing":
            return "↑ increasing"
        elif trend_value == "decreasing":
            return "↓ decreasing"
        elif trend_value == "stable":
            return "→ stable"
        elif trend_value == "shorter":
            return "↓ shorter"
        elif trend_value == "longer":
            return "↑ longer"
        else:
            return f"→ {trend_value}"  # Fallback for unknown values
    
    print("\nTrends:")
    print(f"  - Speaking speed          : {get_trend_display(trends['speaking_speed'])}")
    print(f"  - Sentence length         : {get_trend_display(trends['sentence_length'])}")
    print(f"  - Filler word usage       : {get_trend_display(trends['filler_usage'])}")

def print_analysis_results(analysis_results):
    """Print all analysis results to console."""
    print("\n" + "="*50)
    print("📊 ANALYSIS RESULTS")
    print("="*50)
    
    # Extract data
    audio_info = analysis_results["audio_info"]
    top_words = analysis_results["top_words"]
    word_categories = analysis_results["word_categories"]
    sentence_analysis = analysis_results["sentence_analysis"]
    vocabulary = analysis_results["vocabulary"]
    filler_analysis = analysis_results["filler_analysis"]
    drift_data = analysis_results["confidence_drift"]
    
    total_words = audio_info["total_words"]
    
    # Print sections
    print("\n==============================")
    print("TOP USED WORDS")
    print("==============================")
    for item in top_words:
        print(f"{item['word']:>10} : {item['count']}")

    print("\n==============================")
    print("WORD CATEGORY BREAKDOWN")
    print("==============================")
    if total_words > 0:
        for category, data in word_categories.items():
            print(f"- {category:<35} : {data['count']:3d} words ({data['percentage']:5.1f}%)")
    else:
        print("No words detected in transcript.")

    print("\n==============================")
    print("SENTENCE ANALYSIS")
    print("==============================")
    if sentence_analysis["total_sentences"] > 0:
        print(f"- Total sentences           : {sentence_analysis['total_sentences']}")
        print(f"- Avg sentence length       : {sentence_analysis['avg_sentence_length']:.1f} words")
        print(f"- Longest sentence          : {sentence_analysis['longest_sentence_length']} words")
        print(f"- Long sentences (>{sentence_analysis['long_sentences']['threshold']} words) : {sentence_analysis['long_sentences']['count']} ({sentence_analysis['long_sentences']['percentage']:.1f}%)")
        if sentence_analysis['longest_sentence_preview']:
            print(f"\nLongest sentence preview:")
            print(f"  \"{sentence_analysis['longest_sentence_preview']}\"")
            
        raw_issues = sentence_analysis.get("raw_speech_issues")

        if raw_issues:
            print("\nRaw Speech Issues:")
            print(f"- Max sentence length       : {raw_issues['max_sentence_length']}")
            print(f"- Very long sentences (>40) : {raw_issues['very_long_sentences']}")
    else:
        print("No sentences detected in transcript.")

    print("\n==============================")
    print("PAUSE ANALYSIS")
    print("==============================")

    pause = analysis_results.get("pause_analysis")

    if pause:
        print(f"- Total pauses          : {pause['total_pauses']}")
        print(f"- Long pauses (>1.5s)   : {pause['long_pauses']}")
        print(f"- Avg pause duration    : {pause['avg_pause_duration']} sec")

    print("\n==============================")
    print("VOCABULARY & REPETITION")
    print("==============================")
    print(f"- Total words               : {vocabulary['total_words']}")
    print(f"- Unique words              : {vocabulary['unique_words']}")
    print(f"- Repetition ratio          : {vocabulary['repetition_ratio']:.2f}")
    print(f"  (Higher = more diverse vocabulary)")

    if vocabulary['repeated_trigrams'] or vocabulary['repeated_bigrams']:
        print("\nRepeated phrases:")
        
        if vocabulary['repeated_trigrams']:
            print("  3-word phrases:")
            for item in vocabulary['repeated_trigrams']:
                print(f"    \"{item['phrase']}\" ({item['count']})")
        
        if vocabulary['repeated_bigrams']:
            print("  2-word phrases:")
            for item in vocabulary['repeated_bigrams']:
                print(f"    \"{item['phrase']}\" ({item['count']})")
    else:
        print("\nNo repeated phrases detected.")

    print("\n==============================")
    print("FILLER ANALYSIS")
    print("==============================")
    print(f"Filler Words:")
    print(f"  - Count                   : {filler_analysis['filler_words']['count']}")
    print(f"  - Per 100 words           : {filler_analysis['filler_words']['per_100_words']:.1f}")
    
    print(f"\nFiller Phrases:")
    print(f"  - Count                   : {filler_analysis['filler_phrases']['count']}")
    print(f"  - Per 100 words           : {filler_analysis['filler_phrases']['per_100_words']:.1f}")
    
    if filler_analysis['filler_phrases']['details']:
        print(f"\n  Most common filler phrases:")
        sorted_phrases = sorted(
            filler_analysis['filler_phrases']['details'].items(),
            key=lambda x: x[1],
            reverse=True
        )
        for phrase, count in sorted_phrases[:5]:
            print(f"    \"{phrase}\" ({count})")
    
    print(f"\nTotal Fillers:")
    print(f"  - Count                   : {filler_analysis['total_fillers']['count']}")
    print(f"  - Per 100 words           : {filler_analysis['total_fillers']['per_100_words']:.1f}")

    print_confidence_drift(drift_data)

    print("\n==============================")
    print("FLUENCY METRICS")
    print("==============================")
    print(f"Audio duration      : {audio_info['duration_seconds']:.2f} sec")
    print(f"Total words         : {audio_info['total_words']}")
    print(f"Words per minute    : {audio_info['words_per_minute']:.2f}")

    print("\n==============================")
    print("SPEAKING SCORE")
    print("==============================")

    score = analysis_results.get("score")

    if score:
        print(f"- Fluency     : {score['fluency']}/10")
        print(f"- Confidence  : {score['confidence']}/10")
        print(f"- Clarity     : {score['clarity']}/10")
        print(f"- Overall     : {score['overall']}/10")

    print("\n==============================")
    print("FEEDBACK")
    print("==============================")

    fb = analysis_results.get("feedback", {})

    print("\nTop Issues:")
    for i, issue in enumerate(fb.get("issues", []), 1):
        print(f"{i}. ❌ {issue}")

    print("\nStrengths:")
    for s in fb.get("strengths", []):
        print(f"- ✅ {s}")

def save_json(analysis_results, base_path, audio_filename):
    """
    Save analysis results to JSON file.
    
    Args:
        analysis_results (dict): Complete analysis results
        base_path (str): Base directory path
        audio_filename (str): Original audio filename
    """
    json_path = os.path.join(
        base_path,
        f"analysis_{audio_filename.replace('.wav', '')}.json"
    )
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(analysis_results, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Analysis saved to: {json_path}")