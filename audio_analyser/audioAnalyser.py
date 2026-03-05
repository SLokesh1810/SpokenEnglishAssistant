import os
import re
import json
from collections import Counter
import whisper
import librosa
import torch
import audio_analyser.convertor as convertor
import gc

# ----------------------------
# WORD CATEGORIES (Expanded)
# ----------------------------
WORD_CATEGORIES = {

    "Self references": {
        "i", "me", "my", "mine", "myself",
        "we", "us", "our", "ours",
        "personally", "myself"
    },

    "Connectors (basic)": {
        "and", "so", "but", "or", "nor",
        "because", "though", "although",
        "yet", "also"
    },

    "Connectors (advanced)": {
        "however", "therefore", "moreover",
        "furthermore", "consequently",
        "meanwhile", "nevertheless",
        "additionally", "hence",
        "thus", "overall"
    },

    "Action verbs": {
        "do", "does", "did",
        "make", "made",
        "go", "went",
        "work", "worked",
        "take", "took",
        "get", "got",
        "give", "gave",
        "use", "used",
        "try", "tried",
        "start", "started",
        "stop", "stopped",
        "create", "created",
        "build", "built",
        "run", "ran",
        "move", "moved",
        "speak", "talk",
        "learn", "improve",
        "develop", "practice",
        "achieve", "complete",
        "solve", "handle",
        "manage", "lead"
    },

    "Emotion words": {
        "feel", "felt",
        "bad", "good",
        "nervous", "confident",
        "happy", "sad",
        "angry", "excited",
        "afraid", "worried",
        "anxious", "proud",
        "love", "hate",
        "motivated", "demotivated",
        "stressed", "overwhelmed",
        "frustrated", "embarrassed",
        "inferior"
    },

    "Planning words": {
        "goal", "goals",
        "want", "wanted",
        "plan", "planned",
        "will", "would",
        "should", "need",
        "hope", "wish",
        "aim", "intend",
        "strategy", "future",
        "target", "vision",
        "prepare", "improve",
        "next", "step",
        "focus", "objective"
    },

    "Weak language (confidence killers)": {
        "maybe", "probably",
        "actually", "basically",
        "kind", "sort",
        "almost", "somewhat",
        "just", "like",
        "i think", "i guess",
        "i feel", "i believe",
        "i mean"
    }
}

FILLER_WORDS = {
    "um", "uh", "ah", "er",
    "so", "like", "yeah",
    "well", "okay", "right",
    "basically", "actually",
    "literally", "honestly",
    "just"
}

FILLER_PHRASES = {
    "you know",
    "i mean",
    "kind of",
    "sort of",
    "i guess",
    "i think",
    "what to say",
    "how to say",
    "at the end of the day",
    "to be honest",
    "to be frank"
}

# Whisper model cache to avoid reloading models multiple times
_MODEL_CACHE = {}

# Regex Patterns
WORD_PATTERN = re.compile(r"\b[a-zA-Z']+\b")
SENTENCE_PATTERN = re.compile(r'[.!?]+')

# -----------------------------
# MODEL LOADING
# -----------------------------
def get_or_load_model(model_size, device):
    """Load model once and cache it."""
    cache_key = (model_size, device)
    if cache_key not in _MODEL_CACHE:
        print(f"Loading Whisper model ({model_size})...")
        _MODEL_CACHE[cache_key] = whisper.load_model(model_size).to(device)
    return _MODEL_CACHE[cache_key]

# ----------------------------
# FILLER DETECTION FUNCTIONS
# ----------------------------
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
        # Count occurrences of this phrase
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
    
    # Count filler words
    filler_word_count = count_filler_words(words)
    
    # Count filler phrases
    filler_phrase_count, phrase_details = count_filler_phrases(transcript)
    
    # Total fillers (words + phrases)
    total_filler_instances = filler_word_count + filler_phrase_count
    
    # Calculate percentages
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

# ----------------------------
# CONFIDENCE DRIFT FUNCTIONS
# ----------------------------
def analyze_text_segment(segment_text, segment_duration):
    """
    Analyze a text segment and return speaking metrics.
    
    Args:
        segment_text (str): The text to analyze
        segment_duration (float): Duration in seconds
    
    Returns:
        dict: Contains word_count, wpm, avg_sentence_length, filler analysis
    """
    # Extract words
    words = re.findall(r"\b[a-zA-Z']+\b", segment_text.lower())
    word_count = len(words)
    
    # Calculate WPM
    wpm = (word_count / segment_duration) * 60 if segment_duration > 0 else 0
    
    # Split into sentences
    sents = re.split(r'[.!?]+', segment_text)
    sents = [s.strip() for s in sents if s.strip()]
    
    # Calculate avg sentence length
    sent_lengths = []
    for sent in sents:
        sent_words = re.findall(r"\b[a-zA-Z']+\b", sent)
        sent_lengths.append(len(sent_words))
    
    avg_sent_len = sum(sent_lengths) / len(sent_lengths) if sent_lengths else 0
    
    # Comprehensive filler analysis
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
    
    segment_1 = transcript_text[:first_end]
    segment_2 = transcript_text[first_end:second_end]
    segment_3 = transcript_text[second_end:]
    
    return segment_1, segment_2, segment_3

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
    
    dur_1 = (words_1 / total_words) * total_duration if total_words > 0 else 0
    dur_2 = (words_2 / total_words) * total_duration if total_words > 0 else 0
    dur_3 = (words_3 / total_words) * total_duration if total_words > 0 else 0
    
    return dur_1, dur_2, dur_3

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
    
    # Split transcript
    seg_1_text, seg_2_text, seg_3_text = split_transcript_into_thirds(transcript)
    
    # Calculate durations
    dur_1, dur_2, dur_3 = calculate_segment_durations(
        seg_1_text, seg_2_text, seg_3_text, audio_duration, total_words
    )
    
    # Analyze each segment
    segment_1 = analyze_text_segment(seg_1_text, dur_1)
    segment_2 = analyze_text_segment(seg_2_text, dur_2)
    segment_3 = analyze_text_segment(seg_3_text, dur_3)
    
    # Calculate trends
    wpm_trend = "increasing" if segment_3['wpm'] > segment_1['wpm'] else "decreasing" if segment_3['wpm'] < segment_1['wpm'] else "stable"
    sent_trend = "shorter" if segment_3['avg_sentence_length'] < segment_1['avg_sentence_length'] else "longer" if segment_3['avg_sentence_length'] > segment_1['avg_sentence_length'] else "stable"
    filler_trend = "decreasing" if segment_3['fillers']['total_fillers']['per_100_words'] < segment_1['fillers']['total_fillers']['per_100_words'] else "increasing" if segment_3['fillers']['total_fillers']['per_100_words'] > segment_1['fillers']['total_fillers']['per_100_words'] else "stable"
    
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

def print_confidence_drift(drift_data):
    """
    Print the session progression analysis.
    
    Args:
        drift_data (dict or None): Results from analyze_confidence_drift()
    """
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
    print("\nTrends:")
    print(f"  - Speaking speed          : ↑ {trends['speaking_speed']}" if trends['speaking_speed'] == "increasing" else f"  - Speaking speed          : ↓ {trends['speaking_speed']}" if trends['speaking_speed'] == "decreasing" else f"  - Speaking speed          : → {trends['speaking_speed']}")
    print(f"  - Sentence length         : ↓ {trends['sentence_length']}" if trends['sentence_length'] == "shorter" else f"  - Sentence length         : ↑ {trends['sentence_length']}" if trends['sentence_length'] == "longer" else f"  - Sentence length         : → {trends['sentence_length']}")
    print(f"  - Filler word usage       : ↓ {trends['filler_usage']}" if trends['filler_usage'] == "decreasing" else f"  - Filler word usage       : ↑ {trends['filler_usage']}" if trends['filler_usage'] == "increasing" else f"  - Filler word usage       : → {trends['filler_usage']}")

# ----------------------------
# ANALYSIS FUNCTIONS
# ----------------------------
def transcribe_and_analyze(audio_array, AUDIO_PATH, BASE_PATH, AUDIO_FILENAME, audio_duration_sec, MODEL_SIZE, TOP_K, LONG_SENTENCE_THRESHOLD, TOP_PHRASES, return_json=False, save_json=False):
    """Transcribe audio and perform analysis."""
    # Device setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_fp16 = device == "cuda"

    print(f"Using device : {device}")
    if device == "cuda":
        print(f"GPU detected : {torch.cuda.get_device_name(0)}")

    # Load model
    model = get_or_load_model(MODEL_SIZE, device)

    # Transcribe
    print("Transcribing audio...")
    print(f"Audio path: {AUDIO_PATH}")

    result = model.transcribe(
        audio_array,
        language='en',
        verbose=False,
        fp16=use_fp16,
        word_timestamps=False,          
        beam_size=1,
        temperature=0.0,
        condition_on_previous_text=False
    )

    transcript = result["text"].strip()
    
    # Save transcript
    transcript_path = os.path.join(BASE_PATH, "transcript_" + AUDIO_FILENAME.replace(".wav", "") + ".txt")
    with open(transcript_path, "w", encoding="utf-8") as f:
        f.write(transcript)
    
    return analyse_n_print(transcript, audio_duration_sec, TOP_K, LONG_SENTENCE_THRESHOLD, TOP_PHRASES, return_json, save_json, BASE_PATH, AUDIO_FILENAME)

def cleaner(text):
    """Remove punctuation and return list of words."""
    return WORD_PATTERN.findall(text.lower())

def categorize_word(words):
    """Count words by category."""
    category_counts = {
        category: sum(1 for word in words if word in word_set)
        for category, word_set in WORD_CATEGORIES.items()
    }
    return category_counts

def analyse_sentences(transcript, LONG_SENTENCE_THRESHOLD):
    """Analyze sentence structure and length."""
    sentences = SENTENCE_PATTERN.split(transcript)
    sentences = [s.strip() for s in sentences if s.strip()]

    sentence_lengths = []
    longest_sentence = ""
    longest_length = 0

    for sentence in sentences:
        words_in_sentence = re.findall(r"\b[a-zA-Z']+\b", sentence)
        length = len(words_in_sentence)
        sentence_lengths.append(length)
        
        if length > longest_length:
            longest_length = length
            longest_sentence = sentence

    num_sentences = len(sentence_lengths)
    avg_sentence_length = sum(sentence_lengths) / num_sentences if num_sentences > 0 else 0
    long_sentence_count = sum(1 for length in sentence_lengths if length > LONG_SENTENCE_THRESHOLD)
    long_sentence_percentage = (long_sentence_count / num_sentences * 100) if num_sentences > 0 else 0

    return num_sentences, avg_sentence_length, longest_sentence, longest_length, long_sentence_count, long_sentence_percentage

def repetition_analysis(words, transcript, TOP_PHRASES):
    """Analyze vocabulary repetition and common phrases."""
    total_words = len(words)
    unique_words = len(set(words))
    repetition_ratio = unique_words / total_words if total_words > 0 else 0

    transcript_words = re.findall(r"\b[a-zA-Z']+\b", transcript)
    
    # Bigrams
    bigrams = [f"{w1} {w2}" for w1, w2 in zip(transcript_words, transcript_words[1:])]
    bigram_counts = Counter(bigrams)
    top_bigrams = [(phrase, count) for phrase, count in bigram_counts.most_common(TOP_PHRASES * 2) if count > 1][:TOP_PHRASES]

    # Trigrams  
    trigrams = [f"{w1} {w2} {w3}" for w1, w2, w3 in zip(transcript_words, transcript_words[1:], transcript_words[2:])]
    trigram_counts = Counter(trigrams)
    top_trigrams = [(phrase, count) for phrase, count in trigram_counts.most_common(TOP_PHRASES * 2) if count > 1][:TOP_PHRASES]

    return unique_words, repetition_ratio, top_bigrams, top_trigrams
    
def fluency_analysis(words, audio_duration_sec):
    """Calculate fluency metrics."""
    total_words = len(words)
    wpm = (total_words / audio_duration_sec) * 60 if audio_duration_sec > 0 else 0
    return total_words, wpm

def analyse_n_print(transcript, audio_duration_sec, TOP_K, LONG_SENTENCE_THRESHOLD, TOP_PHRASES, return_json=False, save_json=False, BASE_PATH=None, AUDIO_FILENAME=None):
    """Main analysis and printing function."""
    # Step 1: Clean words
    clean_words = cleaner(transcript)

    # Step 2: Top K words
    word_counts = Counter(clean_words)
    top_words = word_counts.most_common(TOP_K)

    # Step 3: Categorize words
    category_counts = categorize_word(clean_words)

    # Step 4: Sentence analysis
    num_sentences, avg_sentence_length, longest_sentence, longest_length, long_sentence_count, long_sentence_percentage = analyse_sentences(transcript, LONG_SENTENCE_THRESHOLD)

    # Step 5: Repetition analysis
    unique_words, repetition_ratio, top_bigrams, top_trigrams = repetition_analysis(clean_words, transcript, TOP_PHRASES)

    # Step 6: Fluency metrics
    total_words, wpm = fluency_analysis(clean_words, audio_duration_sec)

    # Step 7: Filler analysis
    filler_analysis_results = analyze_filler_usage(clean_words, transcript)

    # Step 8: Confidence drift analysis
    drift_data = analyze_confidence_drift(transcript, total_words, audio_duration_sec)

    # ----------------------------
    # BUILD JSON OUTPUT
    # ----------------------------
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
                "threshold": LONG_SENTENCE_THRESHOLD
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
        "filler_analysis": filler_analysis_results,
        "confidence_drift": drift_data
    }

    # ----------------------------
    # SAVE JSON IF REQUESTED
    # ----------------------------
    if save_json and BASE_PATH and AUDIO_FILENAME:
        json_path = os.path.join(BASE_PATH, "analysis_" + AUDIO_FILENAME.replace(".wav", "") + ".json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(analysis_results, f, indent=2, ensure_ascii=False)
        print(f"\n✓ Analysis saved to: {json_path}")

    # ----------------------------
    # PRINT OUTPUT (if not return_json only)
    # ----------------------------
    if not return_json:
        print("\n==============================")
        print("TOP USED WORDS")
        print("==============================")
        for word, count in top_words:
            print(f"{word:>10} : {count}")

        print("\n==============================")
        print("WORD CATEGORY BREAKDOWN")
        print("==============================")
        if total_words > 0:
            for category, count in category_counts.items():
                percentage = (count / total_words) * 100
                print(f"- {category:<35} : {count:3d} words ({percentage:5.1f}%)")
        else:
            print("No words detected in transcript.")

        print("\n==============================")
        print("SENTENCE ANALYSIS")
        print("==============================")
        if num_sentences > 0:
            print(f"- Total sentences           : {num_sentences}")
            print(f"- Avg sentence length       : {avg_sentence_length:.1f} words")
            print(f"- Longest sentence          : {longest_length} words")
            print(f"- Long sentences (>{LONG_SENTENCE_THRESHOLD} words) : {long_sentence_count} ({long_sentence_percentage:.1f}%)")
            if longest_sentence:
                print(f"\nLongest sentence preview:")
                print(f"  \"{longest_sentence[:150]}{'...' if len(longest_sentence) > 150 else ''}\"")
        else:
            print("No sentences detected in transcript.")

        print("\n==============================")
        print("VOCABULARY & REPETITION")
        print("==============================")
        print(f"- Total words               : {total_words}")
        print(f"- Unique words              : {unique_words}")
        print(f"- Repetition ratio          : {repetition_ratio:.2f}")
        print(f"  (Higher = more diverse vocabulary)")

        if top_bigrams or top_trigrams:
            print("\nRepeated phrases:")
            
            if top_trigrams:
                print("  3-word phrases:")
                for phrase, count in top_trigrams:
                    print(f"    \"{phrase}\" ({count})")
            
            if top_bigrams:
                print("  2-word phrases:")
                for phrase, count in top_bigrams:
                    print(f"    \"{phrase}\" ({count})")
        else:
            print("\nNo repeated phrases detected.")

        # Print filler analysis
        print("\n==============================")
        print("FILLER ANALYSIS")
        print("==============================")
        print(f"Filler Words:")
        print(f"  - Count                   : {filler_analysis_results['filler_words']['count']}")
        print(f"  - Per 100 words           : {filler_analysis_results['filler_words']['per_100_words']:.1f}")
        
        print(f"\nFiller Phrases:")
        print(f"  - Count                   : {filler_analysis_results['filler_phrases']['count']}")
        print(f"  - Per 100 words           : {filler_analysis_results['filler_phrases']['per_100_words']:.1f}")
        
        if filler_analysis_results['filler_phrases']['details']:
            print(f"\n  Most common filler phrases:")
            # Sort by count descending
            sorted_phrases = sorted(
                filler_analysis_results['filler_phrases']['details'].items(),
                key=lambda x: x[1],
                reverse=True
            )
            for phrase, count in sorted_phrases[:5]:  # Show top 5
                print(f"    \"{phrase}\" ({count})")
        
        print(f"\nTotal Fillers:")
        print(f"  - Count                   : {filler_analysis_results['total_fillers']['count']}")
        print(f"  - Per 100 words           : {filler_analysis_results['total_fillers']['per_100_words']:.1f}")

        # Print confidence drift
        print_confidence_drift(drift_data)

        print("\n==============================")
        print("FLUENCY METRICS")
        print("==============================")
        print(f"Audio duration      : {audio_duration_sec:.2f} sec")
        print(f"Total words         : {total_words}")
        print(f"Words per minute    : {wpm:.2f}")

    # Return JSON if requested
    if return_json:
        return analysis_results

def main(BASE_PATH, AUDIO_FILENAME, MODEL_SIZE="small", TOP_K=5, LONG_SENTENCE_THRESHOLD=25, TOP_PHRASES=5, return_json=False, save_json=False):
    """Main entry point for audio analysis."""
    AUDIO_PATH = os.path.join(BASE_PATH, AUDIO_FILENAME)
    
    # Ensure audio exists
    if not os.path.exists(AUDIO_PATH):
        print("Audio not found, converting from video...")
        AUDIO_PATH = convertor.convert(AUDIO_PATH)

    # Load audio for duration
    audio_array, sr = librosa.load(AUDIO_PATH, sr=16000)
    audio_duration_sec = librosa.get_duration(y=audio_array, sr=sr)

    # Check if transcript exists
    transcript_path = os.path.join(BASE_PATH, "transcript_" + AUDIO_FILENAME.replace(".wav", "") + ".txt")
    if os.path.exists(transcript_path):
        print("Transcript already exists, skipping transcription.")
        with open(transcript_path, "r", encoding="utf-8") as f:
            transcript = f.read()
        result = analyse_n_print(transcript, audio_duration_sec, TOP_K, LONG_SENTENCE_THRESHOLD, TOP_PHRASES, return_json, save_json, BASE_PATH, AUDIO_FILENAME)
    else:
        print("Transcript not found, starting transcription and analysis...")
        result = transcribe_and_analyze(audio_array, AUDIO_PATH, BASE_PATH, AUDIO_FILENAME, audio_duration_sec, MODEL_SIZE, TOP_K, LONG_SENTENCE_THRESHOLD, TOP_PHRASES, return_json, save_json)

    # Cleanup
    del audio_array
    gc.collect()
    
    return result if return_json else None