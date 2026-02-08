import os
import re
from collections import Counter
import whisper
import librosa
import torch
import convertor

# ----------------------------
# CONFIG
# ----------------------------
BASE_PATH = "data/"
AUDIO_FILENAME = "audio2.wav"
AUDIO_PATH = os.path.join(BASE_PATH, AUDIO_FILENAME)
MODEL_SIZE = "small"

TOP_K = 5
LONG_SENTENCE_THRESHOLD = 25
TOP_PHRASES = 5

# ----------------------------
# WORD CATEGORIES
# ----------------------------
WORD_CATEGORIES = {
    "Self references": {
        "i", "my", "me", "myself", "mine", "we", "us", "our", "ours"
    },
    "Connectors": {
        "so", "and", "but", "because", "however", "therefore", "also", 
        "moreover", "furthermore", "yet", "though", "although", "or", "nor"
    },
    "Action verbs": {
        "do", "make", "go", "work", "take", "get", "give", "use", "try",
        "start", "stop", "create", "build", "run", "move", "speak", "talk"
    },
    "Emotion words": {
        "feel", "bad", "good", "nervous", "confident", "happy", "sad",
        "angry", "excited", "afraid", "worried", "anxious", "proud", "love", "hate"
    },
    "Planning words": {
        "goal", "want", "try", "plan", "will", "would", "should", "need",
        "hope", "wish", "aim", "intend", "strategy", "future"
    }
}

# Filler words for confidence tracking
FILLER_WORDS = {"so", "like", "um", "uh", "you know", "i mean", "kind of", "sort of"}

def transcribe_and_analyze():
    # ----------------------------
    # DEVICE SETUP
    # ----------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_fp16 = device == "cuda"

    print(f"Using device : {device}")
    if device == "cuda":
        print(f"GPU detected : {torch.cuda.get_device_name(0)}")

    # ----------------------------
    # LOAD MODEL (once)
    # ----------------------------
    print("Loading Whisper model...")
    model = whisper.load_model(MODEL_SIZE).to(device)

    # ----------------------------
    # TRANSCRIBE (FAST SETTINGS)
    # ----------------------------
    print("Transcribing audio...")
    print(f"Audio path: {AUDIO_PATH}")

    # Load audio as numpy array first
    audio_array, _ = librosa.load(AUDIO_PATH, sr=16000)

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
    with open(os.path.join(BASE_PATH, "transcript_"+ AUDIO_FILENAME.replace(".wav", "") +".txt"), "w", encoding="utf-8") as f:
        f.write(transcript)
    
    print_analysis(transcript, audio_duration_sec)

def print_analysis(transcript, audio_duration_sec):
    # ----------------------------
    # WORD COUNT (CLEAN)
    # ----------------------------
    # remove punctuation
    clean_words = re.findall(r"\b[a-zA-Z']+\b", transcript.lower())

    word_counts = Counter(clean_words)
    top_words = word_counts.most_common(TOP_K)

    # ----------------------------
    # WORD CATEGORY BREAKDOWN
    # ----------------------------
    category_counts = {category: 0 for category in WORD_CATEGORIES}

    for word in clean_words:
        for category, word_set in WORD_CATEGORIES.items():
            if word in word_set:
                category_counts[category] += 1

    # ----------------------------
    # SENTENCE LENGTH ANALYSIS
    # ----------------------------
    # Split transcript into sentences
    sentences = re.split(r'[.!?]+', transcript)
    # Filter out empty sentences
    sentences = [s.strip() for s in sentences if s.strip()]

    sentence_lengths = []
    longest_sentence = ""
    longest_length = 0

    for sentence in sentences:
        # Count words in this sentence
        words_in_sentence = re.findall(r"\b[a-zA-Z']+\b", sentence)
        length = len(words_in_sentence)
        sentence_lengths.append(length)
        
        # Track longest sentence
        if length > longest_length:
            longest_length = length
            longest_sentence = sentence

    # Calculate metrics
    num_sentences = len(sentence_lengths)
    avg_sentence_length = sum(sentence_lengths) / num_sentences if num_sentences > 0 else 0
    long_sentence_count = sum(1 for length in sentence_lengths if length > LONG_SENTENCE_THRESHOLD)
    long_sentence_percentage = (long_sentence_count / num_sentences * 100) if num_sentences > 0 else 0

    # ----------------------------
    # REPETITION INTENSITY ANALYSIS
    # ----------------------------
    total_words = len(clean_words)
    unique_words = len(set(clean_words))
    repetition_ratio = unique_words / total_words if total_words > 0 else 0

    # Generate n-grams (2-word and 3-word phrases)
    # Keep original casing for phrases by using transcript words
    transcript_words = re.findall(r"\b[a-zA-Z']+\b", transcript)

    # Bigrams (2-word phrases)
    bigrams = [" ".join(transcript_words[i:i+2]) for i in range(len(transcript_words) - 1)]
    bigram_counts = Counter(bigrams)
    # Filter out phrases that appear only once
    top_bigrams = [(phrase, count) for phrase, count in bigram_counts.most_common(TOP_PHRASES * 2) if count > 1][:TOP_PHRASES]

    # Trigrams (3-word phrases)
    trigrams = [" ".join(transcript_words[i:i+3]) for i in range(len(transcript_words) - 2)]
    trigram_counts = Counter(trigrams)
    # Filter out phrases that appear only once
    top_trigrams = [(phrase, count) for phrase, count in trigram_counts.most_common(TOP_PHRASES * 2) if count > 1][:TOP_PHRASES]


    # ----------------------------
    # FLUENCY METRICS (approx)
    # ----------------------------
    total_words = len(clean_words)
    wpm = (total_words / audio_duration_sec) * 60 if audio_duration_sec > 0 else 0

    # ----------------------------
    # OUTPUT
    # ----------------------------
    # print("\n==============================")
    # print("TRANSCRIPT")
    # print("==============================")
    # print(transcript)

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
            print(f"- {category:<20} : {count:3d} words ({percentage:5.1f}%)")
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

    print("\n==============================")
    print("FLUENCY METRICS")
    print("==============================")
    print(f"Audio duration      : {audio_duration_sec:.2f} sec")
    print(f"Total words         : {total_words}")
    print(f"Words per minute    : {wpm:.2f}")


if __name__ == "__main__":
    # ----------------------------
    # ENSURE AUDIO EXISTS
    # ----------------------------
    if not os.path.exists(AUDIO_PATH):
        print("Audio not found, converting from video...")
        AUDIO_PATH = convertor.convert(AUDIO_PATH)

    # ----------------------------
    # LOAD AUDIO (for duration only)
    # ----------------------------
    y, sr = librosa.load(AUDIO_PATH, sr=None)
    audio_duration_sec = librosa.get_duration(y=y, sr=sr)

    if os.path.exists(os.path.join(BASE_PATH, "transcript_"+ AUDIO_FILENAME.replace(".wav", "") +".txt")):
        print("Transcript already exists, skipping transcription.")
        with open(os.path.join(BASE_PATH, "transcript_"+ AUDIO_FILENAME.replace(".wav", "") +".txt"), "r", encoding="utf-8") as f:
            transcript = f.read()
            print_analysis(transcript, audio_duration_sec)
    else:
        print("Transcript not found, starting transcription and analysis...")
        transcribe_and_analyze()