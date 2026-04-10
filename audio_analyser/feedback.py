def generate_feedback(results):
    feedback = []

    wpm = results["audio_info"]["words_per_minute"]
    filler = results["filler_analysis"]["total_fillers"]["per_100_words"]
    repetition = results["vocabulary"]["repetition_ratio"]
    avg_len = results["sentence_analysis"]["avg_sentence_length"]
    raw = results["sentence_analysis"]["raw_speech_issues"]

    # 🔹 Fluency
    if wpm < 110:
        feedback.append("⚠️ You are speaking too slowly. Try to increase your pace.")
    elif wpm > 160:
        feedback.append("⚠️ You are speaking too fast. Slow down for better clarity.")
    else:
        feedback.append("✅ Your speaking pace is good.")

    # 🔹 Fillers
    if filler > 7:
        feedback.append("❌ You use too many filler words. Try pausing instead of saying 'um', 'like'.")
    elif filler > 4:
        feedback.append("⚠️ Moderate filler usage. Try to reduce it.")
    else:
        feedback.append("✅ Good control over filler words.")

    # 🔹 Vocabulary
    if repetition < 0.35:
        feedback.append("❌ Low vocabulary diversity. Try using more varied words.")
    elif repetition < 0.5:
        feedback.append("⚠️ Moderate vocabulary usage. Can improve.")
    else:
        feedback.append("✅ Good vocabulary diversity.")

    # 🔹 Sentence flow
    if avg_len > 25:
        feedback.append("⚠️ Your sentences are too long. Break them into smaller ones.")
    elif avg_len < 10:
        feedback.append("⚠️ Your sentences are too short. Try forming complete thoughts.")
    else:
        feedback.append("✅ Good sentence structure.")

    # 🔹 Rambling detection
    if raw["max_sentence_length"] > 80:
        feedback.append("❌ You tend to ramble without pauses. Try structuring your thoughts better.")
    elif raw["max_sentence_length"] > 40:
        feedback.append("⚠️ Some long unbroken speech detected.")
    
    return feedback