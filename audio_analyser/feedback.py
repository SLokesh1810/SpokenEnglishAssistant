def generate_feedback(results):
    issues = []
    strengths = []

    wpm = results["audio_info"]["words_per_minute"]
    filler = results["filler_analysis"]["total_fillers"]["per_100_words"]
    repetition = results["vocabulary"]["repetition_ratio"]
    avg_len = results["sentence_analysis"]["avg_sentence_length"]
    raw = results["sentence_analysis"]["raw_speech_issues"]

    # -------------------------
    # FLUENCY
    # -------------------------
    if wpm < 110:
        issues.append(("Low speaking speed", 2, "Try increasing your pace."))
    elif wpm > 160:
        issues.append(("Too fast speaking", 2, "Slow down for clarity."))
    else:
        strengths.append("Good speaking pace")

    # -------------------------
    # FILLERS
    # -------------------------
    if filler > 8:
        issues.append(("High filler usage", 3, "Pause instead of using filler words."))
    elif filler > 5:
        issues.append(("Moderate filler usage", 2, "Reduce fillers for better clarity."))
    else:
        strengths.append("Good control over filler words")

    # -------------------------
    # VOCABULARY
    # -------------------------
    if repetition < 0.3:
        issues.append(("Low vocabulary diversity", 3, "Use more varied words."))
    elif repetition < 0.5:
        issues.append(("Moderate vocabulary diversity", 2, "Try improving word variety."))
    else:
        strengths.append("Good vocabulary usage")

    # -------------------------
    # SENTENCE STRUCTURE
    # -------------------------
    if avg_len > 25:
        issues.append(("Very long sentences", 2, "Break sentences into smaller parts."))
    elif avg_len < 10:
        issues.append(("Very short sentences", 1, "Try forming complete thoughts."))
    else:
        strengths.append("Good sentence structure")

    # -------------------------
    # RAMBLING
    # -------------------------
    if raw["max_sentence_length"] > 100:
        issues.append(("Excessive rambling", 3, "You speak too long without pauses."))
    elif raw["max_sentence_length"] > 50:
        issues.append(("Some rambling detected", 2, "Try structuring your thoughts."))

    # -------------------------
    # SORT ISSUES BY SEVERITY
    # -------------------------
    issues.sort(key=lambda x: x[1], reverse=True)

    # Format nicely
    formatted_issues = [
        f"{title} ({'CRITICAL' if sev==3 else 'HIGH' if sev==2 else 'LOW'}) - {msg}"
        for title, sev, msg in issues
    ]

    return {
        "issues": formatted_issues[:3],   # top 3
        "strengths": strengths
    }