def clamp(score):
    return max(0, min(10, round(score, 2)))


def fluency_score(wpm):
    # Ideal range: 120–150
    if wpm < 100:
        return clamp(4 + (wpm / 100) * 3)
    elif 100 <= wpm <= 150:
        return clamp(7 + (wpm - 100) / 50 * 3)
    else:
        return clamp(10 - (wpm - 150) / 50 * 3)


def confidence_score(filler_per_100, weak_word_pct):
    score = 10

    score -= filler_per_100 * 0.3   # heavy penalty
    score -= weak_word_pct * 0.2

    return clamp(score)


def clarity_score(repetition_ratio, avg_sentence_length):
    score = 5

    # vocabulary diversity
    score += repetition_ratio * 5

    # sentence balance
    if 12 <= avg_sentence_length <= 20:
        score += 2
    else:
        score -= 1

    return clamp(score)


def overall_score(fluency, confidence, clarity):
    return clamp((fluency + confidence + clarity) / 3)