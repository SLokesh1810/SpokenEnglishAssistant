"""
Configuration constants for audio analysis.
"""

import re

# ----------------------------
# WORD CATEGORIES
# ----------------------------
WORD_CATEGORIES = {
    "Self references": {
        "i", "me", "my", "mine", "myself",
        "we", "us", "our", "ours",
        "personally"
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

# ----------------------------
# FILLER WORDS AND PHRASES
# ----------------------------
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

# ----------------------------
# REGEX PATTERNS
# ----------------------------
WORD_PATTERN = re.compile(r"\b[a-zA-Z']+\b")
SENTENCE_PATTERN = re.compile(r'[.!?]+')

# ----------------------------
# DEFAULT PARAMETERS
# ----------------------------
DEFAULT_MODEL_SIZE = "small"
DEFAULT_TOP_K = 5
DEFAULT_LONG_SENTENCE_THRESHOLD = 25
DEFAULT_TOP_PHRASES = 5