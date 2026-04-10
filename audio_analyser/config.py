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
    },
    "Discourse markers": {
        "so", "well", "okay", "right",
        "basically", "actually",
        "you know", "i mean",
        "like", "anyway",
        "by the way"
    },
    "Weak language (confidence killers)": {
        "maybe", "probably",
        "actually", "basically",
        "kind", "sort", "kinda", "sorta",
        "almost", "somewhat",
        "just", "like",
        "i think", "i guess",
        "i feel", "i believe",
        "i mean",
        "i'm not sure",
        "i don't know",
        "i guess so",
        "i think so"
    },
    "Professional language": {
        "specifically", "particularly",
        "for example", "for instance",
        "in conclusion", "overall",
        "in summary", "primarily",
        "effectively", "efficiently",
        "significantly", "notably"
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
    "just",
    "hmm", "huh"
}

FILLER_PHRASES = {
    "you know",
    "i mean",
    "kind of",
    "sort of",
    "i guess",
    "i think",
    "i feel like",
    "i don't know",
    "you see",
    "let me think",
    "how do i say",
    "what i mean is",
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