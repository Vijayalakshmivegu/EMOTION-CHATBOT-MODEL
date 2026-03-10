CRISIS_KEYWORDS = [
    "suicide",
    "kill myself",
    "end my life",
    "self harm",
    "cut myself",
    "no reason to live"
]

def detect_crisis(text):
    text = text.lower()
    for keyword in CRISIS_KEYWORDS:
        if keyword in text:
            return True
    return False

def crisis_response():
    return (
        "I'm really sorry you're feeling this way. "
        "You are not alone and help is available. "
        "Please consider reaching out to a trusted person "
        "or a mental health professional."
    )