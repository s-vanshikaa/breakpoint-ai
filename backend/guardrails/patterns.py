import re

ADVERSARIAL_PATTERNS = [
    r"\bignore\b.{0,40}\b(instructions?|policy|policies|rules?)\b",
    r"\bdisregard\b.{0,40}\b(instructions?|policy|policies|rules?)\b",
    r"\bforget\b.{0,40}\b(instructions?|rules?)\b",
    r"\bno longer (bound|restricted)\b",
    r"\bnew instructions supersede\b",
    r"\bsystem:?\s*you are now\b",
    r"\bdebug mode\b",
    r"\boverride:?\b",
    r"\bauthorized override\b",
    r"\bas the (system )?administrator\b",
    r"\bskip (the )?approval\b",
    r"\bno approval needed\b",
]

COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in ADVERSARIAL_PATTERNS]


def find_adversarial_pattern(text: str) -> str | None:
    for pattern in COMPILED_PATTERNS:
        if pattern.search(text):
            return pattern.pattern
    return None
