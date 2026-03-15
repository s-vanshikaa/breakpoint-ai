from guardrails.patterns import find_adversarial_pattern

REFUSAL_MESSAGE = (
    "I can't follow instructions that ask me to override my policies or "
    "reveal protected information."
)


def check_input(prompt: str) -> str | None:
    matched_pattern = find_adversarial_pattern(prompt)
    if matched_pattern:
        return f"Input guardrail: matched adversarial pattern '{matched_pattern}'."
    return None
