"""Equation detection. PyMuPDF returns formulas as plain text, so a text block counts as an equation
when it is mostly symbols and digits with few words: "E = mc^2", "x_i = 5", "2 + 2 = 4".
Prose such as "Refund within 30 days" does not match. Detected equations are rendered as images, and
the vision model transcribes them to LaTeX when it is enabled."""

import re

MATH_CHARS = set("=+−-×÷±<>≤≥≈≠∑∫∏√∂∇∞^_{}()/*|αβγδεθλμπσφω")


def looks_like_equation(text: str) -> bool:
    t = text.strip()
    if not t or len(t) > 300:
        return False
    compact = re.sub(r"\s+", "", t)
    math = sum(c in MATH_CHARS for c in compact)
    if math == 0:
        return False
    digits = sum(c.isdigit() for c in compact)
    words = re.findall(r"[A-Za-z]{3,}", t)
    has_operator = any(c in compact for c in "=+−-×÷±<>≤≥≈≠∑∫∏√^_*/")
    return len(words) <= 3 and has_operator and (math + digits) / len(compact) >= 0.25
