"""Map free-form model outputs to answer choices and score them."""

import re

ARTICLES = {"a", "an", "the"}
NUMBER_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6",
    "seven": "7", "eight": "8", "nine": "9", "ten": "10", "eleven": "11", "twelve": "12",
}
ANSWER_RE = re.compile(r"answer\s*(?:is)?\s*[:：]\s*", re.I)
LETTER_RE = re.compile(r"^\(?([A-Z])\)?\s*(?:[.):]|$)")


def normalize(s: str) -> str:
    s = s.lower().replace("_", " ")
    s = re.sub(r"[^\w\s+&-]", " ", s)
    toks = [NUMBER_WORDS.get(t, t) for t in s.split() if t not in ARTICLES]
    return " ".join(toks)


def extract_final(output: str) -> str:
    """Text after the last 'Answer:' marker; otherwise the last non-empty line."""
    t = (output or "").strip()
    marks = list(ANSWER_RE.finditer(t))
    if marks:
        t = t[marks[-1].end():]
    else:
        lines = [ln for ln in t.splitlines() if ln.strip()]
        t = lines[-1] if lines else ""
    t = t.strip().strip("*`\"' ").strip()
    return t.splitlines()[0].strip() if t else ""


def _word_search(needle: str, hay: str):
    return re.search(rf"(?<![\w-]){re.escape(needle)}(?![\w-])", hay)


def match_choice(output: str, choices: list, letters: bool) -> tuple:
    """Return (choice index or None, rule used)."""
    t = extract_final(output)
    if not t:
        return None, "empty"
    if letters:
        m = LETTER_RE.match(t)
        if m and 0 <= ord(m.group(1)) - 65 < len(choices):
            return ord(m.group(1)) - 65, "letter"
    nt = normalize(t)
    nc = [normalize(c) for c in choices]
    for i, c in enumerate(nc):
        if nt == c:
            return i, "exact"
    hits = [i for i, c in enumerate(nc) if c and _word_search(c, nt)]
    if len(hits) == 1:
        return hits[0], "contains"
    if len(hits) > 1:
        longest = max(hits, key=lambda i: len(nc[i]))
        if all(_word_search(nc[i], nc[longest]) for i in hits):
            return longest, "contains_longest"
        first = min(hits, key=lambda i: _word_search(nc[i], nt).start())
        return first, "contains_first"
    inside = [i for i, c in enumerate(nc) if nt and _word_search(nt, c)]
    if len(inside) == 1:
        return inside[0], "partial"
    return None, "nomatch"


def chance_level(choices: list) -> float:
    return 1.0 / max(1, len(choices))
