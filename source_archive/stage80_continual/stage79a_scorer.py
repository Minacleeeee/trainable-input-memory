"""
Stage 79A strict response scorer (v2).

Replaces the Stage 77 substring scorer. The old scorer did ``expected in resp``
which produced substring false positives and missed negation / multi-answer /
hedge cases. This version tokenizes and applies explicit rejection rules.

Design policy (documented, pinned by test_scorer.py):

* Matching is whole-token / whole-phrase, never raw substring. The expected
  answer's token sequence must appear contiguously in the response tokens.
* Case-insensitive, whitespace-normalized.
* Negation scopes over the expected answer only when a negation marker appears
  in the *same clause* (comma/semicolon/period-delimited) and directly before it
  (skipping only auxiliaries/determiners); an intervening content word means the
  negation modifies that other word, not the expected answer. This is the
  documented false-positive control: "it is not red, it is mauve" ACCEPTS mauve,
  while "it is not mauve" REJECTS mauve.
* "X, not Y" / "X not Y" is rejected: a negation marker within a few tokens
  immediately *after* the expected answer contradicts it.
* Multi-answer is rejected on sight: "or", a slash "/", and the presence of any
  *other known target answer* token sequence in the response.
* Hedged / non-committal responses ("maybe", "not sure", ...) are rejected.
* Overly long responses (more than MAX_TOKENS) are rejected: these short
  factual / S-R tasks expect 1-4 word answers.

The scorer returns (correct: bool, reason: str) and is fully deterministic.
"""
import re

# Negation markers that can scope over a following answer.
NEGATIONS = {
    "not", "n't", "never", "no", "cannot", "can't", "don't", "doesn't",
    "isn't", "won't", "ain't", "neither", "nor", "wasn't", "weren't",
    "haven't", "hasn't", "shouldn't", "wouldn't", "couldn't",
}

# Hedged / non-committal markers.
HEDGES = {
    "maybe", "perhaps", "possibly", "unsure", "unknown", "unclear",
    "guess", "dunno", "sure", "cant", "cannot", "uncertain",
}

# Clause delimiters (tokenized as their own tokens).
CLAUSE_BREAKS = {",", ";", ".", ":", "!", "?", "—", "-"}

# Tokens skipped when scanning backward for the negation that scopes over the
# expected answer (auxiliaries, determiners, light verbs).
SKIP_BACK = {
    "is", "are", "was", "were", "it", "the", "a", "an", "of", "to", "be",
    "this", "that", "its", "answer", "response", "they", "he", "she",
    "will", "would", "should", "do", "does", "did", "can", "could",
}

# These short tasks expect 1-4 word answers; anything longer is non-committal
# rambling / degenerate generation.
MAX_TOKENS = 14


def _tokenize(text):
    """Lowercase, normalize whitespace, split into word/punctuation tokens.

    Apostrophes inside contractions (don't) stay attached; punctuation marks
    become standalone tokens.
    """
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    tokens = re.findall(r"[a-z0-9']+|[^\sa-z0-9']", text)
    return tokens


def _find_subsequence(haystack, needle):
    """Return start index of contiguous ``needle`` in ``haystack``, else -1."""
    n = len(needle)
    if n == 0:
        return -1
    for i in range(len(haystack) - n + 1):
        if haystack[i:i + n] == needle:
            return i
    return -1


def strict_score(response, expected, known_answers=None):
    """Score a generated response against an expected short answer.

    Parameters
    ----------
    response : str
        Greedy decoded generation (already lowercased/whitespace-normalized by
        the caller; we re-normalize anyway).
    expected : str
        The gold target answer.
    known_answers : iterable of str, optional
        All *other* target answers in the dataset. If the response contains any
        of these as a whole-token phrase, it is a multi-answer / cross-talk
        response and is rejected.

    Returns
    -------
    (correct : bool, reason : str)
    """
    if response is None:
        return False, "empty_response"
    resp_tokens = _tokenize(response)
    exp_tokens = _tokenize(expected)

    if not exp_tokens:
        return False, "empty_expected"

    # 1) Length bound: overly long responses are rejected.
    if len(resp_tokens) > MAX_TOKENS:
        return False, "too_long"

    # 2) Expected answer must appear as a contiguous token phrase.
    idx = _find_subsequence(resp_tokens, exp_tokens)
    if idx < 0:
        return False, "expected_not_found"

    end = idx + len(exp_tokens)

    # 3) Slash alternatives: "X/Y" -> reject.
    if "/" in resp_tokens:
        return False, "slash_alternatives"

    # 4) "or" as a token -> alternatives ("X or Y").
    if "or" in resp_tokens:
        return False, "or_alternatives"

    # 5) Hedge markers anywhere -> non-committal.
    for t in resp_tokens:
        if t in HEDGES:
            return False, f"hedged:{t}"

    # 6) Leading "no," construction.
    if len(resp_tokens) >= 2 and resp_tokens[0] == "no" and resp_tokens[1] == ",":
        return False, "leading_no"

    # 7) Other known answers present -> multi-answer / cross-talk.
    if known_answers:
        for other in known_answers:
            ot = _tokenize(other)
            if not ot:
                continue
            if ot == exp_tokens:
                continue  # the expected itself
            if _find_subsequence(resp_tokens, ot) >= 0:
                return False, f"contains_other_answer:{other}"

    # 8) Contradiction AFTER expected: "X, not Y" / "X not Y".
    #    Look up to 3 tokens immediately after the expected phrase.
    after_window = resp_tokens[end:end + 4]
    for t in after_window:
        if t in NEGATIONS:
            return False, f"negation_after:{t}"

    # 9) Negation BEFORE expected within the same clause.
    #    Find the last clause break strictly before `idx`.
    break_pos = -1
    for j in range(idx - 1, -1, -1):
        if resp_tokens[j] in CLAUSE_BREAKS:
            break_pos = j
            break
    # Scan backward from idx-1 down to break_pos+1.
    for j in range(idx - 1, break_pos, -1):
        t = resp_tokens[j]
        if t in SKIP_BACK:
            continue
        if t in NEGATIONS:
            return False, f"negation_before:{t}"
        # any other content word -> negation (if any) modifies this word,
        # not the expected answer; stop scanning.
        break

    return True, "correct"


def score_items(items, known_answers=None):
    """Score a list of (response, expected) pairs. Returns (n_correct, details)."""
    correct = 0
    details = []
    for resp, exp in items:
        ok, reason = strict_score(resp, exp, known_answers=known_answers)
        if ok:
            correct += 1
        details.append({"expected": exp, "response": resp,
                         "correct": ok, "reason": reason})
    return correct, details
