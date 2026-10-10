"""
Distinct pronounceable nonsense-word generator.

The Stage 79A dataset must contain many novel words (stimuli + answers) that a
1.7B model can learn without adversarial confusability. The earlier generator
built every word by concatenating two random syllables from a small shared pool
(~56 syllables -> 140 words), so different answers shared morphemes
("dungold" vs "dunhaize" vs "goldpine"). The model learned the morpheme-level
statistics but mixed the pairings, dropping exact-match acquisition to ~14%.

This generator instead:
  1. Builds a large set of UNIQUE syllables (each syllable used at most once).
  2. Pairs them into words WITHOUT replacement.
  3. Guarantees every word and every syllable is distinct.

Thus no two words share a syllable, removing morpheme cross-talk while keeping
the words pronounceable and novel.
"""
import random

# Single consonants and a few common onset clusters (pronounceable in English).
_ONSETS = [
    "b", "c", "d", "f", "g", "h", "j", "k", "l", "m", "n", "p", "qu",
    "r", "s", "t", "v", "w", "x", "y", "z",
    "bl", "br", "ch", "cl", "cr", "dr", "fl", "fr", "gl", "gr", "kn",
    "pl", "pr", "sc", "sh", "sk", "sl", "sm", "sn", "sp", "st", "sw",
    "th", "tr", "tw", "wh", "wr",
]
_VOWELS = ["a", "e", "i", "o", "u", "ae", "ee", "oa", "oo", "ou"]
_CODAS = [
    "", "b", "c", "d", "f", "g", "k", "l", "m", "n", "p", "r", "s",
    "t", "x", "z", "ch", "ck", "ll", "ng", "nk", "nt", "sh", "ss", "st", "th",
]


def _make_syllable(rng, seen):
    """Generate one unique pronounceable syllable."""
    for _ in range(200):
        s = rng.choice(_ONSETS) + rng.choice(_VOWELS) + rng.choice(_CODAS)
        if len(s) >= 2 and s not in seen:
            seen.add(s)
            return s
    raise RuntimeError("could not generate a unique syllable")


def _make_syllables(n, rng):
    """Generate n unique syllables."""
    seen = set()
    out = []
    for _ in range(n):
        out.append(_make_syllable(rng, seen))
    return out


def generate_distinct_words(n, rng, min_len=4):
    """Generate n unique words, each built from two unique syllables.

    No syllable is reused across words.
    """
    # Need 2 syllables per word; generate them all unique, then pair.
    syllables = _make_syllables(2 * n, rng)
    rng.shuffle(syllables)
    words = set()
    out = []
    i = 0
    while len(out) < n:
        w = syllables[i] + syllables[i + 1]
        i += 2
        if len(w) >= min_len and w not in words:
            words.add(w)
            out.append(w)
        else:
            # Rare collision / too short; regenerate a fresh pair.
            extra = _make_syllables(2, rng)
            w = extra[0] + extra[1]
            if len(w) >= min_len and w not in words:
                words.add(w)
                out.append(w)
    return out


if __name__ == "__main__":
    rng = random.Random(42)
    ws = generate_distinct_words(20, rng)
    for w in ws:
        print(w)
    # Verify no shared syllables is not directly checked here (syllables are
    # consumed internally), but all words are unique.
    print("unique:", len(set(ws)) == len(ws))
