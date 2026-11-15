"""Grading (spec section 4.3).

Strict grader = the SQ2 grader (scoring.classify: garbled? -> declined? -> contains an accepted answer? ->
wrong) with one check added between "declined?" and "contains an accepted answer?":

  Hedged? Split the normalised reply on " or " and "/". If there are at least 2 non-empty parts, each of at
  most 6 words, and the parts do not all match the same alias set -> hedged, counted as wrong.

Implementation choices (the spec leaves them open):
  - "normalised reply" for splitting = scoring.light_normalize (lower case, accents and quotes unified,
    whitespace collapsed; it keeps "/" so the split is possible). Each part is then normalised with
    scoring.normalize_answer; parts that normalise to "" are dropped; words are counted after that.
  - "the same alias set" = the gold aliases: the reply is not a hedge when every part matches an accepted
    answer (e.g. "USA / United States"). A reply that is itself exactly an accepted answer (e.g. "AC/DC") is
    not a hedge either.
  - Methods may not use gold answers. For discarding candidates they use `hedged_form` (the same split rule)
    with "the same alias set" read as "all parts are the same answer as each other" (pilot.metrics
    .same_answer), so that check is gold-free as the spec requires.

Labels use scoring's codes: C_ = 1 correct, A_ = 0 declined ("I don't know"), W_ = -1 wrong (garbled and
hedged replies count as wrong).
"""
import re

from pilot.metrics import same_answer
from scoring import A_, C_, W_, abstention_type, alias_match, classify, is_degenerate, light_normalize, normalize_answer

HEDGE_MAX_WORDS = 6
_SPLIT = re.compile(r" or |/")


def hedge_parts(text):
    """Non-empty normalised parts, or None if the reply does not have the hedge form."""
    parts = [normalize_answer(p) for p in _SPLIT.split(light_normalize(text))]
    parts = [p for p in parts if p]
    if len(parts) < 2 or any(len(p.split()) > HEDGE_MAX_WORDS for p in parts):
        return None
    return parts


def is_hedged(text, aliases):
    """Grader version (uses the gold aliases)."""
    parts = hedge_parts(text)
    if parts is None:
        return False
    if alias_match(text, aliases, strict=True):
        return False
    return not all(alias_match(p, aliases) for p in parts)


def hedged_form(text):
    """Gold-free version for methods: hedge form and the parts are not all the same answer."""
    parts = hedge_parts(text)
    if parts is None:
        return False
    return not all(same_answer(p, parts[0]) for p in parts[1:])


def grade(text, aliases):
    """Strict grade. Returns (label, garbled, hedged)."""
    if is_degenerate(text):
        return W_, True, False
    if abstention_type(text) is not None:
        return A_, False, False
    if is_hedged(text, aliases):
        return W_, False, True
    if alias_match(text, aliases):
        return C_, False, False
    return W_, False, False


def grade_sq2(text, aliases):
    return int(classify(text, aliases)[0])


def grade_both(text, aliases):
    lab, garbled, hedged = grade(text, aliases)
    return {"label_strict": int(lab), "label_sq2": grade_sq2(text, aliases), "garbled": bool(garbled),
            "hedged": bool(hedged)}


def is_idk(text):
    return abstention_type(text) is not None


def candidate_status(text, rejected_norms):
    """Gold-free check of a proposal in the search loop (spec 5.1 step 4b): 'garbled', 'idk', 'hedged',
    'duplicate' (normalised text equals a rejected answer) or 'valid'."""
    if is_degenerate(text):
        return "garbled"
    if is_idk(text):
        return "idk"
    if hedged_form(text):
        return "hedged"
    if normalize_answer(text) in rejected_norms:
        return "duplicate"
    return "valid"


__all__ = ["A_", "C_", "W_", "grade", "grade_sq2", "grade_both", "is_idk", "is_hedged", "hedged_form",
           "candidate_status", "hedge_parts", "normalize_answer"]
