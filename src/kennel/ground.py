"""
Ground claims against a set of source passages via token overlap.
"""

import re
from typing import Any


_STOP = set("a an the is are of to in for and or on with at by it this that be as".split())


def ground(
    claims: list[str],
    passages: list[dict[str, str]],
) -> list[dict[str, Any]]:
    """
    For each claim find the best matching passage by token overlap.

    Args:
        claims: List of claim strings.
        passages: List of {"id": str, "text": str} dicts.  Entries missing
            "id" or "text" are silently skipped.

    Returns:
        List of {"claim": str, "source_id": str | None, "score": float}.
        *source_id* is None when the best score is < 0.2 (ungrounded).
    """
    # Normalise passages: keep only those with both id and text
    valid_passages = []
    for p in passages:
        if isinstance(p, dict) and "id" in p and "text" in p:
            id_val = p["id"]
            text_val = p["text"]
            if isinstance(id_val, str) and isinstance(text_val, str):
                valid_passages.append((id_val, text_val))

    # Pre-tokenise passages: lowercase, split on non-alphanumeric
    passage_tokens = []
    for pid, ptext in valid_passages:
        # Normalise: lowercase, keep alphanumeric sequences
        tokens = [t.lower() for t in re.findall(r'\b\w+\b', ptext)]
        passage_tokens.append((pid, set(tokens) - _STOP))

    result = []
    for claim in claims:
        if not isinstance(claim, str):
            # If claim is not a string, treat it as empty
            token_set = set()
        else:
            # Tokenise claim the same way
            claim_tokens = [t.lower() for t in re.findall(r'\b\w+\b', claim)]
            token_set = set(claim_tokens) - _STOP

        best_score = 0.0
        best_id = None

        if token_set:
            # Compute Jaccard similarity with each passage
            for pid, ptokens in passage_tokens:
                if not ptokens:
                    continue
                intersection = len(token_set & ptokens)
                # Coverage of the claim's content words; a single shared word never grounds a claim
                score = intersection / len(token_set) if intersection >= 2 else 0.0
                if score > best_score:
                    best_score = score
                    best_id = pid

        # Apply threshold
        if best_score < 0.2:
            best_id = None

        result.append({
            "claim": claim,
            "source_id": best_id,
            "score": best_score
        })

    return result
