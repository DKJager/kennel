"""
Reconcile answers from the crowd into agreement and dissent.
"""

from typing import Any
import re


def _normalize_sentence(sentence: str) -> str:
    """Normalize a sentence: lowercase, strip punctuation."""
    # Strip punctuation from start/end and internal punctuation
    normalized = sentence.lower().strip()
    # Remove all punctuation characters
    normalized = re.sub(r"(?<=\w)['\u2019](?=\w)", '', normalized)  # keep contractions whole
    normalized = re.sub(r'[^\w\s]', ' ', normalized)  # dashes etc. separate words
    # Collapse multiple spaces
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    return normalized


def _tokenize(text: str) -> set[str]:
    """Tokenize text into word tokens."""
    if not text:
        return set()
    tokens = re.findall(r'\b\w+\b', text.lower())
    return set(tokens)


def _jaccard_similarity(set_a: set[str], set_b: set[str]) -> float:
    """Compute Jaccard similarity between two token sets."""
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def _split_into_sentences(answer: str) -> list[str]:
    """Split answer text into sentences."""
    if not answer:
        return []
    text = re.sub(r'[*_#`]+', ' ', answer)
    text = re.sub(r'(?m)^\s*(?:[-+]|\d+[.)])\s+', '', text)
    for ab in ("e.g.", "i.e.", "etc.", "vs.", "Dr.", "Mr.", "Mrs.", "U.S.", "approx."):
        text = re.sub(re.escape(ab), ab.replace(".", "<DOT>"), text, flags=re.I)
    parts = re.split(r'[.!?]+|\n+', text)
    out = []
    for s in parts:
        s = s.replace("<DOT>", ".").strip()
        if len(re.findall(r'\b\w+\b', s)) >= 2:
            out.append(s)
    return out


def reconcile(answers: list[dict[str, str]]) -> dict[str, Any]:
    """
    Compute agreement, dissent, and confidence from a list of model answers.

    Each answer dict must have "model", "answer", and optionally "error".
    Answers whose "error" is non-empty are ignored.

    Returns:
        A dict with keys:
        - "agreement": list of normalised claim sentences supported by >=2 models
        - "dissent": list of claim sentences from exactly one model
        - "confidence": float in [0, 1] = agreeing models / answering models

        An empty *answers* list or empty token sets yield safe defaults:
        "agreement"=[], "dissent"=[], "confidence"=0.0.
    """
    if not answers:
        return {"agreement": [], "dissent": [], "confidence": 0.0}

    # Collect valid answers (no error)
    valid_answers = []
    for ans in answers:
        if ans.get("error", ""):
            continue
        if "answer" not in ans or "model" not in ans:
            continue
        valid_answers.append(ans)

    # If no valid answers, return defaults
    if not valid_answers:
        return {"agreement": [], "dissent": [], "confidence": 0.0}

    # Extract claims from each model
    model_claims = {}  # model -> list of (normalized, original) pairs
    for ans in valid_answers:
        model = ans["model"]
        sentences = _split_into_sentences(ans["answer"])
        normalized_pairs = []
        for sent in sentences:
            normalized = _normalize_sentence(sent)
            if normalized:  # Skip empty after normalization
                normalized_pairs.append((normalized, sent))
        if normalized_pairs:
            model_claims[model] = normalized_pairs

    # If no models produced any claims
    if not model_claims:
        return {"agreement": [], "dissent": [], "confidence": 0.0}

    # Build mapping from normalized claim to list of supporting models
    claim_support = {}  # normalized -> (original, set of models)
    for model, pairs in model_claims.items():
        for normalized, original in pairs:
            if normalized not in claim_support:
                claim_support[normalized] = (original, set())
            claim_support[normalized][1].add(model)

    # Group similar claims (Jaccard >= 0.6)
    processed = set()
    grouped_claims = []  # list of (merged_normalized, original, set of models)

    normalized_list = list(claim_support.keys())
    for i in range(len(normalized_list)):
        if normalized_list[i] in processed:
            continue

        # Start a new group with this claim
        current_claim = normalized_list[i]
        current_original, current_models = claim_support[current_claim]
        processed.add(current_claim)

        # Find similar claims
        for j in range(i + 1, len(normalized_list)):
            other_claim = normalized_list[j]
            if other_claim in processed:
                continue

            tokens_a = _tokenize(current_claim)
            tokens_b = _tokenize(other_claim)
            if _jaccard_similarity(tokens_a, tokens_b) >= 0.6:
                current_models.update(claim_support[other_claim][1])
                processed.add(other_claim)
                # Keep the longer original text as representative
                if len(claim_support[other_claim][0]) > len(current_original):
                    current_original = claim_support[other_claim][0]

        grouped_claims.append((current_claim, current_original, current_models))

    # Separate agreement (>=2 models) and dissent (exactly 1 model)
    agreement = []
    dissent = []
    for normalized, _original, models in grouped_claims:
        if len(models) >= 2:
            agreement.append(normalized)
        else:
            dissent.append(normalized)

    # Calculate confidence: agreeing models / answering models
    agreeing_models = set()
    for _, _, models in grouped_claims:
        if len(models) >= 2:
            agreeing_models.update(models)

    answering_models = set(model_claims.keys())
    confidence = len(agreeing_models) / len(answering_models) if answering_models else 0.0

    return {
        "agreement": agreement,
        "dissent": dissent,
        "confidence": confidence
    }
