"""
Reinforcement from Subjective Input (RSI) vote store.

Stores per-model helpful/unhelpful votes in a JSONL file and computes smoothed
weights (helpful+1)/(total+2).
"""

import json
import os
from typing import Dict


class Votes:
    """JSONL-backed vote store with Laplace-smoothed model weights."""

    def __init__(self, path: str) -> None:
        """
        Initialise the store; *path* is a directory.  The JSONL file is created
        inside that directory if it does not exist.
        """
        os.makedirs(path, exist_ok=True)
        self._file_path = os.path.join(path, "votes.jsonl")

    def record(self, question_id: str, model: str, helpful: bool) -> None:
        """
        Append one vote record to the JSONL file.

        Args:
            question_id: Arbitrary non-empty string identifying the question.
            model: The model name that produced the answer.
            helpful: Whether the user marked the answer as helpful.
        """
        record = {
            "question_id": question_id,
            "model": model,
            "helpful": helpful,
        }
        with open(self._file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def weights(self, models: list[str]) -> Dict[str, float]:
        """
        Return Laplace-smoothed weight for each model in *models*.

        Returns:
            Mapping from model name to (helpful+1)/(total+2).  Models with no
            votes yet receive 0.5.
        """
        counts: Dict[str, int] = {}
        totals: Dict[str, int] = {}

        for model in models:
            counts[model] = 0
            totals[model] = 0

        if os.path.exists(self._file_path):
            with open(self._file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    model = record.get("model")
                    if model is None or model not in counts:
                        continue
                    totals[model] += 1
                    if record.get("helpful") is True:
                        counts[model] += 1

        result: Dict[str, float] = {}
        for model in models:
            helpful = counts.get(model, 0)
            total = totals.get(model, 0)
            result[model] = (helpful + 1) / (total + 2)

        return result
