"""
Run a question against multiple models in parallel and collect answers.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable


def ask_crowd(
    question: str,
    models: list[str],
    ask_fn: Callable[[str, str], str],
) -> list[dict[str, str]]:
    """
    Ask *question* to every model in *models* in parallel threads.

    Args:
        question: The question string.
        models: List of model identifiers (>=2 entries).
        ask_fn: Callable(model_name, question) -> answer string.

    Returns:
        List of {"model": str, "answer": str, "error": str | None}.  A failed
        model populates "error" and sets "answer" to an empty string; it never
        aborts the whole crowd.
    """
    if len(models) < 2:
        raise ValueError("models must contain at least 2 entries")

    results = []

    def query_one(model_name: str) -> dict[str, str]:
        error_msg = ""
        answer_text = ""
        try:
            answer_text = ask_fn(model_name, question)
        except Exception as e:
            error_msg = str(e)
        return {
            "model": model_name,
            "answer": answer_text,
            "error": error_msg if error_msg else None
        }

    with ThreadPoolExecutor(max_workers=len(models)) as executor:
        future_to_model = {
            executor.submit(query_one, model): model
            for model in models
        }
        for future in as_completed(future_to_model):
            results.append(future.result())

    return results
