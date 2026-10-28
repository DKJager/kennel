"""
OpenAI-compatible chat completion client via urllib.

Uses Bearer token authentication; never logs or prints the key.
"""

import json
import urllib.request
import urllib.error
from typing import Any, Callable


class KennelError(Exception):
    """Raised when the chat completion request fails (HTTP or JSON issues)."""


def chat(
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    timeout: int = 60,
    opener: Callable[[urllib.request.Request, int | None], bytes] | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
) -> str:
    """
    POST {base_url}/chat/completions with a Bearer header, return the assistant message content.

    Args:
        base_url: The base URL of the OpenAI-compatible API (e.g. https://api.tokenfactory.nebius.com/v1).
        api_key: Bearer token (never logged).
        model: Model identifier string.
        messages: List of {"role": ..., "content": ...} dicts.
        timeout: Request timeout in seconds.
        opener: Optional injectable callable(request, timeout) -> bytes for testing.

    Returns:
        The text content of the first choice's message.

    Raises:
        KennelError: On any HTTP error, unexpected status, or JSON parsing failure.
    """
    url = f"{base_url}/chat/completions"

    body: dict = {"model": model, "messages": messages}
    if max_tokens is not None:
        body["max_tokens"] = int(max_tokens)
    if temperature is not None:
        body["temperature"] = float(temperature)
    payload = json.dumps(body).encode("utf-8")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    req = urllib.request.Request(url, data=payload, headers=headers, method="POST")

    try:
        if opener is not None:
            response_bytes = opener(req, timeout)
        else:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                response_bytes = resp.read()
    except urllib.error.HTTPError as e:
        try:
            body = e.read().decode("utf-8")
        except Exception:
            body = ""
        raise KennelError(f"HTTP {e.code}: {body}") from e
    except urllib.error.URLError as e:
        raise KennelError(f"URL error: {e.reason}") from e
    except Exception as e:
        raise KennelError(f"Request failed: {e}") from e

    try:
        response_text = response_bytes.decode("utf-8")
        data = json.loads(response_text)
    except json.JSONDecodeError as e:
        raise KennelError(f"Invalid JSON response: {e}") from e
    except UnicodeDecodeError as e:
        raise KennelError(f"Response decoding failed: {e}") from e

    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as e:
        raise KennelError(f"Unexpected response structure: {e}") from e

    return content
