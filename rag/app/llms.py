"""
LLM client wrappers
Author: Ravi

Provides a switchable interface for local LLMs. Currently supports:
- ollama (via local Ollama API)
- echo (a simple deterministic debug LLM)

The interface exposes an async `generate(prompt, model_name, **kwargs)` function that returns text.
"""
from typing import Optional, Tuple
import asyncio
import logging

from .config import OLLAMA_API_URL

logger = logging.getLogger(__name__)


class LLMError(Exception):
    pass


class OllamaClient:
    """Minimal Ollama client using the local Ollama HTTP API.
    This client is intentionally small: send a prompt and return the generated text.
    """

    def __init__(self, base_url: str = OLLAMA_API_URL, timeout: int = 60):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._lock = asyncio.Lock()

    async def generate(self, prompt: str, model: str = "ollama/llama2", max_tokens: int = 512) -> str:
        """Call Ollama's generate endpoint. Returns generated text.

        Note: the Ollama installation must be running locally and exposing the API.
        """
        # Import here to avoid requiring httpx at module import time
        try:
            import httpx
        except Exception as e:
            logger.exception("httpx is required for OllamaClient")
            raise LLMError("httpx is required for OllamaClient") from e

        async with self._lock:
            async with httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout) as client:
                url = "/api/generate"
                payload = {"model": model, "prompt": prompt, "max_tokens": max_tokens}
                try:
                    resp = await client.post(url, json=payload)
                    resp.raise_for_status()
                    # Try strict JSON first
                    try:
                        data = resp.json()
                    except Exception:
                        # Some Ollama versions stream or emit newline-delimited JSON
                        # Fall back to parsing text: attempt to parse each line as JSON,
                        # otherwise return the raw text body.
                        text = resp.text
                        import json as _json

                        # Attempt to parse each non-empty line as JSON and extract known fields
                        for line in (l.strip() for l in text.splitlines()):
                            if not line:
                                continue
                            try:
                                obj = _json.loads(line)
                            except Exception:
                                continue
                            # obj parsed successfully; extract common patterns
                            if isinstance(obj, dict):
                                if "text" in obj:
                                    return obj["text"]
                                if "output" in obj:
                                    out = obj["output"]
                                    if isinstance(out, list):
                                        return "".join(str(x) for x in out)
                                    return str(out)
                                if "choices" in obj and isinstance(obj["choices"], list) and obj["choices"]:
                                    choice = obj["choices"][0]
                                    if isinstance(choice, dict) and "text" in choice:
                                        return choice["text"]
                                    return str(choice)
                                # fallback to stringified object
                                return str(obj)
                        # If line-based parsing didn't find JSON, attempt to decode multiple concatenated JSON objects
                        try:
                            decoder = _json.JSONDecoder()
                            pos = 0
                            L = len(text)
                            while pos < L:
                                # skip whitespace
                                while pos < L and text[pos].isspace():
                                    pos += 1
                                if pos >= L:
                                    break
                                obj, idx = decoder.raw_decode(text, pos)
                                pos = idx
                                # process obj
                                if isinstance(obj, dict):
                                    if "text" in obj:
                                        return obj["text"]
                                    if "output" in obj:
                                        out = obj["output"]
                                        if isinstance(out, list):
                                            return "".join(str(x) for x in out)
                                        return str(out)
                                    if "choices" in obj and isinstance(obj["choices"], list) and obj["choices"]:
                                        choice = obj["choices"][0]
                                        if isinstance(choice, dict) and "text" in choice:
                                            return choice["text"]
                                        return str(choice)
                                    return str(obj)
                                # otherwise continue to next object
                            # last resort: return raw text
                            return text
                        except Exception:
                            # Log truncated response for debugging and fall back to raw text
                            logger.debug("Ollama response (truncated): %s", text[:1000])
                            return text

                    # Ollama may return different shapes depending on version; try common keys
                    if isinstance(data, dict):
                        # possible keys: "text", "output", "choices"
                        if "text" in data:
                            return data["text"]
                        if "output" in data:
                            out = data["output"]
                            if isinstance(out, list):
                                return "".join(str(x) for x in out)
                            return str(out)
                        if "choices" in data and isinstance(data["choices"], list) and data["choices"]:
                            choice = data["choices"][0]
                            if isinstance(choice, dict) and "text" in choice:
                                return choice["text"]
                            return str(choice)
                        # fallback to raw text
                        return str(data)
                    # if it's a plain list/string
                    return str(data)
                except Exception as e:
                    logger.exception("Ollama request failed")
                    raise LLMError("Ollama request failed") from e


class EchoClient:
    """Simple LLM that echoes the prompt (useful for testing)."""

    async def generate(self, prompt: str, **kwargs) -> str:
        await asyncio.sleep(0)
        return f"ECHO:\n{prompt}"


class LLMFactory:
    """Create LLM clients by name. Keeps the surface open to add more local LLMs later.
    Supported names: "ollama", "echo". Note: model-spec strings are handled at call time.
    """

    def __init__(self):
        self._ollama = OllamaClient()
        self._echo = EchoClient()

    def get(self, name: str):
        n = (name or "ollama").lower()
        if n in ("ollama", "ollama_local"):
            return self._ollama
        if n in ("echo", "dummy"):
            return self._echo
        raise ValueError(f"Unknown llm name: {name}")


llm_factory = LLMFactory()


def _resolve_client_and_model(llm_name: Optional[str]) -> Tuple[object, Optional[str]]:
    """Resolve the intended client and (optional) model spec from the llm_name parameter.

    Behavior:
    - If llm_name is one of the known client aliases (ollama, echo), return (client, None)
    - If llm_name looks like a model spec (contains '/' or ':' or the substring 'llama'),
      route to the Ollama client and return the model spec as second element.
    - Otherwise, raise ValueError.
    """
    if not llm_name:
        return llm_factory.get("ollama"), None
    n = llm_name.strip()
    ln = n.lower()
    # Known aliases
    if ln in ("ollama", "ollama_local"):
        return llm_factory.get("ollama"), None
    if ln in ("echo", "dummy"):
        return llm_factory.get("echo"), None
    # Heuristic: if user supplied a model spec like "llama3.2:latest" or "ollama/llama2",
    # treat as Ollama model identifier and route accordingly.
    if "/" in n or ":" in n or "llama" in ln:
        return llm_factory.get("ollama"), n
    # Unknown name
    raise ValueError(f"Unknown llm name: {llm_name}")


async def generate_text(prompt: str, llm_name: Optional[str] = "ollama", **kwargs) -> str:
    """Generate text using the resolved client.

    If llm_name encodes a model spec (e.g. 'llama3.2:latest'), it's passed to the Ollama client
    as the `model` parameter. For known client names (e.g. 'echo') the appropriate client is used.
    """
    client, model_spec = _resolve_client_and_model(llm_name)
    # If it's an OllamaClient, pass model_spec as `model` if provided.
    if isinstance(client, OllamaClient):
        if model_spec:
            return await client.generate(prompt, model=model_spec, **kwargs)
        return await client.generate(prompt, **kwargs)
    # Otherwise, just call the client
    return await client.generate(prompt, **kwargs)
