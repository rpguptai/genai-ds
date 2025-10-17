"""
Unit tests for LLM routing
Author: Ravi

Validates that model-spec strings are routed to the Ollama client and passed as `model`.
"""
import asyncio
import pytest

from app.llms import _resolve_client_and_model, generate_text, OllamaClient


def test_resolve_model_spec():
    client, model = _resolve_client_and_model("llama3.2:latest")
    # Should pick the Ollama client and return the model spec
    assert isinstance(client, OllamaClient)
    assert model == "llama3.2:latest"


@pytest.mark.asyncio
async def test_generate_text_routes_model(monkeypatch):
    captured = {}

    async def fake_generate(self, prompt, model=None, **kwargs):
        captured['prompt'] = prompt
        captured['model'] = model
        return "ok"

    monkeypatch.setattr(OllamaClient, "generate", fake_generate)

    res = await generate_text("hello", llm_name="llama3.2:latest")
    assert res == "ok"
    assert captured['model'] == "llama3.2:latest"
    assert "hello" in captured['prompt']

