"""
eval/lib/llm_clients.py
═══════════════════════
Cliente LLM unificado via API OpenAI-compatible — atende Groq, Cerebras e Google
com a MESMA interface. Usado por:
  • 01_build_golden.py  (gerador de QA — Gemini)
  • rag_runner.py       (gerador do RAG — Qwen-32b @ Groq)
  • judges.py           (juízes — Llama-3.3-70B e gpt-oss-120b @ Groq/Cerebras)

Provedores e endpoints:
  groq      -> https://api.groq.com/openai/v1                 (GROQ_API_KEY)
  cerebras  -> https://api.cerebras.ai/v1                     (CEREBRAS_API_KEY)
  google    -> .../v1beta/openai/                             (GOOGLE_GENAI/API_KEY)
"""
from __future__ import annotations

import json
import re
from typing import Optional

from .common import get_env, with_retry

PROVIDERS = {
    "groq":     ("https://api.groq.com/openai/v1",                       "GROQ_API_KEY"),
    "cerebras": ("https://api.cerebras.ai/v1",                           "CEREBRAS_API_KEY"),
    "google":   ("https://generativelanguage.googleapis.com/v1beta/openai/", "GOOGLE_GENAI_API_KEY"),
    "openai":   (None,                                                   "OPENAI_API_KEY"),
}

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def strip_reasoning(text: str) -> str:
    """Remove blocos <think>...</think> (modelos de raciocínio: Qwen3, gpt-oss)."""
    return _THINK_RE.sub("", text or "").strip()


def extract_json(text: str) -> dict | list:
    """Extrai o primeiro objeto/array JSON do texto, tolerando cercas ```json e
    raciocínio. Levanta ValueError se não houver JSON válido."""
    if not text:
        raise ValueError("resposta vazia")
    t = strip_reasoning(text)
    # remove cercas de código
    t = re.sub(r"^```(?:json)?", "", t.strip(), flags=re.IGNORECASE).strip()
    t = re.sub(r"```$", "", t).strip()
    try:
        return json.loads(t)
    except Exception:
        pass
    # procura o maior trecho {...} ou [...]
    for open_c, close_c in (("{", "}"), ("[", "]")):
        start, end = t.find(open_c), t.rfind(close_c)
        if 0 <= start < end:
            try:
                return json.loads(t[start:end + 1])
            except Exception:
                continue
    raise ValueError(f"sem JSON válido em: {text[:200]!r}")


class LLMClient:
    def __init__(self, provider: str, model: str, temperature: float = 0.0,
                 max_tokens: int = 2048, reasoning_effort: Optional[str] = None):
        from openai import OpenAI
        if provider not in PROVIDERS:
            raise ValueError(f"provedor desconhecido: {provider}")
        base_url, key_name = PROVIDERS[provider]
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        # Modelos de raciocínio (Qwen3, gpt-oss) gastam o orçamento de max_tokens
        # pensando e devolvem a resposta truncada — ou nem a devolvem. No Groq,
        # reasoning_effort="none" desliga o <think> no qwen/qwen3-32b.
        self.reasoning_effort = reasoning_effort
        self._client = OpenAI(base_url=base_url,
                              api_key=get_env(key_name, required=True))

    def chat(self, user: str, system: Optional[str] = None,
             json_mode: bool = False, temperature: Optional[float] = None) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})
        kwargs = dict(
            model=self.model,
            messages=messages,
            temperature=self.temperature if temperature is None else temperature,
            max_tokens=self.max_tokens,
        )
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if self.reasoning_effort:
            kwargs["reasoning_effort"] = self.reasoning_effort

        def _call():
            resp = self._client.chat.completions.create(**kwargs)
            return resp.choices[0].message.content

        return with_retry(_call, tries=4, base_delay=3.0,
                          label=f"{self.provider}:{self.model}")

    def chat_json(self, user: str, system: Optional[str] = None,
                  temperature: Optional[float] = None) -> dict | list:
        """Chat exigindo JSON; tenta json_mode e faz parsing robusto."""
        try:
            raw = self.chat(user, system, json_mode=True, temperature=temperature)
            return extract_json(raw)
        except Exception:
            # alguns modelos não suportam response_format — tenta sem ele
            raw = self.chat(user, system, json_mode=False, temperature=temperature)
            return extract_json(raw)


def build_client(spec: dict, default_temp: float = 0.0) -> LLMClient:
    """spec = {provider, model, temperature?, max_tokens?}."""
    return LLMClient(
        provider=spec["provider"],
        model=spec["model"],
        temperature=spec.get("temperature", default_temp),
        max_tokens=spec.get("max_tokens", 2048),
        reasoning_effort=spec.get("reasoning_effort"),
    )
