"""
eval/lib/llm_clients.py
═══════════════════════
Cliente LLM unificado via API OpenAI-compatible — Groq, Cerebras, Google,
OpenRouter. Usado por 01_build_golden, rag_runner (gerador) e judges (juízes).

Provedores e endpoints:
  groq       -> https://api.groq.com/openai/v1         (GROQ_API_KEY[_N])
  cerebras   -> https://api.cerebras.ai/v1             (CEREBRAS_API_KEY[_N])
  openrouter -> https://openrouter.ai/api/v1           (OPENROUTER_API_KEY[_N])
  google     -> .../v1beta/openai/                     (GOOGLE_GENAI_API_KEY)

═══ DOIS NÍVEIS DE ROTAÇÃO ═══

1) ROTAÇÃO DE CHAVES (dentro de um provedor). Contas free têm teto por chave
   (Groq: 1.000 req/dia por modelo). No .env, GROQ_API_KEY_1..N. Quando a chave
   atual dá erro de limite/recusa (429/403/5xx), passa para a próxima chave do
   MESMO provedor.

2) ROTAÇÃO DE ENDPOINT (entre provedores). O MESMO modelo (mesmos pesos) pode
   ser servido por mais de um provedor com ids diferentes:
       Groq:     qwen/qwen3.8-27b     Cerebras: qwen-3.8-27b
   Declarar os dois como `backends` de um gerador só distribui a carga entre
   provedores — é balanceamento, não troca de modelo (o gerador avaliado
   continua sendo "qwen3", servido ora no Groq, ora na Cerebras). Quando TODAS
   as chaves de um backend se esgotam, passa para o próximo backend.

   Compatibilidade: um spec clássico {provider, model} vira um backend único e
   o comportamento é idêntico ao de antes.
"""
from __future__ import annotations

import os
import re
from typing import List, Optional

from .common import get_env, with_retry

PROVIDERS = {
    "groq":       ("https://api.groq.com/openai/v1",                        "GROQ_API_KEY"),
    "cerebras":   ("https://api.cerebras.ai/v1",                            "CEREBRAS_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1",                          "OPENROUTER_API_KEY"),
    "google":     ("https://generativelanguage.googleapis.com/v1beta/openai/", "GOOGLE_GENAI_API_KEY"),
    "openai":     (None,                                                    "OPENAI_API_KEY"),
}

# Erros que disparam TROCA DE CHAVE (rotação), não falha definitiva:
#   429/rate limit  — teto da chave atingido
#   401/invalid_api_key — chave inválida/revogada: pula para a próxima chave
#       (ex.: uma conta free expirou enquanto outras seguem válidas)
#   403/forbidden, 5xx, over capacity — recusa de borda / indisponibilidade
# Quando TODAS as chaves falham, o erro final é propagado ao with_retry.
_ROTATE_ON = ("429", "rate limit", "rate_limit", "too many requests",
              "401", "invalid_api_key", "invalid api key",
              "403", "forbidden", "500", "502", "503", "over capacity",
              "service unavailable")

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def strip_reasoning(text: str) -> str:
    return _THINK_RE.sub("", text or "").strip()


def extract_json(text: str) -> dict | list:
    import json
    if not text:
        raise ValueError("resposta vazia")
    t = strip_reasoning(text)
    t = re.sub(r"^```(?:json)?", "", t.strip(), flags=re.IGNORECASE).strip()
    t = re.sub(r"```$", "", t).strip()
    try:
        return json.loads(t)
    except Exception:
        pass
    for open_c, close_c in (("{", "}"), ("[", "]")):
        start, end = t.find(open_c), t.rfind(close_c)
        if 0 <= start < end:
            try:
                return json.loads(t[start:end + 1])
            except Exception:
                continue
    raise ValueError(f"sem JSON válido em: {text[:200]!r}")


def _resolve_keys(provider: str, key_name: str) -> List[str]:
    get_env(key_name, required=False)      # garante .env carregado
    chaves: List[str] = []
    i, faltas = 1, 0
    while faltas < 4 and i <= 32:
        v = os.getenv(f"{key_name}_{i}")
        if v:
            chaves.append(v.strip()); faltas = 0
        else:
            faltas += 1
        i += 1
    avulsa = os.getenv(key_name)
    if avulsa:
        chaves.append(avulsa.strip())
    vistos, unicas = set(), []
    for k in chaves:
        if k and k not in vistos:
            vistos.add(k); unicas.append(k)
    if not unicas:
        raise RuntimeError(
            f"Nenhuma chave para o provedor '{provider}': defina {key_name} "
            f"ou {key_name}_1, {key_name}_2, ... no .env")
    return unicas


def _should_rotate(err: Exception) -> bool:
    s = str(err).lower()
    return any(tok in s for tok in _ROTATE_ON)


def _normalize_backends(provider, model, backends) -> List[dict]:
    """Produz a lista de backends {provider, model}. Aceita tanto o formato
    clássico (provider+model) quanto uma lista `backends` para rotação entre
    provedores."""
    if backends:
        out = []
        for b in backends:
            out.append({"provider": b["provider"], "model": b["model"]})
        return out
    return [{"provider": provider, "model": model}]


class _Backend:
    """Um provedor+modelo, com suas chaves e sub-clientes (rotação de chave)."""
    def __init__(self, provider: str, model: str, temperature, max_tokens,
                 reasoning_effort):
        from openai import OpenAI
        if provider not in PROVIDERS:
            raise ValueError(f"provedor desconhecido: {provider}")
        base_url, key_name = PROVIDERS[provider]
        self.provider = provider
        self.model = model
        self.keys = _resolve_keys(provider, key_name)
        self.clients = [OpenAI(base_url=base_url, api_key=k,
                               timeout=60.0, max_retries=0) for k in self.keys]
        self.kidx = 0

    def _advance_key(self):
        self.kidx = (self.kidx + 1) % len(self.clients)


class LLMClient:
    def __init__(self, provider: str = None, model: str = None,
                 temperature: float = 0.0, max_tokens: int = 2048,
                 reasoning_effort: Optional[str] = None,
                 backends: Optional[list] = None):
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort

        specs = _normalize_backends(provider, model, backends)
        self._backends = [
            _Backend(s["provider"], s["model"], temperature, max_tokens,
                     reasoning_effort) for s in specs
        ]
        self._bidx = 0                         # backend atual
        n_keys = sum(len(b.clients) for b in self._backends)
        if len(self._backends) > 1 or n_keys > 1:
            rota = " → ".join(f"{b.provider}:{b.model}({len(b.clients)}k)"
                              for b in self._backends)
            print(f"[rotação] {rota}")

    # identidade "lógica" do gerador/juiz (o 1º backend nomeia o modelo)
    @property
    def provider(self): return self._backends[0].provider
    @property
    def model(self):    return self._backends[0].model

    def _advance_backend(self):
        self._bidx = (self._bidx + 1) % len(self._backends)

    def chat(self, user: str, system: Optional[str] = None,
             json_mode: bool = False, temperature: Optional[float] = None) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})

        def _call():
            erro_final = None
            # tenta cada BACKEND; dentro de cada um, tenta cada CHAVE
            for _b in range(len(self._backends)):
                be = self._backends[self._bidx]
                kwargs = dict(
                    model=be.model, messages=messages,
                    temperature=self.temperature if temperature is None else temperature,
                    max_tokens=self.max_tokens,
                )
                if json_mode:
                    kwargs["response_format"] = {"type": "json_object"}
                if self.reasoning_effort:
                    kwargs["reasoning_effort"] = self.reasoning_effort

                esgotou_chaves = False
                for _k in range(len(be.clients)):
                    cli = be.clients[be.kidx]
                    try:
                        resp = cli.chat.completions.create(**kwargs)
                        return resp.choices[0].message.content
                    except Exception as e:                       # noqa: BLE001
                        erro_final = e
                        if _should_rotate(e):
                            be._advance_key()
                            if _k == len(be.clients) - 1:
                                esgotou_chaves = True
                            continue
                        raise          # erro não-rotacionável
                # todas as chaves deste backend falharam → próximo backend
                if esgotou_chaves and len(self._backends) > 1:
                    print(f"  [rotação] backend {be.provider}:{be.model} "
                          f"esgotado; tentando o próximo")
                    self._advance_backend()
                    continue
                break
            raise erro_final

        return with_retry(_call, tries=4, base_delay=3.0,
                          label=f"{self.provider}:{self.model}")

    def chat_json(self, user: str, system: Optional[str] = None,
                  temperature: Optional[float] = None) -> dict | list:
        try:
            raw = self.chat(user, system, json_mode=True, temperature=temperature)
            return extract_json(raw)
        except Exception:
            raw = self.chat(user, system, json_mode=False, temperature=temperature)
            return extract_json(raw)


def build_client(spec: dict, default_temp: float = 0.0) -> LLMClient:
    """spec = {provider, model, ...} OU {backends: [{provider, model}, ...], ...}."""
    return LLMClient(
        provider=spec.get("provider"),
        model=spec.get("model"),
        temperature=spec.get("temperature", default_temp),
        max_tokens=spec.get("max_tokens", 2048),
        reasoning_effort=spec.get("reasoning_effort"),
        backends=spec.get("backends"),
    )
