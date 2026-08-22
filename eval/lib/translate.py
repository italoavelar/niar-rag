"""
eval/lib/translate.py
═════════════════════
Tradução de consulta PT→EN para o BM25 cross-lingual (CLIR).

Não é uma contribuição — é a MEDIDA DE JUSTIÇA que impede o BM25 de ser um
espantalho no corpus bilíngue: sem traduzir a query, o BM25 nunca casaria uma
pergunta em PT com um documento em inglês. As traduções são cacheadas em disco
(uma chamada de LLM por pergunta única). O denso/fusão não precisam disso.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from tqdm import tqdm

from .common import ensure_dir
from .llm_clients import LLMClient, strip_reasoning

_SYS = ("Você traduz consultas de busca do português para o inglês, preservando "
        "termos técnicos jurídicos e de saúde. Responda APENAS com a tradução, "
        "em uma linha, sem aspas nem explicação.")


def translate_queries(queries: Sequence[Tuple[str, str]], bm25cfg: dict,
                      cache_path) -> Dict[str, str]:
    """Retorna {query_pt: query_en} para todas as queries, usando cache em disco.
    Se o tradutor não estiver disponível (sem chave/erro), degrada para identidade
    (query_en = query_pt) — nesse caso A_bm25_mt ≈ A_bm25 e o aviso é impresso."""
    cache_path = Path(cache_path)
    cache: Dict[str, str] = {}
    if cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            cache = {}

    uniq = list(dict.fromkeys(q for _, q in queries))
    missing = [q for q in uniq if q not in cache]

    if missing:
        tcfg = bm25cfg.get("translator") or {"provider": "groq",
                                             "model": "llama-3.1-8b-instant"}
        try:
            client = LLMClient(provider=tcfg["provider"], model=tcfg["model"],
                               temperature=0, max_tokens=256)
        except Exception as e:
            print(f"[BM25/CLIR] tradutor indisponível ({e}); usando identidade "
                  f"(A_bm25_mt ≈ A_bm25).")
            client = None

        for q in tqdm(missing, desc="[BM25/CLIR] traduzindo queries", leave=False):
            if client is None:
                cache[q] = q
                continue
            try:
                raw = strip_reasoning(client.chat(q, system=_SYS))
                cache[q] = raw.strip().strip('"').split("\n")[0].strip() or q
            except Exception:
                cache[q] = q          # fallback por-query: mantém PT (no-op)
        ensure_dir(cache_path.parent)
        cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2),
                              encoding="utf-8")

    return {q: cache.get(q, q) for _, q in queries}
