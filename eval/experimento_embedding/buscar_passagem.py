#!/usr/bin/env python3
"""Procura uma expressão dentro de um documento do corpus, para escolher âncora.

Ferramenta de bancada, não de pipeline: serve para encontrar a passagem que
realmente sustenta uma afirmação, antes de gravá-la como citação. Busca literal
sobre o texto normalizado (sem acento, sem caixa), não semântica — quem sabe o
que procura não quer o trecho parecido, quer o trecho exato.

Uso:
  python eval/experimento_embedding/buscar_passagem.py DOC "expressao"
  python eval/experimento_embedding/buscar_passagem.py DOC "a|b|c"   (alternativas)
"""

from __future__ import annotations

import json
import re
import sys
import unicodedata
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
CORPUS = RAIZ / "data/processed/documents.jsonl"

for _f in (sys.stdout, sys.stderr):
    try:
        _f.reconfigure(encoding="utf-8")
    except Exception:
        pass


def normalizar(t: str) -> str:
    t = (t or "").replace("­", "")
    t = unicodedata.normalize("NFKD", t.lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    return " ".join(t.split())


def main() -> None:
    if len(sys.argv) < 3:
        print(__doc__)
        return
    doc, expr = sys.argv[1], sys.argv[2]
    termos = [normalizar(x) for x in expr.split("|") if x.strip()]
    janela = int(sys.argv[3]) if len(sys.argv) > 3 else 300

    achados = 0
    with CORPUS.open(encoding="utf-8") as fh:
        for i, linha in enumerate(fh):
            if not linha.strip():
                continue
            r = json.loads(linha)
            if r["metadata"]["document_id"] != doc:
                continue
            norm = normalizar(r["text"])
            for termo in termos:
                pos = norm.find(termo)
                if pos < 0:
                    continue
                achados += 1
                ini = max(0, pos - 60)
                print(f"── trecho {i} · termo {termo!r}")
                print(f"   …{norm[ini:pos + janela]}…\n")
                break
    if not achados:
        print(f"nada encontrado em {doc} para {expr!r}")


if __name__ == "__main__":
    main()
