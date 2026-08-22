#!/usr/bin/env python3
"""
Cenário SEM RAG (geração) — Qwen responde SOZINHO, sem contexto recuperado
(pelo conhecimento próprio). É o baseline que mede o valor da ferramenta:
espera-se pior correção/citação, mais alucinação e menos recusa nas perguntas
fora de escopo. NÃO precisa da recuperação.

Condição ÚNICA (sem ICL): a comparação com/sem ICL fica só nos cenários COM RAG.

Uso:  python eval/generation_eval/scenario_no_rag.py [--limit N] [--no-judge]
"""
from __future__ import annotations
import argparse
from _common import Setup, run_and_save


def run(S=None, cfg=None):
    S = S or Setup(cfg)
    # baseline único (Qwen sozinho, sem ICL) — a variação de ICL é só COM RAG.
    run_and_save(S, "no_rag", None, icl=False, no_rag=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int); ap.add_argument("--no-judge", action="store_true")
    a = ap.parse_args()
    run(S=Setup(limit=a.limit, no_judge=a.no_judge))
