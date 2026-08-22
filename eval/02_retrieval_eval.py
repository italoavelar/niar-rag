#!/usr/bin/env python3
"""
02_retrieval_eval.py — MODULARIZADO.
════════════════════════════════════
A avaliação de recuperação foi separada em eval/retrieval_eval/, um script por
cenário, cada um gerando seu próprio CSV (results/retrieval/scenario_<sys>.csv):

    scenario_A_bm25.py            → A_bm25 e A_bm25_mt (ablação de tradução CLIR)
    scenario_B_dense_gemini.py    → B_dense_gemini
    scenario_B_dense_bge.py       → B_dense_bge_m3
    scenario_C_fusion.py          → C_fusion (usa os vencedores de A e B)

Rode um cenário isoladamente, por exemplo:
    python eval/retrieval_eval/scenario_A_bm25.py
    python eval/retrieval_eval/scenario_C_fusion.py     # depois de A e B

Ou todos de uma vez (na ordem certa) + artefatos combinados p/ 03/05/exportadores:
    python eval/retrieval_eval/run_all.py

Este arquivo é apenas um atalho para o run_all.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "retrieval_eval"))
import run_all  # noqa: E402

if __name__ == "__main__":
    run_all.main()
