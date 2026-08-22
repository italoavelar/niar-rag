#!/usr/bin/env python3
"""
03_generation_eval.py — MODULARIZADO.
═════════════════════════════════════
A avaliação de geração foi separada em eval/generation_eval/, um script por
cenário, cada um avaliado COM e SEM ICL e julgado pelos 2 juízes:

    scenario_A_bm25.py     → geração a partir do contexto do BM25
    scenario_B_dense.py    → geração a partir do denso implantado (dense_for_fusion)
    scenario_C_fusion.py   → geração a partir da fusão de deploy
    scenario_no_rag.py     → SEM RAG (Qwen sozinho) — mede o valor da ferramenta

Cada cenário gera judged_<tag>.jsonl (bruto) + scenario_<tag>.csv (organizado),
tag = <cenário>_icl{0,1}.

Rode um cenário isolado, por exemplo:
    python eval/generation_eval/scenario_C_fusion.py --limit 30
    python eval/generation_eval/scenario_no_rag.py
Ou todos:
    python eval/generation_eval/run_all.py

Este arquivo é apenas um atalho para o run_all.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "generation_eval"))
import run_all  # noqa: E402

if __name__ == "__main__":
    run_all.main()
