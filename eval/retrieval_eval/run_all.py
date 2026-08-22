#!/usr/bin/env python3
"""
run_all.py — roda TODOS os cenários de recuperação na ordem correta
(A → B_gemini → B_bge → C) e, ao final, gera os artefatos COMBINADOS
(rankings.json, metrics.json, per_query_ndcg.json, metrics.csv) consumidos por
03_generation_eval, 05_report e pelos exportadores. Substitui o antigo
02_retrieval_eval.py monolítico.

Uso:  python eval/retrieval_eval/run_all.py [--rebuild]
"""
from __future__ import annotations
import argparse
from _common import load_config, set_seed, setup_io, finalize
import scenario_A_bm25
import scenario_B_dense_gemini
import scenario_B_dense_bge
import scenario_C_fusion


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--rebuild", action="store_true", help="recomputa caches dense")
    args = ap.parse_args()
    cfg = load_config(args.config)
    setup_io(); set_seed(cfg.get("seed", 42))

    scenario_A_bm25.run(cfg)
    scenario_B_dense_gemini.run(cfg, rebuild=args.rebuild)
    scenario_B_dense_bge.run(cfg, rebuild=args.rebuild)
    scenario_C_fusion.run(cfg)      # usa os vencedores de A e B
    finalize(cfg)


if __name__ == "__main__":
    main()
