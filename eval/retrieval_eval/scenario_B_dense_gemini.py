#!/usr/bin/env python3
"""
Cenário B — Dense com o embedding Gemini (proprietário, incumbente).
Busca semântica no corpus (via Qdrant, conforme config). Gera B_dense_gemini.

Uso:  python eval/retrieval_eval/scenario_B_dense_gemini.py [--rebuild]
"""
from __future__ import annotations
import argparse
from _common import (load_config, load_corpus, load_gold, resolve, setup_io,
                     set_seed, save_scenario, dense_ranking)

NAME = "gemini"


def run(cfg=None, rebuild=False):
    if cfg is None:
        cfg = load_config()
    setup_io(); set_seed(cfg.get("seed", 42))
    corpus = load_corpus(resolve(cfg["paths"]["corpus"]))
    queries, qrels, meta = load_gold(cfg)
    print(f"══ Cenário B — Dense ({NAME}, proprietário) ══")
    ranking = dense_ranking(cfg, NAME, corpus, queries, rebuild=rebuild)
    save_scenario(cfg, f"B_dense_{NAME}", ranking, qrels, meta, corpus)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--rebuild", action="store_true")
    run(rebuild=ap.parse_args().rebuild)
