#!/usr/bin/env python3
"""
Cenário B — Dense com o embedding BGE-m3 (aberto, auto-hospedável, desafiante).
Busca semântica no corpus (embute localmente em cache, roda em CPU).
Gera B_dense_bge_m3.

Uso:  python eval/retrieval_eval/scenario_B_dense_bge.py [--rebuild]
"""
from __future__ import annotations
import argparse
from _common import (load_config, load_corpus, load_gold, resolve, setup_io,
                     set_seed, save_scenario, dense_ranking)

NAME = "bge_m3"


def run(cfg=None, rebuild=False):
    if cfg is None:
        cfg = load_config()
    setup_io(); set_seed(cfg.get("seed", 42))
    corpus = load_corpus(resolve(cfg["paths"]["corpus"]))
    queries, qrels, meta = load_gold(cfg)
    print(f"══ Cenário B — Dense ({NAME}, aberto) ══")
    ranking = dense_ranking(cfg, NAME, corpus, queries, rebuild=rebuild)
    save_scenario(cfg, f"B_dense_{NAME}", ranking, qrels, meta, corpus)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--rebuild", action="store_true")
    run(rebuild=ap.parse_args().rebuild)
