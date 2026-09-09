#!/usr/bin/env python3
"""
Cenário B — Dense com o embedding Qwen3-Embedding-0.6B (aberto, desafiante da E1).
Busca semântica no corpus, a partir do cache de vetores gerado no Colab.
Gera B_dense_qwen3_0_6b.

O modelo é ASSIMÉTRICO: a consulta leva prompt_name="query" e o documento vai
sem prompt. Isso está declarado em eval/config.yaml (query_prompt_name) e é
respeitado por lib/embedders.py:SentenceTransformerEmbedder. Com
`offline: true`, as consultas vêm prontas de dense_qwen3_0_6b_queries.npz e
nenhum modelo é carregado.

Uso:  python eval/retrieval_eval/scenario_B_dense_qwen.py [--rebuild]
"""
from __future__ import annotations
import argparse
from _common import (load_config, load_corpus, load_gold, resolve, setup_io,
                     set_seed, save_scenario, dense_ranking)

NAME = "qwen3_0_6b"


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
