#!/usr/bin/env python3
"""
Cenário A — BM25 (busca léxica, bilíngue).
Gera dois sistemas/CSV: A_bm25 (sem tradução) e, se translate_query=true,
A_bm25_mt (consulta traduzida PT→EN, CLIR) — a ablação de justiça no estrato EN.

Uso:  python eval/retrieval_eval/scenario_A_bm25.py
"""
from __future__ import annotations
import argparse
from _common import (load_config, load_corpus, load_gold, resolve, setup_io,
                     set_seed, save_scenario, indexes_dir)
from lib.retrievers import BM25Retriever
from lib.translate import translate_queries


def run(cfg=None):
    if cfg is None:
        cfg = load_config()
    setup_io(); set_seed(cfg.get("seed", 42))
    corpus = load_corpus(resolve(cfg["paths"]["corpus"]))
    queries, qrels, meta = load_gold(cfg)
    top_k = cfg["retrieval"]["top_k"]
    bcfg = cfg["retrieval"]["bm25"]

    print("══ Cenário A — BM25 ══")
    bm = BM25Retriever(corpus, bcfg, cache_dir=indexes_dir(cfg))
    save_scenario(cfg, "A_bm25", bm.run_queries(queries, top_k), qrels, meta, corpus)

    if bcfg.get("translate_query"):
        translations = translate_queries(queries, bcfg,
                                          indexes_dir(cfg) / "query_translations.json")
        print("══ Cenário A — BM25 + tradução de consulta (CLIR) ══")
        save_scenario(cfg, "A_bm25_mt",
                      bm.run_queries(queries, top_k, translations=translations), qrels, meta, corpus)


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()  # aceita -h
    run()
