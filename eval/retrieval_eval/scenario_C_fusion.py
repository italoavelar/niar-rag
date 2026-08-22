#!/usr/bin/env python3
"""
Cenário C — Fusão de DEPLOY: RRF do MELHOR BM25 (A) com o MELHOR denso (B),
escolhidos por nDCG@5. É a única fusão — os vencedores de cada família.

DEPENDE dos rankings salvos por A e B (rode-os antes). Gera C_fusion + fusion_meta.json.

Uso:  python eval/retrieval_eval/scenario_C_fusion.py
"""
from __future__ import annotations
import argparse, json, sys
from _common import (load_config, load_corpus, load_gold, resolve, setup_io,
                     save_scenario, rankings_dir, out_dir)
from lib.retrievers import rrf
from lib.metrics import ndcg_at_k


def run(cfg=None):
    if cfg is None:
        cfg = load_config()
    setup_io()
    corpus = load_corpus(resolve(cfg["paths"]["corpus"]))
    queries, qrels, meta = load_gold(cfg)
    rdir = rankings_dir(cfg)
    loaded = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in rdir.glob("*.json")}
    bms = [s for s in loaded if s.startswith("A_bm25")]
    dns = [s for s in loaded if s.startswith("B_dense")]
    if not (bms and dns):
        sys.exit("! faltam rankings de A e/ou B — rode os cenários A e B antes do C.")

    k5 = cfg["retrieval"]["ndcg_k"][0]
    def ndcg5(system):
        v = [ndcg_at_k(loaded[system].get(q, []), qrels[q], k5) for q in qrels]
        return sum(v) / len(v) if v else 0.0

    best_bm = max(bms, key=ndcg5)
    best_dn = max(dns, key=ndcg5)
    print(f"══ Cenário C — Fusão de deploy: RRF({best_bm} + {best_dn}) ══")
    fused = rrf({"bm25": loaded[best_bm], "dense": loaded[best_dn]},
                k=cfg["retrieval"]["fusion"]["rrf_k"])
    top_k = cfg["retrieval"]["top_k"]
    fused = {q: r[:top_k] for q, r in fused.items()}
    save_scenario(cfg, "C_fusion", fused, qrels, meta, corpus)
    (out_dir(cfg) / "fusion_meta.json").write_text(
        json.dumps({"bm25": best_bm, "dense": best_dn}, ensure_ascii=False), encoding="utf-8")
    print(f"  (fusão de {best_bm} + {best_dn})")


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    run()
