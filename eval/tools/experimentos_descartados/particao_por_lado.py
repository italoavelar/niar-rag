#!/usr/bin/env python3
"""Alocação PARTICIONADA do top-5 por lado da comparação (sem fusão, sem cross-encoder).

Hipótese que este script testa (09/set/2026):

O pipeline atual (`decompose_eval.py`) decompõe a pergunta em duas sub-perguntas
justamente para escapar do "centroide semântico" que faz uma busca densa única
encher o top-5 com um documento só. Mas depois ele (i) funde as duas listas por
RRF numa lista única e (ii) repontua TODOS os candidatos contra a PERGUNTA
ORIGINAL (linha 216-218: `pares.append((texto, ...))` com `texto` = pergunta
inteira). Os passos (i) e (ii) desfazem a decomposição: a competição volta a ser
global e volta a ser resolvida a favor do lado semanticamente dominante.

Aqui a decomposição NÃO é desfeita: cada sub-pergunta recupera com o SEU próprio
vetor, e o orçamento de 5 vagas é ALOCADO a priori entre os lados (ex.: 3+2),
sem nenhuma competição entre eles. Custo: só embedding de query + produto de
matriz. Zero cross-encoder, zero aumento de pool.

Salva também r1/r2 (top-100 de cada sub-pergunta) para reuso em experimentos
futuros, evitando recomputar.

Uso:
  python eval/tools/particao_por_lado.py
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]   # .../eval/tools/experimentos_descartados/x.py
EVAL = PROJECT_ROOT / "eval"
sys.path.insert(0, str(EVAL))
sys.path.insert(0, str(EVAL / "retrieval_eval"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from _common import load_config, load_corpus, resolve, setup_io   # noqa: E402
from embedding_text import (                                       # noqa: E402
    corpus_embedding_fingerprint, bge_cache_filename,
)
from lib import metrics as M                                       # noqa: E402

OUT_DIR = EVAL / "results_decomp"
CACHE_FILE = OUT_DIR / "subqueries_cache.json"
RETRIEVE_TOP_N = 100


from lib.metrics import document_of as doc_of  # noqa: E402  (uma implementação só)


def load_gold_comparative():
    gold = [json.loads(l) for l in open(EVAL / "data/golden_qa.jsonl", encoding="utf-8")
            if l.strip()]
    alvo = [r for r in gold if r.get("qrels") and r.get("question_type") == "comparative"]
    qrels = {r["qid"]: {k: int(v) for k, v in r["qrels"].items()} for r in alvo}
    pergunta = {r["qid"]: r["question"] for r in alvo}
    req_docs = {r["qid"]: sorted({doc_of(c) for c in r["qrels"]}) for r in alvo}
    return pergunta, qrels, req_docs


def dense_search(qvec, matrix, ids, top_n):
    scores = matrix @ qvec.astype(np.float32)
    idx = np.argpartition(-scores, top_n - 1)[:top_n]
    idx = idx[np.argsort(-scores[idx])]
    return [ids[i] for i in idx]


def main() -> None:
    cfg = load_config()
    setup_io()
    t0 = time.time()

    corpus = load_corpus(resolve(cfg["paths"]["corpus"]))
    fp = corpus_embedding_fingerprint(corpus.values())
    npz = np.load(EVAL / "results/indexes" / bge_cache_filename("bge_m3", fp), allow_pickle=True)
    matrix = npz["matrix"].astype(np.float32)
    ids = [str(i) for i in npz["ids"]]
    print(f"[{time.time()-t0:5.1f}s] corpus + matriz ({len(ids)} chunks)")

    perguntas, qrels, req_docs = load_gold_comparative()
    subqs_cache = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
    subqs = {qid: subqs_cache[q] for qid, q in perguntas.items()}

    from lib.embedders import build_embedder
    embedder = build_embedder("bge_m3", cfg)
    print(f"[{time.time()-t0:5.1f}s] BGE-m3 carregado")

    r1, r2, r_orig = {}, {}, {}
    for qid, texto in perguntas.items():
        r1[qid] = dense_search(embedder.embed_query(subqs[qid]["sub1"]), matrix, ids, RETRIEVE_TOP_N)
        r2[qid] = dense_search(embedder.embed_query(subqs[qid]["sub2"]), matrix, ids, RETRIEVE_TOP_N)
        r_orig[qid] = dense_search(embedder.embed_query(texto), matrix, ids, RETRIEVE_TOP_N)
    (OUT_DIR / "subquery_rankings.json").write_text(
        json.dumps({"sub1": r1, "sub2": r2, "original": r_orig}, ensure_ascii=False),
        encoding="utf-8")
    print(f"[{time.time()-t0:5.1f}s] buscas por sub-pergunta concluidas e salvas")

    # ── Onde está o chunk relevante de cada documento, na busca de cada lado ──
    print("\n=== Posicao do chunk relevante de cada documento exigido, por lista ===")
    print(f"{'qid':<8}{'documento':<42}{'orig':>7}{'sub1':>7}{'sub2':>7}{'melhor':>8}")
    achou_em_5, achou_em_5_orig = 0, 0
    total_docs = 0
    for qid in sorted(perguntas):
        for d in req_docs[qid]:
            rel_d = [c for c in qrels[qid] if doc_of(c) == d]
            def pos(lst):
                for i, c in enumerate(lst, start=1):
                    if c in rel_d:
                        return i
                return None
            p_o, p_1, p_2 = pos(r_orig[qid]), pos(r1[qid]), pos(r2[qid])
            melhor_sub = min([p for p in (p_1, p_2) if p], default=None)
            total_docs += 1
            if melhor_sub and melhor_sub <= 5:
                achou_em_5 += 1
            if p_o and p_o <= 5:
                achou_em_5_orig += 1
            f = lambda p: str(p) if p else "-"
            print(f"{qid:<8}{d[:40]:<42}{f(p_o):>7}{f(p_1):>7}{f(p_2):>7}{f(melhor_sub):>8}")
    print(f"\ndocumentos exigidos, total: {total_docs}")
    print(f"  com chunk relevante no top-5 da busca ORIGINAL         : {achou_em_5_orig}/{total_docs}")
    print(f"  com chunk relevante no top-5 da MELHOR sub-pergunta    : {achou_em_5}/{total_docs}")

    # ── Estratégias de alocação particionada ────────────────────────────────
    def aloca(qid, k1, k2):
        """top-k1 do lado 1 + top-k2 do lado 2, sem fusao nem competicao."""
        out = []
        for c in r1[qid][:k1]:
            if c not in out:
                out.append(c)
        for c in r2[qid][:k2]:
            if c not in out:
                out.append(c)
        # se deduplicacao deixou vagas, completa alternando mais fundo
        i1, i2 = k1, k2
        while len(out) < 5 and (i1 < RETRIEVE_TOP_N or i2 < RETRIEVE_TOP_N):
            if i1 < RETRIEVE_TOP_N and r1[qid][i1] not in out:
                out.append(r1[qid][i1])
            i1 += 1
            if len(out) >= 5:
                break
            if i2 < RETRIEVE_TOP_N and r2[qid][i2] not in out:
                out.append(r2[qid][i2])
            i2 += 1
        return out[:5]

    def cobertura(top5, qid):
        rel_in = set(top5) & set(qrels[qid].keys())
        return {doc_of(c) for c in rel_in}

    print("\n=== Alocacao particionada (sem fusao, sem cross-encoder) ===")
    print(f"{'estrategia':<28}{'ambos docs':>12}{'>=1 doc':>10}{'nDCG@5':>10}")
    qs = sorted(perguntas)
    baselines = {}
    for nome, k1, k2 in [("3+2", 3, 2), ("2+3", 2, 3), ("2+2 (+1 do lado 1)", 2, 2),
                         ("4+1", 4, 1), ("1+4", 1, 4)]:
        tops = {q: aloca(q, k1, k2) for q in qs}
        full = sum(1 for q in qs if cobertura(tops[q], q) == set(req_docs[q]))
        any_ = sum(1 for q in qs if cobertura(tops[q], q))
        nd = float(np.mean([M.ndcg_at_k(tops[q], qrels[q], 5) for q in qs]))
        baselines[nome] = tops
        print(f"{nome:<28}{full:>7}/{len(qs):<4}{any_:>6}/{len(qs):<3}{nd:>10.4f}")

    # baseline: busca original crua, top-5
    tops_o = {q: r_orig[q][:5] for q in qs}
    full_o = sum(1 for q in qs if cobertura(tops_o[q], q) == set(req_docs[q]))
    any_o = sum(1 for q in qs if cobertura(tops_o[q], q))
    nd_o = float(np.mean([M.ndcg_at_k(tops_o[q], qrels[q], 5) for q in qs]))
    print(f"{'[baseline] busca original':<28}{full_o:>7}/{len(qs):<4}{any_o:>6}/{len(qs):<3}{nd_o:>10.4f}")

    (OUT_DIR / "particao_top5.json").write_text(
        json.dumps(baselines["3+2"], ensure_ascii=False), encoding="utf-8")
    print(f"\nsalvo: {OUT_DIR/'particao_top5.json'} (estrategia 3+2)")
    print(f"tempo total: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
