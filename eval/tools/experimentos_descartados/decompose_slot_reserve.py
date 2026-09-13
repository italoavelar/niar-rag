#!/usr/bin/env python3
"""Decomposição + RRF + reranker, com RESERVA DE VAGA POR LADO no top-5 final.

Diagnóstico que motiva este script (09/set/2026): sob a métrica de "ambos os
documentos comparados aparecem no top-5" (a que importa de verdade para uma
comparativa — ver golden_qa.jsonl question_type=comparative, qrels tipicamente
com 2 documentos-fonte distintos), o pipeline `decompose_eval.py` cobre só
5/25. Rastreando ONDE o documento que falta estava no pool fundido (RRF) de
cada consulta: em 19 dos 26 casos de documento-faltante, o candidato JÁ ESTAVA
dentro dos top-30 que o cross-encoder pontuou — às vezes na 1ª, 2ª ou 3ª
posição do pool fundido — e mesmo assim não entrou no top-5 final, porque a
seleção original é um sort global por score do cross-encoder contra a
pergunta INTEIRA, que tende a favorecer um lado da comparação inteiro.
Só 7/26 casos são de pool pequeno demais (candidato existia, mas além da
janela de 30 que o cross-encoder vê). Zero casos de documento ausente do pool
fundido inteiro (a fusão RRF já resolve o recall: 25/25 comparativas têm
ambos os documentos em algum lugar do pool fundido).

Este script isola a mudança: MESMO pool (30, igual ao decompose_eval.py),
MESMO cross-encoder, MESMA decomposição (reaproveita o cache) — muda só a
montagem do top-5 final:
  1. Reserva a vaga do melhor candidato de CADA subconsulta (sub1 e sub2),
     pelo score do cross-encoder, entre os candidatos que aquela subconsulta
     trouxe no seu próprio top-100 (não o pool fundido inteiro — queremos o
     melhor candidato QUE AQUELE LADO endossa).
  2. As vagas restantes (3, ou 4 se as duas reservas coincidirem no mesmo
     chunk) são preenchidas pelos próximos melhores por score, no pool
     fundido inteiro, sem essa restrição.
  3. As 5 escolhidas são então ordenadas entre si por score do cross-encoder
     (a reserva garante QUEM entra, não a posição).

Custo: mesmo de eval/tools/decompose_eval.py --rerank (pool=30, cross-encoder
BAAI/bge-reranker-v2-m3 em CPU) — não paga API nem GPU nova, só CPU.

Uso:
  python eval/tools/decompose_slot_reserve.py
"""

from __future__ import annotations

import json
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
    corpus_embedding_fingerprint, bge_cache_filename, build_embedding_text,
)
from lib import metrics as M                                       # noqa: E402
from lib import stats as S                                         # noqa: E402
from lib.retrievers import rrf                                     # noqa: E402

OUT_DIR = EVAL / "results_decomp"
CACHE_FILE = OUT_DIR / "subqueries_cache.json"
RETRIEVE_TOP_N = 100     # igual ao decompose_eval.py — top-N de CADA subconsulta
RERANK_POOL = 30         # igual ao decompose_eval.py — isola o efeito da SELEÇÃO,
                         # não do tamanho do pool (esse é o próximo experimento)
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"


def load_gold_comparative():
    gold = [json.loads(l) for l in open(EVAL / "data/golden_qa.jsonl", encoding="utf-8")
            if l.strip()]
    alvo = [r for r in gold if r.get("qrels") and r.get("question_type") == "comparative"]
    qrels = {r["qid"]: {k: int(v) for k, v in r["qrels"].items()} for r in alvo}
    pergunta = {r["qid"]: r["question"] for r in alvo}
    return pergunta, qrels


def build_bge_embedder(cfg):
    from lib.embedders import build_embedder
    return build_embedder("bge_m3", cfg)


def load_doc_matrix(corpus, fp):
    path = EVAL / "results/indexes" / bge_cache_filename("bge_m3", fp)
    npz = np.load(path, allow_pickle=True)
    matrix = npz["matrix"].astype(np.float32)
    ids = [str(i) for i in npz["ids"]]
    assert ids == list(corpus.keys()), "ordem dos ids nao confere com o corpus"
    return matrix, ids


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
    matrix, ids = load_doc_matrix(corpus, fp)
    print(f"[{time.time()-t0:5.1f}s] corpus + matriz carregados ({len(ids)} chunks)")

    perguntas, qrels = load_gold_comparative()
    print(f"[{time.time()-t0:5.1f}s] {len(perguntas)} comparativas")

    subqs_cache = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
    subqs = {qid: subqs_cache[q] for qid, q in perguntas.items()}
    faltando = [qid for qid in perguntas if perguntas[qid] not in subqs_cache]
    if faltando:
        raise SystemExit(f"faltam subconsultas em cache para: {faltando} "
                          f"(rode decompose_eval.py primeiro para gerar o cache)")

    embedder = build_bge_embedder(cfg)
    print(f"[{time.time()-t0:5.1f}s] BGE-m3 carregado")

    fused, side1_set, side2_set = {}, {}, {}
    for qid in perguntas:
        v1 = embedder.embed_query(subqs[qid]["sub1"])
        v2 = embedder.embed_query(subqs[qid]["sub2"])
        r1 = dense_search(v1, matrix, ids, RETRIEVE_TOP_N)
        r2 = dense_search(v2, matrix, ids, RETRIEVE_TOP_N)
        side1_set[qid] = set(r1)
        side2_set[qid] = set(r2)
        fundida = rrf({"sub1": {qid: r1}, "sub2": {qid: r2}})
        fused[qid] = fundida[qid]
    print(f"[{time.time()-t0:5.1f}s] fusao concluida para {len(fused)} consultas")

    print(f"\n=== Reordenando pool fundido (top-{RERANK_POOL} por consulta) ===")
    from sentence_transformers import CrossEncoder
    ce = CrossEncoder(RERANK_MODEL, max_length=384)

    pares, indice = [], []
    for qid, texto in perguntas.items():
        for cid in fused[qid][:RERANK_POOL]:
            pares.append((texto, build_embedding_text(corpus[cid])))
            indice.append((qid, cid))
    print(f"pontuando {len(pares)} pares...")
    scores = ce.predict(pares, batch_size=32, show_progress_bar=True)

    por_q: dict[str, dict[str, float]] = {}
    for (qid, cid), s in zip(indice, scores):
        por_q.setdefault(qid, {})[cid] = float(s)

    def top5_com_reserva(qid):
        pool = fused[qid][:RERANK_POOL]
        sc = por_q[qid]
        ranked_pool = sorted(pool, key=lambda c: sc[c], reverse=True)

        cand1 = [c for c in ranked_pool if c in side1_set[qid]]
        cand2 = [c for c in ranked_pool if c in side2_set[qid]]
        reservado = []
        if cand1:
            reservado.append(cand1[0])
        if cand2 and cand2[0] not in reservado:
            reservado.append(cand2[0])

        restante = [c for c in ranked_pool if c not in reservado]
        escolhidos = reservado + restante[: max(0, 5 - len(reservado))]
        escolhidos = escolhidos[:5]
        return sorted(escolhidos, key=lambda c: sc[c], reverse=True)

    top5_reservado = {qid: top5_com_reserva(qid) for qid in perguntas}
    (OUT_DIR / "decomposed_rr_top5_reserva.json").write_text(
        json.dumps(top5_reservado, ensure_ascii=False), encoding="utf-8")

    # também recomputa o top5 "sem reserva" com ESTE MESMO pool/scores, pra
    # comparar maçã com maçã (isola só o efeito da selecao, nao reintroduz
    # variancia de reamostrar o cross-encoder)
    top5_livre = {qid: sorted(fused[qid][:RERANK_POOL], key=lambda c: por_q[qid].get(c, -1e9),
                              reverse=True)[:5]
                  for qid in perguntas}

    print(f"\n[{time.time()-t0:5.1f}s] salvo: {OUT_DIR / 'decomposed_rr_top5_reserva.json'}")

    from lib.metrics import document_of as doc_of  # uma implementação só

    def req_docs(qid):
        return sorted({doc_of(c) for c in qrels[qid]})

    def full_cov(top5, qid):
        top_docs = {doc_of(c) for c in top5}
        return all(d in top_docs for d in req_docs(qid))

    def any_cov(top5, qid):
        top_docs = {doc_of(c) for c in top5}
        return any(d in top_docs for d in req_docs(qid))

    print("\n=== RESULTADO: reserva de vaga por lado vs selecao livre (MESMO pool/scores) ===")
    for nome, top5s in [("livre (baseline, igual ao decompose_eval.py)", top5_livre),
                        ("COM RESERVA por lado", top5_reservado)]:
        n = len(perguntas)
        full = sum(1 for qid in perguntas if full_cov(top5s[qid], qid))
        any_ = sum(1 for qid in perguntas if any_cov(top5s[qid], qid))
        ndcg = np.mean([M.ndcg_at_k(top5s[qid], qrels[qid], 5) for qid in perguntas])
        print(f"  {nome}")
        print(f"    ambos os documentos no top-5 : {full}/{n}")
        print(f"    pelo menos 1 documento no top-5: {any_}/{n}")
        print(f"    nDCG@5 medio: {ndcg:.4f}")

    xb = [M.ndcg_at_k(top5_livre[qid], qrels[qid], 5) for qid in perguntas]
    xa = [M.ndcg_at_k(top5_reservado[qid], qrels[qid], 5) for qid in perguntas]
    d, p = S.paired_randomization(xa, xb, n=10000, seed=42)
    _, lo, hi = S.bootstrap_mean_diff_ci(xa, xb, n_boot=10000, alpha=0.05, seed=42)
    melhora = sum(1 for a, b in zip(xa, xb) if a > b + 1e-9)
    piora = sum(1 for a, b in zip(xa, xb) if a < b - 1e-9)
    print(f"\n  delta nDCG@5 (reserva - livre) = {d:+.4f}  IC95%=[{lo:+.4f}; {hi:+.4f}]  "
          f"p={p:.4f}  (melhoram={melhora} pioram={piora} n={len(perguntas)})")

    print(f"\ntempo total: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
