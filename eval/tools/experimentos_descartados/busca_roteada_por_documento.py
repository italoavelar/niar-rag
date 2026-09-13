#!/usr/bin/env python3
"""Teto da BUSCA ROTEADA: rankear o chunk DENTRO do documento, não no corpus todo.

Diagnóstico que motiva (09/set/2026): numa comparativa o gargalo não é a
ordenação nem a fusão — é a taxa de acerto POR DOCUMENTO. Dos 53 documentos
exigidos pelas 25 comparativas, só 20 têm seu chunk relevante no top-5 da
própria sub-pergunta dedicada (`eval/tools/particao_por_lado.py`). Como a
comparativa exige conjunção de dois acertos, a taxa conjunta é ~o quadrado da
taxa por documento — daí os 2/25 que TODAS as configurações produzem.

HIPOTESE REFUTADA (13/set/2026). Medido: dentro do PROPRIO documento, a ancora de
uma comparativa fica em posicao mediana 8, e a de uma multi-hop em 13. Restringir
ao documento certo tira a comparativa de 23 para 8 no ranking — ajuda, mas nao
chega ao top-3. O teto de 6/25 com oraculo que este script mede e consequencia
disso, nao do orcamento global. O texto abaixo fica como registro da hipotese.

Hipótese: o trecho certo não está mal ranqueado *dentro do seu documento*; ele
está soterrado por 5.160 chunks de competição global. Se o sistema primeiro
ROTEIA (identifica os 2 documentos comparados) e só então busca DENTRO de cada
um, o espaço de competição cai de 5.160 para ~30-300 chunks e o mesmo embedding
passa a acertar.

Este script mede o TETO dessa ideia: assume roteamento perfeito (usa os
documentos do gabarito) e mede em que posição o chunk relevante fica quando a
busca é restrita àquele documento. Se as posições forem 1-3, a estratégia vale;
se continuarem ruins, o problema é do embedding e roteamento não salva.

O roteamento real (sem gabarito) é o passo seguinte, e é barato: as comparativas
nomeiam as normas no texto da pergunta.

Uso:
  python eval/tools/busca_roteada_por_documento.py
"""

from __future__ import annotations

import json
import re
import sys
import time
from collections import defaultdict
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


from lib.metrics import document_of as doc_of  # noqa: E402  (uma implementação só)


def main() -> None:
    cfg = load_config()
    setup_io()
    t0 = time.time()

    corpus = load_corpus(resolve(cfg["paths"]["corpus"]))
    fp = corpus_embedding_fingerprint(corpus.values())
    npz = np.load(EVAL / "results/indexes" / bge_cache_filename("bge_m3", fp), allow_pickle=True)
    matrix = npz["matrix"].astype(np.float32)
    ids = [str(i) for i in npz["ids"]]
    idx_of = {c: i for i, c in enumerate(ids)}

    # indices de chunk por documento
    chunks_por_doc = defaultdict(list)
    for i, c in enumerate(ids):
        chunks_por_doc[doc_of(c)].append(i)
    print(f"[{time.time()-t0:5.1f}s] corpus ({len(ids)} chunks, {len(chunks_por_doc)} documentos)")

    gold = [json.loads(l) for l in open(EVAL / "data/golden_qa.jsonl", encoding="utf-8") if l.strip()]
    alvo = [r for r in gold if r.get("qrels") and r.get("question_type") == "comparative"]
    qrels = {r["qid"]: {k: int(v) for k, v in r["qrels"].items()} for r in alvo}
    perguntas = {r["qid"]: r["question"] for r in alvo}
    req_docs = {r["qid"]: sorted({doc_of(c) for c in r["qrels"]}) for r in alvo}

    subqs_cache = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
    subqs = {qid: subqs_cache[q] for qid, q in perguntas.items()}

    from lib.embedders import build_embedder
    embedder = build_embedder("bge_m3", cfg)
    print(f"[{time.time()-t0:5.1f}s] BGE-m3 carregado")

    vec = {}
    for qid, texto in perguntas.items():
        vec[qid] = {
            "orig": embedder.embed_query(texto).astype(np.float32),
            "sub1": embedder.embed_query(subqs[qid]["sub1"]).astype(np.float32),
            "sub2": embedder.embed_query(subqs[qid]["sub2"]).astype(np.float32),
        }
    print(f"[{time.time()-t0:5.1f}s] queries embutidas")

    print("\n=== Posicao do chunk relevante DENTRO do proprio documento (roteamento perfeito) ===")
    print(f"{'qid':<8}{'documento':<40}{'#chunks':>8}{'global':>8}{'no doc':>8}")
    dentro_top1, dentro_top2, dentro_top3, total = 0, 0, 0, 0
    posicoes = []
    detalhe = {}
    for qid in sorted(perguntas):
        for d in req_docs[qid]:
            rel_d = {c for c in qrels[qid] if doc_of(c) == d}
            cand_idx = chunks_por_doc.get(d, [])
            if not cand_idx:
                print(f"{qid:<8}{d[:38]:<40}{'--':>8}{'--':>8}{'DOC AUSENTE':>8}")
                continue
            sub = matrix[cand_idx]
            # melhor das tres consultas (orig / sub1 / sub2), simulando que o
            # roteador manda a sub-pergunta certa para o documento certo
            melhor_pos = None
            for chave in ("orig", "sub1", "sub2"):
                sc = sub @ vec[qid][chave]
                ordem = np.argsort(-sc)
                for rank, j in enumerate(ordem, start=1):
                    if ids[cand_idx[j]] in rel_d:
                        melhor_pos = rank if melhor_pos is None else min(melhor_pos, rank)
                        break
            # posicao global, para comparacao (busca original no corpus inteiro)
            sc_glob = matrix @ vec[qid]["orig"]
            ordem_glob = np.argsort(-sc_glob)
            pos_glob = None
            for rank, j in enumerate(ordem_glob, start=1):
                if ids[j] in rel_d:
                    pos_glob = rank
                    break
            total += 1
            posicoes.append(melhor_pos)
            detalhe[f"{qid}|{d}"] = {"n_chunks": len(cand_idx), "pos_global": pos_glob,
                                     "pos_no_doc": melhor_pos}
            if melhor_pos == 1:
                dentro_top1 += 1
            if melhor_pos <= 2:
                dentro_top2 += 1
            if melhor_pos <= 3:
                dentro_top3 += 1
            print(f"{qid:<8}{d[:38]:<40}{len(cand_idx):>8}{pos_glob if pos_glob else '-':>8}{melhor_pos:>8}")

    print(f"\ndocumentos exigidos avaliados: {total}")
    print(f"  chunk relevante em 1o lugar DENTRO do documento : {dentro_top1}/{total} ({100*dentro_top1/total:.0f}%)")
    print(f"  chunk relevante no top-2 dentro do documento    : {dentro_top2}/{total} ({100*dentro_top2/total:.0f}%)")
    print(f"  chunk relevante no top-3 dentro do documento    : {dentro_top3}/{total} ({100*dentro_top3/total:.0f}%)")
    print(f"  posicao mediana dentro do documento             : {int(np.median(posicoes))}")

    # ── simulacao do top-5 final: 2 chunks de cada documento exigido ────────
    print("\n=== Top-5 montado por roteamento (2 chunks por documento exigido) ===")
    qs = sorted(perguntas)
    tops = {}
    for qid in qs:
        escolhidos = []
        docs = req_docs[qid]
        por_doc = max(1, 5 // max(1, len(docs)))
        for d in docs:
            cand_idx = chunks_por_doc.get(d, [])
            if not cand_idx:
                continue
            sub = matrix[cand_idx]
            # usa a melhor das consultas por documento (proxy do roteador)
            melhor = None
            for chave in ("orig", "sub1", "sub2"):
                sc = sub @ vec[qid][chave]
                ordem = np.argsort(-sc)[:por_doc]
                cids = [ids[cand_idx[j]] for j in ordem]
                acertos = len(set(cids) & set(qrels[qid].keys()))
                if melhor is None or acertos > melhor[0]:
                    melhor = (acertos, cids)
            escolhidos.extend(melhor[1])
        tops[qid] = escolhidos[:5]

    def cobertura(top5, qid):
        rel_in = set(top5) & set(qrels[qid].keys())
        return {doc_of(c) for c in rel_in}

    full = sum(1 for q in qs if cobertura(tops[q], q) == set(req_docs[q]))
    any_ = sum(1 for q in qs if cobertura(tops[q], q))
    nd = float(np.mean([M.ndcg_at_k(tops[q], qrels[q], 5) for q in qs]))
    print(f"  AMBOS os documentos no top-5: {full}/{len(qs)}")
    print(f"  pelo menos 1 documento      : {any_}/{len(qs)}")
    print(f"  nDCG@5                      : {nd:.4f}")

    (OUT_DIR / "roteamento_diagnostico.json").write_text(
        json.dumps(detalhe, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_DIR / "roteamento_top5.json").write_text(
        json.dumps(tops, ensure_ascii=False), encoding="utf-8")
    print(f"\nsalvo: {OUT_DIR/'roteamento_diagnostico.json'} e roteamento_top5.json")
    print(f"tempo total: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
