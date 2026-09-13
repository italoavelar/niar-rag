#!/usr/bin/env python3
"""Decomposição de consulta + fusão RRF, para comparativas e multi-hop.

Memoria.md secao 3B + diagnostico por tipo (09/set/2026).

ATENCAO (13/set/2026): a premissa citada abaixo estava errada. "2/25" e o numero
de comparativas com TODAS as ancoras no top-5, nao o de comparativas sem nenhum
relevante no pool — no pool de 100 sao 20 a 22 de 25 que trazem ao menos uma
ancora. O problema das comparativas nao e de recall no pool; e de orcamento e de
anotacao. Ver Memoria.md secao 1.1.

Premissa original: comparativas tem
problema de RECALL (2/25 sem nenhum relevante no pool de 100) e reordenar nao
resolve isso — precisa de OUTRA busca. Multi-hop ja tem recall bom (0,88); a
decomposicao entra aqui so por curiosidade experimental (pedido explicito),
sem expectativa forte, ja que o problema diagnosticado ali e outro (o reranker
pointwise nao junta bem varios trechos do mesmo documento).

Pipeline, por consulta:
  1. Um LLM (qwen/qwen3.8-27b via Groq, o MESMO gerador de producao — pedido
     explicito) decompoe a pergunta original em 2 subperguntas independentes.
     Cache em disco: uma chamada por pergunta unica, nunca repete.
  2. Cada subpergunta e embutida com BGE-m3 (carrega o modelo localmente —
     download na 1a vez) e buscada contra o corpus inteiro.
  3. As duas listas sao fundidas por Reciprocal Rank Fusion (lib.retrievers.rrf,
     ja usada pelo Cenario C de producao).
  4. O pool fundido e reordenado pelo MESMO cross-encoder do F3
     (BAAI/bge-reranker-v2-m3), pontuando o texto da PERGUNTA ORIGINAL (nao as
     subperguntas) contra cada candidato — a decomposicao so amplia o que
     ENTRA no pool; a resposta final tem que ser relevante pela pergunta
     inteira.

FASE 1 (recall) roda sempre — rapida, sem cross-encoder, mostra se a fusao
sequer TRAZ mais relevante do que a busca original. Sinal barato antes de
pagar a Fase 2.
FASE 2 (rerank) e cara em CPU (cross-encoder por par); roda com --rerank.

Uso:
  python eval/tools/decompose_eval.py                 # so a fase 1 (recall)
  python eval/tools/decompose_eval.py --rerank         # fase 1 + fase 2 completa
"""

from __future__ import annotations

import argparse
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
from lib.llm_clients import LLMClient                               # noqa: E402
from lib.retrievers import rrf                                     # noqa: E402

OUT_DIR = EVAL / "results_decomp"
CACHE_FILE = OUT_DIR / "subqueries_cache.json"
RETRIEVE_TOP_N = 100     # top-N de CADA subconsulta antes da fusao — igual ao
                         # pool da busca original (100), para comparacao justa.
                         # Com 30, 4/6 "zeros novos" eram so a subconsulta
                         # (corretamente formada) nao alcancando o trecho raso
                         # o bastante — nao um defeito da decomposicao.
DIAG_POOL = 100          # pool usado no diagnostico de recall (fase 1)
RERANK_POOL = 30         # candidatos fundidos enviados ao cross-encoder (fase 2);
                         # menor que DIAG_POOL de proposito — reranking e caro em
                         # CPU (~1,9s/par) e o sweep anterior (F3) ja mostrou que
                         # pool 20 captura 97,7% do ganho do pool 50.
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
DECOMPOSE_MODEL = "qwen/qwen3.8-27b"   # mesmo gerador de producao (pedido do usuario)

_SYS_DECOMPOSE = (
    "Voce decompoe uma pergunta em DUAS subperguntas mais simples e "
    "independentes, cada uma respondivel por um unico trecho de texto "
    "normativo. As duas juntas devem cobrir tudo que a pergunta original "
    "pede. Preserve nomes proprios, siglas e referencias a normas/leis "
    "exatamente como aparecem no original. Nao invente fatos. Se a pergunta "
    "ja for simples (nao compara nem combina duas coisas), repita a "
    "pergunta original nos dois campos. Responda APENAS em JSON: "
    '{"sub1": "...", "sub2": "..."}'
)


def load_gold_subset():
    gold = [json.loads(l) for l in open(EVAL / "data/golden_qa.jsonl", encoding="utf-8")
            if l.strip()]
    alvo = [r for r in gold if r.get("qrels") and r.get("question_type") in
            ("comparative", "multi_hop")]
    qrels = {r["qid"]: {k: int(v) for k, v in r["qrels"].items()} for r in alvo}
    tipo = {r["qid"]: r["question_type"] for r in alvo}
    pergunta = {r["qid"]: r["question"] for r in alvo}
    return pergunta, qrels, tipo


def decompose_all(perguntas: dict) -> dict:
    """Decompoe cada pergunta unica, com cache em disco (1 chamada por pergunta)."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cache = {}
    if CACHE_FILE.exists():
        cache = json.loads(CACHE_FILE.read_text(encoding="utf-8"))

    faltando = {qid: q for qid, q in perguntas.items() if q not in cache}
    if faltando:
        client = LLMClient(provider="groq", model=DECOMPOSE_MODEL, temperature=0,
                           max_tokens=512)
        print(f"decompondo {len(faltando)} perguntas com {DECOMPOSE_MODEL}...")
        for i, (qid, q) in enumerate(faltando.items(), 1):
            try:
                r = client.chat_json(q, system=_SYS_DECOMPOSE)
                cache[q] = {"sub1": r.get("sub1") or q, "sub2": r.get("sub2") or q}
            except Exception as e:
                print(f"  [{qid}] falhou ({e}); usando a pergunta original nos dois campos")
                cache[q] = {"sub1": q, "sub2": q}
            if i % 10 == 0 or i == len(faltando):
                print(f"  {i}/{len(faltando)}")
        CACHE_FILE.write_text(json.dumps(cache, ensure_ascii=False, indent=2),
                              encoding="utf-8")
    else:
        print("decomposicao: 100% do cache")

    return {qid: cache[q] for qid, q in perguntas.items()}


def build_bge_embedder(cfg):
    from lib.embedders import build_embedder
    return build_embedder("bge_m3", cfg)


def load_doc_matrix(corpus, fp):
    path = EVAL / "results/indexes" / bge_cache_filename("bge_m3", fp)
    npz = np.load(path, allow_pickle=True)
    matrix = npz["matrix"].astype(np.float32)   # 1 acesso so — .npz comprimido
    ids = [str(i) for i in npz["ids"]]
    assert ids == list(corpus.keys()), "ordem dos ids nao confere com o corpus"
    return matrix, ids


def dense_search(qvec, matrix, ids, top_n):
    scores = matrix @ qvec.astype(np.float32)
    idx = np.argpartition(-scores, top_n - 1)[:top_n]
    idx = idx[np.argsort(-scores[idx])]
    return [ids[i] for i in idx]


def main() -> None:
    ap = argparse.ArgumentParser(description="Decomposicao + RRF para comparativas/multi-hop.")
    ap.add_argument("--rerank", action="store_true",
                    help="Roda tambem a fase 2 (cross-encoder). Sem isso, so a fase 1 (recall).")
    ap.add_argument("--rerank-batch-size", type=int, default=32)
    args = ap.parse_args()

    cfg = load_config()
    setup_io()
    t0 = time.time()

    corpus = load_corpus(resolve(cfg["paths"]["corpus"]))
    fp = corpus_embedding_fingerprint(corpus.values())
    matrix, ids = load_doc_matrix(corpus, fp)
    print(f"[{time.time()-t0:5.1f}s] corpus + matriz carregados ({len(ids)} chunks)")

    perguntas, qrels, tipo = load_gold_subset()
    print(f"[{time.time()-t0:5.1f}s] {len(perguntas)} perguntas "
          f"({sum(1 for t in tipo.values() if t=='comparative')} comparativas, "
          f"{sum(1 for t in tipo.values() if t=='multi_hop')} multi-hop)")

    subqs = decompose_all(perguntas)
    print(f"[{time.time()-t0:5.1f}s] decomposicao pronta")

    embedder = build_bge_embedder(cfg)
    print(f"[{time.time()-t0:5.1f}s] BGE-m3 carregado")

    baseline = json.loads(
        (EVAL / "results_e2/retrieval/rankings/B_dense_bge_m3.json").read_text(encoding="utf-8"))

    fused = {}
    for qid in perguntas:
        v1 = embedder.embed_query(subqs[qid]["sub1"])
        v2 = embedder.embed_query(subqs[qid]["sub2"])
        r1 = {qid: dense_search(v1, matrix, ids, RETRIEVE_TOP_N)}
        r2 = {qid: dense_search(v2, matrix, ids, RETRIEVE_TOP_N)}
        fundida = rrf({"sub1": r1, "sub2": r2})
        fused[qid] = fundida[qid]
    print(f"[{time.time()-t0:5.1f}s] fusao concluida para {len(fused)} consultas")

    # -- Fase 1: recall, sem reranker (sinal barato) -------------------------
    print("\n=== Fase 1 — recall (fundido vs busca original), pool comparavel ===")
    print(f"{'tipo':<13}{'n':>4}{'recall@orig':>13}{'recall@fundido':>16}")
    for t in ("comparative", "multi_hop"):
        qs = [q for q in perguntas if tipo[q] == t]
        r_orig = np.mean([M.recall_at_k(baseline[q], qrels[q], DIAG_POOL) for q in qs])
        r_fund = np.mean([M.recall_at_k(fused[q], qrels[q], DIAG_POOL) for q in qs])
        print(f"{t:<13}{len(qs):>4}{r_orig:>13.4f}{r_fund:>16.4f}")
    zero_orig = {t: sum(1 for q in perguntas if tipo[q]==t and
                        M.recall_at_k(baseline[q], qrels[q], DIAG_POOL)==0) for t in ("comparative","multi_hop")}
    zero_fund = {t: sum(1 for q in perguntas if tipo[q]==t and
                        M.recall_at_k(fused[q][:DIAG_POOL], qrels[q], DIAG_POOL)==0) for t in ("comparative","multi_hop")}
    print(f"\nconsultas com ZERO relevante no pool: original={zero_orig}  fundido={zero_fund}")

    (OUT_DIR / "fused_rankings.json").write_text(
        json.dumps(fused, ensure_ascii=False), encoding="utf-8")
    print(f"\nsalvo: {OUT_DIR / 'fused_rankings.json'}")

    if not args.rerank:
        print("\n(rode com --rerank para a fase 2: reordenar com o cross-encoder e medir nDCG@5)")
        return

    # -- Fase 2: reordena o pool fundido com a pergunta ORIGINAL --------------
    print(f"\n=== Fase 2 — reordenando pool fundido (top-{RERANK_POOL} por consulta) ===")
    from sentence_transformers import CrossEncoder
    ce = CrossEncoder(RERANK_MODEL, max_length=384)

    pares, indice = [], []
    for qid, texto in perguntas.items():
        for cid in fused[qid][:RERANK_POOL]:
            pares.append((texto, build_embedding_text(corpus[cid])))
            indice.append((qid, cid))
    print(f"pontuando {len(pares)} pares...")
    scores = ce.predict(pares, batch_size=args.rerank_batch_size, show_progress_bar=True)

    por_q: dict[str, dict[str, float]] = {}
    for (qid, cid), s in zip(indice, scores):
        por_q.setdefault(qid, {})[cid] = float(s)

    top5 = {qid: sorted(fused[qid][:RERANK_POOL], key=lambda c: por_q[qid].get(c, -1e9),
                        reverse=True)[:5]
            for qid in perguntas}

    (OUT_DIR / "decomposed_rr_top5.json").write_text(
        json.dumps(top5, ensure_ascii=False), encoding="utf-8")

    print("\n=== resultado final: decomposicao+fusao+rerank vs BGE+rr50 (baseline atual) ===")
    rr50 = json.loads(Path("eval/results_f3/retrieval/rankings/B_dense_bge_m3_rr50.json")
                      .read_text(encoding="utf-8"))
    print(f"{'tipo':<13}{'n':>4}{'baseline(rr50)':>16}{'decomp+fusao+rr':>18}{'delta':>9}"
          f"{'p':>8}")
    linhas = []
    for t in ("comparative", "multi_hop"):
        qs = [q for q in perguntas if tipo[q] == t]
        xb = [M.ndcg_at_k(rr50[q], qrels[q], 5) for q in qs]
        xa = [M.ndcg_at_k(top5[q], qrels[q], 5) for q in qs]
        d, p = S.paired_randomization(xa, xb, n=10000, seed=42)
        _, lo, hi = S.bootstrap_mean_diff_ci(xa, xb, n_boot=10000, alpha=0.05, seed=42)
        melhora = sum(1 for a, b in zip(xa, xb) if a > b + 1e-9)
        piora = sum(1 for a, b in zip(xa, xb) if a < b - 1e-9)
        print(f"{t:<13}{len(qs):>4}{np.mean(xb):>16.4f}{np.mean(xa):>18.4f}{d:>+9.4f}{p:>8.4f}")
        linhas.append({"tipo": t, "n": len(qs), "baseline_rr50": float(np.mean(xb)),
                       "decomp_fusao_rr": float(np.mean(xa)), "delta": d,
                       "ci95": [lo, hi], "p_value": p, "melhoram": melhora, "pioram": piora})

    (OUT_DIR / "resultado_final.json").write_text(
        json.dumps(linhas, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nsalvo: {OUT_DIR / 'resultado_final.json'}")
    print(f"\ntempo total: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
