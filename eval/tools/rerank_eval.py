#!/usr/bin/env python3
"""F3 — reranker cross-encoder sobre pool ampliado (Memoria.md secao 3A).

Recupera o pool que o denso ja trouxe, reordena com um cross-encoder que le
consulta e trecho JUNTOS, e entrega os 5 melhores. Custo de API: zero — roda
offline sobre o rankings/<sistema>.json ja em disco.

ECONOMIA DE INFERENCIA. O score do cross-encoder para um par (consulta, trecho)
nao depende do tamanho do pool. Entao pontuamos o top-`--max-candidates` UMA vez
e derivamos todos os pools desse mesmo conjunto de scores, em vez de reprocessar
por pool. Sao 3x menos passagens pelo modelo.

O QUE O RERANKER NAO MUDA. Reordenar um pool nao acrescenta documento nenhum:
recall@100 fica identico ao da linha de base quando o pool e 100. O ganho
aparece em nDCG@5, P@5 e MRR — metricas de ORDEM. Isso e esperado, nao bug.

GEORISK. A media esconde regressao: um reranker pode subir o agregado e piorar
consultas que ja iam bem. O finalize() calcula GeoRisk sobre todas as variantes
salvas, que e a leitura pedida na Memoria.md secao 3A.

Uso:
  python eval/tools/rerank_eval.py --rankings eval/results_e2/retrieval/rankings/B_dense_bge_m3.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
EVAL = PROJECT_ROOT / "eval"
sys.path.insert(0, str(EVAL))
sys.path.insert(0, str(EVAL / "retrieval_eval"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from _common import (                                    # noqa: E402
    load_config, load_corpus, load_gold, resolve, setup_io, set_seed,
    save_scenario, finalize, out_dir, ks, metric_cols,
)
from lib import metrics as M                             # noqa: E402
from lib import stats as S                               # noqa: E402
from embedding_text import build_embedding_text          # noqa: E402

DEFAULT_MODEL = "BAAI/bge-reranker-large"


def texto_do_chunk(corpus, chunk_id, contextual=True) -> str:
    doc = corpus.get(chunk_id)
    if doc is None:
        return ""
    return build_embedding_text(doc) if contextual else (doc.get("text") or "")


def main() -> None:
    ap = argparse.ArgumentParser(description="F3 — reranker cross-encoder.")
    ap.add_argument("--rankings", required=True,
                    help="rankings/<sistema>.json produzido por um cenario denso.")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="Cross-encoder do HuggingFace.")
    ap.add_argument("--pools", default="20,50,100",
                    help="Tamanhos de pool a reordenar, separados por virgula.")
    ap.add_argument("--max-candidates", type=int, default=100,
                    help="Quantos candidatos pontuar por consulta (>= maior pool).")
    ap.add_argument("--out", default="results_f3", help="Saida, relativa a eval/.")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--scores",
                    help="JSON de scores pre-computados (Colab/GPU); "
                         "se dado, nenhum modelo e carregado aqui.")
    ap.add_argument("--raw-text", action="store_true",
                    help="Pontua o texto cru em vez do contextual (padrao: contextual).")
    args = ap.parse_args()

    pools = sorted({int(p) for p in args.pools.split(",") if p.strip()})
    if max(pools) > args.max_candidates:
        raise SystemExit(f"--max-candidates ({args.max_candidates}) < maior pool ({max(pools)}).")

    cfg = load_config()
    setup_io()
    set_seed(cfg.get("seed", 42))

    cfg_out = deepcopy(cfg)
    cfg_out["paths"]["results_dir"] = args.out

    corpus = load_corpus(resolve(cfg["paths"]["corpus"]))
    queries, qrels, meta = load_gold(cfg)
    pergunta = dict(queries)
    ndcg_k, recall_k, precision_k, f1_k, mrr_k, _ = ks(cfg)
    primary = f"ndcg@{ndcg_k[0]}"

    base_path = Path(args.rankings)
    base = json.loads(base_path.read_text(encoding="utf-8"))
    sistema = base_path.stem

    qids = [q for q in base if q in qrels]
    print(f"== F3 — reranker sobre {sistema} ==")
    print(f"modelo    : {args.model}")
    print(f"consultas : {len(qids)} com qrels (de {len(base)} no ranking)")
    print(f"pools     : {pools} | candidatos pontuados: {args.max_candidates}")
    print(f"texto     : {'cru' if args.raw_text else 'contextual (mesmo do indice)'}")
    print(f"saida     : {out_dir(cfg_out)}\n")

    pares, indice = [], []
    for qid in qids:
        for cid in base[qid][:args.max_candidates]:
            pares.append((pergunta[qid], texto_do_chunk(corpus, cid, not args.raw_text)))
            indice.append((qid, cid))

    por_query: dict[str, dict[str, float]] = {}

    if args.scores:
        # Pontuacao feita fora (Colab/GPU). Em CPU, um cross-encoder de 560M
        # sobre 7500 pares leva horas; na GPU sao minutos.
        bruto = json.loads(Path(args.scores).read_text(encoding="utf-8"))
        por_query = {q: {c: float(s) for c, s in d.items()}
                     for q, d in bruto.get("scores", bruto).items()}
        faltando = [(q, c) for q, c in indice if c not in por_query.get(q, {})]
        if faltando:
            raise SystemExit(
                f"Scores incompletos: {len(faltando)} pares sem pontuacao "
                f"(ex.: {faltando[:2]}). Regere com o mesmo rankings.json."
            )
        print(f"scores carregados de {args.scores} ({len(pares)} pares cobertos)")
    else:
        print(f"pontuando {len(pares)} pares (consulta x trecho)...")
        from sentence_transformers import CrossEncoder

        modelo = CrossEncoder(args.model, max_length=args.max_length)
        scores = modelo.predict(pares, batch_size=args.batch_size,
                                show_progress_bar=True)
        for (qid, cid), s in zip(indice, scores):
            por_query.setdefault(qid, {})[cid] = float(s)

    # Linha de base: o ranking denso como veio.
    variantes = {f"{sistema}_base": {q: base[q] for q in qids}}

    for pool in pools:
        novo = {}
        for qid in qids:
            cands = base[qid][:pool]
            resto = base[qid][pool:]
            reordenado = sorted(cands, key=lambda c: por_query[qid].get(c, -1e9),
                                reverse=True)
            novo[qid] = reordenado + resto
        variantes[f"{sistema}_rr{pool}"] = novo

    for nome, ranking in variantes.items():
        save_scenario(cfg_out, nome, ranking, qrels, meta, corpus)

    finalize(cfg_out, rewrite_csv=False)

    # -- significancia de cada pool contra a linha de base -----------------
    res = {n: M.evaluate_run(r, qrels, ndcg_k, recall_k, precision_k, f1_k, mrr_k)
           for n, r in variantes.items()}
    linha_base = f"{sistema}_base"
    xb = [res[linha_base]["per_query"][primary][q] for q in qids]
    sg = cfg["metrics"]["significance"]
    seed = cfg.get("seed", 42)

    linhas, pvals = [], []
    for pool in pools:
        nome = f"{sistema}_rr{pool}"
        xa = [res[nome]["per_query"][primary][q] for q in qids]
        delta, p = S.paired_randomization(xa, xb, n=sg["n_permutations"], seed=seed)
        _, lo, hi = S.bootstrap_mean_diff_ci(xa, xb, n_boot=sg["n_bootstrap"],
                                             alpha=sg["alpha"], seed=seed)
        piora = sum(1 for a, b in zip(xa, xb) if a < b - 1e-9)
        melhora = sum(1 for a, b in zip(xa, xb) if a > b + 1e-9)
        linhas.append({"pool": pool, "sistema": nome,
                       primary: res[nome]["mean"][primary], "delta": delta,
                       "ci_lo": lo, "ci_hi": hi, "p_value": p,
                       "consultas_melhoram": melhora, "consultas_pioram": piora})
        pvals.append(p)

    for linha, p_corr in zip(linhas, S.correct(pvals, sg.get("correction", "holm"))):
        linha["p_corrigido"] = p_corr

    od = out_dir(cfg_out)
    with (od / "rerank_ganho.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(linhas[0]))
        w.writeheader()
        for linha in linhas:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v)
                        for k, v in linha.items()})

    (od / "rerank_meta.json").write_text(json.dumps({
        "modelo": args.model, "sistema_base": sistema,
        "rankings_origem": str(base_path), "pools": pools,
        "max_candidates": args.max_candidates,
        "texto": "cru" if args.raw_text else "contextual",
        "max_length": args.max_length, "n_pares": len(pares),
        "n_queries": len(qids), "metrica": primary,
        "linha_base": res[linha_base]["mean"][primary],
        "ganhos": linhas,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    base_v = res[linha_base]["mean"][primary]
    print("\n" + "=" * 92)
    print(f"F3 — ganho sobre a linha de base ({primary} = {base_v:.4f})")
    print("=" * 92)
    print(f"{'pool':>6}{primary:>10}{'delta':>10}{'IC95%':>22}{'p':>9}{'p_holm':>9}"
          f"{'melhora':>9}{'piora':>7}")
    for linha in linhas:
        print(f"{linha['pool']:>6}{linha[primary]:>10.4f}{linha['delta']:>+10.4f}"
              f"   [{linha['ci_lo']:+.4f}, {linha['ci_hi']:+.4f}]"
              f"{linha['p_value']:>9.4f}{linha['p_corrigido']:>9.4f}"
              f"{linha['consultas_melhoram']:>9}{linha['consultas_pioram']:>7}")
    print("=" * 92)
    print(f"\nartefatos em {od}")


if __name__ == "__main__":
    main()
