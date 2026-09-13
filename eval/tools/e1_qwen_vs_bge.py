#!/usr/bin/env python3
"""E1 — Qwen3-Embedding-0.6B vs BAAI/bge-m3, apenas o cenário DENSO.

Memoria.md seção 6 (fase F7): mesmo protocolo, mesmo gold set, nDCG@5 primário,
mais o TETO DE REORDENAÇÃO de cada um — sem o teto, um modelo com nDCG@5 pior
mas pool melhor (o melhor candidato a reranker) seria descartado por engano.

Reusa eval/retrieval_eval/_common.py, então as planilhas saem no MESMO formato
dos demais cenários:

  scenario_B_dense_qwen3_0_6b.csv   por consulta: top-5 recuperado (id, qrel,
  scenario_B_dense_bge_m3.csv       trecho), métricas da consulta, tipo, língua
                                    e a linha MÉDIA ao final
  metrics.csv / metrics.json        uma linha por sistema, com GeoRisk
  per_query_ndcg.json               nDCG@5 por consulta
  rankings.json / rankings/*.json   top-100 recuperado

Mais três artefatos específicos da E1:

  comparacao_dense_qwen3_0_6b_vs_bge_m3.csv   head-to-head + vencedor
  e1_rerank_ceiling.csv                        teto @20/50/100 e a margem
  e1_significancia.json                        randomização pareada + IC bootstrap

ONDE ESCREVE. Em eval/results_e1/retrieval/, não em eval/results/retrieval/. Os
rankings que estão lá são do corpus PRÉ-correção — 5,5% dos chunks daqueles
rankings não existem mais no corpus atual — e o finalize() do pipeline agrega
tudo que encontra em rankings/. Misturar produziria uma metrics.csv com linhas
incomparáveis entre si. Nada do que já existe é sobrescrito.

Uso:
  # depois do Colab, sem carregar modelo nenhum (usa os .npz de consulta):
  python eval/tools/e1_qwen_vs_bge.py --offline

  # carregando os modelos localmente (baixa os pesos):
  python eval/tools/e1_qwen_vs_bge.py

No Windows, rode com PYTHONIOENCODING=utf-8 — o pipeline imprime caracteres que
o console cp1252 não aceita.
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

from _common import (                                    # noqa: E402
    load_config, load_corpus, load_gold, resolve, setup_io, set_seed,
    save_scenario, finalize, out_dir, indexes_dir, ks, metric_cols,
)
from lib import metrics as M                             # noqa: E402
from lib import stats as S                               # noqa: E402
from lib.embedders import build_embedder                 # noqa: E402
from lib.retrievers import DenseRetriever                # noqa: E402

SYSTEMS = ["qwen3_0_6b", "bge_m3"]
ROTULO = {"qwen3_0_6b": "Qwen3-0.6B", "bge_m3": "BGE-M3"}
CEILING_POOLS = (20, 50, 100)


def rerank_ceiling(ranked, rel, pool: int, k: int = 5) -> float:
    """Melhor nDCG@k alcançável reordenando perfeitamente o top-`pool` já trazido."""
    cand = sorted(ranked[:pool], key=lambda c: rel.get(c, 0), reverse=True)
    return M.ndcg_at_k(cand, rel, k)


def checar_prompt(name: str, embedder, cfg) -> None:
    """O Qwen sem prompt_name na consulta sai subestimado — falha alto e cedo."""
    esperado = cfg["embedders"][name].get("query_prompt_name") or None
    obtido = getattr(embedder, "query_prompt_name", None) or None
    if obtido == "None":
        obtido = None
    if esperado != obtido:
        raise SystemExit(
            f"[{name}] query_prompt_name={obtido!r}, esperado {esperado!r}.\n"
            "  Vetores de consulta com o prompt errado viciam a E1.\n"
            "  Regere o .npz de consulta com eval/colab_leme_completo.py."
        )
    if esperado:
        print(f"  [{name}] prompt de consulta: {esperado!r}")


def filtrar_queries(name: str, queries, embedder, qrels):
    """Restringe as consultas as que tem vetor pre-computado.

    O Colab embutiu o queries.csv (as 75 respondiveis); o gold set tem 100,
    sendo 25 fora-de-escopo, que existem para a Etapa 03 testar a RECUSA. Elas
    nao entram em metrica nenhuma — evaluate_run itera sobre os qrels —, entao
    pular as sem vetor nao altera resultado algum.

    Se faltar vetor de uma consulta COM qrel, aborta: um ranking vazio contaria
    como nDCG 0 e rebaixaria o modelo silenciosamente.
    """
    if not hasattr(embedder, "has_query"):
        return queries
    tem = [(qid, txt) for qid, txt in queries if embedder.has_query(txt)]
    faltam = [qid for qid, txt in queries if not embedder.has_query(txt)]
    criticas = [qid for qid in faltam if qid in qrels]
    if criticas:
        raise SystemExit(
            f"[{name}] {len(criticas)} consultas COM qrel sem vetor pre-computado "
            f"(ex.: {criticas[:3]}). Isso zeraria o nDCG delas. Regere "
            "o .npz de consulta cobrindo o gold set inteiro."
        )
    if faltam:
        print(f"  [{name}] {len(faltam)} consultas fora-de-escopo sem vetor: nao "
              "recuperadas (nao entram em metrica)")
    return tem


def main() -> None:
    ap = argparse.ArgumentParser(description="E1 — Qwen3-0.6B vs BGE-m3 (denso).")
    ap.add_argument("--offline", action="store_true",
                    help="Usa os .npz de consulta do Colab; não carrega modelos.")
    ap.add_argument("--rebuild", action="store_true",
                    help="Ignora o cache de documentos e re-embute o corpus.")
    ap.add_argument("--out", default="results_e1",
                    help="Diretório de saída, relativo a eval/ (padrão: results_e1).")
    args = ap.parse_args()

    cfg = load_config()
    setup_io()
    set_seed(cfg.get("seed", 42))

    if args.offline:
        for name in SYSTEMS:
            cfg["embedders"][name]["offline"] = True

    # Saída isolada: preserva eval/results/ intacto.
    cfg_out = deepcopy(cfg)
    cfg_out["paths"]["results_dir"] = args.out

    corpus = load_corpus(resolve(cfg["paths"]["corpus"]))
    queries, qrels, meta = load_gold(cfg)
    ndcg_k, recall_k, precision_k, f1_k, mrr_k, top_k = ks(cfg)
    primary = f"ndcg@{ndcg_k[0]}"
    mcols = metric_cols(cfg)

    print("== E1 - Qwen3-Embedding-0.6B vs BAAI/bge-m3 (cenario denso) ==")
    print(f"corpus    : {len(corpus)} chunks")
    print(f"gold set  : {len(meta)} consultas ({len(qrels)} respondiveis)")
    print(f"top_k     : {top_k} | metrica primaria: {primary}")
    print(f"saida     : {out_dir(cfg_out)}")
    print(f"modo      : {'offline (vetores prontos)' if args.offline else 'carregando modelos'}\n")

    rankings, results = {}, {}
    for name in SYSTEMS:
        embedder = build_embedder(name, cfg)
        checar_prompt(name, embedder, cfg)
        retriever = DenseRetriever(corpus, embedder,
                                   cache_dir=indexes_dir(cfg),   # cache real
                                   rebuild=args.rebuild)
        rankings[name] = retriever.run_queries(
            filtrar_queries(name, queries, embedder, qrels), top_k)
        save_scenario(cfg_out, f"B_dense_{name}", rankings[name], qrels, meta, corpus)
        results[name] = M.evaluate_run(rankings[name], qrels, ndcg_k, recall_k,
                                       precision_k, f1_k, mrr_k)

    # Artefatos combinados no formato do pipeline (inclui GeoRisk).
    finalize(cfg_out, rewrite_csv=False)

    od = out_dir(cfg_out)
    qids = sorted(qrels)

    # -- head-to-head, no formato de comparacao_dense_gemini_bge_m3.csv -----
    melhor = max(SYSTEMS, key=lambda n: results[n]["mean"][primary])
    comp = od / f"comparacao_dense_{SYSTEMS[0]}_vs_{SYSTEMS[1]}.csv"
    with comp.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["modelo"] + mcols + [f"vencedor_{primary.replace('@', '')}"])
        for name in SYSTEMS:
            mean = results[name]["mean"]
            w.writerow([ROTULO[name]] + [f"{mean[c]:.4f}" for c in mcols]
                       + [name == melhor])

    # -- teto de reordenacao -----------------------------------------------
    tetos = {
        name: {pool: float(np.mean([rerank_ceiling(rankings[name][q], qrels[q], pool)
                                    for q in qids]))
               for pool in CEILING_POOLS}
        for name in SYSTEMS
    }
    with (od / "e1_rerank_ceiling.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["sistema", "modelo", primary]
                   + [f"teto@{p}" for p in CEILING_POOLS] + ["margem@20"])
        for name in SYSTEMS:
            atual = results[name]["mean"][primary]
            w.writerow([name, cfg["embedders"][name]["model"], f"{atual:.4f}"]
                       + [f"{tetos[name][p]:.4f}" for p in CEILING_POOLS]
                       + [f"{tetos[name][20] - atual:.4f}"])

    # -- significancia sobre a metrica primaria ----------------------------
    a, b = SYSTEMS
    xa = [results[a]["per_query"][primary][q] for q in qids]
    xb = [results[b]["per_query"][primary][q] for q in qids]
    sg = cfg["metrics"]["significance"]
    seed = cfg.get("seed", 42)
    delta, pval = S.paired_randomization(xa, xb, n=sg["n_permutations"], seed=seed)
    _, lo, hi = S.bootstrap_mean_diff_ci(xa, xb, n_boot=sg["n_bootstrap"],
                                         alpha=sg["alpha"], seed=seed)
    cliff = S.cliffs_delta(xa, xb)
    margem = sg["noninferiority_margin"]

    (od / "e1_significancia.json").write_text(json.dumps({
        "metric": primary, "contrast": f"{a} - {b}", "n_queries": len(qids),
        "delta": delta, "ci95": [lo, hi], "p_value": pval, "cliffs_delta": cliff,
        "test": "paired_randomization", "n_permutations": sg["n_permutations"],
        "n_bootstrap": sg["n_bootstrap"], "seed": seed,
        "noninferiority_margin": margem, "ci_dentro_da_margem": bool(lo > -margem),
        "rerank_ceiling": {n: {f"teto@{p}": tetos[n][p] for p in CEILING_POOLS}
                           for n in SYSTEMS},
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    # -- resumo no terminal ------------------------------------------------
    print("\n" + "=" * 78)
    print("E1 - teto de reordenacao")
    print("=" * 78)
    print(f"{'sistema':<14}{primary:>10}"
          + "".join(f"{'teto@' + str(p):>10}" for p in CEILING_POOLS)
          + f"{'margem@20':>12}")
    for name in SYSTEMS:
        atual = results[name]["mean"][primary]
        print(f"{name:<14}{atual:>10.4f}"
              + "".join(f"{tetos[name][p]:>10.4f}" for p in CEILING_POOLS)
              + f"{tetos[name][20] - atual:>12.4f}")
    print("-" * 78)
    print(f"{primary}: {ROTULO[a]} - {ROTULO[b]} = {delta:+.4f}   "
          f"IC95% [{lo:+.4f}, {hi:+.4f}]   p = {pval:.4f}   Cliff d = {cliff:+.3f}")
    print(f"margem de nao-inferioridade = {margem}: IC "
          f"{'DENTRO' if lo > -margem else 'FORA'} da margem")
    print("=" * 78)
    print(f"\nplanilhas em {od}:")
    for nome in sorted(p.name for p in od.glob("*.csv")):
        print("  -", nome)


if __name__ == "__main__":
    main()
