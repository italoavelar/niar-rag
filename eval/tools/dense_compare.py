#!/usr/bin/env python3
"""Compara dois embedders no cenário DENSO, com o mesmo gold set e protocolo.

Generaliza o runner da E1. O par é escolhido por `--systems`:

  E1  --systems qwen3_0_6b,bge_m3  --out results_e1   (Qwen vs BGE, aberto x aberto)
  E2  --systems gemini,bge_m3      --out results_e2   (proprietário x aberto)

Cada sistema é recuperado pelo caminho que sua config manda: `use_qdrant: true`
consulta a coleção (Gemini); caso contrário usa o cache .npz em memória (BGE,
Qwen). É o mesmo `dense_ranking` dos cenários do pipeline.

Reusa eval/retrieval_eval/_common.py, então as planilhas saem no MESMO formato
dos demais cenários:

  scenario_B_dense_<sistema>.csv   por consulta: top-5 recuperado (id, qrel,
                                   TRECHO completo), métricas, tipo, língua,
                                   e a linha MÉDIA ao final
  metrics.csv / metrics.json       uma linha por sistema, com GeoRisk
  per_query_ndcg.json              nDCG@5 por consulta
  rankings.json / rankings/*.json  top-100 recuperado

Mais três artefatos da comparação:

  comparacao_dense_<a>_vs_<b>.csv  head-to-head + vencedor
  rerank_ceiling.csv               teto @20/50/100 e a margem sobre o nDCG@5
  significancia.json               randomização pareada + IC bootstrap + TOST

ONDE ESCREVE. Em eval/<--out>/retrieval/, nunca em eval/results/retrieval/. Os
rankings de lá são do corpus PRÉ-correção e o finalize() agrega tudo que acha em
rankings/; misturar produziria uma metrics.csv com linhas incomparáveis.

No Windows, rode com PYTHONIOENCODING=utf-8.
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
from lib.retrievers import DenseRetriever, QdrantDenseRetriever   # noqa: E402

ROTULOS = {"gemini": "Gemini", "bge_m3": "BGE-M3", "qwen3_0_6b": "Qwen3-0.6B"}
CEILING_POOLS = (20, 50, 100)


def rotulo(name: str) -> str:
    return ROTULOS.get(name, name)


def rerank_ceiling(ranked, rel, pool: int, k: int = 5) -> float:
    """Melhor nDCG@k alcançável reordenando perfeitamente o top-`pool` já trazido."""
    cand = sorted(ranked[:pool], key=lambda c: rel.get(c, 0), reverse=True)
    return M.ndcg_at_k(cand, rel, k)


def checar_prompt(name: str, embedder, cfg) -> None:
    """Modelo assimétrico sem o prompt na consulta sai subestimado."""
    esperado = cfg["embedders"][name].get("query_prompt_name") or None
    obtido = getattr(embedder, "query_prompt_name", None) or None
    if obtido == "None":
        obtido = None
    if esperado != obtido:
        raise SystemExit(
            f"[{name}] query_prompt_name={obtido!r}, esperado {esperado!r}.\n"
            "  Consulta embutida com o prompt errado vicia a comparação.\n"
            "  Regere o .npz de consulta com eval/colab_leme_completo.py."
        )
    if esperado:
        print(f"  [{name}] prompt de consulta: {esperado!r}")


def filtrar_queries(name: str, queries, embedder, qrels):
    """Restringe às consultas com vetor pré-computado, quando for o caso.

    O Colab embutiu o queries.csv (as 75 respondíveis); o gold set tem 100, e as
    25 fora-de-escopo existem para a Etapa 03 testar a RECUSA. Elas não entram em
    métrica — evaluate_run itera sobre os qrels —, então pulá-las não altera
    resultado. Embedders que geram vetor na hora (Gemini) atendem as 100.

    Falta de vetor de consulta COM qrel aborta: ranking vazio contaria nDCG 0.
    """
    if not hasattr(embedder, "has_query"):
        return queries
    tem = [(qid, txt) for qid, txt in queries if embedder.has_query(txt)]
    faltam = [qid for qid, txt in queries if not embedder.has_query(txt)]
    criticas = [qid for qid in faltam if qid in qrels]
    if criticas:
        raise SystemExit(
            f"[{name}] {len(criticas)} consultas COM qrel sem vetor pré-computado "
            f"(ex.: {criticas[:3]}). Isso zeraria o nDCG delas. Regere "
            "o .npz de consulta cobrindo o gold set inteiro."
        )
    if faltam:
        print(f"  [{name}] {len(faltam)} consultas fora-de-escopo sem vetor: não "
              "recuperadas (não entram em métrica)")
    return tem


def recuperar(name: str, cfg, corpus, queries, qrels, top_k, rebuild=False):
    """Mesma bifurcação de _common.dense_ranking, com prompt e filtro checados."""
    ecfg = cfg["embedders"][name]
    embedder = build_embedder(name, cfg)
    checar_prompt(name, embedder, cfg)
    alvo = filtrar_queries(name, queries, embedder, qrels)

    if ecfg.get("use_qdrant"):
        print(f"  [{name}] via Qdrant, coleção {ecfg['qdrant_collection']!r}")
        ret = QdrantDenseRetriever(embedder, ecfg["qdrant_collection"])
    else:
        ret = DenseRetriever(corpus, embedder, cache_dir=indexes_dir(cfg),
                             rebuild=rebuild)
    return ret.run_queries(alvo, top_k)


def main() -> None:
    ap = argparse.ArgumentParser(description="Compara dois embedders densos.")
    ap.add_argument("--systems", default="qwen3_0_6b,bge_m3",
                    help="Dois nomes de eval/config.yaml, separados por vírgula.")
    ap.add_argument("--out", default="results_e1",
                    help="Diretório de saída, relativo a eval/.")
    ap.add_argument("--offline", action="store_true",
                    help="Usa .npz de consulta nos embedders locais (não afeta os de Qdrant).")
    ap.add_argument("--rebuild", action="store_true",
                    help="Ignora o cache de documentos e re-embute o corpus.")
    args = ap.parse_args()

    systems = [s.strip() for s in args.systems.split(",") if s.strip()]
    if len(systems) != 2:
        raise SystemExit(f"--systems precisa de exatamente dois nomes; veio {systems}.")

    cfg = load_config()
    setup_io()
    set_seed(cfg.get("seed", 42))

    for name in systems:
        if name not in cfg["embedders"]:
            raise SystemExit(f"Embedder {name!r} não existe em eval/config.yaml.")
        # offline só faz sentido para quem embute localmente
        if args.offline and not cfg["embedders"][name].get("use_qdrant"):
            cfg["embedders"][name]["offline"] = True

    cfg_out = deepcopy(cfg)
    cfg_out["paths"]["results_dir"] = args.out

    corpus = load_corpus(resolve(cfg["paths"]["corpus"]))
    queries, qrels, meta = load_gold(cfg)
    ndcg_k, recall_k, precision_k, f1_k, mrr_k, top_k = ks(cfg)
    primary = f"ndcg@{ndcg_k[0]}"
    mcols = metric_cols(cfg)

    a, b = systems
    print(f"== Denso: {rotulo(a)} vs {rotulo(b)} ==")
    print(f"corpus    : {len(corpus)} chunks")
    print(f"gold set  : {len(meta)} consultas ({len(qrels)} respondiveis)")
    print(f"top_k     : {top_k} | metrica primaria: {primary}")
    print(f"saida     : {out_dir(cfg_out)}\n")

    rankings, results = {}, {}
    for name in systems:
        rankings[name] = recuperar(name, cfg, corpus, queries, qrels, top_k,
                                   rebuild=args.rebuild)
        save_scenario(cfg_out, f"B_dense_{name}", rankings[name], qrels, meta, corpus)
        results[name] = M.evaluate_run(rankings[name], qrels, ndcg_k, recall_k,
                                       precision_k, f1_k, mrr_k)

    finalize(cfg_out, rewrite_csv=False)

    od = out_dir(cfg_out)
    qids = sorted(qrels)

    # -- head-to-head -------------------------------------------------------
    melhor = max(systems, key=lambda n: results[n]["mean"][primary])
    with (od / f"comparacao_dense_{a}_vs_{b}.csv").open(
            "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["modelo"] + mcols + [f"vencedor_{primary.replace('@', '')}"])
        for name in systems:
            mean = results[name]["mean"]
            w.writerow([rotulo(name)] + [f"{mean[c]:.4f}" for c in mcols]
                       + [name == melhor])

    # -- teto de reordenacao ------------------------------------------------
    tetos = {
        name: {pool: float(np.mean([rerank_ceiling(rankings[name][q], qrels[q], pool)
                                    for q in qids]))
               for pool in CEILING_POOLS}
        for name in systems
    }
    with (od / "rerank_ceiling.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["sistema", "modelo", primary]
                   + [f"teto@{p}" for p in CEILING_POOLS] + ["margem@20"])
        for name in systems:
            atual = results[name]["mean"][primary]
            w.writerow([name, cfg["embedders"][name]["model"], f"{atual:.4f}"]
                       + [f"{tetos[name][p]:.4f}" for p in CEILING_POOLS]
                       + [f"{tetos[name][20] - atual:.4f}"])

    # -- significancia ------------------------------------------------------
    xa = [results[a]["per_query"][primary][q] for q in qids]
    xb = [results[b]["per_query"][primary][q] for q in qids]
    sg = cfg["metrics"]["significance"]
    seed = cfg.get("seed", 42)
    delta, pval = S.paired_randomization(xa, xb, n=sg["n_permutations"], seed=seed)
    _, lo, hi = S.bootstrap_mean_diff_ci(xa, xb, n_boot=sg["n_bootstrap"],
                                         alpha=sg["alpha"], seed=seed)
    cliff = S.cliffs_delta(xa, xb)
    margem = sg["noninferiority_margin"]

    (od / "significancia.json").write_text(json.dumps({
        "metric": primary, "contrast": f"{a} - {b}", "n_queries": len(qids),
        "delta": delta, "ci95": [lo, hi], "p_value": pval, "cliffs_delta": cliff,
        "test": "paired_randomization", "n_permutations": sg["n_permutations"],
        "n_bootstrap": sg["n_bootstrap"], "seed": seed,
        "noninferiority_margin": margem,
        # A nao-inferioridade e DIRECIONAL: testa se o PRIMEIRO sistema
        # (desafiante) nao fica abaixo do segundo por mais que a margem.
        # Passe o desafiante primeiro em --systems, senao o veredito responde
        # a pergunta errada.
        "nao_inferioridade": {
            "desafiante": a, "referencia": b, "margem": margem,
            "limite_inferior": lo,
            "dentro_da_margem": bool(lo > -margem),
        },
        "rerank_ceiling": {n: {f"teto@{p}": tetos[n][p] for p in CEILING_POOLS}
                           for n in systems},
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    # -- resumo -------------------------------------------------------------
    print("\n" + "=" * 78)
    print("Teto de reordenacao")
    print("=" * 78)
    print(f"{'sistema':<14}{primary:>10}"
          + "".join(f"{'teto@' + str(p):>10}" for p in CEILING_POOLS)
          + f"{'margem@20':>12}")
    for name in systems:
        atual = results[name]["mean"][primary]
        print(f"{name:<14}{atual:>10.4f}"
              + "".join(f"{tetos[name][p]:>10.4f}" for p in CEILING_POOLS)
              + f"{tetos[name][20] - atual:>12.4f}")
    print("-" * 78)
    print(f"{primary}: {rotulo(a)} - {rotulo(b)} = {delta:+.4f}   "
          f"IC95% [{lo:+.4f}, {hi:+.4f}]   p = {pval:.4f}   Cliff d = {cliff:+.3f}")
    veredito = "DENTRO" if lo > -margem else "FORA"
    print(f"nao-inferioridade de {rotulo(a)} em relacao a {rotulo(b)} "
          f"(margem {margem}): limite inferior {lo:+.4f} -> {veredito} da margem")
    if lo > 0:
        print(f"  (atencao: {rotulo(a)} esta ACIMA de {rotulo(b)}; o veredito de "
              f"nao-inferioridade so interessa na direcao desafiante -> referencia)")
    print("=" * 78)
    print(f"\nplanilhas em {od}:")
    for nome in sorted(p.name for p in od.glob("*.csv")):
        print("  -", nome)


if __name__ == "__main__":
    main()
