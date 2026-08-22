"""
eval/retrieval_eval/_common.py
══════════════════════════════
Infra compartilhada pelos cenários de recuperação (um script por cenário nesta
pasta). Cada cenário:
  1) roda sua busca (top-k) e salva o ranking em results/retrieval/rankings/<sys>.json
  2) grava seu CSV de resultados POR CONSULTA (+ média) em
     results/retrieval/scenario_<sys>.csv

O run_all.py roda todos na ordem (A, B×2, C) e, ao final, gera os artefatos
COMBINADOS que o restante do pipeline consome (rankings.json, metrics.json,
per_query_ndcg.json, metrics.csv, com GeoRisk).
"""
from __future__ import annotations
import csv, json, sys
from pathlib import Path

EVAL = Path(__file__).resolve().parent.parent      # .../eval
sys.path.insert(0, str(EVAL))
from lib.common import (load_config, load_corpus, load_jsonl, resolve,   # noqa: E402
                        ensure_dir, set_seed, setup_io)
from lib import metrics as M                                             # noqa: E402


def load_gold(cfg):
    rows = load_jsonl(resolve(cfg["paths"]["golden_qa"]))
    queries = [(r["qid"], r["question"]) for r in rows]
    qrels = {r["qid"]: {k: int(v) for k, v in r["qrels"].items()}
             for r in rows if r.get("qrels")}
    meta = {r["qid"]: (r.get("question_type", ""), r.get("source_lang", ""),
                       r.get("question", "")) for r in rows}
    return queries, qrels, meta


def ks(cfg):
    r = cfg["retrieval"]
    return r["ndcg_k"], r["recall_k"], r["precision_k"], r["f1_k"], r["mrr_k"], r["top_k"]


def metric_cols(cfg):
    r = cfg["retrieval"]
    return ([f"ndcg@{k}" for k in r["ndcg_k"]] + [f"p@{k}" for k in r["precision_k"]]
            + [f"recall@{k}" for k in r["recall_k"]] + [f"f1@{k}" for k in r["f1_k"]]
            + [f"mrr@{r['mrr_k']}", "map"])


def out_dir(cfg):      return ensure_dir(resolve(cfg["paths"]["results_dir"]) / "retrieval")
def rankings_dir(cfg): return ensure_dir(out_dir(cfg) / "rankings")
def indexes_dir(cfg):  return ensure_dir(resolve(cfg["paths"]["results_dir"]) / "indexes")


def save_scenario(cfg, system, ranking, qrels, meta, corpus):
    """Salva o ranking + grava o CSV do cenário. Cada linha (por consulta) traz, para
    cada top-K chunk RECUPERADO: o id (doc{i}), o grau de qrel (qrel{i}) e o TEXTO
    completo do trecho (trecho{i}); depois as métricas. A última linha é a MÉDIA. Com
    o texto no próprio CSV, as etapas seguintes o consomem sem recuperar nem consultar
    o corpus de novo.
        qrel: 2 = chunk-fonte · 1 = vizinho (±1 pág.) · 0 = não relevante.
    K = generation.context_top_k (o que vai para o gerador).

    O CSV lista as 100 consultas, não só as 75 respondíveis: as fora-de-escopo
    também são recuperadas, e é justamente o contexto delas que a Etapa 03 usa
    para testar a RECUSA (o modelo precisa recusar TENDO material plausível em
    mãos). A coluna `respondivel` separa as duas, e as colunas de métrica ficam
    VAZIAS nas fora-de-escopo: nDCG/recall são indefinidos sem documento
    relevante, não zero. A MÉDIA continua sendo só sobre as respondíveis."""
    ndcg_k, recall_k, precision_k, f1_k, mrr_k, _ = ks(cfg)
    res = M.evaluate_run(ranking, qrels, ndcg_k, recall_k, precision_k, f1_k, mrr_k)
    mcols = metric_cols(cfg)
    topk = int(cfg.get("generation", {}).get("context_top_k", 5))

    (rankings_dir(cfg) / f"{system}.json").write_text(
        json.dumps(ranking, ensure_ascii=False), encoding="utf-8")

    doc_cols = [x for i in range(1, topk + 1) for x in (f"doc{i}", f"qrel{i}", f"trecho{i}")]
    csv_path = out_dir(cfg) / f"scenario_{system}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["qid", "respondivel", "question"] + doc_cols + mcols
                   + ["tipo", "lingua"])
        for qid in sorted(meta):
            tipo, lang, question = meta.get(qid, ("", "", ""))
            answerable = qid in qrels
            rel = qrels.get(qid, {})
            retrieved = ranking.get(qid, [])[:topk]
            docrow = []
            for i in range(topk):
                if i < len(retrieved):
                    cid = retrieved[i]
                    docrow += [cid, rel.get(cid, 0), (corpus.get(cid, {}).get("text", "") or "")]
                else:
                    docrow += ["", "", ""]
            # fora-de-escopo: métrica indefinida (não há relevante) → célula vazia
            metrics = ([f"{res['per_query'][c].get(qid, 0.0):.4f}" for c in mcols]
                       if answerable else [""] * len(mcols))
            w.writerow([qid, "sim" if answerable else "não", question]
                       + docrow + metrics + [tipo, lang])
        # rótulo "MÉDIA" é lido por 05_report.py (_per_query_recall) — não renomear
        w.writerow(["MÉDIA", f"{len(qrels)} respondíveis", ""] + [""] * len(doc_cols)
                   + [f"{res['mean'][c]:.4f}" for c in mcols] + ["", ""])
    primary = f"ndcg@{ndcg_k[0]}"
    print(f"  ✓ {system:<16} {primary}={res['mean'][primary]:.4f}  →  {csv_path.name}")
    return res["mean"]


def dense_ranking(cfg, name, corpus, queries, rebuild=False):
    """Busca densa top-k para um embedder: via Qdrant (se use_qdrant) ou local
    (embute o corpus em cache). Retorna {qid: [chunk_id, ...]}."""
    from lib.embedders import build_embedder
    from lib.retrievers import QdrantDenseRetriever, DenseRetriever
    ecfg = cfg["embedders"][name]
    top_k = cfg["retrieval"]["top_k"]
    if ecfg.get("use_qdrant"):
        ret = QdrantDenseRetriever(build_embedder(name, cfg), ecfg["qdrant_collection"])
        return ret.run_queries(queries, top_k)
    dr = DenseRetriever(corpus, build_embedder(name, cfg),
                        cache_dir=indexes_dir(cfg), rebuild=rebuild)
    return dr.run_queries(queries, top_k)


def finalize(cfg, rewrite_csv=True):
    """Combina os rankings/<sys>.json em rankings.json + metrics.json +
    per_query_ndcg.json + metrics.csv (com GeoRisk) — o que 03/05/exportadores usam.

    Com rewrite_csv, reescreve também os scenario_<sys>.csv a partir dos rankings
    JÁ em disco — sem refazer busca nem gastar API. É o caminho para atualizar o
    formato dos CSVs sem re-rodar a recuperação inteira."""
    od, rdir = out_dir(cfg), rankings_dir(cfg)
    ndcg_k, recall_k, precision_k, f1_k, mrr_k, _ = ks(cfg)
    primary = ndcg_k[0]
    queries, qrels, meta = load_gold(cfg)
    systems = {p.stem: json.loads(p.read_text(encoding="utf-8"))
               for p in sorted(rdir.glob("*.json"))}
    if not systems:
        print("! nenhum ranking em", rdir, "— rode os cenários antes."); return
    if rewrite_csv:
        corpus = load_corpus(resolve(cfg["paths"]["corpus"]))
        for s, rk in systems.items():
            save_scenario(cfg, s, rk, qrels, meta, corpus)
    table, per_query_primary = {}, {}
    for s, rk in systems.items():
        res = M.evaluate_run(rk, qrels, ndcg_k, recall_k, precision_k, f1_k, mrr_k)
        table[s] = res["mean"]
        per_query_primary[s] = res["per_query"][f"ndcg@{primary}"]
    for s, g in M.geo_risk(per_query_primary, alpha=cfg["metrics"]["georisk_alpha"]).items():
        table[s]["georisk"] = g
    (od / "rankings.json").write_text(json.dumps(systems, ensure_ascii=False), encoding="utf-8")
    (od / "metrics.json").write_text(json.dumps(table, ensure_ascii=False, indent=2), encoding="utf-8")
    (od / "per_query_ndcg.json").write_text(json.dumps(per_query_primary, ensure_ascii=False), encoding="utf-8")
    cols = metric_cols(cfg) + ["georisk"]
    with (od / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["system"] + cols)
        for s, m in sorted(table.items(), key=lambda x: -x[1].get(f"ndcg@{primary}", 0)):
            w.writerow([s] + [f"{m.get(c, 0):.4f}" for c in cols])
    print(f"\n✓ artefatos combinados em {od}: rankings.json · metrics.json · metrics.csv")
    width = 22 + 10 * len(cols)
    print("=" * width)
    print(f"{'system':<22}" + "".join(f"{c:>10}" for c in cols))
    print("-" * width)
    for s, m in sorted(table.items(), key=lambda x: -x[1].get(f"ndcg@{primary}", 0)):
        print(f"{s:<22}" + "".join(f"{m.get(c, 0):>10.4f}" for c in cols))
    print("=" * width)
