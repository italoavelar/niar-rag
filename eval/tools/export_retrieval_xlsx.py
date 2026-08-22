#!/usr/bin/env python3
"""
export_retrieval_xlsx.py
════════════════════════
Consolida a RECUPERAÇÃO (Etapa 02) numa planilha ORGANIZADA, results/recuperacao.xlsx:

  • resumo         médias de TODOS os sistemas (nDCG@5/@10, P@5, Recall@5/@10/@100,
                   F1@5, MRR@10, MAP, GeoRisk) + a linha Δ da ablação CLIR do BM25.
  • <sistema>      1 aba por sistema (A_bm25, A_bm25_mt, B_dense_gemini,
                   B_dense_bge_m3, C_fusion): métricas POR CONSULTA + tipo e língua.
  • ablação_bm25   BM25 sem vs. com tradução (CLIR) lado a lado, por consulta, com Δ
                   e a coluna de língua — o ganho concentra-se no estrato PT→EN.

Lê results/retrieval/rankings.json + o gold set (qrels) e RECALCULA as métricas por
consulta com lib.metrics — então roda sobre a saída existente do 02, sem alterá-lo.

Fallback para CSV (um por aba) se faltar openpyxl.
Uso:  python eval/tools/export_retrieval_xlsx.py
"""
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lib.common import load_config, load_jsonl, resolve, setup_io
from lib import metrics as M


def main():
    setup_io()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--out")
    args = ap.parse_args()

    cfg = load_config(args.config)
    rcfg = cfg["retrieval"]
    ndcg_k, recall_k = rcfg["ndcg_k"], rcfg["recall_k"]
    precision_k, f1_k, mrr_k = rcfg["precision_k"], rcfg["f1_k"], rcfg["mrr_k"]

    rdir = resolve(cfg["paths"]["results_dir"]) / "retrieval"
    rpath = rdir / "rankings.json"
    if not rpath.exists():
        sys.exit(f"! {rpath} não encontrado — rode 02_retrieval_eval.py antes.")
    rankings_all = json.loads(rpath.read_text(encoding="utf-8"))

    gold = load_jsonl(resolve(cfg["paths"]["golden_qa"]))
    qrels = {r["qid"]: {k: int(v) for k, v in r["qrels"].items()}
             for r in gold if r.get("qrels")}
    meta = {r["qid"]: (r.get("question_type", ""), r.get("source_lang", ""),
                       (r.get("question", "") or "")[:120]) for r in gold}

    metric_cols = ([f"ndcg@{k}" for k in ndcg_k] + [f"p@{k}" for k in precision_k]
                   + [f"recall@{k}" for k in recall_k] + [f"f1@{k}" for k in f1_k]
                   + [f"mrr@{mrr_k}", "map"])

    per_system, pq_all, means = {}, {}, {}
    for sysname, rk in rankings_all.items():
        res = M.evaluate_run(rk, qrels, ndcg_k, recall_k, precision_k, f1_k, mrr_k)
        pq_all[sysname] = res["per_query"]
        means[sysname] = res["mean"]
        rows = []
        for qid in sorted(qrels):
            tipo, lang, q = meta.get(qid, ("", "", ""))
            row = {"qid": qid, "tipo": tipo, "lingua": lang}
            for c in metric_cols:
                row[c] = round(res["per_query"].get(c, {}).get(qid, 0.0), 4)
            row["pergunta"] = q
            rows.append(row)
        per_system[sysname] = rows

    # resumo (usa metrics.json p/ trazer GeoRisk; senão, médias recalculadas)
    mjson = rdir / "metrics.json"
    table = json.loads(mjson.read_text(encoding="utf-8")) if mjson.exists() else means
    resumo = []
    for s in rankings_all:
        row = {"sistema": s}
        for c in metric_cols + ["georisk"]:
            row[c] = round(table.get(s, {}).get(c, means.get(s, {}).get(c, 0.0)), 4)
        resumo.append(row)
    if "A_bm25" in means and "A_bm25_mt" in means:
        d = {"sistema": "Δ CLIR (A_bm25_mt − A_bm25)"}
        for c in metric_cols:
            d[c] = round(means["A_bm25_mt"].get(c, 0) - means["A_bm25"].get(c, 0), 4)
        d["georisk"] = None
        resumo.append(d)

    # ablação CLIR do BM25, por consulta (métricas-chave, sem/com/Δ + língua)
    ablation = []
    if "A_bm25" in pq_all and "A_bm25_mt" in pq_all:
        key = [f"ndcg@{ndcg_k[0]}", f"recall@{recall_k[0]}", f"p@{precision_k[0]}", f"mrr@{mrr_k}"]
        for qid in sorted(qrels):
            tipo, lang, q = meta.get(qid, ("", "", ""))
            row = {"qid": qid, "tipo": tipo, "lingua": lang}
            for c in key:
                sem = pq_all["A_bm25"].get(c, {}).get(qid, 0.0)
                com = pq_all["A_bm25_mt"].get(c, {}).get(qid, 0.0)
                row[f"{c} (sem)"] = round(sem, 4)
                row[f"{c} (com)"] = round(com, 4)
                row[f"Δ {c}"] = round(com - sem, 4)
            ablation.append(row)

    out = Path(args.out) if args.out else resolve(cfg["paths"]["results_dir"]) / "recuperacao.xlsx"
    sys_cols = ["qid", "tipo", "lingua"] + metric_cols + ["pergunta"]
    sheet = lambda s: s[:31]
    try:
        import pandas as pd
        out.parent.mkdir(parents=True, exist_ok=True)
        legenda = pd.DataFrame([
            ("qrel = 2", "chunk-fonte (de onde a pergunta nasceu) — mais relevante"),
            ("qrel = 1", "vizinho (±1 página do mesmo documento)"),
            ("qrel = 0", "não relevante (não consta no qrel)"),
            ("nDCG", "ganho exponencial 2^grau−1  →  fonte = 3, vizinho = 1, irrelevante = 0"),
            ("Δ CLIR", "ganho da tradução de consulta no BM25 (A_bm25_mt − A_bm25); efeito no estrato EN"),
        ], columns=["campo", "significado"])
        with pd.ExcelWriter(out, engine="openpyxl") as xw:
            legenda.to_excel(xw, sheet_name="legenda", index=False)
            pd.DataFrame(resumo).to_excel(xw, sheet_name="resumo", index=False)
            for s, rows in per_system.items():
                pd.DataFrame(rows)[sys_cols].to_excel(xw, sheet_name=sheet(s), index=False)
            if ablation:
                pd.DataFrame(ablation).to_excel(xw, sheet_name="ablação_bm25", index=False)
        print(f"✓ recuperação → {out}")
        print(f"  abas: resumo, {', '.join(sheet(s) for s in per_system)}"
              + (", ablação_bm25" if ablation else ""))
    except Exception as e:
        base = out.with_suffix("")
        def dump(name, rows):
            if not rows:
                return
            p = Path(f"{base}_{name}.csv")
            with p.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader(); w.writerows(rows)
            print(f"  (sem openpyxl) → {p}")
        dump("resumo", resumo)
        for s, rows in per_system.items():
            dump(s, [{k: r[k] for k in sys_cols} for r in rows])
        dump("ablacao_bm25", ablation)
        print(f"! xlsx não gerado ({e}); CSVs organizados salvos no lugar.")


if __name__ == "__main__":
    main()
