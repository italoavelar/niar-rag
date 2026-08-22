#!/usr/bin/env python3
"""
export_judged_xlsx.py
═════════════════════
Consolida os results/generation/judged_*.jsonl (Etapa 03) em UMA planilha
ORGANIZADA, por resposta, com as notas e justificativas dos 2 juízes lado a lado
— para inspeção humana, curadoria e material suplementar do artigo.

Abas geradas em results/respostas_julgadas.xlsx:
  • respostas       1 linha por (config × pergunta): resposta gerada, contexto
                    recuperado, resposta-referência, nota de cada juiz×critério e
                    a média dos juízes por critério.
  • justificativas  formato longo (config, qid, juiz, critério, nota, justificativa)
                    — ideal para auditar o raciocínio dos juízes / filtrar/pivotar.
  • resumo          média (dos 2 juízes) por config × critério.

Cada 'config' vem do nome do arquivo (judged_<config>.jsonl), então a comparação
COM vs. SEM RAG e os cenários A/B/C aparecem automaticamente como linhas/abas.

Se o openpyxl não estiver instalado, cai para CSV (um por aba) — ainda organizado.

Uso:
  python eval/tools/export_judged_xlsx.py
  python eval/tools/export_judged_xlsx.py --gen-dir <dir> --out <arquivo.xlsx>
"""
from __future__ import annotations
import argparse, collections, csv, glob, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lib.common import load_config, load_jsonl, resolve, setup_io


def flatten(gen_dir: Path, criteria: list[str]):
    rows_wide, rows_long = [], []
    files = sorted(glob.glob(str(gen_dir / "judged_*.jsonl")))
    for fp in files:
        config = Path(fp).stem.replace("judged_", "")
        for rec in load_jsonl(fp):
            scores = rec.get("scores", {}) or {}
            wide = {
                "config": config,
                "qid": rec.get("qid"),
                "tipo": rec.get("question_type"),
                "pergunta": rec.get("question"),
                "resposta_gerada": rec.get("answer"),
                "chunks_recuperados": ", ".join(rec.get("context_chunk_ids", []) or []),
                "resposta_referencia": rec.get("reference_answer"),
            }
            per_crit = {c: [] for c in criteria}
            for judge, js in scores.items():
                js = js if isinstance(js, dict) else {}
                just = js.get("_justifications", {}) or {}
                for c in criteria:
                    val = js.get(c)
                    wide[f"{judge}·{c}"] = val
                    if isinstance(val, (int, float)):
                        per_crit[c].append(val)
                    rows_long.append({
                        "config": config, "qid": rec.get("qid"),
                        "tipo": rec.get("question_type"), "juiz": judge,
                        "criterio": c, "nota": val, "justificativa": just.get(c, ""),
                    })
            for c in criteria:
                v = per_crit[c]
                wide[f"média·{c}"] = round(sum(v) / len(v), 2) if v else None
            rows_wide.append(wide)
    return rows_wide, rows_long, files


def summarize(rows_long, criteria):
    agg = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows_long:
        if isinstance(r["nota"], (int, float)):
            agg[r["config"]][r["criterio"]].append(r["nota"])
    out = []
    for conf, d in sorted(agg.items()):
        row = {"config": conf, "n_respostas": max((len(v) for v in d.values()), default=0)}
        for c in criteria:
            v = d.get(c, [])
            row[c] = round(sum(v) / len(v), 2) if v else None
        out.append(row)
    return out


def refusal_summary(gen_dir):
    """Recusa OBJETIVA por config: recusa correta (nas fora-de-escopo) e falsa recusa
    (nas respondíveis), por casamento da frase de recusa — sem depender dos juízes."""
    from lib.refusal import refusal_rates
    out = []
    for fp in sorted(glob.glob(str(gen_dir / "judged_*.jsonl"))):
        out.append({"config": Path(fp).stem.replace("judged_", ""),
                    **refusal_rates(load_jsonl(fp))})
    return out


def main():
    setup_io()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--gen-dir")
    ap.add_argument("--out")
    args = ap.parse_args()

    cfg = load_config(args.config)
    criteria = cfg["judges"]["criteria"]
    gen_dir = Path(args.gen_dir) if args.gen_dir else \
        resolve(cfg["paths"]["results_dir"]) / "generation"
    out = Path(args.out) if args.out else \
        resolve(cfg["paths"]["results_dir"]) / "respostas_julgadas.xlsx"

    rows_wide, rows_long, files = flatten(gen_dir, criteria)
    if not files:
        sys.exit(f"! nenhum judged_*.jsonl em {gen_dir} — rode 03_generation_eval.py antes.")
    resumo = summarize(rows_long, criteria)
    recusa = refusal_summary(gen_dir)

    try:
        import pandas as pd
        out.parent.mkdir(parents=True, exist_ok=True)
        with pd.ExcelWriter(out, engine="openpyxl") as xw:
            pd.DataFrame(rows_wide).to_excel(xw, sheet_name="respostas", index=False)
            pd.DataFrame(rows_long).to_excel(xw, sheet_name="justificativas", index=False)
            pd.DataFrame(resumo).to_excel(xw, sheet_name="resumo", index=False)
            pd.DataFrame(recusa).to_excel(xw, sheet_name="recusa", index=False)
        print(f"✓ {len(rows_wide)} respostas de {len(files)} config(s) → {out}")
        print(f"  configs: {', '.join(Path(f).stem.replace('judged_','') for f in files)}")
    except Exception as e:
        base = out.with_suffix("")
        for name, rows in [("respostas", rows_wide), ("justificativas", rows_long),
                           ("resumo", resumo), ("recusa", recusa)]:
            if not rows:
                continue
            p = Path(f"{base}_{name}.csv")
            with p.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader(); w.writerows(rows)
            print(f"  (sem openpyxl) → {p}")
        print(f"! xlsx não gerado ({e}); CSVs organizados salvos no lugar.")


if __name__ == "__main__":
    main()
