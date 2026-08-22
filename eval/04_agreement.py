#!/usr/bin/env python3
"""
04_agreement.py
═══════════════
Concordância entre os DOIS juízes (valida a confiabilidade do LLM-as-judge ANTES
de confiar nos scores). Por critério e no agregado:

  • κ de Cohen ponderado (quadrático)  — concordância ordinal corrigida por acaso
  • α de Krippendorff (ordinal)        — robusto a faltas/escala
  • Spearman ρ                          — correlação de postos
  • % de concordância exata e dentro de ±1
Cada métrica (κ, α, ρ) acompanha IC 95% por bootstrap pareado.

Também lista os casos de MAIOR divergência (|Δ|>=2) p/ adjudicação humana.

Uso:  python 04_agreement.py
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.common import load_config, load_jsonl, resolve, ensure_dir, setup_io
import lib.stats as st


def _pairs(records, j1, j2, crit):
    a, b, meta = [], [], []
    for r in records:
        s = r.get("scores", {})
        x = (s.get(j1) or {}).get(crit)
        y = (s.get(j2) or {}).get(crit)
        if isinstance(x, int) and isinstance(y, int):
            a.append(x); b.append(y); meta.append(r)
    return np.array(a), np.array(b), meta


def _metric_fns():
    """Funções de concordância (reutilizadas no ponto-estimativa e no bootstrap)."""
    from sklearn.metrics import cohen_kappa_score
    from scipy.stats import spearmanr

    def kappa(a, b):
        try:
            return float(cohen_kappa_score(a, b, weights="quadratic"))
        except Exception:
            return None

    def kripp(a, b):
        try:
            import krippendorff
            return float(krippendorff.alpha(reliability_data=[list(a), list(b)],
                                            level_of_measurement="ordinal"))
        except Exception:
            return None

    def spear(a, b):
        try:
            r = spearmanr(a, b).correlation
            return float(r) if r == r else None
        except Exception:
            return None

    return {"kappa_w": kappa, "krippendorff": kripp, "spearman": spear}


def _agreement(a, b, n_boot=1000):
    out = {"n": int(len(a))}
    keys = ("kappa_w", "krippendorff", "spearman")
    if len(a) < 2:
        for k in keys:
            out[k] = out[k + "_lo"] = out[k + "_hi"] = None
        out["exact"] = out["within1"] = None
        return out
    for name, fn in _metric_fns().items():
        out[name] = fn(a, b)
        lo, hi = st.bootstrap_ci_pairs(fn, a, b, n_boot=n_boot)
        out[name + "_lo"], out[name + "_hi"] = lo, hi
    out["exact"] = float(np.mean(a == b))
    out["within1"] = float(np.mean(np.abs(a - b) <= 1))
    return out


def main():
    setup_io()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    args = ap.parse_args()
    cfg = load_config(args.config)

    j1, j2 = [m["name"] for m in cfg["judges"]["models"]][:2]
    criteria = cfg["judges"]["criteria"]
    gen_dir = resolve(cfg["paths"]["results_dir"]) / "generation"
    files = sorted(glob.glob(str(gen_dir / "judged_*.jsonl")))
    if not files:
        sys.exit("! nenhum judged_*.jsonl — rode 03_generation_eval.py antes.")

    records = []
    for fp in files:
        for r in load_jsonl(fp):
            r["_config"] = Path(fp).stem.replace("judged_", "")
            records.append(r)
    print(f"Juízes: {j1} vs {j2}  |  {len(records)} respostas julgadas  |  {len(files)} configs")

    # ── Concordância por critério (pooled) ───────────────────────────────────
    nb = cfg["metrics"]["significance"].get("agreement_bootstrap", 1000)
    rows, disagreements = [], []
    for crit in criteria:
        a, b, meta = _pairs(records, j1, j2, crit)
        ag = _agreement(a, b, n_boot=nb)
        rows.append({"criterion": crit, **ag})
        for x, y, r in zip(a, b, meta):
            if abs(int(x) - int(y)) >= 2:
                disagreements.append({"config": r.get("_config"), "qid": r.get("qid"),
                                      "criterion": crit, j1: int(x), j2: int(y),
                                      "question": r.get("question", "")[:120]})

    out = ensure_dir(resolve(cfg["paths"]["results_dir"]) / "agreement")
    cols = ["criterion", "n", "kappa_w", "kappa_w_lo", "kappa_w_hi",
            "krippendorff", "krippendorff_lo", "krippendorff_hi",
            "spearman", "spearman_lo", "spearman_hi", "exact", "within1"]
    with open(out / "agreement.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(cols)
        for r in rows:
            w.writerow([r["criterion"]] + [_fmt(r.get(c)) for c in cols[1:]])
    (out / "disagreements.json").write_text(
        json.dumps(disagreements, ensure_ascii=False, indent=2), encoding="utf-8")

    # ── Console ───────────────────────────────────────────────────────────────
    W = 86
    print("\n" + "=" * W)
    print(f"{'critério':<16}{'n':>5}{'κ_quad':>9}{'κ IC95%':>17}"
          f"{'Kripp.α':>9}{'Spear':>8}{'exato':>8}{'±1':>7}")
    print("-" * W)
    for r in rows:
        ci = ("—" if r.get("kappa_w_lo") is None
              else f"[{r['kappa_w_lo']:.2f},{r['kappa_w_hi']:.2f}]")
        print(f"{r['criterion']:<16}{r['n']:>5}{_fmt(r['kappa_w']):>9}{ci:>17}"
              f"{_fmt(r['krippendorff']):>9}{_fmt(r['spearman']):>8}"
              f"{_pct(r['exact']):>8}{_pct(r['within1']):>7}")
    print("=" * W)
    print(_interpret(rows))
    print(f"\n{len(disagreements)} divergências fortes (|Δ|≥2) → {out/'disagreements.json'}")
    print(f"✓ {out/'agreement.csv'}")


def _fmt(x):
    return f"{x:.3f}" if isinstance(x, float) else "—"


def _pct(x):
    return f"{100*x:.0f}%" if isinstance(x, float) else "—"


def _interpret(rows):
    ks = [r["kappa_w"] for r in rows if isinstance(r["kappa_w"], float)]
    if not ks:
        return "Sem κ disponível."
    avg = sum(ks) / len(ks)
    lvl = ("excelente" if avg >= .8 else "boa" if avg >= .6 else
           "moderada" if avg >= .4 else "fraca")
    return (f"κ quadrático médio = {avg:.3f} → concordância {lvl}. "
            f"{'Scores confiáveis.' if avg>=.6 else 'Considere 3º juiz/adjudicação ou refinar a rubrica.'}")


if __name__ == "__main__":
    main()
