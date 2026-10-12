#!/usr/bin/env python3
"""
04_agreement.py  (versão N-juízes; funciona com 2 OU 3+ juízes)
═══════════════════════════════════════════════════════════════
Concordância entre juízes (valida a confiabilidade do LLM-as-judge ANTES de
confiar nos scores). Com 3 juízes reporta:

  • POR PAR (κ de Cohen ponderado, α de Krippendorff, Spearman ρ, exato, ±1),
    para cada um dos pares — ex.: llama×glm, llama×gemma, glm×gemma;
  • OVERALL por critério: α de Krippendorff sobre TODOS os juízes de uma vez
    (o estimador correto para >2 avaliadores), com IC 95% bootstrap;
  • cobertura de MAIORIA: fração de itens em que ≥2 juízes dão a MESMA nota
    exata, e a concordância média |Δ| do trio.

Cada κ/α/ρ par-a-par acompanha IC 95% por bootstrap pareado. Lista os casos de
maior divergência (amplitude do trio ≥2) para adjudicação humana.

Uso:  python 04_agreement.py
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.common import load_config, load_jsonl, resolve, ensure_dir, setup_io
import lib.stats as st


def _pairs(records, j1, j2, crit):
    """Notas pareadas (j1, j2) para um critério, só onde ambos pontuaram."""
    a, b, meta = [], [], []
    for r in records:
        s = r.get("scores", {})
        x = (s.get(j1) or {}).get(crit)
        y = (s.get(j2) or {}).get(crit)
        if isinstance(x, int) and isinstance(y, int):
            a.append(x); b.append(y); meta.append(r)
    return np.array(a), np.array(b), meta


def _matrix(records, judges, crit):
    """Matriz [n_juízes × n_itens] para Krippendorff overall; itens onde TODOS
    os juízes pontuaram (listwise). Retorna (matriz, metas)."""
    cols, metas = [], []
    for r in records:
        s = r.get("scores", {})
        vals = [(s.get(j) or {}).get(crit) for j in judges]
        if all(isinstance(v, int) for v in vals):
            cols.append(vals); metas.append(r)
    if not cols:
        return np.empty((len(judges), 0)), metas
    return np.array(cols).T, metas          # linhas = juízes


def _metric_fns():
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


def _kripp_overall(matrix):
    """α de Krippendorff ordinal sobre N juízes (matriz juízes×itens)."""
    try:
        import krippendorff
        if matrix.shape[1] < 2:
            return None
        return float(krippendorff.alpha(reliability_data=matrix.tolist(),
                                        level_of_measurement="ordinal"))
    except Exception:
        return None


def _majority(matrix):
    """Cobertura de maioria: fração de itens com ≥2 juízes na MESMA nota, e a
    amplitude média do trio (max−min por item)."""
    if matrix.shape[1] == 0:
        return None, None
    n_itens = matrix.shape[1]
    tem_maioria = 0
    amplitudes = []
    for j in range(n_itens):
        col = matrix[:, j]
        vals, counts = np.unique(col, return_counts=True)
        if counts.max() >= 2:
            tem_maioria += 1
        amplitudes.append(int(col.max() - col.min()))
    return tem_maioria / n_itens, float(np.mean(amplitudes))


def main():
    setup_io()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    args = ap.parse_args()
    cfg = load_config(args.config)

    criteria = cfg["judges"]["criteria"]
    gen_dir = resolve(cfg["paths"]["results_dir"]) / "generation"
    # ignora arquivos de teste (*_TESTE.jsonl) — só os julgamentos oficiais
    files = sorted(fp for fp in glob.glob(str(gen_dir / "judged_*.jsonl"))
                   if not Path(fp).stem.endswith("_TESTE"))
    if not files:
        sys.exit("! nenhum judged_*.jsonl — rode a geração antes.")

    records = []
    for fp in files:
        for r in load_jsonl(fp):
            r["_config"] = Path(fp).stem.replace("judged_", "")
            records.append(r)

    # Juízes vêm dos PRÓPRIOS registros (robusto a trocas de painel: Gemma→Mistral
    # etc. não exigem mexer no config). Para não misturar painéis de rodadas
    # diferentes que coexistam na pasta, usa-se o CONJUNTO de juízes que pontua
    # junto no MAIOR número de registros (o painel dominante). Ordem de 1ª
    # aparição p/ saída estável.
    from collections import Counter
    paineis = Counter()
    ordem = []
    for r in records:
        js = tuple(j for j in r.get("scores", {})
                   if isinstance(r["scores"].get(j), dict))
        if js:
            paineis[frozenset(js)] += 1
        for j in js:
            if j not in ordem:
                ordem.append(j)
    if paineis:
        painel_dom = paineis.most_common(1)[0][0]            # frozenset dominante
        judges = [j for j in ordem if j in painel_dom]
    else:                                                     # fallback: config
        judges = [m["name"] for m in cfg["judges"]["models"]]

    # mantém só os registros que têm o painel dominante (descarta rodadas antigas
    # com outro conjunto de juízes, se houver)
    painel_set = set(judges)
    records = [r for r in records
               if painel_set.issubset(
                   {j for j in r.get("scores", {})
                    if isinstance(r["scores"].get(j), dict)})]

    pares = list(combinations(judges, 2))
    print(f"Juízes ({len(judges)}): {', '.join(judges)}  |  "
          f"{len(pares)} par(es)  |  {len(records)} respostas  |  {len(files)} configs")

    nb = cfg["metrics"]["significance"].get("agreement_bootstrap", 1000)
    out = ensure_dir(resolve(cfg["paths"]["results_dir"]) / "agreement")

    # ── 1) Concordância POR PAR ──────────────────────────────────────────────
    cols = ["par", "criterion", "n", "kappa_w", "kappa_w_lo", "kappa_w_hi",
            "krippendorff", "krippendorff_lo", "krippendorff_hi",
            "spearman", "spearman_lo", "spearman_hi", "exact", "within1"]
    linhas_par, disagreements = [], []
    for (j1, j2) in pares:
        for crit in criteria:
            a, b, meta = _pairs(records, j1, j2, crit)
            ag = _agreement(a, b, n_boot=nb)
            linhas_par.append({"par": f"{j1}×{j2}", "criterion": crit, **ag})
    with open(out / "agreement_pairs.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(cols)
        for r in linhas_par:
            w.writerow([r["par"], r["criterion"]] + [_fmt(r.get(c)) for c in cols[2:]])

    # ── 2) OVERALL (Krippendorff N-juízes) + MAIORIA, por critério ───────────
    linhas_overall = []
    for crit in criteria:
        M, metas = _matrix(records, judges, crit)
        alpha = _kripp_overall(M)
        cov, amp = _majority(M)
        linhas_overall.append({"criterion": crit, "n": int(M.shape[1]),
                               "kripp_overall": alpha,
                               "maioria_cobertura": cov, "amplitude_media": amp})
        # divergências fortes do trio (amplitude ≥2) → adjudicação humana
        for k in range(M.shape[1]):
            col = M[:, k]
            if int(col.max() - col.min()) >= 2:
                r = metas[k]
                disagreements.append({
                    "config": r.get("_config"), "qid": r.get("qid"),
                    "criterion": crit,
                    **{j: int(col[i]) for i, j in enumerate(judges)},
                    "question": r.get("question", "")[:120]})
    with open(out / "agreement_overall.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["criterion", "n", "kripp_overall", "maioria_cobertura", "amplitude_media"])
        for r in linhas_overall:
            w.writerow([r["criterion"], r["n"], _fmt(r["kripp_overall"]),
                        _fmt(r["maioria_cobertura"]), _fmt(r["amplitude_media"])])

    (out / "disagreements.json").write_text(
        json.dumps(disagreements, ensure_ascii=False, indent=2), encoding="utf-8")

    # ── Console ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("OVERALL por critério (Krippendorff α sobre os 3 juízes + maioria)")
    print("=" * 72)
    print(f"{'critério':<16}{'n':>5}{'Kripp.α':>10}{'maioria':>10}{'ampl.méd':>10}")
    print("-" * 72)
    for r in linhas_overall:
        print(f"{r['criterion']:<16}{r['n']:>5}{_fmt(r['kripp_overall']):>10}"
              f"{_pct(r['maioria_cobertura']):>10}{_fmt(r['amplitude_media']):>10}")
    print("=" * 72)
    print(_interpret_overall(linhas_overall))

    print("\n" + "-" * 72)
    print("POR PAR — κ quadrático médio (detalhe em agreement_pairs.csv)")
    for (j1, j2) in pares:
        ks = [r["kappa_w"] for r in linhas_par
              if r["par"] == f"{j1}×{j2}" and isinstance(r["kappa_w"], float)]
        media = sum(ks) / len(ks) if ks else None
        print(f"  {j1}×{j2:<14} κ̄ = {_fmt(media)}")
    print(f"\n{len(disagreements)} divergências fortes do trio (amplitude≥2) "
          f"→ {out/'disagreements.json'}")
    print(f"✓ {out/'agreement_pairs.csv'}")
    print(f"✓ {out/'agreement_overall.csv'}")


def _fmt(x):
    return f"{x:.3f}" if isinstance(x, float) else "—"


def _pct(x):
    return f"{100*x:.0f}%" if isinstance(x, float) else "—"


def _interpret_overall(rows):
    ks = [r["kripp_overall"] for r in rows if isinstance(r["kripp_overall"], float)]
    if not ks:
        return "Sem α disponível."
    avg = sum(ks) / len(ks)
    lvl = ("excelente" if avg >= .8 else "boa" if avg >= .667 else
           "aceitável" if avg >= .6 else "fraca")
    nota = ("Scores confiáveis." if avg >= .667
            else "Abaixo do limiar usual (α≥0,667): refine a rubrica ou use "
                 "voto de maioria como sinal primário.")
    return f"α de Krippendorff médio = {avg:.3f} → concordância {lvl}. {nota}"


if __name__ == "__main__":
    main()
