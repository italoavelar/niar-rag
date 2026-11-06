#!/usr/bin/env python3
"""
04_agreement.py  (versão N-juízes; funciona com 2 OU 3+ juízes)
═══════════════════════════════════════════════════════════════
Concordância entre juízes (valida a confiabilidade do LLM-as-judge ANTES de
confiar nos scores). Com N juízes reporta:

  • POR PAR (κ de Cohen ponderado, α de Krippendorff, Spearman ρ, exato, ±1);
  • OVERALL por critério: α de Krippendorff sobre TODOS os juízes de uma vez
    (o estimador correto para >2 avaliadores), com IC 95% bootstrap;
  • cobertura de MAIORIA ao lado da cobertura esperada por ACASO.

SEPARA POR CENÁRIO (rag / sem-rag). ── auditoria nov/2026 ──
Juntar rag e sem-rag num pool só INFLA o α: os juízes concordam facilmente que
"resposta sem contexto é pior", e essa concordância ENTRE braços entra no α como
se fosse concordância sobre a nota — o agregado fica MAIOR que qualquer cenário
isolado. Além disso, faithfulness e hallucination medem sustentação NO CONTEXTO;
no sem-rag não há contexto, então a nota ali é indefinida e é EXCLUÍDA (="na") do
cálculo. Por isso a análise roda três vezes: GERAL (agregado, mantido só para
mostrar a inflação), RAG e SEM-RAG. O número honesto para o artigo é o POR
CENÁRIO; o geral fica como comparação.

SEPARA POR RUBRICA: julgamento da escala 1..5 e da escala mista não são
comparáveis. O campo `_rubrica` de cada bloco decide o lote; o descarte — por
rubrica, por painel ou por cenário — é CONTADO, em vez de sumir em silêncio.

Uso:  python 04_agreement.py
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.common import load_config, load_jsonl, resolve, ensure_dir, setup_io
import lib.stats as st

# Critérios que medem sustentação NO CONTEXTO — indefinidos sem contexto (sem-rag)
CRIT_DEPENDE_CONTEXTO = {"faithfulness", "hallucination"}


def _cenario(r) -> str:
    """'rag' ou 'norag'. Usa o campo `cenario` (primário); se faltar, infere pela
    presença de context_chunk_ids (que o sem-rag não tem, por desenho)."""
    c = r.get("cenario")
    if c in ("rag", "norag"):
        return c
    return "rag" if r.get("context_chunk_ids") else "norag"


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


def _maioria_por_acaso(n_juizes: int, n_niveis: int) -> float:
    """Fração de itens com maioria (≥2 iguais) se os juízes sorteassem a nota."""
    from itertools import product
    if n_juizes < 2 or n_niveis < 1:
        return float("nan")
    casos = list(product(range(n_niveis), repeat=n_juizes))
    com = sum(1 for c in casos
              if max(c.count(v) for v in set(c)) >= 2)
    return com / len(casos)


def _majority(matrix):
    """Cobertura de maioria: fração de itens com ≥2 juízes na MESMA nota, a
    amplitude média (max−min por item) e a cobertura ESPERADA POR ACASO."""
    if matrix.shape[1] == 0:
        return None, None, None
    n_itens = matrix.shape[1]
    tem_maioria = 0
    amplitudes = []
    for j in range(n_itens):
        col = matrix[:, j]
        vals, counts = np.unique(col, return_counts=True)
        if counts.max() >= 2:
            tem_maioria += 1
        amplitudes.append(int(col.max() - col.min()))
    niveis = len(np.unique(matrix))
    acaso = _maioria_por_acaso(matrix.shape[0], max(niveis, 2))
    return tem_maioria / n_itens, float(np.mean(amplitudes)), acaso


def analisar(records, judges, criteria, nb, out, rotulo, suf):
    """Roda a análise de concordância (por par + overall + maioria) sobre um
    recorte de `records`, com os `criteria` indicados. Escreve CSVs com sufixo
    `suf` e devolve (linhas_overall, disagreements) para o console.

    `rotulo` é o nome do recorte (GERAL / RAG / SEM-RAG), só para impressão."""
    pares = list(combinations(judges, 2))

    # ── 1) POR PAR ──────────────────────────────────────────────────────────
    cols = ["par", "criterion", "n", "kappa_w", "kappa_w_lo", "kappa_w_hi",
            "krippendorff", "krippendorff_lo", "krippendorff_hi",
            "spearman", "spearman_lo", "spearman_hi", "exact", "within1"]
    linhas_par, disagreements = [], []
    for (j1, j2) in pares:
        for crit in criteria:
            a, b, meta = _pairs(records, j1, j2, crit)
            ag = _agreement(a, b, n_boot=nb)
            linhas_par.append({"par": f"{j1}×{j2}", "criterion": crit, **ag})
    with open(out / f"agreement_pairs{suf}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(cols)
        for r in linhas_par:
            w.writerow([r["par"], r["criterion"]] + [_fmt(r.get(c)) for c in cols[2:]])

    # ── 2) OVERALL + MAIORIA ────────────────────────────────────────────────
    linhas_overall = []
    for crit in criteria:
        M, metas = _matrix(records, judges, crit)
        alpha = _kripp_overall(M)
        cov, amp, acaso = _majority(M)
        linhas_overall.append({"criterion": crit, "n": int(M.shape[1]),
                               "kripp_overall": alpha,
                               "maioria_cobertura": cov,
                               "maioria_acaso": acaso, "amplitude_media": amp})
        for k in range(M.shape[1]):
            col = M[:, k]
            if int(col.max() - col.min()) >= 2:
                r = metas[k]
                disagreements.append({
                    "cenario": rotulo, "config": r.get("_config"),
                    "qid": r.get("qid"), "criterion": crit,
                    **{j: int(col[i]) for i, j in enumerate(judges)},
                    "question": r.get("question", "")[:120]})
    with open(out / f"agreement_overall{suf}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["criterion", "n", "kripp_overall", "maioria_cobertura",
                    "maioria_acaso", "amplitude_media"])
        for r in linhas_overall:
            w.writerow([r["criterion"], r["n"], _fmt(r["kripp_overall"]),
                        _fmt(r["maioria_cobertura"]), _fmt(r["maioria_acaso"]),
                        _fmt(r["amplitude_media"])])

    # ── console ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print(f"[{rotulo}]  OVERALL por critério (Krippendorff α sobre {len(judges)} "
          f"juízes + maioria)  ·  n_registros={len(records)}")
    print("=" * 72)
    print(f"{'critério':<16}{'n':>5}{'Kripp.α':>10}{'maioria':>10}"
          f"{'(acaso)':>10}{'ampl.méd':>10}")
    print("-" * 72)
    for r in linhas_overall:
        print(f"{r['criterion']:<16}{r['n']:>5}{_fmt(r['kripp_overall']):>10}"
              f"{_pct(r['maioria_cobertura']):>10}{_pct(r['maioria_acaso']):>10}"
              f"{_fmt(r['amplitude_media']):>10}")
    print("-" * 72)
    print(_interpret_overall(linhas_overall))

    print(f"\n[{rotulo}] POR PAR — κ quadrático médio:")
    for (j1, j2) in pares:
        ks = [r["kappa_w"] for r in linhas_par
              if r["par"] == f"{j1}×{j2}" and isinstance(r["kappa_w"], float)]
        media = sum(ks) / len(ks) if ks else None
        print(f"  {j1}×{j2:<14} κ̄ = {_fmt(media)}")
    print(f"✓ agreement_pairs{suf}.csv · agreement_overall{suf}.csv")
    return linhas_overall, disagreements


def main():
    setup_io()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    args = ap.parse_args()
    cfg = load_config(args.config)

    criteria = cfg["judges"]["criteria"]
    gen_dir = resolve(cfg["paths"]["results_dir"]) / "generation"
    files = sorted(fp for fp in glob.glob(str(gen_dir / "judged_*.jsonl"))
                   if not Path(fp).stem.endswith("_TESTE"))
    if not files:
        sys.exit("! nenhum judged_*.jsonl — rode a geração antes.")

    records = []
    for fp in files:
        for r in load_jsonl(fp):
            r["_config"] = Path(fp).stem.replace("judged_", "")
            records.append(r)
    n_lidos = len(records)

    # ── RUBRICA: nunca misturar escalas ─────────────────────────────────────
    def _rubrica_do(r):
        rs = {(s or {}).get("_rubrica", "1-5 (sem marcador)")
              for s in (r.get("scores") or {}).values() if isinstance(s, dict)}
        return rs.pop() if len(rs) == 1 else "MISTA NO MESMO REGISTRO"

    por_rubrica = Counter(_rubrica_do(r) for r in records)
    if len(por_rubrica) > 1:
        print("! Ha mais de uma RUBRICA na pasta de resultados:")
        for nome, n in por_rubrica.most_common():
            print(f"    {n:5d} registros  rubrica {nome}")
        alvo = por_rubrica.most_common(1)[0][0]
        print(f"  Usando so a dominante ({alvo}) e DESCARTANDO o resto.")
        records = [r for r in records if _rubrica_do(r) == alvo]
    elif por_rubrica:
        print(f"Rubrica: {por_rubrica.most_common(1)[0][0]}")

    # ── Juízes vêm dos PRÓPRIOS registros (painel dominante) ────────────────
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
        painel_dom = paineis.most_common(1)[0][0]
        judges = [j for j in ordem if j in painel_dom]
    else:
        judges = [m["name"] for m in cfg["judges"]["models"]]

    painel_set = set(judges)
    antes = len(records)
    records = [r for r in records
               if painel_set.issubset(
                   {j for j in r.get("scores", {})
                    if isinstance(r["scores"].get(j), dict)})]
    if antes != len(records):
        print(f"! {antes - len(records)} de {antes} registros descartados por nao "
              f"terem o painel {', '.join(judges)}")
    if n_lidos != len(records):
        print(f"  (lidos {n_lidos}, em analise {len(records)})")

    print(f"Juízes ({len(judges)}): {', '.join(judges)}  |  "
          f"{len(records)} respostas  |  {len(files)} configs")

    # Notas fora de escala rejeitadas na coleta (lib/judges.py)
    fora = Counter()
    for r in records:
        for j, s in (r.get("scores") or {}).items():
            if isinstance(s, dict):
                for c in (s.get("_fora_escala") or {}):
                    fora[(j, c)] += 1
    if fora:
        print(f"! {sum(fora.values())} notas fora da escala, rejeitadas na coleta:")
        for (j, c), n in fora.most_common(8):
            print(f"    {n:4d}  {j} / {c}")

    nb = cfg["metrics"]["significance"].get("agreement_bootstrap", 1000)
    out = ensure_dir(resolve(cfg["paths"]["results_dir"]) / "agreement")

    # ── Três recortes: GERAL, RAG, SEM-RAG ──────────────────────────────────
    rag = [r for r in records if _cenario(r) == "rag"]
    norag = [r for r in records if _cenario(r) == "norag"]
    print(f"\nCenários: rag={len(rag)}  sem-rag={len(norag)}  "
          f"(outros/indef.={len(records)-len(rag)-len(norag)})")

    todos_disagreements = []

    # GERAL (agregado) — mantido só para mostrar a inflação; todos os critérios
    _, d0 = analisar(records, judges, criteria, nb, out,
                     "GERAL (agregado — pode inflar)", "")
    todos_disagreements += d0

    # RAG — todos os critérios fazem sentido (há contexto)
    if rag:
        _, d1 = analisar(rag, judges, criteria, nb, out, "RAG", "_rag")
        todos_disagreements += d1

    # SEM-RAG — faithfulness/hallucination saem (sem contexto, indefinidos)
    if norag:
        criteria_norag = [c for c in criteria if c not in CRIT_DEPENDE_CONTEXTO]
        print(f"\n[SEM-RAG] faithfulness/hallucination EXCLUÍDOS (sem contexto): "
              f"critérios = {', '.join(criteria_norag)}")
        _, d2 = analisar(norag, judges, criteria_norag, nb, out, "SEM-RAG", "_norag")
        todos_disagreements += d2

    (out / "disagreements.json").write_text(
        json.dumps(todos_disagreements, ensure_ascii=False, indent=2),
        encoding="utf-8")

    print("\n" + "=" * 72)
    print("LEITURA: o α do artigo é o POR CENÁRIO (RAG / SEM-RAG), não o GERAL.")
    print("  O GERAL agrega rag+sem-rag e sobe-estima o α (concordância entre")
    print("  braços entra como se fosse concordância sobre a nota). faithfulness e")
    print("  hallucination NÃO têm sentido no sem-rag (sem contexto) e saem de lá.")
    print(f"  {len(todos_disagreements)} divergências fortes → {out/'disagreements.json'}")
    print("=" * 72)


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
