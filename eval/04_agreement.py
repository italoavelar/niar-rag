#!/usr/bin/env python3
"""
04_agreement.py  (versão N-juízes; funciona com 2 OU 3+ juízes)
═══════════════════════════════════════════════════════════════
Concordância entre juízes (valida a confiabilidade do LLM-as-judge ANTES de
confiar nos scores). Com N juízes reporta:

  • POR PAR (κ de Cohen ponderado, α de Krippendorff, Spearman ρ, exato, ±1),
    para cada um dos pares — ex.: llama×glm, llama×gemma, glm×gemma;
  • OVERALL por critério: α de Krippendorff sobre TODOS os juízes de uma vez
    (o estimador correto para >2 avaliadores), com IC 95% bootstrap;
  • cobertura de MAIORIA: fração de itens em que ≥2 juízes dão a MESMA nota
    exata, SEMPRE ao lado da cobertura esperada por ACASO — ela não é
    corrigida por acaso e, sozinha, é lida como se fosse concordância:
    com 3 juízes e critério binário o acaso é 100%.

Cada κ/α/ρ par-a-par acompanha IC 95% por bootstrap pareado. Lista os casos de
maior divergência (amplitude ≥2 no painel) para adjudicação humana.

SEPARA POR RUBRICA: julgamento da escala 1..5 e da escala mista não são
comparáveis. O campo `_rubrica` de cada bloco de notas decide qual lote
entra, e o que for descartado — por rubrica ou por painel — é CONTADO
na saída, em vez de sumir em silêncio.

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
    """Fração de itens com maioria (≥2 iguais) se os juízes sorteassem a nota.

    Existe porque a cobertura de maioria NÃO é corrigida por acaso e, sem esta
    linha de base, é lida como se fosse concordância. Dois casos reais aqui:
    com 3 juízes e escala binária a maioria é **100% por construção** — casa dos
    pombos, 3 notas em 2 valores sempre repetem —, e com 3 juízes e escala
    ternária o acaso já dá 78%."""
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
    n_lidos = len(records)

    # ── RUBRICA: nunca misturar escalas ─────────────────────────────────────
    # Os judged_*.jsonl da rubrica 1..5 e os da escala mista tem o MESMO formato
    # e podem ter os MESMOS nomes de juiz, entao pooling silencioso era possivel:
    # kappa sobre notas 1..5 somadas a notas 0/1 nao mede nada. Cada bloco de
    # notas agora carrega `_rubrica` (lib/judges.py). Registro sem o campo e da
    # rubrica antiga, por eliminacao.
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
        print("  Para nao perder dado, arquive os julgamentos antigos:")
        print("    mkdir -p eval/results/generation/arquivo-rubrica-antiga")
        print("    mv eval/results/generation/judged_*.jsonl "
              "eval/results/generation/arquivo-rubrica-antiga/   # os antigos")
        records = [r for r in records if _rubrica_do(r) == alvo]
    elif por_rubrica:
        print(f"Rubrica: {por_rubrica.most_common(1)[0][0]}")

    # Juízes vêm dos PRÓPRIOS registros (robusto a trocas de painel: Gemma→Mistral
    # etc. não exigem mexer no config). Para não misturar painéis de rodadas
    # diferentes que coexistam na pasta, usa-se o CONJUNTO de juízes que pontua
    # junto no MAIOR número de registros (o painel dominante). Ordem de 1ª
    # aparição p/ saída estável.
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
    antes = len(records)
    records = [r for r in records
               if painel_set.issubset(
                   {j for j in r.get("scores", {})
                    if isinstance(r["scores"].get(j), dict)})]
    # O descarte TEM de aparecer. A versao anterior filtrava e so imprimia o
    # total depois, entao perder metade dos registros era invisivel.
    if antes != len(records):
        print(f"! {antes - len(records)} de {antes} registros descartados por nao "
              f"terem o painel {', '.join(judges)}")
    if n_lidos != len(records):
        print(f"  (lidos {n_lidos}, em analise {len(records)})")

    pares = list(combinations(judges, 2))
    print(f"Juízes ({len(judges)}): {', '.join(judges)}  |  "
          f"{len(pares)} par(es)  |  {len(records)} respostas  |  {len(files)} configs")

    # Notas que o juiz emitiu fora da escala do critetio foram rejeitadas na
    # coleta (lib/judges.py) e nao entram em kappa/alpha. Contar aqui para o
    # descarte nao ser invisivel: muitas rejeicoes significam juiz ignorando a
    # rubrica, o que e problema de prompt, nao de concordancia.
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
        cov, amp, acaso = _majority(M)
        linhas_overall.append({"criterion": crit, "n": int(M.shape[1]),
                               "kripp_overall": alpha,
                               "maioria_cobertura": cov,
                               "maioria_acaso": acaso, "amplitude_media": amp})
        # divergências fortes no painel (amplitude ≥2) → adjudicação humana
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
        w.writerow(["criterion", "n", "kripp_overall", "maioria_cobertura",
                    "maioria_acaso", "amplitude_media"])
        for r in linhas_overall:
            w.writerow([r["criterion"], r["n"], _fmt(r["kripp_overall"]),
                        _fmt(r["maioria_cobertura"]), _fmt(r["maioria_acaso"]),
                        _fmt(r["amplitude_media"])])

    (out / "disagreements.json").write_text(
        json.dumps(disagreements, ensure_ascii=False, indent=2), encoding="utf-8")

    # ── Console ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print(f"OVERALL por critério (Krippendorff α sobre os {len(judges)} "
          f"juízes + maioria)")
    print("=" * 72)
    print(f"{'critério':<16}{'n':>5}{'Kripp.α':>10}{'maioria':>10}"
          f"{'(acaso)':>10}{'ampl.méd':>10}")
    print("-" * 72)
    for r in linhas_overall:
        print(f"{r['criterion']:<16}{r['n']:>5}{_fmt(r['kripp_overall']):>10}"
              f"{_pct(r['maioria_cobertura']):>10}{_pct(r['maioria_acaso']):>10}"
              f"{_fmt(r['amplitude_media']):>10}")
    print("\n  'maioria' nao e corrigida por acaso — compare SEMPRE com a coluna")
    print("  (acaso). Em criterio binario com 3 juizes o acaso e 100%: tres notas")
    print("  em dois valores sempre repetem, entao a coluna nao informa nada ali.")
    print("=" * 72)
    print(_interpret_overall(linhas_overall))

    print("\n" + "-" * 72)
    print("POR PAR — κ quadrático médio (detalhe em agreement_pairs.csv)")
    for (j1, j2) in pares:
        ks = [r["kappa_w"] for r in linhas_par
              if r["par"] == f"{j1}×{j2}" and isinstance(r["kappa_w"], float)]
        media = sum(ks) / len(ks) if ks else None
        print(f"  {j1}×{j2:<14} κ̄ = {_fmt(media)}")
    print(f"\n{len(disagreements)} divergências fortes no painel (amplitude≥2) "
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
