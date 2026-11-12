#!/usr/bin/env python3
"""
eval/tools/significancia_geracao.py
═══════════════════════════════════
Testes de significância para os contrastes de geração, pareados por pergunta
(qid). Simetria com a recuperação, que usa McNemar exato.

Contrastes:
  A) RAG vs. sem-RAG  (por gerador; pareado: mesmo qid nos dois cenários)
  B) qwen3-27b vs. gpt-oss-120b  (por cenário; pareado: mesmo qid)

Testes por tipo de critério:
  • BINÁRIOS (citation, refusal, hallucination) → McNemar exato (binomial nos
    discordantes b, c). Reporta b, c e p exato.
  • TERNÁRIOS (faithfulness, answer_relevance, completeness, correctness) →
    Wilcoxon signed-rank (pareado, ordinal). Reporta estatística e p.

Agregação: voto de maioria do painel aberto (llama + glm + mistral; gemini fora).
Só pares onde AMBOS os lados têm nota inteira (ignora "na"). faithfulness e
hallucination só existem no RAG, logo não entram no contraste RAG vs sem-RAG.

Uso:  python eval/tools/significancia_geracao.py
"""
from __future__ import annotations
import csv
import glob
import json
import sys
from collections import defaultdict
from math import comb
from pathlib import Path

PAINEL_ABERTO = ["llama", "glm", "mistral"]
TERNARIOS = ["faithfulness", "answer_relevance", "completeness", "correctness"]
BINARIOS = ["citation", "refusal", "hallucination"]
SO_RAG = {"faithfulness", "hallucination"}
GEN_DIR = Path("eval/results/generation")


def gerador_do_arquivo(nome: str):
    if "qwen3_27b" in nome:
        return "qwen3_27b"
    if "gptoss_120b" in nome:
        return "gptoss_120b"
    return None


def voto_maioria(valores):
    nums = [v for v in valores if isinstance(v, int)]
    if not nums:
        return None
    cont = {}
    for v in nums:
        cont[v] = cont.get(v, 0) + 1
    maxc = max(cont.values())
    return sorted(v for v, c in cont.items() if c == maxc)[0]


def nota(r, crit):
    s = r.get("scores", {})
    return voto_maioria([(s.get(j) or {}).get(crit) for j in PAINEL_ABERTO])


def carrega():
    """{(gerador, cenario, qid): registro}."""
    out = {}
    for fp in sorted(glob.glob(str(GEN_DIR / "judged_*.jsonl"))):
        if fp.endswith("_TESTE.jsonl"):
            continue
        g = gerador_do_arquivo(Path(fp).name)
        if g is None:
            continue
        for l in open(fp, encoding="utf-8"):
            if not l.strip():
                continue
            r = json.loads(l)
            cen = r.get("cenario", "rag" if r.get("context_chunk_ids") else "norag")
            out[(g, cen, r.get("qid"))] = r
    return out


def mcnemar_exato(pares):
    """pares: lista de (x, y) binários. McNemar exato (binomial bicaudal)."""
    b = sum(1 for x, y in pares if x == 1 and y == 0)   # 1→0
    c = sum(1 for x, y in pares if x == 0 and y == 1)   # 0→1
    n = b + c
    if n == 0:
        return b, c, 1.0
    k = min(b, c)
    # p bicaudal exato: 2 * soma binomial(0.5) até k, limitado a 1
    p = sum(comb(n, i) for i in range(0, k + 1)) * (0.5 ** n) * 2
    return b, c, min(p, 1.0)


def wilcoxon(pares):
    """pares: lista de (x, y) ordinais. Wilcoxon signed-rank via scipy."""
    try:
        from scipy.stats import wilcoxon as _w
    except Exception:
        return None, None, None
    difs = [x - y for x, y in pares if x != y]
    if len(difs) < 1:
        return 0, 1.0, len(pares)
    xs = [x for x, y in pares]
    ys = [y for x, y in pares]
    try:
        res = _w(xs, ys, zero_method="wilcox", alternative="two-sided")
        return float(res.statistic), float(res.pvalue), len(pares)
    except Exception as e:
        return None, None, len(pares)


def coleta_pares(dados, chave_a, chave_b, crit):
    """Notas pareadas (a, b) por qid, só onde ambos têm nota inteira."""
    (ga, cena), (gb, cenb) = chave_a, chave_b
    pares = []
    qids = {qid for (g, c, qid) in dados if (g, c) == (ga, cena)}
    for qid in qids:
        ra = dados.get((ga, cena, qid))
        rb = dados.get((gb, cenb, qid))
        if ra is None or rb is None:
            continue
        x = nota(ra, crit)
        y = nota(rb, crit)
        if isinstance(x, int) and isinstance(y, int):
            pares.append((x, y))
    return pares


def roda_contraste(dados, nome, chave_a, chave_b, criterios):
    linhas = []
    print(f"\n=== {nome} ===")
    print(f"{'criterio':<18}{'teste':<10}{'n':>5}{'b/W':>8}{'c':>6}{'p':>10}  sig")
    print("-" * 64)
    for crit in criterios:
        pares = coleta_pares(dados, chave_a, chave_b, crit)
        if not pares:
            continue
        if crit in BINARIOS:
            b, c, p = mcnemar_exato(pares)
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
            print(f"{crit:<18}{'McNemar':<10}{len(pares):>5}{b:>8}{c:>6}{p:>10.4f}  {sig}")
            linhas.append([nome, crit, "McNemar_exato", len(pares), b, c, round(p, 4), sig])
        else:
            W, p, n = wilcoxon(pares)
            if p is None:
                print(f"{crit:<18}{'Wilcoxon':<10}{len(pares):>5}  (scipy ausente)")
                continue
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
            print(f"{crit:<18}{'Wilcoxon':<10}{n:>5}{W:>8.0f}{'':>6}{p:>10.4f}  {sig}")
            linhas.append([nome, crit, "Wilcoxon", n, round(W, 1), "", round(p, 4), sig])
    return linhas


def main():
    dados = carrega()
    if not dados:
        sys.exit("! nenhum judged de geração.")
    print(f"Registros: {len(dados)}  | painel: {'+'.join(PAINEL_ABERTO)}")

    todas = []
    # A) RAG vs sem-RAG, por gerador (faithfulness/halluc. fora: só existem no rag)
    crit_rag_vs = [c for c in (TERNARIOS + BINARIOS) if c not in SO_RAG]
    for g in ("qwen3_27b", "gptoss_120b"):
        todas += roda_contraste(
            dados, f"A) {g}: RAG vs sem-RAG",
            (g, "rag"), (g, "norag"), crit_rag_vs)

    # B) qwen3 vs gpt-oss, por cenário
    for cen in ("rag", "norag"):
        crit = (TERNARIOS + BINARIOS) if cen == "rag" else crit_rag_vs
        todas += roda_contraste(
            dados, f"B) qwen3 vs gpt-oss ({cen})",
            ("qwen3_27b", cen), ("gptoss_120b", cen), crit)

    out = GEN_DIR / "significancia_geracao.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["contraste", "criterio", "teste", "n", "b_ou_W", "c", "p", "sig"])
        w.writerows(todas)
    print(f"\n✓ {out}")
    print("Legenda: *** p<0,001 · ** p<0,01 · * p<0,05 · n.s. não significativo")
    print("McNemar: b = 1→0 (A acerta, B não); c = 0→1 (B acerta, A não).")


if __name__ == "__main__":
    main()
