#!/usr/bin/env python3
"""Monta o top-5 da comparativa a partir dos dois pools reranqueados.

A RECUPERAÇÃO JÁ MELHOROU (medido em 29/09): o pool unido cobre as duas citações
em 59 das 75 comparativas contra 48 da consulta única. O que falta saber é se a
SELEÇÃO consegue sacar isso — e é exatamente aí que a decomposição falhou três
vezes antes, com reserva de vaga (no-op), partição 3+2 (piorou) e fusão RRF (não
mexeu no top-5). A diferença agora é que há reranker sobre os pools das
sub-perguntas, que nenhuma daquelas tentativas tinha.

QUATRO ESTRATÉGIAS, porque "decomposição" não é uma coisa só e reprovar a pior
não reprova a ideia:

    score global    junta tudo e ordena por score. É o controle: se a fusão não
                    fizer nada, o resultado tende ao da consulta única.
    reserva         garante a vaga do melhor de CADA lado, depois preenche por
                    score. É a variante que foi no-op em 2026.
    intercalado     alterna: melhor do sub1, melhor do sub2, 2º do sub1…
                    Reserva forte, sem depender de score entre lados.
    RRF             fusão por posição recíproca nos dois rankings.

ORÇAMENTO IGUAL, SENÃO NÃO VALE. O pool decomposto de 100+100 vê 166 trechos
distintos contra 100 da consulta única; comparar Junta@5 direto premiaria ter
visto mais material. Por isso cada estratégia é medida também em 50+50, que usa
85 trechos — MENOS que a consulta única. É essa a coluna que decide.

Uso:
  python eval/experimento_embedding/fundir_subperguntas.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parents[1]

GABARITO = RAIZ / "eval/data/golden_dezembro.jsonl"
POOLS = RAIZ / "eval/results/indexes/pools_subperguntas.json"
SCORES = RAIZ / "eval/results/indexes/rerank_scores_subperguntas.json"
RERANK_UNICO = RAIZ / "eval/results/retrieval/rankings/B_bge_m3_rerank.json"
K = 5

for _f in (sys.stdout, sys.stderr):
    try:
        _f.reconfigure(encoding="utf-8")
    except Exception:
        pass


def ordenar(cands, notas):
    return sorted(cands, key=lambda c: -notas.get(c, -1e9))


def score_global(o1, o2, n1, n2, k):
    juntos = {}
    for c in o1:
        juntos[c] = max(juntos.get(c, -1e9), n1.get(c, -1e9))
    for c in o2:
        juntos[c] = max(juntos.get(c, -1e9), n2.get(c, -1e9))
    return sorted(juntos, key=lambda c: -juntos[c])[:k]


def reserva(o1, o2, n1, n2, k):
    saida = []
    for lista in (ordenar(o1, n1), ordenar(o2, n2)):
        for c in lista:
            if c not in saida:
                saida.append(c)
                break
    resto = [c for c in score_global(o1, o2, n1, n2, 10 ** 6) if c not in saida]
    return (saida + resto)[:k]


def intercalado(o1, o2, n1, n2, k):
    a, b = ordenar(o1, n1), ordenar(o2, n2)
    saida, i, j = [], 0, 0
    while len(saida) < k and (i < len(a) or j < len(b)):
        for lista, idx in ((a, "i"), (b, "j")):
            pos = i if idx == "i" else j
            while pos < len(lista) and lista[pos] in saida:
                pos += 1
            if pos < len(lista) and len(saida) < k:
                saida.append(lista[pos])
                pos += 1
            if idx == "i":
                i = pos
            else:
                j = pos
    return saida[:k]


def rrf(o1, o2, n1, n2, k, const=30):
    pontos = defaultdict(float)
    for lista, notas in ((o1, n1), (o2, n2)):
        for pos, c in enumerate(ordenar(lista, notas), start=1):
            pontos[c] += 1.0 / (const + pos)
    return sorted(pontos, key=lambda c: -pontos[c])[:k]


ESTRATEGIAS = {"score global": score_global, "reserva": reserva,
               "intercalado": intercalado, "RRF": rrf}


def main() -> None:
    gold = {g["qid"]: g for g in
            (json.loads(l) for l in GABARITO.open(encoding="utf-8") if l.strip())}
    pools = json.loads(POOLS.read_text(encoding="utf-8"))
    scores = json.loads(SCORES.read_text(encoding="utf-8"))["scores"]
    unico = json.loads(RERANK_UNICO.read_text(encoding="utf-8"))

    comp = [q for q in pools if q in gold and f"{q}#sub1" in scores]
    print(f"comparativas: {len(comp)}\n")

    def junta(ordem, q) -> int:
        t = set(ordem[:K])
        return int(all(any(c in t for c in gr) for gr in gold[q]["qrels_grupos"]))

    base = sum(junta(unico[q], q) for q in comp if q in unico)
    print(f"linha de base — consulta única reranqueada: {base} de {len(comp)}"
          f"  ({base/len(comp):.3f})\n")

    print(f"{'estratégia':16s}{'50+50 (85 trechos)':>22s}{'100+100 (166)':>18s}")
    print("─" * 56)
    resultados = {}
    for nome, fn in ESTRATEGIAS.items():
        linha = f"{nome:16s}"
        for prof in (50, 100):
            acertos, por_origem = 0, defaultdict(int)
            for q in comp:
                o1 = pools[q]["sub1"][:prof]
                o2 = pools[q]["sub2"][:prof]
                n1 = scores[f"{q}#sub1"]
                n2 = scores[f"{q}#sub2"]
                v = junta(fn(o1, o2, n1, n2, K), q)
                acertos += v
                por_origem[gold[q]["origem"]] += v
            linha += f"{acertos:14d} de {len(comp):<5d}"
            resultados[(nome, prof)] = (acertos, dict(por_origem))
        print(linha)

    melhor = max(resultados.items(), key=lambda kv: kv[1][0])
    print(f"\nmelhor: {melhor[0][0]} em {melhor[0][1]}+{melhor[0][1]} — "
          f"{melhor[1][0]} de {len(comp)} ({melhor[1][0]/len(comp):.3f})")

    print("\n── por procedência, na profundidade 50+50 (orçamento menor que o único) ──")
    n_por = defaultdict(int)
    for q in comp:
        n_por[gold[q]["origem"]] += 1
    base_por = defaultdict(int)
    for q in comp:
        if q in unico:
            base_por[gold[q]["origem"]] += junta(unico[q], q)
    print(f"{'origem':24s}{'n':>4s}{'única':>8s}" +
          "".join(f"{nome[:11]:>13s}" for nome in ESTRATEGIAS))
    for o in sorted(n_por):
        linha = f"{o:24s}{n_por[o]:4d}{base_por[o]:8d}"
        for nome in ESTRATEGIAS:
            linha += f"{resultados[(nome, 50)][1].get(o, 0):13d}"
        print(linha)


if __name__ == "__main__":
    main()
