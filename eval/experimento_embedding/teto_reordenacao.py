#!/usr/bin/env python3
"""Quanto ainda dá para ganhar só reordenando o que já foi recuperado.

Separa as falhas em duas, porque elas têm remédios diferentes e o mesmo número
agregado esconde as duas:

  DE ORDEM        tudo que a pergunta precisa está dentro do top-100, mas o
                  corte em 5 deixou algo de fora. Uma seleção melhor resolve.
  IMPOSSÍVEL      algo necessário não está nem no top-100. Nenhuma reordenação
                  alcança — só mudar a recuperação.

O "teto" é onde o Junta@5 chegaria se a escolha das 5 vagas fosse perfeita
dentro do pool já recuperado. É o orçamento total de qualquer trabalho de
ordenação: reranker melhor, MMR, teto por documento, decomposição na hora de
selecionar. Nada disso passa do teto.

Uso:
  python eval/experimento_embedding/teto_reordenacao.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parents[1]
RANKINGS = RAIZ / "eval/results/retrieval/rankings"
GABARITO = RAIZ / "eval/data/golden_dezembro.jsonl"
K = 5

for _f in (sys.stdout, sys.stderr):
    try:
        _f.reconfigure(encoding="utf-8")
    except Exception:
        pass

TIPOS = ("comparative", "multi_hop", "factual")
BRACOS = ("B_bge_m3_rerank", "B_dense_gemini", "B_dense_bge_m3")
ROTULO = {"B_bge_m3_rerank": "BGE+reranker", "B_dense_gemini": "Gemini",
          "B_dense_bge_m3": "BGE-m3"}


def junta(ordem, grupos, k=K) -> bool:
    topo = set(ordem[:k])
    return all(any(c in topo for c in gr) for gr in grupos)


def main() -> None:
    gold = {g["qid"]: g for g in
            (json.loads(l) for l in GABARITO.open(encoding="utf-8") if l.strip())}
    rk = {b: json.loads((RANKINGS / f"{b}.json").read_text(encoding="utf-8"))
          for b in BRACOS if (RANKINGS / f"{b}.json").exists()}

    print(f"ACERTOS ABSOLUTOS — todas as citações no top-{K}\n")
    print(f"{'braço':16s}" + "".join(f"{t:>18s}" for t in TIPOS))
    print("─" * (16 + 18 * len(TIPOS)))
    for b, r in rk.items():
        linha = f"{ROTULO[b]:16s}"
        for t in TIPOS:
            qs = [q for q, g in gold.items()
                  if g["question_type"] == t and g["qrels_grupos"] and q in r]
            a = sum(1 for q in qs if junta(r[q], gold[q]["qrels_grupos"]))
            linha += f"{a:11d} de {len(qs):<4d}"
        print(linha)

    base = "B_bge_m3_rerank"
    if base not in rk:
        return
    r = rk[base]
    print(f"\n\nTETO DA REORDENAÇÃO — {ROTULO[base]}\n")
    print(f"{'tipo':14s}{'acerta hoje':>14s}{'de ordem':>11s}{'impossível':>12s}"
          f"{'hoje':>8s}{'teto':>8s}")
    print("─" * 67)
    th = to = ti = tn = 0
    for t in TIPOS:
        qs = [q for q, g in gold.items()
              if g["question_type"] == t and g["qrels_grupos"] and q in r]
        h = o = i = 0
        for q in qs:
            grupos = gold[q]["qrels_grupos"]
            if junta(r[q], grupos):
                h += 1
                continue
            pos = set(r[q])
            if all(any(c in pos for c in gr) for gr in grupos):
                o += 1
            else:
                i += 1
        th, to, ti, tn = th + h, to + o, ti + i, tn + len(qs)
        print(f"{t:14s}{h:9d} de {len(qs):<3d}{o:11d}{i:12d}"
              f"{h/len(qs):8.3f}{(h+o)/len(qs):8.3f}")
    print("─" * 67)
    print(f"{'TOTAL':14s}{th:9d} de {tn:<3d}{to:11d}{ti:12d}"
          f"{th/tn:8.3f}{(th+to)/tn:8.3f}")
    print(f"\n  hoje {th/tn:.3f} · teto da reordenação {(th+to)/tn:.3f} · Gemini 0,493")
    print(f"  {to} perguntas têm tudo no pool e não no top-{K}: é aí que cabe ganho.")
    print(f"  {ti} não têm o material no pool: só mudando a recuperação.")


if __name__ == "__main__":
    main()
