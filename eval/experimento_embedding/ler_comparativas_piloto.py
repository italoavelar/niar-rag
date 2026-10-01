#!/usr/bin/env python3
"""Taxa POR CITAÇÃO contra taxa POR PERGUNTA, e leitura das citações que somem.

DUAS CONTAS DIFERENTES. `Junta@5` é por PERGUNTA e exige as duas citações no
top-5. Mas o que a busca faz, ela faz uma citação de cada vez. Comparar famílias
pela taxa por pergunta esconde isso: a taxa por pergunta é aproximadamente o
quadrado da taxa por citação, então uma diferença modesta em uma vira uma
diferença grande na outra. Este script imprime as duas lado a lado.

A LEITURA. Depois, despeja pergunta, resposta e citação das comparativas do
piloto cuja citação não aparece nem no top-100 — para conferir se a
citação sustenta a resposta ou se o gabarito está apontando texto de moldura.
Se for moldura, o problema é de gabarito e nenhum trabalho de recuperação o
conserta.

Uso:
  python eval/experimento_embedding/ler_comparativas_piloto.py
  python eval/experimento_embedding/ler_comparativas_piloto.py --todas
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parents[1]
GABARITO = RAIZ / "eval/data/golden_dezembro.jsonl"
RERANK = RAIZ / "eval/results/retrieval/rankings/B_bge_m3_rerank.json"
DENSO = RAIZ / "eval/results/retrieval/rankings/B_dense_bge_m3.json"

for _f in (sys.stdout, sys.stderr):
    try:
        _f.reconfigure(encoding="utf-8")
    except Exception:
        pass


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--todas", action="store_true",
                    help="despeja todas as comparativas do piloto, não só as sumidas")
    args = ap.parse_args()

    gold = {g["qid"]: g for g in
            (json.loads(l) for l in GABARITO.open(encoding="utf-8") if l.strip())}
    rr = json.loads(RERANK.read_text(encoding="utf-8"))
    denso = json.loads(DENSO.read_text(encoding="utf-8"))

    comp = [q for q, g in gold.items()
            if g["question_type"] == "comparative" and g["qrels_grupos"] and q in rr]
    por_origem = defaultdict(list)
    for q in comp:
        por_origem[gold[q]["origem"]].append(q)

    print("AS DUAS CONTAS, no braço BGE+reranker\n")
    cab = (f"{'origem':24s}{'citações':>10s}{'no top-5':>10s}{'taxa/citação':>14s}"
           f"{'perguntas':>11s}{'acerta':>8s}{'taxa/pergunta':>15s}")
    print(cab)
    print("─" * len(cab))
    for origem, qs in sorted(por_origem.items()):
        n_cit = n_ok = n_perg = n_acerta = 0
        for q in qs:
            topo = set(rr[q][:5])
            chegou = [any(c in topo for c in gr) for gr in gold[q]["qrels_grupos"]]
            n_cit += len(chegou)
            n_ok += sum(chegou)
            n_perg += 1
            n_acerta += all(chegou)
        print(f"{origem:24s}{n_cit:10d}{n_ok:10d}{n_ok/n_cit:14.3f}"
              f"{n_perg:11d}{n_acerta:8d}{n_acerta/n_perg:15.3f}")
    print("\n  taxa/citação  = das citações exigidas, quantas a busca trouxe")
    print("  taxa/pergunta = das perguntas, quantas tiveram TODAS as suas citações")

    print("\n\n" + "═" * 78)
    print("CITAÇÕES DO PILOTO PARA CONFERÊNCIA")
    print("═" * 78)

    alvos = por_origem.get("piloto", [])
    mostradas = 0
    for q in sorted(alvos):
        g = gold[q]
        pos = {c: i for i, c in enumerate(denso[q])}
        linhas = []
        for gr, cit in zip(g["qrels_grupos"], g["qrels_text"]):
            p = min((pos[c] for c in gr if c in pos), default=None)
            linhas.append((p, cit))
        sumiu = any(p is None for p, _ in linhas)
        if not args.todas and not sumiu:
            continue
        mostradas += 1
        print(f"\n[{q}]  documentos: {', '.join(g['source_docs'])}")
        print(f"  PERGUNTA: {g['question']}")
        print(f"  RESPOSTA: {g['reference_answer'][:330]}")
        for p, cit in linhas:
            onde = "FORA do top-100" if p is None else f"posição {p+1}"
            print(f"  ── citação ({onde}):")
            print(f"     {cit[:330]}")
    print(f"\n{mostradas} pergunta(s) impressa(s).")


if __name__ == "__main__":
    main()
