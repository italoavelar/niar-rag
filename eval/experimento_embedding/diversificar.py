#!/usr/bin/env python3
"""Limita quantos trechos do MESMO documento ocupam o topo do ranking.

O PROBLEMA QUE ISTO ATACA. Numa pergunta comparativa a resposta só existe
cruzando dois documentos, e o ranking se entope com um lado só: medido em
28/09/2026, o BGE com reranker traz os dois lados em 14 de 75, e em 34 traz um
lado apenas. Não é falta de recall — é o top-5 gasto com cinco trechos ótimos do
mesmo documento.

A REGRA. Desce o ranking na ordem e descarta o trecho cujo documento já ocupa
`teto` posições na saída. O que sobra vai para o fim, então nada se perde: é
reordenação pura, e `recall@100` fica idêntico.

A TENSÃO QUE PRECISA SER MEDIDA, NÃO SUPOSTA. Comparativa quer diversidade;
multi-hop dentro de um documento quer o contrário — ela precisa de DUAS regiões
do MESMO documento, e um teto de 1 a torna impossível por construção. Então isto
não pode ser avaliado só na média: um teto que melhore o agregado escondendo a
destruição do multi_hop seria um ganho falso. O relatório sai sempre por tipo.

Uso:
  python eval/experimento_embedding/diversificar.py
  python eval/experimento_embedding/diversificar.py --aplicar
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

AQUI = Path(__file__).resolve().parent
PROJECT_ROOT = AQUI.parents[1]
RANKINGS = PROJECT_ROOT / "eval/results/retrieval/rankings"
CORPUS = PROJECT_ROOT / "data/processed/documents.jsonl"

for _f in (sys.stdout, sys.stderr):
    try:
        _f.reconfigure(encoding="utf-8")
    except Exception:
        pass

TETOS = (1, 2, 3)
BASES = ("B_bge_m3_rerank", "B_dense_gemini", "B_dense_bge_m3")


def diversificar(ordem: list[str], doc_de, teto: int) -> list[str]:
    """Reordena pondo no fim o que excede `teto` trechos por documento."""
    contagem: dict[str, int] = defaultdict(int)
    mantidos, adiados = [], []
    for cid in ordem:
        d = doc_de(cid)
        if contagem[d] < teto:
            contagem[d] += 1
            mantidos.append(cid)
        else:
            adiados.append(cid)
    return mantidos + adiados


def junta(ordem, grupos, k=5) -> float:
    t = set(ordem[:k])
    return 1.0 if all(any(c in t for c in gr) for gr in grupos) else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--gabarito", default=str(PROJECT_ROOT / "eval/data/golden_dezembro.jsonl"))
    ap.add_argument("--aplicar", action="store_true")
    args = ap.parse_args()

    doc_por_chunk = {}
    with CORPUS.open(encoding="utf-8") as fh:
        for linha in fh:
            if linha.strip():
                r = json.loads(linha)
                doc_por_chunk[r["id"]] = r["metadata"]["document_id"]
    doc_de = lambda c: doc_por_chunk.get(c, c)  # noqa: E731

    gold = {g["qid"]: g for g in
            (json.loads(l) for l in open(args.gabarito, encoding="utf-8") if l.strip())}
    respondiveis = [q for q, g in gold.items() if g["qrels_grupos"]]
    tipos = sorted({gold[q]["question_type"] for q in respondiveis})

    print(f"gabarito: {Path(args.gabarito).name} — {len(respondiveis)} respondíveis, k={args.k}\n")

    for base in BASES:
        caminho = RANKINGS / f"{base}.json"
        if not caminho.exists():
            continue
        ranking = json.loads(caminho.read_text(encoding="utf-8"))
        alvos = [q for q in respondiveis if q in ranking]
        if not alvos:
            continue

        print(f"══ {base}  ({len(alvos)} perguntas) ══")
        cab = f"{'teto':>6s} {'GERAL':>8s}" + "".join(f"{t[:11]:>13s}" for t in tipos)
        print(cab)
        print("─" * len(cab))

        for teto in (None,) + TETOS:
            notas, por_tipo = [], defaultdict(list)
            for q in alvos:
                ordem = (ranking[q] if teto is None
                         else diversificar(ranking[q], doc_de, teto))
                v = junta(ordem, gold[q]["qrels_grupos"], args.k)
                notas.append(v)
                por_tipo[gold[q]["question_type"]].append(v)
            rot = "sem" if teto is None else str(teto)
            linha = f"{rot:>6s} {sum(notas)/len(notas):8.3f}"
            for t in tipos:
                v = por_tipo[t]
                linha += f"{sum(v)/len(v):13.3f}" if v else f"{'—':>13s}"
            print(linha)

            if args.aplicar and teto is not None:
                saida = RANKINGS / f"{base}_div{teto}.json"
                saida.write_text(json.dumps(
                    {q: diversificar(o, doc_de, teto) for q, o in ranking.items()},
                    ensure_ascii=False), encoding="utf-8")
        print()

    if args.aplicar:
        print(f"✓ rankings diversificados gravados em {RANKINGS}")
    else:
        print("(relatório apenas — use --aplicar para gravar os rankings)")


if __name__ == "__main__":
    main()
