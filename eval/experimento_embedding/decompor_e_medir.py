#!/usr/bin/env python3
"""A decomposição melhora a RECUPERAÇÃO da comparativa? (antes de gastar GPU)

A ORDEM DAS PERGUNTAS IMPORTA. A decomposição já foi refutada três vezes como
estratégia de SELEÇÃO — reserva de vaga foi no-op, partição piorou, fusão RRF
não mexeu no top-5. O que nunca foi medido no corpus atual é se ela melhora a
RECUPERAÇÃO: se o material que a consulta única não alcança entra no pool quando
se busca um lado de cada vez.

Isso importa porque 27 das 75 comparativas falham por recuperação, não por
ordem: a citação não está nem no top-100 da consulta única. Nenhuma reordenação
as alcança. Só a decomposição poderia.

E é barato de responder: basta buscar, sem reranker nenhum. Se o pool unido não
cobrir mais citações que o pool único, o assunto termina aqui e a rodada de GPU
não acontece. Se cobrir, aí vale reranquear — e só aí.

Compara três pools, todos no mesmo tamanho total para não ganhar por volume:

    única@100        o top-100 da pergunta inteira
    união@50+50      top-50 de cada sub-pergunta
    união@100+100    top-100 de cada — teto do que a decomposição alcança

Uso:
  python eval/experimento_embedding/decompor_e_medir.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parents[1]
sys.path.insert(0, str(RAIZ / "eval"))

GABARITO = RAIZ / "eval/data/golden_dezembro.jsonl"
SUBS = RAIZ / "eval/results/indexes/subperguntas.json"
INDICE = RAIZ / "eval/results/indexes/dense_bge_m3_context-v1_c656b175b4ece716.npz"
DENSO = RAIZ / "eval/results/retrieval/rankings/B_dense_bge_m3.json"
SAIDA = RAIZ / "eval/results/indexes/pools_subperguntas.json"

for _f in (sys.stdout, sys.stderr):
    try:
        _f.reconfigure(encoding="utf-8")
    except Exception:
        pass


def cobre(pool, grupos) -> bool:
    p = set(pool)
    return all(any(c in p for c in gr) for gr in grupos)


def quantas(pool, grupos) -> int:
    p = set(pool)
    return sum(1 for gr in grupos if any(c in p for c in gr))


def main() -> None:
    gold = {g["qid"]: g for g in
            (json.loads(l) for l in GABARITO.open(encoding="utf-8") if l.strip())}
    subs = json.loads(SUBS.read_text(encoding="utf-8"))
    denso = json.loads(DENSO.read_text(encoding="utf-8"))

    comp = [q for q, g in gold.items()
            if g["question_type"] == "comparative" and g["qrels_grupos"]
            and g["question"] in subs and q in denso]
    print(f"comparativas com sub-perguntas e ranking: {len(comp)}\n")
    if not comp:
        return

    with np.load(INDICE, allow_pickle=True) as d:
        matriz = d["matrix"].astype(np.float32)
        ids = [str(x) for x in d["ids"]]

    from lib.embedders import build_embedder
    from lib.common import load_config
    emb = build_embedder("bge_m3", load_config())

    def buscar(texto: str, k: int) -> list[str]:
        v = emb.embed_query(texto).astype(np.float32)
        s = matriz @ v
        return [ids[i] for i in np.argsort(s)[::-1][:k]]

    pools = {}
    for i, q in enumerate(comp, start=1):
        s = subs[gold[q]["question"]]
        pools[q] = {"sub1": buscar(s["sub1"], 100), "sub2": buscar(s["sub2"], 100)}
        if i % 20 == 0:
            print(f"  buscadas {i}/{len(comp)}")
    SAIDA.write_text(json.dumps(pools, ensure_ascii=False), encoding="utf-8")
    print(f"\n✓ pools salvos em {SAIDA.name}\n")

    variantes = {
        "única@100": lambda q: denso[q][:100],
        "união@50+50": lambda q: pools[q]["sub1"][:50] + pools[q]["sub2"][:50],
        "união@100+100": lambda q: pools[q]["sub1"] + pools[q]["sub2"],
        "única + união": lambda q: denso[q][:100] + pools[q]["sub1"][:50]
                                   + pools[q]["sub2"][:50],
    }

    print(f"{'pool':18s}{'cobre AS DUAS':>16s}{'citações cobertas':>20s}{'trechos':>10s}")
    print("─" * 64)
    total_cit = sum(len(gold[q]["qrels_grupos"]) for q in comp)
    for nome, fn in variantes.items():
        ok = sum(1 for q in comp if cobre(fn(q), gold[q]["qrels_grupos"]))
        cit = sum(quantas(fn(q), gold[q]["qrels_grupos"]) for q in comp)
        tam = sum(len(set(fn(q))) for q in comp) / len(comp)
        print(f"{nome:18s}{ok:10d} de {len(comp):<3d}{cit:13d} de {total_cit:<4d}{tam:10.0f}")

    ganho = [q for q in comp
             if cobre(pools[q]["sub1"] + pools[q]["sub2"], gold[q]["qrels_grupos"])
             and not cobre(denso[q][:100], gold[q]["qrels_grupos"])]
    perda = [q for q in comp
             if cobre(denso[q][:100], gold[q]["qrels_grupos"])
             and not cobre(pools[q]["sub1"] + pools[q]["sub2"], gold[q]["qrels_grupos"])]
    print(f"\n  a decomposição RESGATA {len(ganho)}: {ganho}")
    print(f"  a decomposição PERDE   {len(perda)}: {perda}")
    print("\n  'cobre AS DUAS' é o que decide: só vale reranquear o que está no pool.")


if __name__ == "__main__":
    main()
