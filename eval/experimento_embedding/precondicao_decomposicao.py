#!/usr/bin/env python3
"""A decomposição de consulta traz o lado que falta? Precondição, antes do pipeline.

POR QUE ESTA MEDIÇÃO EXISTE. A decomposição já foi testada três vezes e falhou
(ver `eval/tools/experimentos_descartados/README.md`): fusão RRF não mudou o
top-5 (p=0,62), reserva de vaga foi no-op em 22 de 25, partição 3+2 piorou. Mas
nenhum desses testes verificou a PRECONDIÇÃO — se cada sub-pergunta, buscando
sozinha, alcança o lado que ela nomeia. Se não alcança, reservar vaga para ela
não podia mudar nada, e o no-op de 2026 se explica sem misticismo.

O QUE SE MEDE. Para cada comparativa, busca-se com sub1 e com sub2
separadamente e pergunta-se:

    lado próprio     a sub-pergunta traz a citação DO DOCUMENTO QUE ELA NOMEIA?
    união            juntando o top-5 das duas, as duas citações aparecem?

A união é o teto da decomposição como estratégia de RECUPERAÇÃO — e é o único
número que interessa, porque é ele que pode alcançar as 26 comparativas cujo
material não está no pool da consulta única. Se a união não melhorar, a
decomposição está encerrada, agora com o diagnóstico e não só com o resultado.

Uso:
  python eval/experimento_embedding/precondicao_decomposicao.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parents[1]
sys.path.insert(0, str(RAIZ / "eval"))

CACHE = RAIZ / "eval/results_decomp/subqueries_cache.json"
GABARITO = RAIZ / "eval/data/golden_dezembro.jsonl"
INDICE = RAIZ / "eval/results/indexes/dense_bge_m3_context-v1_c656b175b4ece716.npz"
RANKING = RAIZ / "eval/results/retrieval/rankings/B_bge_m3_rerank.json"
K = 5

for _f in (sys.stdout, sys.stderr):
    try:
        _f.reconfigure(encoding="utf-8")
    except Exception:
        pass


def main() -> None:
    cache = json.loads(CACHE.read_text(encoding="utf-8"))
    gold = {g["qid"]: g for g in
            (json.loads(l) for l in GABARITO.open(encoding="utf-8") if l.strip())}
    por_texto = {g["question"]: q for q, g in gold.items()}
    rr = json.loads(RANKING.read_text(encoding="utf-8"))

    alvos = [(por_texto[t], t) for t in cache
             if t in por_texto and gold[por_texto[t]]["question_type"] == "comparative"]
    if not alvos:
        print("nenhuma comparativa com sub-perguntas em cache.")
        return
    print(f"comparativas com sub-perguntas em cache: {len(alvos)}\n")

    with np.load(INDICE, allow_pickle=True) as d:
        matriz = d["matrix"].astype(np.float32)
        ids = [str(x) for x in d["ids"]]

    doc_de = {}
    with (RAIZ / "data/processed/documents.jsonl").open(encoding="utf-8") as fh:
        for linha in fh:
            if linha.strip():
                r = json.loads(linha)
                doc_de[r["id"]] = r["metadata"]["document_id"]

    from lib.embedders import build_embedder
    from lib.common import load_config
    emb = build_embedder("bge_m3", load_config())

    def buscar(texto: str, k: int) -> list[str]:
        v = emb.embed_query(texto).astype(np.float32)
        s = matriz @ v
        return [ids[i] for i in np.argsort(s)[::-1][:k]]

    proprio = irmao = 0
    uniao_ok = unica_ok = 0
    resgatadas = []

    for qid, texto in alvos:
        g = gold[qid]
        grupos = g["qrels_grupos"]
        # cada grupo de citação pertence ao documento do seu primeiro trecho
        dono = [doc_de.get(gr[0]) for gr in grupos if gr]
        subs = [cache[texto]["sub1"], cache[texto]["sub2"]]
        topos = [buscar(s, K) for s in subs]

        # a sub-pergunta alcança a citação de ALGUM documento; é "próprio" se o
        # documento que ela alcança é diferente do que a outra alcança.
        alcanca = []
        for topo in topos:
            t = set(topo)
            alcanca.append({d for gr, d in zip(grupos, dono)
                            if any(c in t for c in gr)})
        if alcanca[0] and alcanca[1] and alcanca[0] != alcanca[1]:
            proprio += 1
        elif alcanca[0] and alcanca[0] == alcanca[1]:
            irmao += 1

        juntos = set(topos[0]) | set(topos[1])
        u = all(any(c in juntos for c in gr) for gr in grupos)
        uniao_ok += u
        topo5 = set(rr.get(qid, [])[:K])
        s1 = all(any(c in topo5 for c in gr) for gr in grupos)
        unica_ok += s1
        if u and not s1:
            resgatadas.append(qid)

    n = len(alvos)
    print(f"{'as duas sub-perguntas alcançam documentos DIFERENTES':52s} {proprio:3d} de {n}")
    print(f"{'as duas alcançam o MESMO documento (decomposição inútil)':52s} {irmao:3d} de {n}")
    print()
    print(f"{'consulta única, top-5 reranqueado — acerta':52s} {unica_ok:3d} de {n}")
    print(f"{'união do top-5 das duas sub-perguntas — acerta':52s} {uniao_ok:3d} de {n}")
    print()
    if resgatadas:
        print(f"resgatadas só pela decomposição: {len(resgatadas)} → {resgatadas}")
    else:
        print("nenhuma pergunta é resgatada pela decomposição.")
    print(f"\n(k={K} por sub-pergunta, então a união vê até {2*K} trechos contra "
          f"{K} da consulta única — a comparação favorece a decomposição de propósito: "
          f"se nem assim ganhar, está encerrada.)")


if __name__ == "__main__":
    main()
