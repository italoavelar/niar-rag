#!/usr/bin/env python3
"""Pacote de reranking dos pools das sub-perguntas, no mesmo formato do Colab.

O QUE MUDA EM RELAÇÃO AO PACOTE DA CONSULTA ÚNICA. Lá cada par era (pergunta
inteira, trecho). Aqui cada par é (SUB-pergunta, trecho do pool DAQUELA
sub-pergunta) — é esse o ponto da decomposição: cada lado é pontuado por quem o
procura, e não pela pergunta inteira, que tende a favorecer um lado só.

A chave de cada entrada é "<qid>#sub1" ou "<qid>#sub2". O script do Colab não
precisa saber disso: ele pontua pares e devolve scores por chave. A fusão
acontece de volta aqui, onde as estratégias podem ser comparadas sem gastar GPU
de novo.

POR QUE O POOL DE 100 POR LADO. Medido em 29/09/2026: o pool unido de 100+100
cobre as duas citações em 59 das 75 comparativas, contra 48 da consulta única.
Cortar para 50+50 cobre 52 — ainda melhor que a consulta única, com menos
material, mas deixa 7 fora do alcance. Como o reranking é a parte cara e o
pool é a parte barata, vale levar o pool cheio.

DUAS FONTES, porque há dois experimentos de decomposição:

    comparativas  o de 29/09: só as 75 comparativas, com um prompt que ANUNCIAVA
                  ao modelo que a pergunta comparava duas normas. Serve de
                  controle.
    cega          o de 30/09: as 225 respondíveis dos três tipos, com o modelo
                  decidindo sozinho se a pergunta pede uma ou duas informações.
                  É esta que pode ir para produção, porque não consulta o tipo.

POR QUE ISTO PRECISA DO COLAB. Medido em 30/09 nesta máquina: o
`bge-reranker-v2-m3` leva ~2,4 s por par em CPU. O lote cego tem ~15 mil pares,
ou seja ~10 horas locais contra minutos em GPU.

Uso:
  python eval/experimento_embedding/preparar_rerank_subperguntas.py --aplicar
  python eval/experimento_embedding/preparar_rerank_subperguntas.py --fonte cega --aplicar
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parents[1]
sys.path.insert(0, str(RAIZ))

from src.embedding_text import build_embedding_text  # noqa: E402

GABARITO = RAIZ / "eval/data/golden_dezembro.jsonl"
CORPUS = RAIZ / "data/processed/documents.jsonl"
IDX = RAIZ / "eval/results/indexes"

FONTES = {
    "comparativas": {
        "subs": IDX / "subperguntas.json",
        "pools": IDX / "pools_subperguntas.json",
        "saida": IDX / "rerank_entrada_subperguntas.json",
        "base": "subperguntas",
        "volta_como": "rerank_scores_subperguntas.json",
    },
    "cega": {
        "subs": IDX / "decomposicao_cega.json",
        "pools": IDX / "pools_decomposicao_cega.json",
        "saida": IDX / "rerank_entrada_decomposicao_cega.json",
        "base": "decomposicao_cega",
        "volta_como": "rerank_scores_decomposicao_cega.json",
    },
}

for _f in (sys.stdout, sys.stderr):
    try:
        _f.reconfigure(encoding="utf-8")
    except Exception:
        pass


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fonte", choices=sorted(FONTES), default="comparativas")
    ap.add_argument("--profundidade", type=int, default=100)
    ap.add_argument("--max-length", type=int, default=384)
    ap.add_argument("--modelo", default="BAAI/bge-reranker-v2-m3")
    ap.add_argument("--aplicar", action="store_true")
    args = ap.parse_args()

    F = FONTES[args.fonte]
    for chave in ("subs", "pools"):
        if not F[chave].exists():
            print(f"✗ falta {F[chave].name} — gere os pools dessa fonte antes.")
            return

    gold = {g["qid"]: g for g in
            (json.loads(l) for l in GABARITO.open(encoding="utf-8") if l.strip())}
    subs = json.loads(F["subs"].read_text(encoding="utf-8"))
    pools = json.loads(F["pools"].read_text(encoding="utf-8"))
    SAIDA = F["saida"]

    registros = {}
    with CORPUS.open(encoding="utf-8") as fh:
        for linha in fh:
            if linha.strip():
                r = json.loads(linha)
                registros[r["id"]] = r

    perguntas, precisa = {}, set()
    for qid, p in pools.items():
        g = gold.get(qid)
        if not g:
            continue
        s = subs.get(g["question"])
        if not s:
            continue
        for lado in ("sub1", "sub2"):
            cands = [c for c in p[lado][:args.profundidade] if c in registros]
            perguntas[f"{qid}#{lado}"] = {"pergunta": s[lado], "candidatos": cands}
            precisa.update(cands)

    trechos = {cid: build_embedding_text(registros[cid]) for cid in sorted(precisa)}
    pares = sum(len(v["candidatos"]) for v in perguntas.values())

    print(f"fonte          : {args.fonte}")
    print(f"perguntas com pool: {len(pools)}")
    print(f"sub-perguntas  : {len(perguntas)}")
    print(f"pares          : {pares}")
    print(f"trechos únicos : {len(trechos)}")
    print(f"custo em CPU local, a 2,4 s por par: "
          f"{pares * 2.4 / 3600:.1f} h  → use o Colab")

    if not args.aplicar:
        print("\n(relatório apenas — use --aplicar para gravar)")
        return

    SAIDA.write_text(json.dumps({
        "modelo": args.modelo,
        "max_length": args.max_length,
        "max_candidates": args.profundidade,
        "texto": "contextual",
        "base": F["base"],
        "perguntas": perguntas,
        "trechos": trechos,
    }, ensure_ascii=False), encoding="utf-8")
    mb = SAIDA.stat().st_size / 1024 / 1024
    print(f"\n✓ {SAIDA}  ({mb:.1f} MB)")
    print("\nNo Colab: mesmo colab_rerank.py, trocando o nome do arquivo de entrada.")
    print(f"Volta como rerank_scores_bge_reranker_v2_m3.json — renomeie para")
    print(f"{F['volta_como']} e ponha em eval/results/indexes/,")
    print("para não sobrescrever o da consulta única.")


if __name__ == "__main__":
    main()
