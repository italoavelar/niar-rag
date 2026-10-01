#!/usr/bin/env python3
"""Empacota um arquivo único para o reranker rodar em GPU no Colab.

POR QUE O TEXTO VAI PRONTO NO PACOTE. O reranker pontua o texto CONTEXTUAL — o
trecho com o prefixo que `src/embedding_text.py` monta (título, emissor, seção).
Se o Colab recalculasse esse prefixo, bastaria uma diferença de versão para
pontuar um texto e o pipeline local medir outro, em silêncio. Foi exatamente esse
o defeito que quebrou o índice BGE em 27/09: o `.npz` saiu sem
`embedding_text_profile` e o cenário morreu depois de 59 minutos de CPU gastos.
Aqui o texto sai daqui pronto, e o Colab só multiplica matriz.

POR QUE UM ARQUIVO SÓ, E COM OS TRECHOS DEDUPLICADOS. Mandar (pergunta, texto)
par a par daria ~50 MB, porque o mesmo trecho aparece no pool de muitas
perguntas. Separando em `perguntas` (só ids) e `trechos` (texto uma vez), o
pacote cai para poucos megabytes e sobe rápido.

Uso:
  python eval/experimento_embedding/preparar_rerank_colab.py
  python eval/experimento_embedding/preparar_rerank_colab.py --rankings B_dense_bge_m3 --aplicar
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
PROJECT_ROOT = AQUI.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.embedding_text import build_embedding_text  # noqa: E402

RANKINGS = PROJECT_ROOT / "eval/results/retrieval/rankings"
CORPUS = PROJECT_ROOT / "data/processed/documents.jsonl"
GOLDEN = PROJECT_ROOT / "eval/data/golden_acervo.jsonl"
SAIDA = PROJECT_ROOT / "eval/results/indexes/rerank_entrada.json"

for _f in (sys.stdout, sys.stderr):
    try:
        _f.reconfigure(encoding="utf-8")
    except Exception:
        pass


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rankings", default="B_dense_bge_m3",
                    help="Nome do braço cujo pool será reordenado.")
    ap.add_argument("--max-candidates", type=int, default=100)
    ap.add_argument("--max-length", type=int, default=384)
    ap.add_argument("--modelo", default="BAAI/bge-reranker-v2-m3")
    ap.add_argument("--aplicar", action="store_true")
    args = ap.parse_args()

    caminho = RANKINGS / f"{args.rankings}.json"
    if not caminho.exists():
        print(f"✗ ranking não encontrado: {caminho}")
        print("  Rode o braço antes (eval/retrieval_eval/scenario_B_dense_bge.py).")
        return
    ranking = json.loads(caminho.read_text(encoding="utf-8"))

    registros = {}
    with CORPUS.open(encoding="utf-8") as fh:
        for linha in fh:
            if linha.strip():
                r = json.loads(linha)
                registros[r["id"]] = r

    gold = {g["qid"]: g for g in
            (json.loads(l) for l in GOLDEN.open(encoding="utf-8") if l.strip())}

    perguntas, precisa = {}, set()
    ausentes = 0
    for qid, lista in ranking.items():
        if qid not in gold:
            continue
        candidatos = [c for c in lista[:args.max_candidates] if c in registros]
        ausentes += len(lista[:args.max_candidates]) - len(candidatos)
        perguntas[qid] = {"pergunta": gold[qid]["question"], "candidatos": candidatos}
        precisa.update(candidatos)

    trechos = {cid: build_embedding_text(registros[cid]) for cid in sorted(precisa)}
    pares = sum(len(v["candidatos"]) for v in perguntas.values())

    pacote = {
        "modelo": args.modelo,
        "max_length": args.max_length,
        "max_candidates": args.max_candidates,
        "texto": "contextual",
        "base": args.rankings,
        "perguntas": perguntas,
        "trechos": trechos,
    }

    print(f"braço     : {args.rankings}")
    print(f"perguntas : {len(perguntas)}")
    print(f"pares     : {pares}")
    print(f"trechos únicos: {len(trechos)}  (sem deduplicar seriam {pares})")
    if ausentes:
        print(f"✗ {ausentes} ids do ranking não existem no corpus atual — ranking velho?")
    if not perguntas:
        print("✗ nenhuma pergunta casou com o gabarito. Ranking e gabarito são de épocas diferentes.")
        return

    if not args.aplicar:
        print("\n(relatório apenas — use --aplicar para gravar)")
        return

    SAIDA.parent.mkdir(parents=True, exist_ok=True)
    SAIDA.write_text(json.dumps(pacote, ensure_ascii=False), encoding="utf-8")
    mb = SAIDA.stat().st_size / 1024 / 1024
    print(f"\n✓ {SAIDA}  ({mb:.1f} MB)")
    print("\nSuba no Colab: este arquivo + eval/experimento_embedding/colab_rerank.py")
    print("Traga de volta: rerank_scores_bge_reranker_v2_m3.json")
    print("Depois, aqui:")
    print("  python eval/tools/rerank_eval.py \\")
    print(f"    --rankings eval/results/retrieval/rankings/{args.rankings}.json \\")
    print("    --scores rerank_scores_bge_reranker_v2_m3.json \\")
    print("    --out results_dezembro_rerank")


if __name__ == "__main__":
    main()
