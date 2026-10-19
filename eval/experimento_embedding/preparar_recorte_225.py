#!/usr/bin/env python3
"""Réplica do experimento de recorte na população final — o pacote para o Colab.

POR QUE A RÉPLICA EXISTE. O recorte de 1.200 foi escolhido em 22/09 sobre 91
perguntas que eram **56 factuais e 35 multi-hop: zero comparativa e zero
irrespondível**, e em 15 dos 54 documentos. Em 05/10 o corte por janela mostrou
que a vantagem do 1.200 se concentra nas perguntas nascidas de passagem de 1400
caracteres — a mais próxima do tamanho dele. Duas objeções, as duas legítimas.

ESTA RÉPLICA RESPONDE AS DUAS: 54 documentos (não 15) e as **225 respondíveis do
subconjunto balanceado**, com 75 comparativas e as multi-hop de evidência
dispersa que não existiam em setembro. É um experimento melhor que o original,
não uma repetição.

SÓ O LADO ABERTO. O BGE-m3 roda em GPU do Colab sem custo de API. O braço Gemini
exigiria ~100 mil chamadas de embedding de documento e fica fora: a pergunta aqui
é sobre o RECORTE, e o recorte é a mesma variável nos dois embeddings.

O QUE SOBE E O QUE DESCE. Sobe um pacote com o texto de embedding de cada trecho
das quatro condições, já pronto — como no `colab_rerank.py`, o Colab não remonta
nada, senão uma diferença de versão faria o Colab pontuar um texto e a análise
local medir outro. Desce, por condição, apenas a ORDEM do ranking truncada em 300
— alguns KB em vez das centenas de MB que as matrizes custariam.

Uso:
  python eval/experimento_embedding/preparar_recorte_225.py
  python eval/experimento_embedding/preparar_recorte_225.py --condicoes T400 T1200
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parents[1]
sys.path.insert(0, str(RAIZ / "src"))

GABARITO = RAIZ / "eval/data/golden_dezembro.jsonl"
DADOS = AQUI / "dados_225"
PACOTE = DADOS / "recorte225_pacote.json.gz"
PERGUNTAS = DADOS / "perguntas.json"

CONDICOES = ("T400", "T1200", "T3500", "PC")
PROFUNDIDADE = 300   # fundo de sobra: o maior orçamento gasta ~31 unidades

for _f in (sys.stdout, sys.stderr):
    try:
        _f.reconfigure(encoding="utf-8")
    except Exception:
        pass


def montar_perguntas() -> list[dict]:
    """As 225 respondíveis, na forma que o `analisar_recorte.py` já consome.

    `evidencia` é a lista de citações LITERAIS. O casamento na análise é por
    substring no texto devolvido, e não por id de trecho — tem de ser, porque
    cada condição recorta o corpus com ids próprios.

    `janela` não existe nestas perguntas: o campo era do sorteio de passagens de
    setembro. Fica em 0 e o corte que interessa aqui é por `question_type`.
    """
    saida = []
    for linha in GABARITO.open(encoding="utf-8"):
        if not linha.strip():
            continue
        g = json.loads(linha)
        if not g.get("qrels_grupos"):
            continue
        cits = g.get("qrels_text") or []
        if not cits:
            continue
        saida.append({
            "n": len(saida) + 1,
            "qid": g["qid"],
            "question": g["question"],
            "evidencia": cits,
            "n_evidencias": len(cits),
            "question_type": g["question_type"],
            "source_lang": g["source_lang"],
            "origem": g["origem"],
            "janela": 0,
        })
    return saida


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--condicoes", nargs="+", default=list(CONDICOES))
    ap.add_argument("--profundidade", type=int, default=PROFUNDIDADE)
    args = ap.parse_args()

    from embedding_text import build_embedding_text

    perguntas = montar_perguntas()
    PERGUNTAS.write_text(json.dumps(perguntas, ensure_ascii=False, indent=1),
                         encoding="utf-8")
    import collections
    tipos = collections.Counter(p["question_type"] for p in perguntas)
    print(f"✓ {PERGUNTAS.name}  {len(perguntas)} perguntas respondíveis")
    for t, n in sorted(tipos.items()):
        print(f"    {t:14s} {n:3d}")
    print(f"    citações exigidas: {sum(p['n_evidencias'] for p in perguntas)}\n")

    pacote = {"modelo": "BAAI/bge-m3", "profundidade": args.profundidade,
              "perguntas": [p["question"] for p in perguntas], "condicoes": {}}
    for cond in args.condicoes:
        arq = DADOS / f"{cond}.jsonl"
        if not arq.exists():
            print(f"✗ falta {arq.name} — rode antes:")
            print("    python eval/experimento_embedding/bench_tamanho.py recortar \\")
            print("      --amostra dados/amostra_completa.json --destino dados_225 \\")
            print("      --tamanhos 400 1200 3500")
            return
        textos, ids = [], []
        for linha in arq.open(encoding="utf-8"):
            if linha.strip():
                r = json.loads(linha)
                textos.append(build_embedding_text(r))
                ids.append(r["id"])
        pacote["condicoes"][cond] = {"ids": ids, "textos": textos}
        print(f"  {cond:7s} {len(textos):6d} trechos")

    total = sum(len(v["textos"]) for v in pacote["condicoes"].values())
    DADOS.mkdir(parents=True, exist_ok=True)
    with gzip.open(PACOTE, "wt", encoding="utf-8") as f:
        json.dump(pacote, f, ensure_ascii=False)
    mb = PACOTE.stat().st_size / 1e6
    print(f"\n✓ {PACOTE.name}  {mb:.1f} MB  ({total} trechos no total)")
    print(f"\nSuba no Colab com GPU: {PACOTE.name} e colab_recorte_225.py")
    print("Volta um emb_<condição>.npz por condição, com a ordem do ranking.")
    print("Ponha os .npz em eval/experimento_embedding/resultados_225/ e rode:")
    print("  python eval/experimento_embedding/analisar_recorte.py --modelo bge \\")
    print("    --dados dados_225 --resultados resultados_225 --agrupar-por tipo")


if __name__ == "__main__":
    main()
