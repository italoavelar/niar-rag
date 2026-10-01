#!/usr/bin/env python3
"""Três fechamentos antes de decidir o que vai para produção.

(1) LIMIAR DE ABSTENÇÃO, ESCOLHIDO E MEDIDO EM METADES DIFERENTES.
    O 0,50 foi escolhido olhando as mesmas 125 irrespondíveis em
    que seria avaliado — ajuste no conjunto de teste, e o número em produção
    seria pior. Aqui o limiar é escolhido na metade A e medido na metade B, com
    semente fixa. A diferença entre as duas é a estimativa honesta do otimismo.

(2) O RERANKER PIORA 38 CONSULTAS. ELAS TÊM ALGO EM COMUM?
    Se tiverem, existe regra barata de "não reranquear quando X" e o ganho sobe
    sem custo nenhum. Se não tiverem, é ruído e não há o que fazer.

(3) VALE ANOTAR O POOL?
    A pergunta prática: entre os trechos que a busca traz no top-5 e que NÃO
    estão no gabarito, quantos parecem responder mesmo assim? Se forem muitos, o
    nDCG está punindo acerto e o pool precisa de julgamento. Se forem poucos, o
    gabarito derivado já é quase completo no topo e o pool pode esperar.

    Dois indicadores independentes, nenhum deles julgamento humano:
      nota do reranker  — é um modelo TREINADO para julgar o par; se dá a um
                          trecho não anotado nota igual à do anotado, ou o
                          trecho é relevante ou o juiz não serve.
      sinal da resposta — fração dos termos distintivos da resposta-referência
                          que o trecho contém. Independente do reranker.
    Concordar entre si é o que dá confiança em qualquer dos dois.

Uso:
  python eval/experimento_embedding/fechamento_dezembro.py
"""

from __future__ import annotations

import json
import random
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parents[1]

# ACERVO INTEIRO DE PROPÓSITO: o limiar de abstenção é estimado contra os
# negativos, e aqui são 125 irrespondíveis em vez das 75 da vista balanceada do
# artigo. Mais negativos, metades maiores, limiar menos ruidoso. Ver a nota em
# `abstencao_e_idioma.py`.
GABARITO = RAIZ / "eval/data/golden_acervo.jsonl"
SCORES = RAIZ / "eval/results/indexes/rerank_scores_bge_reranker_v2_m3.json"
R = RAIZ / "eval/results/retrieval/rankings"
CORPUS = RAIZ / "data/processed/documents.jsonl"
SEMENTE = 2026
K = 5

for _f in (sys.stdout, sys.stderr):
    try:
        _f.reconfigure(encoding="utf-8")
    except Exception:
        pass


def sinais(texto: str) -> set[str]:
    base = "".join(c for c in unicodedata.normalize("NFKD", texto or "")
                   if not unicodedata.combining(c))
    return ({w.lower() for w in re.findall(r"[A-Za-z]{5,}", base)}
            | set(re.findall(r"\d[\d.,]*", base))
            | {s.lower() for s in re.findall(r"\b[A-Z]{3,}\b", base)})


def main() -> None:
    gold = {g["qid"]: g for g in
            (json.loads(l) for l in GABARITO.open(encoding="utf-8") if l.strip())}
    notas = json.loads(SCORES.read_text(encoding="utf-8"))["scores"]
    rr = json.loads((R / "B_bge_m3_rerank.json").read_text(encoding="utf-8"))
    bge = json.loads((R / "B_dense_bge_m3.json").read_text(encoding="utf-8"))
    gem = json.loads((R / "B_dense_gemini.json").read_text(encoding="utf-8"))
    textos = {}
    with CORPUS.open(encoding="utf-8") as fh:
        for linha in fh:
            if linha.strip():
                r = json.loads(linha)
                textos[r["id"]] = r["text"]

    # ═══ (1) limiar honesto ═════════════════════════════════════════════════
    print("(1) LIMIAR DE ABSTENÇÃO — escolhido numa metade, medido na outra\n")
    resp = [(q, max(notas[q].values())) for q in notas
            if q in gold and gold[q]["qrels_grupos"] and notas[q]]
    irre = [(q, max(notas[q].values())) for q in notas
            if q in gold and not gold[q]["qrels_grupos"] and notas[q]]
    rnd = random.Random(SEMENTE)
    rnd.shuffle(resp)
    rnd.shuffle(irre)
    mr, mi = len(resp) // 2, len(irre) // 2
    A = ([v for _, v in resp[:mr]], [v for _, v in irre[:mi]])
    B = ([v for _, v in resp[mr:]], [v for _, v in irre[mi:]])
    print(f"  metade A: {len(A[0])} respondíveis, {len(A[1])} irrespondíveis")
    print(f"  metade B: {len(B[0])} respondíveis, {len(B[1])} irrespondíveis\n")

    def avaliar(meia, lim):
        ok_i = sum(1 for v in meia[1] if v < lim)       # recusa certa
        erro_r = sum(1 for v in meia[0] if v < lim)     # recusa errada
        return ok_i / len(meia[1]), erro_r / len(meia[0])

    # critério de escolha: melhor recusa certa com no máximo 10% de perda
    TETO_PERDA = 0.10
    melhor, melhor_lim = -1.0, None
    for i in range(1, 1000):
        lim = i / 1000
        acerto, perda = avaliar(A, lim)
        if perda <= TETO_PERDA and acerto > melhor:
            melhor, melhor_lim = acerto, lim
    a_ac, a_pe = avaliar(A, melhor_lim)
    b_ac, b_pe = avaliar(B, melhor_lim)
    print(f"  critério: maior recusa correta com perda ≤ {TETO_PERDA:.0%}")
    print(f"  limiar escolhido em A: {melhor_lim:.3f}\n")
    print(f"{'':10s}{'recusa correta':>18s}{'recusa indevida':>18s}")
    print("  " + "─" * 44)
    print(f"  {'em A':8s}{a_ac:17.1%}{a_pe:18.1%}   ← onde foi escolhido")
    print(f"  {'em B':8s}{b_ac:17.1%}{b_pe:18.1%}   ← o número honesto")
    print(f"\n  otimismo do ajuste: {a_ac - b_ac:+.1%} de recusa correta")

    # ═══ (2) quem piora com o reranker ══════════════════════════════════════
    print("\n\n(2) AS CONSULTAS QUE O RERANKER PIORA\n")

    def junta(ordem, grupos):
        t = set(ordem[:K])
        return int(all(any(c in t for c in gr) for gr in grupos))

    piores, melhores = [], []
    for q, g in gold.items():
        if not g["qrels_grupos"] or q not in rr or q not in bge:
            continue
        a, b = junta(bge[q], g["qrels_grupos"]), junta(rr[q], g["qrels_grupos"])
        if b < a:
            piores.append(q)
        elif b > a:
            melhores.append(q)
    print(f"  piora: {len(piores)}   melhora: {len(melhores)}\n")
    for campo in ("question_type", "source_lang"):
        cp = defaultdict(int)
        cm = defaultdict(int)
        ctot = defaultdict(int)
        for q, g in gold.items():
            if g["qrels_grupos"] and q in rr:
                ctot[g[campo]] += 1
        for q in piores:
            cp[gold[q][campo]] += 1
        for q in melhores:
            cm[gold[q][campo]] += 1
        print(f"  por {campo}:")
        for k in sorted(ctot):
            print(f"     {k:14s} piora {cp[k]:3d}  melhora {cm[k]:3d}  "
                  f"de {ctot[k]:3d}")
        print()

    # ═══ (3) vale anotar o pool? ════════════════════════════════════════════
    print("\n(3) OS TRECHOS NÃO ANOTADOS DO TOP-5 PARECEM RESPONDER?\n")
    linhas = []
    for q, g in gold.items():
        if not g["qrels_grupos"] or q not in rr or q not in gem:
            continue
        anotados = {c for gr in g["qrels_grupos"] for c in gr}
        alvo = sinais(g.get("reference_answer") or "")
        if not alvo:
            continue
        piso = min((notas[q].get(c, 0.0) for c in anotados if c in notas.get(q, {})),
                   default=None)
        pool = list(dict.fromkeys(rr[q][:K] + gem[q][:K]))
        for c in pool:
            if c in anotados:
                continue
            nota = notas.get(q, {}).get(c)
            cob = len(sinais(textos.get(c, "")) & alvo) / len(alvo)
            linhas.append((q, c, nota, cob, piso))

    com_nota = [x for x in linhas if x[2] is not None and x[4] is not None]
    alto_juiz = [x for x in com_nota if x[2] >= x[4]]
    alto_sinal = [x for x in linhas if x[3] >= 0.30]
    ambos = [x for x in com_nota if x[2] >= x[4] and x[3] >= 0.30]
    print(f"  trechos não anotados no top-5 (união BGE+rr e Gemini): {len(linhas)}")
    print(f"    com nota do reranker ≥ a do PIOR trecho anotado : {len(alto_juiz)}"
          f"  ({100*len(alto_juiz)/max(len(com_nota),1):.0f}% dos comparáveis)")
    print(f"    com ≥30% dos sinais da resposta-referência       : {len(alto_sinal)}"
          f"  ({100*len(alto_sinal)/max(len(linhas),1):.0f}%)")
    print(f"    OS DOIS ao mesmo tempo                           : {len(ambos)}"
          f"  ({100*len(ambos)/max(len(com_nota),1):.0f}%)")
    perguntas_afetadas = len({x[0] for x in ambos})
    print(f"\n  perguntas com ao menos um desses: {perguntas_afetadas} de "
          f"{len({x[0] for x in linhas})}")
    print("\n  Estes são os candidatos que o pool julgaria e o gabarito atual")
    print("  conta como zero. Quanto maior, mais o nDCG está punindo acerto.")

    if ambos:
        print("\n  ── três exemplos para leitura ──")
        for q, c, nota, cob, piso in sorted(ambos, key=lambda x: -x[3])[:3]:
            print(f"\n  [{q}] nota {nota:.3f} (piso anotado {piso:.3f}) · "
                  f"sinal {cob:.0%}")
            print(f"     P: {gold[q]['question'][:100]}")
            print(f"     T: {textos.get(c, '')[:190]}")


if __name__ == "__main__":
    main()
