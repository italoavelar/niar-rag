#!/usr/bin/env python3
"""Regras baratas de seleção do top-5, sobre a lista JÁ reranqueada.

POR QUE REABRIR ALGO JÁ REFUTADO. Diversificação no top-5 foi testada três vezes
e sempre trocou comparativa por multi-hop. MAS todas as três mediram sobre o
conjunto de perguntas ANTIGO, em que 42 das 101 `multi_hop` tinham as duas
citações DENTRO DO MESMO TRECHO (descoberto em 27/09/2026). Nessas, diversidade
é ativamente nociva: a resposta inteira está num trecho só, e qualquer regra que
espalhe o top-5 tira o trecho certo para pôr um diferente.

Ou seja: o resultado negativo pode ser artefato do gabarito, não da técnica.
Com o tipo corrigido — toda `multi_hop` agora exige trechos DIFERENTES — a
medição precisa ser refeita. É barata: os rankings já existem, os vetores já
existem, nada de API nem GPU.

AS REGRAS

    base          o top-5 reranqueado, sem mexer
    MMR           penaliza candidato parecido com o que já foi escolhido
                  (similaridade de cosseno entre trechos)
    vizinhança    recusa candidato a menos de N posições, NO MESMO DOCUMENTO,
                  de alguém já escolhido
    teto/doc      no máximo N trechos por documento (o controle que já falhou)

A REGRA DE VIZINHANÇA É NOVA E É A MAIS BARATA. O recorte de produção tem 240
caracteres de sobreposição: dois trechos vizinhos compartilham texto por
construção, então o segundo quase nada acrescenta. Mas dois trechos do mesmo
documento a 40 posições de distância não são redundantes — são as duas pontas
que uma multi-hop exige. O teto por documento não distingue os dois casos e por
isso destrói multi-hop; a vizinhança distingue, e não precisa de vetor nenhum.

Uso:
  python eval/experimento_embedding/selecao_diversa.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parents[1]

GABARITO = RAIZ / "eval/data/golden_dezembro.jsonl"
RERANK = RAIZ / "eval/results/retrieval/rankings/B_bge_m3_rerank.json"
INDICE = RAIZ / "eval/results/indexes/dense_bge_m3_context-v1_c656b175b4ece716.npz"
CORPUS = RAIZ / "data/processed/documents.jsonl"
K = 5
POOL = 30          # de onde as regras escolhem; além disso o reranker já ordenou mal

for _f in (sys.stdout, sys.stderr):
    try:
        _f.reconfigure(encoding="utf-8")
    except Exception:
        pass


def main() -> None:
    gold = {g["qid"]: g for g in
            (json.loads(l) for l in GABARITO.open(encoding="utf-8") if l.strip())}
    rr = json.loads(RERANK.read_text(encoding="utf-8"))

    doc_de, pos_de = {}, {}
    with CORPUS.open(encoding="utf-8") as fh:
        for i, linha in enumerate(fh):
            if linha.strip():
                r = json.loads(linha)
                doc_de[r["id"]] = r["metadata"]["document_id"]
                pos_de[r["id"]] = i

    with np.load(INDICE, allow_pickle=True) as d:
        matriz = d["matrix"].astype(np.float32)
        linha_de = {str(c): i for i, c in enumerate(d["ids"])}

    def sim(a: str, b: str) -> float:
        ia, ib = linha_de.get(a), linha_de.get(b)
        if ia is None or ib is None:
            return 0.0
        return float(matriz[ia] @ matriz[ib])

    def base(cands, k):
        return cands[:k]

    def mmr(cands, k, lamb):
        escolhidos = []
        restantes = list(cands)
        # relevância = posição no ranking reranqueado, normalizada
        rel = {c: 1.0 - i / max(len(cands) - 1, 1) for i, c in enumerate(cands)}
        while restantes and len(escolhidos) < k:
            melhor, nota_melhor = None, -1e9
            for c in restantes:
                red = max((sim(c, e) for e in escolhidos), default=0.0)
                nota = lamb * rel[c] - (1 - lamb) * red
                if nota > nota_melhor:
                    melhor, nota_melhor = c, nota
            escolhidos.append(melhor)
            restantes.remove(melhor)
        return escolhidos

    def vizinhanca(cands, k, dist):
        escolhidos = []
        for c in cands:
            if len(escolhidos) >= k:
                break
            perto = any(doc_de.get(c) == doc_de.get(e)
                        and abs(pos_de.get(c, 0) - pos_de.get(e, 0)) < dist
                        for e in escolhidos)
            if not perto:
                escolhidos.append(c)
        # se sobrou vaga (regra recusou demais), completa pela ordem original
        for c in cands:
            if len(escolhidos) >= k:
                break
            if c not in escolhidos:
                escolhidos.append(c)
        return escolhidos

    def teto_doc(cands, k, teto):
        conta, escolhidos = defaultdict(int), []
        for c in cands:
            if len(escolhidos) >= k:
                break
            d = doc_de.get(c)
            if conta[d] < teto:
                conta[d] += 1
                escolhidos.append(c)
        for c in cands:
            if len(escolhidos) >= k:
                break
            if c not in escolhidos:
                escolhidos.append(c)
        return escolhidos

    REGRAS = [("base", lambda c, k: base(c, k))]
    REGRAS += [(f"MMR λ={l}", (lambda l: lambda c, k: mmr(c, k, l))(l))
               for l in (0.9, 0.8, 0.7)]
    REGRAS += [(f"vizinhança {d}", (lambda d: lambda c, k: vizinhanca(c, k, d))(d))
               for d in (1, 2, 3, 5)]
    REGRAS += [(f"teto/doc {t}", (lambda t: lambda c, k: teto_doc(c, k, t))(t))
               for t in (2, 3)]

    resp = [q for q, g in gold.items() if g["qrels_grupos"] and q in rr]
    tipos = ("comparative", "multi_hop", "factual")
    print(f"perguntas respondíveis: {len(resp)}  ·  pool de seleção: {POOL}\n")
    cab = f"{'regra':16s}{'GERAL':>12s}" + "".join(f"{t[:11]:>14s}" for t in tipos)
    print(cab)
    print("─" * len(cab))

    for nome, fn in REGRAS:
        acertos, por_tipo = 0, defaultdict(int)
        n_tipo = defaultdict(int)
        for q in resp:
            g = gold[q]
            escolha = set(fn(rr[q][:POOL], K))
            v = int(all(any(c in escolha for c in gr) for gr in g["qrels_grupos"]))
            acertos += v
            por_tipo[g["question_type"]] += v
            n_tipo[g["question_type"]] += 1
        linha = f"{nome:16s}{acertos:5d} ({acertos/len(resp):.3f})"
        for t in tipos:
            linha += f"{por_tipo[t]:7d} ({n_tipo[t]:3d})"
        print(linha)

    print("\n  Números absolutos; entre parênteses, a fração (GERAL) ou o n do tipo.")


if __name__ == "__main__":
    main()
