#!/usr/bin/env python3
"""Por que a comparativa do piloto é três vezes mais difícil que a gerada?

BGE+reranker acerta 2 de 25 nas comparativas do piloto e 12 de 50 nas que foram
escritas em 27/09 — 0,080 contra 0,240. A diferença é grande demais para ser
acaso e a explicação muda o que fazer: se for propriedade da PERGUNTA, é viés de
quem escreveu e vai para as limitações; se for propriedade da CITAÇÃO, diz o que
a busca não consegue achar, que é a pergunta mais produtiva.

Mede-se quatro coisas por procedência, todas sobre o mesmo braço:

  profundidade   em que posição do ranking COMPLETO está o trecho de que a
                 pergunta precisa. É a medida direta de "a busca acha isto?".
  n_evidencias   quantas citações a pergunta exige. Exigir 3 em vez de 2 baixa
                 o teto sem que a busca tenha piorado.
  tamanho        citação curta tem menos sinal para casar.
  especificidade fração de sinais RAROS na citação (números, siglas, termos que
                 aparecem em poucos documentos). Citação de prosa genérica
                 ("deve ser garantida a transparência") não tem onde ancorar.

Uso:
  python eval/experimento_embedding/diagnostico_comparativas.py
"""

from __future__ import annotations

import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parents[1]
GABARITO = RAIZ / "eval/data/golden_dezembro.jsonl"
RANKING = RAIZ / "eval/results/retrieval/rankings/B_bge_m3_rerank.json"
BASE = RAIZ / "eval/results/retrieval/rankings/B_dense_bge_m3.json"
CORPUS = RAIZ / "data/processed/documents.jsonl"

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


def mediana(v):
    v = sorted(v)
    return v[len(v) // 2] if v else None


def main() -> None:
    gold = {g["qid"]: g for g in
            (json.loads(l) for l in GABARITO.open(encoding="utf-8") if l.strip())}
    rr = json.loads(RANKING.read_text(encoding="utf-8"))
    base = json.loads(BASE.read_text(encoding="utf-8"))

    textos, doc_freq = {}, Counter()
    with CORPUS.open(encoding="utf-8") as fh:
        for linha in fh:
            if linha.strip():
                r = json.loads(linha)
                textos[r["id"]] = r["text"]
    # em quantos DOCUMENTOS cada sinal aparece — para medir raridade
    por_doc = defaultdict(set)
    with CORPUS.open(encoding="utf-8") as fh:
        for linha in fh:
            if linha.strip():
                r = json.loads(linha)
                por_doc[r["metadata"]["document_id"]].update(sinais(r["text"]))
    for s in por_doc.values():
        doc_freq.update(s)
    n_docs = len(por_doc)

    comp = [q for q, g in gold.items()
            if g["question_type"] == "comparative" and g["qrels_grupos"] and q in rr]
    por_origem = defaultdict(list)
    for q in comp:
        por_origem[gold[q]["origem"]].append(q)

    print(f"comparativas: {len(comp)}  ·  corpus: {n_docs} documentos\n")
    cab = (f"{'origem':24s}{'n':>4s}{'prof.mediana':>14s}{'>100':>7s}"
           f"{'n_evid':>8s}{'tam.mediano':>13s}{'raros%':>8s}")
    print(cab)
    print("─" * len(cab))

    for origem, qs in sorted(por_origem.items()):
        profs, fora, nev, tams, raros = [], 0, [], [], []
        for q in qs:
            g = gold[q]
            pos = {c: i for i, c in enumerate(base[q])}
            for gr in g["qrels_grupos"]:
                p = min((pos[c] for c in gr if c in pos), default=None)
                if p is None:
                    fora += 1
                else:
                    profs.append(p + 1)
            nev.append(len(g["qrels_grupos"]))
            for cit in g["qrels_text"]:
                tams.append(len(cit))
                s = sinais(cit)
                if s:
                    raros.append(sum(1 for x in s if doc_freq[x] <= 3) / len(s))
        print(f"{origem:24s}{len(qs):4d}{mediana(profs) or 0:14d}{fora:7d}"
              f"{mediana(nev) or 0:8d}{mediana(tams) or 0:13d}"
              f"{100*sum(raros)/len(raros) if raros else 0:7.1f}%")

    print("\n  prof.mediana = posição da citação no ranking DENSO completo (sem reranker)")
    print("  >100 = citações que nem aparecem no top-100")
    print("  raros% = fração dos sinais da citação que ocorrem em ≤3 documentos")

    print("\n\nAs citações mais fundas do piloto (o que a busca não acha):\n")
    fundas = []
    for q in por_origem.get("piloto", []):
        pos = {c: i for i, c in enumerate(base[q])}
        for gr, cit in zip(gold[q]["qrels_grupos"], gold[q]["qrels_text"]):
            p = min((pos[c] for c in gr if c in pos), default=10**6)
            fundas.append((p, q, cit))
    for p, q, cit in sorted(fundas, reverse=True)[:6]:
        onde = "fora do top-100" if p >= 10**6 else f"posição {p+1}"
        print(f"  [{q}] {onde}")
        print(f"      {cit[:150]}")


if __name__ == "__main__":
    main()
