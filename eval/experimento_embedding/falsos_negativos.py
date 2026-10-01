#!/usr/bin/env python3
"""Junta@5 conta erro onde o usuário teria sido servido?

A AFIRMAÇÃO QUE PRECISA SER TESTADA. Disse-se que anotar o pool não afeta
Junta@5, porque "se a pergunta precisa do trecho X e o braço trouxe X, é acerto,
seja o que for julgado". Isso é verdade DADO o conjunto de citações exigidas. Mas
esconde a premissa: que o conjunto exigido é o certo.

Se a resposta também está num trecho Y que ninguém anotou, e o sistema traz Y sem
trazer X, o Junta@5 marca ERRO e o usuário teria recebido a resposta. Nesse caso
a métrica SUBESTIMA o sistema, e a subestimação é justamente o que o pool
corrigiria. Em 27/09 mediu-se que 6,5% das perguntas têm, no mesmo documento e
sem anotação, um trecho com sinal mais forte que o anotado — indício de que o
caso existe.

O QUE SE MEDE AQUI. Só as perguntas que o braço ERRA. Para cada uma, procura-se
no top-5 um trecho NÃO anotado que pareça responder, por dois indicadores
independentes:

    nota do reranker  ≥ a nota do PIOR trecho anotado da mesma pergunta
    sinal da resposta ≥ limiar da fração de termos distintivos da referência

Os que passam nos dois são candidatos a falso negativo da métrica, e vêm
impressos para leitura — nenhum indicador substitui ler.

O QUE ISTO NÃO RESOLVE. O indicador de sinal confunde vocabulário jurídico
genérico com conteúdo; já se viu ele dar 67% a um trecho sobre carteira
profissional numa pergunta sobre penas disciplinares. Por isso o número final
sai como FAIXA, com a leitura humana definindo onde cair dentro dela.

Uso:
  python eval/experimento_embedding/falsos_negativos.py
  python eval/experimento_embedding/falsos_negativos.py --ler 12
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parents[1]

# ACERVO INTEIRO DE PROPÓSITO: isto não compara braços, mede se o GABARITO
# subconta. Quanto mais perguntas erradas houver para examinar, melhor a
# estimativa — no acervo são 108 erros contra os ~104 da vista balanceada. O
# número que sai daqui é uma propriedade do gabarito, não da população, e o
# relatório o cita rotulado como "no acervo completo".
GABARITO = RAIZ / "eval/data/golden_acervo.jsonl"
SCORES = RAIZ / "eval/results/indexes/rerank_scores_bge_reranker_v2_m3.json"
RANK = RAIZ / "eval/results/retrieval/rankings/B_bge_m3_rerank.json"
CORPUS = RAIZ / "data/processed/documents.jsonl"
K = 5

for _f in (sys.stdout, sys.stderr):
    try:
        _f.reconfigure(encoding="utf-8")
    except Exception:
        pass


def sinais(t: str) -> set[str]:
    base = "".join(c for c in unicodedata.normalize("NFKD", t or "")
                   if not unicodedata.combining(c))
    return ({w.lower() for w in re.findall(r"[A-Za-z]{5,}", base)}
            | set(re.findall(r"\d[\d.,]*", base))
            | {s.lower() for s in re.findall(r"\b[A-Z]{3,}\b", base)})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ler", type=int, default=6, help="quantos candidatos imprimir")
    args = ap.parse_args()

    gold = {g["qid"]: g for g in
            (json.loads(l) for l in GABARITO.open(encoding="utf-8") if l.strip())}
    notas = json.loads(SCORES.read_text(encoding="utf-8"))["scores"]
    rr = json.loads(RANK.read_text(encoding="utf-8"))
    textos, doc_de = {}, {}
    with CORPUS.open(encoding="utf-8") as fh:
        for linha in fh:
            if linha.strip():
                r = json.loads(linha)
                textos[r["id"]] = r["text"]
                doc_de[r["id"]] = r["metadata"]["document_id"]

    resp = [q for q, g in gold.items() if g["qrels_grupos"] and q in rr]
    erros = [q for q in resp
             if not all(any(c in set(rr[q][:K]) for c in gr)
                        for gr in gold[q]["qrels_grupos"])]
    print(f"respondíveis: {len(resp)}   ·   ERRA no top-5: {len(erros)}\n")

    candidatos = []
    for q in erros:
        g = gold[q]
        anotados = {c for gr in g["qrels_grupos"] for c in gr}
        alvo = sinais(g.get("reference_answer") or "")
        if not alvo:
            continue
        nq = notas.get(q, {})
        piso = min((nq[c] for c in anotados if c in nq), default=None)
        docs_exigidos = {doc_de.get(c) for gr in g["qrels_grupos"] for c in gr}
        for c in rr[q][:K]:
            if c in anotados:
                continue
            nota = nq.get(c)
            cob = len(sinais(textos.get(c, "")) & alvo) / len(alvo)
            mesmo_doc = doc_de.get(c) in docs_exigidos
            candidatos.append({"q": q, "c": c, "nota": nota, "cob": cob,
                               "piso": piso, "mesmo_doc": mesmo_doc})

    def conta(min_cob, exigir_nota=True, exigir_doc=False):
        sel = [x for x in candidatos if x["cob"] >= min_cob
               and (not exigir_nota or (x["nota"] is not None and x["piso"] is not None
                                        and x["nota"] >= x["piso"]))
               and (not exigir_doc or x["mesmo_doc"])]
        return len(sel), len({x["q"] for x in sel})

    print("Candidatos a FALSO NEGATIVO da métrica — trecho não anotado, no top-5,")
    print("de uma pergunta que o Junta@5 conta como erro:\n")
    print(f"{'critério':46s}{'trechos':>9s}{'perguntas':>11s}")
    print("─" * 66)
    for cob in (0.30, 0.40, 0.50, 0.60):
        t, p = conta(cob)
        print(f"{f'nota ≥ piso anotado  e  sinal ≥ {cob:.0%}':46s}{t:9d}{p:11d}")
    print()
    for cob in (0.40, 0.50):
        t, p = conta(cob, exigir_doc=True)
        print(f"{f'… e no MESMO documento exigido (sinal ≥ {cob:.0%})':46s}{t:9d}{p:11d}")

    t50, p50 = conta(0.50)
    print(f"\n  Faixa para o teto: das {len(erros)} que o braço erra, entre "
          f"{conta(0.60)[1]} e {conta(0.30)[1]} podem ser")
    print(f"  falso negativo da métrica. Junta@5 medido: "
          f"{len(resp)-len(erros)}/{len(resp)} = "
          f"{(len(resp)-len(erros))/len(resp):.3f}")
    for nome, extra in (("otimista", conta(0.30)[1]), ("conservador", conta(0.60)[1])):
        v = (len(resp) - len(erros) + extra) / len(resp)
        print(f"    corrigido ({nome:12s}): {v:.3f}")

    print(f"\n\n── {args.ler} candidatos para leitura, do sinal mais forte ao mais fraco ──")
    fortes = sorted([x for x in candidatos
                     if x["nota"] is not None and x["piso"] is not None
                     and x["nota"] >= x["piso"] and x["cob"] >= 0.40],
                    key=lambda x: -x["cob"])
    for x in fortes[:args.ler]:
        g = gold[x["q"]]
        print(f"\n[{x['q']}] nota {x['nota']:.3f} (piso {x['piso']:.3f}) · "
              f"sinal {x['cob']:.0%} · {'mesmo doc' if x['mesmo_doc'] else 'OUTRO doc'}")
        print(f"   PERGUNTA : {g['question'][:110]}")
        print(f"   RESPOSTA : {(g.get('reference_answer') or '')[:150]}")
        print(f"   NÃO ANOTADO: {textos.get(x['c'], '')[:190]}")


if __name__ == "__main__":
    main()
