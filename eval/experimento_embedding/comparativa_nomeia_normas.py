#!/usr/bin/env python3
"""A comparativa que NOMEIA as duas normas é uma tarefa diferente da que não nomeia?

A AFIRMAÇÃO QUE ESTAVA NO RELATÓRIO, sem teste: "as que nomeiam pressupõem que o
usuário já sabe o que comparar; as que não nomeiam perguntam pelo assunto. São
tarefas distintas e a decomposição só é possível na primeira."

A segunda frase é uma previsão verificável, e a intuição por trás dela é boa: se
o enunciado diz "como o NIST e a UNESCO tratam X", partir em duas sub-perguntas é
mecânico; se diz apenas "como as normas tratam X", o decompositor teria que
ADIVINHAR quais normas comparar, sem ver o corpus. Previsão: ganho de
decomposição concentrado nas que nomeiam.

Este script mede, com os dados que já estão em disco (nenhuma chamada de API).
Detecta se o enunciado menciona cada documento exigido, por acrônimo do emissor e
por número do instrumento, e cruza com o resultado da intercalação.

O DETECTOR É GROSSO, e tem que ser declarado assim: ele acha "NIST" e "2338", não
entende paráfrase ("a lei brasileira de IA" nomeia o PL 2338 para um humano e não
para este código). Portanto ele SUBESTIMA quantas nomeiam. Serve para comparar
dois grupos grandes, não para classificar uma pergunta individual.

Uso:
  python eval/experimento_embedding/comparativa_nomeia_normas.py
  python eval/experimento_embedding/comparativa_nomeia_normas.py --listar
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from collections import defaultdict
from math import comb
from pathlib import Path

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parents[1]

GABARITO = RAIZ / "eval/data/golden_dezembro.jsonl"
POOLS = RAIZ / "eval/results/indexes/pools_subperguntas.json"
SCORES = RAIZ / "eval/results/indexes/rerank_scores_subperguntas.json"
RERANK_UNICO = RAIZ / "eval/results/retrieval/rankings/B_bge_m3_rerank.json"
K = 5
PROF = 50   # a profundidade de orçamento honesto, como em fundir_subperguntas.py

for _f in (sys.stdout, sys.stderr):
    try:
        _f.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Formas de superfície que um enunciado usa para nomear cada emissor. A chave é
# o token que aparece no document_id; os valores são o que um humano escreve.
EMISSOR = {
    "NIST": ["nist"],
    "UNESCO": ["unesco"],
    "WHO": ["who", "oms"],
    "FDA": ["fda"],
    "ANPD": ["anpd"],
    "CFM": ["cfm", "conselho federal de medicina", "codigo de etica medica"],
    "MCTI": ["mcti", "estrategia brasileira"],
    "ICO": ["ico", "dpia"],
    "OECD": ["oecd", "ocde"],
    "IMDRF": ["imdrf"],
    "SBIS": ["sbis"],
    "MS": ["ministerio da saude", "rnds"],
    "UFMG": ["ufmg"],
    "CEP": ["comite de etica em pesquisa", "cep"],
    "EU": ["gdpr", "rgpd", "uniao europeia", "europe", "ehds",
           "espaco europeu", "conselho da europa"],
}

# Instrumentos cujo nome próprio é o identificador, não o emissor.
NOME_PROPRIO = {
    "lgpd": ["lgpd", "lei geral de protecao de dados"],
    "gdpr": ["gdpr", "rgpd"],
    "cf": ["constituicao"],
    "cp": ["codigo penal"],
    "cdc": ["codigo de defesa do consumidor"],
    "eca": ["estatuto da crianca"],
    "european_health_data_space": ["ehds", "espaco europeu"],
    "rnds": ["rnds", "rede nacional de dados em saude"],
    "dpia": ["dpia"],
}


def norm(t: str) -> str:
    base = "".join(c for c in unicodedata.normalize("NFKD", t or "")
                   if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", base.lower())


def apelidos(doc_id: str) -> list[str]:
    """Formas de superfície plausíveis para este documento."""
    saida: list[str] = []
    partes = doc_id.split("_")
    for p in partes:
        if p in EMISSOR:
            saida += EMISSOR[p]
        if p.lower() in NOME_PROPRIO:
            saida += NOME_PROPRIO[p.lower()]
        # número de instrumento: 2338, 14874, 3268, 2314…
        if p.isdigit() and len(p) >= 4 and not (1900 <= int(p) <= 2100):
            saida.append(p)
            if len(p) == 5:                      # 14874 → 14.874
                saida.append(f"{p[:2]}.{p[2:]}")
            elif len(p) == 4:                    # 2338 → 2.338
                saida.append(f"{p[0]}.{p[1:]}")
    chave_nome = "_".join(partes[:-2])
    if chave_nome in NOME_PROPRIO:
        saida += NOME_PROPRIO[chave_nome]
    return list(dict.fromkeys(saida))


def mcnemar(v: int, d: int) -> float:
    n = v + d
    if n == 0:
        return 1.0
    k = min(v, d)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / (2 ** n))


def ordenar(cands, notas):
    return sorted(cands, key=lambda c: -notas.get(c, -1e9))


def intercalado(o1, o2, n1, n2, k):
    a, b = ordenar(o1, n1), ordenar(o2, n2)
    saida, i, j = [], 0, 0
    while len(saida) < k and (i < len(a) or j < len(b)):
        for lista, idx in ((a, "i"), (b, "j")):
            pos = i if idx == "i" else j
            while pos < len(lista) and lista[pos] in saida:
                pos += 1
            if pos < len(lista) and len(saida) < k:
                saida.append(lista[pos])
                pos += 1
            if idx == "i":
                i = pos
            else:
                j = pos
    return saida[:k]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--listar", action="store_true",
                    help="imprime a classificação pergunta por pergunta")
    args = ap.parse_args()

    gold = {g["qid"]: g for g in
            (json.loads(l) for l in GABARITO.open(encoding="utf-8") if l.strip())}
    pools = json.loads(POOLS.read_text(encoding="utf-8"))
    scores = json.loads(SCORES.read_text(encoding="utf-8"))["scores"]
    unico = json.loads(RERANK_UNICO.read_text(encoding="utf-8"))

    comp = [q for q in pools if q in gold and f"{q}#sub1" in scores]

    def junta(ordem, q) -> int:
        t = set(ordem[:K])
        return int(all(any(c in t for c in gr) for gr in gold[q]["qrels_grupos"]))

    classe: dict[str, str] = {}
    detalhe: dict[str, tuple] = {}
    for q in comp:
        texto = norm(gold[q]["question"])
        docs = gold[q].get("source_docs") or []
        achados = []
        for d in docs:
            alias = apelidos(d)
            achados.append(any(a in texto for a in alias))
        n_doc = len(docs)
        n_ach = sum(achados)
        if n_doc >= 2 and n_ach >= 2:
            c = "nomeia as duas"
        elif n_ach == 1:
            c = "nomeia uma"
        else:
            c = "nomeia nenhuma"
        classe[q] = c
        detalhe[q] = (n_doc, n_ach, docs)

    grupos = defaultdict(list)
    for q in comp:
        grupos[classe[q]].append(q)

    ordem_cls = ["nomeia as duas", "nomeia uma", "nomeia nenhuma"]

    print(f"comparativas analisadas: {len(comp)}")
    print(f"detector: acrônimo do emissor + número do instrumento "
          f"(grosso, SUBESTIMA)\n")

    cab = (f"{'classe':18s}{'n':>4s}{'única':>8s}{'intercalado':>13s}"
           f"{'ganho':>8s}{'disc.':>9s}{'p':>9s}")
    print(cab)
    print("─" * len(cab))
    for c in ordem_cls:
        qs = grupos.get(c, [])
        if not qs:
            continue
        base = sum(junta(unico[q], q) for q in qs if q in unico)
        alvo, v, d = 0, 0, 0
        for q in qs:
            o1, o2 = pools[q]["sub1"][:PROF], pools[q]["sub2"][:PROF]
            n1, n2 = scores[f"{q}#sub1"], scores[f"{q}#sub2"]
            a = junta(intercalado(o1, o2, n1, n2, K), q)
            b = junta(unico[q], q) if q in unico else 0
            alvo += a
            v += int(b and not a)
            d += int(a and not b)
        print(f"{c:18s}{len(qs):4d}{base:8d}{alvo:13d}{alvo-base:+8d}"
              f"{f'{v} a {d}':>9s}{mcnemar(v, d):9.4f}")

    print(f"\n  'única' = consulta única reranqueada; 'intercalado' = decomposição")
    print(f"  em duas sub-perguntas com alternância estrita, pool {PROF}+{PROF}")
    print(f"  (85 trechos distintos — orçamento MENOR que os 100 da única).")
    print(f"  'disc.' = pares discordantes, única-acerta a intercalado-acerta.")

    print("\n\n── cruzamento com a procedência ──\n")
    tab = defaultdict(lambda: defaultdict(int))
    for q in comp:
        tab[gold[q]["origem"]][classe[q]] += 1
    cab2 = f"{'origem':26s}{'n':>4s}" + "".join(f"{c[:14]:>16s}" for c in ordem_cls)
    print(cab2)
    print("─" * len(cab2))
    for o in sorted(tab):
        n = sum(tab[o].values())
        linha = f"{o:26s}{n:4d}"
        for c in ordem_cls:
            v = tab[o][c]
            linha += f"{v:9d} ({v/n:4.0%})"
        print(linha)

    if args.listar:
        print("\n\n── pergunta por pergunta ──")
        for c in ordem_cls:
            print(f"\n═══ {c} ═══")
            for q in sorted(grupos.get(c, [])):
                n_doc, n_ach, docs = detalhe[q]
                print(f"\n [{q}] {n_ach}/{n_doc} docs nomeados · "
                      f"{gold[q]['origem']}")
                print(f"   {gold[q]['question'][:150]}")
                print(f"   docs: {', '.join(docs)}")


if __name__ == "__main__":
    main()
