#!/usr/bin/env python3
"""A citação anotada é mesmo o melhor trecho do documento para aquela pergunta?

POR QUE ISTO EXISTE. O plano do benchmark mediu, em 13/09, que em 36 das 75
perguntas respondíveis existia — no documento certo e SEM anotação — um trecho
que casava com a resposta-referência melhor que a âncora anotada
(`benchmark_leme_500.md:44`). Isso é grave: o qrels manda o recuperador buscar
um trecho que não é o melhor, e conta como erro quando ele acha o melhor.

A pergunta agora é se o modo novo de gerar — âncora escolhida por quem acabou de
ler o material, citação extraída do corpus — reduziu isso. Não dá para supor:
mede-se.

COMO SE MEDE. Para cada pergunta respondível, pontua-se TODO trecho dos
documentos citados pela fração de sinais da resposta-referência que ele contém.
Sinal é palavra de 5+ letras, número, ou sigla em maiúsculas — a mesma função que
`converter_piloto.py` usa para escolher âncora. Se algum trecho NÃO anotado
pontua mais que o melhor anotado, a anotação escolheu pior.

O QUE ISTO NÃO É. Não é julgamento de relevância: sobreposição de sinal não sabe
o que responde a pergunta. É um detector barato de um problema específico —
"existe trecho obviamente melhor que o anotado" — e serve para decidir se o pool
da Fase 2 é necessário, não para substituí-lo.

Uso:
  python eval/experimento_embedding/medir_melhor_trecho.py
  python eval/experimento_embedding/medir_melhor_trecho.py --acervo dados/subconjunto_dezembro.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

AQUI = Path(__file__).resolve().parent
PROJECT_ROOT = AQUI.parents[1]
CORPUS = PROJECT_ROOT / "data/processed/documents.jsonl"

for _f in (sys.stdout, sys.stderr):
    try:
        _f.reconfigure(encoding="utf-8")
    except Exception:
        pass


def sem_acento(t: str) -> str:
    base = unicodedata.normalize("NFKD", t or "")
    return "".join(c for c in base if not unicodedata.combining(c))


def sinais(texto: str) -> set[str]:
    base = sem_acento(texto)
    return ({w.lower() for w in re.findall(r"[A-Za-z]{5,}", base)}
            | set(re.findall(r"\d[\d.,]*", base))
            | {s.lower() for s in re.findall(r"\b[A-Z]{3,}\b", base)})


def normalizar(t: str) -> str:
    t = (t or "").replace("­", "")
    return " ".join(sem_acento(t.lower()).split())


def carregar_corpus() -> dict[str, list[tuple[str, str]]]:
    """document_id -> [(chunk_id, texto)]"""
    por_doc: dict[str, list[tuple[str, str]]] = defaultdict(list)
    with CORPUS.open(encoding="utf-8") as fh:
        for linha in fh:
            if not linha.strip():
                continue
            r = json.loads(linha)
            por_doc[r["metadata"]["document_id"]].append((r["id"], r["text"]))
    return por_doc


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--acervo", default="dados/acervo.json")
    args = ap.parse_args()

    caminho = AQUI / args.acervo if not Path(args.acervo).is_absolute() else Path(args.acervo)
    acervo = json.loads(caminho.read_text(encoding="utf-8"))
    por_doc = carregar_corpus()

    por_origem: dict[str, list[bool]] = defaultdict(list)
    exemplos: list[tuple] = []
    sem_citacao_localizada = 0

    for q in acervo:
        ev = [e for e in (q.get("evidencia") or []) if e]
        if not ev:
            continue  # irrespondível: não tem trecho relevante por definição
        docs = [q["documento"]] if q.get("documento") else list(q.get("documentos") or [])
        candidatos = [(cid, txt) for d in docs for cid, txt in por_doc.get(d, [])]
        if not candidatos:
            continue

        alvos = [normalizar(e) for e in ev]
        anotados = {cid for cid, txt in candidatos
                    if any(a in normalizar(txt) for a in alvos)}
        if not anotados:
            sem_citacao_localizada += 1
            continue

        alvo_sinais = sinais(q.get("answer", ""))
        if not alvo_sinais:
            continue

        def pontuar(txt: str) -> float:
            return len(sinais(txt) & alvo_sinais) / len(alvo_sinais)

        notas = {cid: pontuar(txt) for cid, txt in candidatos}
        melhor_anotado = max(notas[c] for c in anotados)
        fora = [(n, c) for c, n in notas.items() if c not in anotados]
        melhor_fora, cid_fora = max(fora) if fora else (0.0, None)

        pior = melhor_fora > melhor_anotado
        por_origem[q["origem"]].append(pior)
        if pior:
            exemplos.append((q["origem"], q["n"], melhor_anotado, melhor_fora,
                             q.get("question", "")[:70]))

    print(f"acervo: {caminho.name} · corpus: {sum(len(v) for v in por_doc.values())} trechos\n")
    print(f"{'origem':26s} {'respondíveis':>12s} {'anotação pior':>14s} {'%':>7s}")
    print("─" * 64)
    tot_n = tot_p = 0
    for origem, marcas in sorted(por_origem.items()):
        n, p = len(marcas), sum(marcas)
        tot_n += n
        tot_p += p
        print(f"{origem:26s} {n:12d} {p:14d} {100*p/max(n,1):6.1f}%")
    print("─" * 64)
    print(f"{'TOTAL':26s} {tot_n:12d} {tot_p:14d} {100*tot_p/max(tot_n,1):6.1f}%")
    if sem_citacao_localizada:
        print(f"\n(citação não localizada no documento: {sem_citacao_localizada})")

    print("\nExemplos onde existe trecho não anotado com sinal mais forte:")
    for origem, n, ma, mf, pergunta in sorted(exemplos, key=lambda x: x[2] - x[3])[:8]:
        print(f"  [{origem[:12]:12s} {str(n):6s}] anotado {ma:.2f} · fora {mf:.2f}  {pergunta}")


if __name__ == "__main__":
    main()
