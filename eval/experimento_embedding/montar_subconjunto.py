#!/usr/bin/env python3
"""Recorta do acervo o subconjunto balanceado de 75 por tipo, sem enviesar.

O PROBLEMA. Em 30/09/2026 o acervo tem 393 perguntas: 118 factuais, 75
multi-hop, 75 comparativas e 125 irrespondíveis. O artigo de dezembro usa 75 de
cada, então é preciso descartar 43 factuais e 50 irrespondíveis. Como se escolhe
define o resultado, e as duas escolhas óbvias estão erradas:

    "as primeiras 75"   pega o arquivo na ordem em que foi montado. As primeiras
                        multi-hop são todas do piloto, e as primeiras
                        irrespondíveis são todas de `norma_ausente`. Seria
                        selecionar por origem e por motivo, não por nada.

    "75 ao acaso"       não erra sistematicamente, mas com n pequeno deixa
                        estrato inteiro de fora por azar — as 8 de
                        jurisprudência podem virar 3.

A SAÍDA É AMOSTRA ESTRATIFICADA PROPORCIONAL. Cada tipo é dividido nos estratos
que importam para ler o resultado, e cada estrato entrega uma fatia proporcional
ao seu tamanho no acervo. Assim o subconjunto de 75 tem a mesma composição do
conjunto de onde saiu, e nenhum estrato desaparece.

    multi_hop / comparative  já estão em 75: entram inteiras, sem sorteio.
    factual                  passou a ser SORTEADA em 30/09, quando a correção
                             de tipo e a retipagem de q0062 levaram o estrato a
                             118. Estrato = origem × idioma, pelo mesmo motivo do
                             multi_hop abaixo.
    unanswerable             estrato = motivo de ausência. São oito, de 8 a 25
                             perguntas cada; é o eixo pelo qual o resultado de
                             recusa se lê, e perder um deles cega uma falha.

A ESTRATIFICAÇÃO É CEGA AO RESULTADO, e isso é o ponto. Os estratos são
propriedades do dado — quem escreveu, em que língua —, nunca a nota de nenhum
braço. Sortear 75 de 118 olhando o acerto seria escolher a conclusão; é por isso
que a regra está escrita aqui e semeada, e não decidida depois de ver a tabela.

O sorteio é semeado. Rodar de novo dá o mesmo subconjunto — o artigo precisa
poder ser refeito.

O ACERVO COMPLETO FICA. Este arquivo é uma vista para dezembro; as 77 perguntas
que sobram não são lixo, são o começo do conjunto de fevereiro.

Uso:
  python eval/experimento_embedding/montar_subconjunto.py
  python eval/experimento_embedding/montar_subconjunto.py --aplicar
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

AQUI = Path(__file__).resolve().parent
DADOS = AQUI / "dados"
ACERVO = DADOS / "acervo.json"
SAIDA = DADOS / "subconjunto_dezembro.json"

POR_TIPO = 75
SEMENTE = 2026

for _f in (sys.stdout, sys.stderr):
    try:
        _f.reconfigure(encoding="utf-8")
    except Exception:
        pass


def estrato(r: dict) -> tuple:
    """O eixo pelo qual o resultado daquele tipo vai ser lido."""
    if r["question_type"] == "unanswerable":
        return (r.get("motivo_ausencia") or "piloto",)
    return (r.get("origem"), r.get("source_lang"))


def escolher(perguntas: list[dict], quantas: int, rng: random.Random) -> list[dict]:
    """Amostra estratificada proporcional, com sobra distribuída pelos maiores.

    A divisão proporcional quase nunca fecha em inteiros. O resto vai para os
    estratos com mais candidatos sobrando, que é onde tirar uma a mais distorce
    menos a proporção.
    """
    if len(perguntas) <= quantas:
        return list(perguntas)

    grupos: dict[tuple, list[dict]] = defaultdict(list)
    for r in perguntas:
        grupos[estrato(r)].append(r)
    for g in grupos.values():
        rng.shuffle(g)

    total = len(perguntas)
    cotas = {k: int(len(v) * quantas / total) for k, v in grupos.items()}
    # Garante ao menos uma de cada estrato: estrato que desaparece cega a leitura.
    for k in grupos:
        cotas[k] = max(1, cotas[k])
    while sum(cotas.values()) > quantas:          # tirar do maior
        k = max(cotas, key=lambda x: (cotas[x], len(grupos[x])))
        if cotas[k] > 1:
            cotas[k] -= 1
        else:
            break
    while sum(cotas.values()) < quantas:          # devolver a quem tem sobra
        k = max(grupos, key=lambda x: len(grupos[x]) - cotas[x])
        if cotas[k] >= len(grupos[k]):
            break
        cotas[k] += 1

    escolhidas = []
    for k, g in grupos.items():
        escolhidas.extend(g[:cotas[k]])
    return escolhidas


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--aplicar", action="store_true")
    args = ap.parse_args()

    acervo = json.loads(ACERVO.read_text(encoding="utf-8"))
    rng = random.Random(SEMENTE)
    por_tipo: dict[str, list[dict]] = defaultdict(list)
    for r in acervo:
        por_tipo[r["question_type"]].append(r)

    subconjunto = []
    for tipo in ("factual", "multi_hop", "comparative", "unanswerable"):
        candidatas = sorted(por_tipo[tipo], key=lambda r: str(r["n"]))
        escolhidas = escolher(candidatas, POR_TIPO, rng)
        subconjunto.extend(escolhidas)
        print(f"\n{tipo}: {len(candidatas)} no acervo → {len(escolhidas)} no subconjunto")
        antes = Counter(estrato(r) for r in candidatas)
        depois = Counter(estrato(r) for r in escolhidas)
        for k in sorted(antes, key=lambda x: -antes[x]):
            a, d = antes[k], depois.get(k, 0)
            marca = "  ← estrato perdido" if d == 0 else ""
            print(f"    {str(k):46s} {a:4d} → {d:3d}   "
                  f"{a/len(candidatas):5.1%} → {d/max(len(escolhidas),1):5.1%}{marca}")

    print(f"\nsubconjunto: {len(subconjunto)} perguntas")
    print(f"  com evidência: {sum(1 for r in subconjunto if r['evidencia'])}")
    print(f"  de recusa    : {sum(1 for r in subconjunto if not r['evidencia'])}")
    docs = set()
    for r in subconjunto:
        if r.get("documento"):
            docs.add(r["documento"])
        for d in (r.get("documentos") or []):
            docs.add(d)
    print(f"  documentos citados: {len(docs)}")
    print(f"  dificuldade  : {dict(Counter(r['difficulty'] for r in subconjunto))}")
    print(f"  idioma       : {dict(Counter(r.get('source_lang') for r in subconjunto))}")
    print(f"  fora do subconjunto (começo do de fevereiro): "
          f"{len(acervo) - len(subconjunto)}")

    if args.aplicar:
        SAIDA.write_text(json.dumps(subconjunto, ensure_ascii=False, indent=1),
                         encoding="utf-8")
        print(f"\n✓ {SAIDA}")
        print(f"  semente {SEMENTE} — rodar de novo dá o mesmo subconjunto.")
        print("  Para o qrels desta vista: gerar_qrels.py aponta para acervo.json; "
              "use --acervo se quiser o qrels só do subconjunto.")
    else:
        print("\n(relatório apenas — use --aplicar para gravar)")


if __name__ == "__main__":
    main()
