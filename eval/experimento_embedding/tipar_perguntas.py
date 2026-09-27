#!/usr/bin/env python3
"""Declara tipo, dificuldade, tema e idioma nas 91 perguntas do experimento.

POR QUE ISTO FALTAVA. As 91 nasceram para uma pergunta só — qual recorte
recupera melhor — e para isso bastavam `evidencia` e `janela`. Para entrarem no
acervo de avaliação junto com o piloto, precisam das mesmas declarações que ele
tem: que tipo de pergunta é, quão difícil, sobre o quê e em que idioma está a
fonte. Sem isso não dá para reportar resultado por tipo, que é onde a diferença
entre "achou um trecho" e "achou tudo" aparece.

O QUE É DERIVADO E O QUE É JULGADO — a distinção importa, porque rótulo
inventado não passa auditoria:

    theme        DERIVADO do corpus. Cada documento já traz `metadata.theme`;
                 a pergunta herda o do documento de onde saiu. Zero julgamento.

    source_lang  DERIVADO de `forma`, que já classifica o documento em
                 articulado_PT / prosa_PT / articulado_EN / prosa_EN.

    question_type DERIVADO DA ESTRUTURA, com uma regra declarada:

        factual    uma citação — um fato, uma resposta.
        multi_hop  duas citações SEPARADAS na passagem. A resposta só existe
                   juntando as duas, e é isso que a métrica de completude cobra.

                 `comparative` NÃO EXISTE aqui, e isso é limitação a declarar:
                 cada uma das 91 saiu de UMA passagem de UM documento, então
                 nenhuma compara duas normas. O piloto tem 25 comparativas; as
                 91 têm zero. Ao reportar por tipo, a comparativa vem só do
                 piloto.

                 `unanswerable` também é zero pelo mesmo motivo — toda pergunta
                 foi escrita a partir de um texto que a responde.

    difficulty   DERIVADO, por regra declarada, e por isso não é comparável ao
                 `difficulty` do piloto, que foi atribuído por quem escreveu:

        easy     1 citação
        medium   2 citações, janela de geração de 600 ou 1400
        hard     2 citações, janela de geração de 3000 — evidência espalhada
                 por um texto largo é a que mais exige do recuperador

                 A PRIMEIRA VERSÃO usava a distância entre as citações dentro da
                 passagem, que seria a medida mais direta de esforço. Não deu:
                 em 10 das 91 a `passagem_texto` foi alargada pelo
                 `reancorar_passagens.py` para cobrir as duas citações depois de
                 uma limpeza do corpus, e ficou de 2x a 25x a janela declarada.
                 Numa delas a separação media 35.792 caracteres. Rótulo tirado
                 de campo corrompido é pior que rótulo grosseiro, então a regra
                 usa só `n_evidencias` e `janela`, que são confiáveis.

                 `separacao_citacoes` continua gravado como diagnóstico, mas
                 vem `null` onde a passagem estourou 1,5x a janela — é lá que
                 ele não significa nada.

Uso:
  python eval/experimento_embedding/tipar_perguntas.py
  python eval/experimento_embedding/tipar_perguntas.py --aplicar
"""

from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from collections import Counter
from pathlib import Path

AQUI = Path(__file__).resolve().parent
PROJECT_ROOT = AQUI.parents[1]
DADOS = AQUI / "dados"
PERGUNTAS = DADOS / "perguntas.json"
CORPUS = PROJECT_ROOT / "data/processed/documents.jsonl"

for _f in (sys.stdout, sys.stderr):
    try:
        _f.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Distância entre citações a partir da qual elas não caberiam juntas num trecho
# de produção (1.200 caracteres). Acima disso, recuperar as duas exige trazer
# duas regiões distintas do documento.
LONGE = 1200


def normalizar(t: str) -> str:
    t = (t or "").replace("­", "")
    t = unicodedata.normalize("NFKD", t.lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    return " ".join(t.split())


def separacao(passagem: str, citacoes: list[str]) -> int | None:
    """Distância em caracteres entre a primeira e a última citação na passagem."""
    p = normalizar(passagem)
    pos = [p.find(normalizar(c)) for c in citacoes]
    if any(i < 0 for i in pos):
        return None
    fim = max(i + len(normalizar(c)) for i, c in zip(pos, citacoes))
    return fim - min(pos)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--aplicar", action="store_true")
    args = ap.parse_args()

    perguntas = json.loads(PERGUNTAS.read_text(encoding="utf-8"))
    temas: dict[str, str] = {}
    for linha in CORPUS.open(encoding="utf-8"):
        r = json.loads(linha)
        temas.setdefault(r["metadata"]["document_id"], r["metadata"].get("theme"))

    sem_tema, sem_separacao, estouradas = [], [], []
    for r in perguntas:
        forma = r.get("forma") or ""
        r["source_lang"] = "en" if forma.endswith("_EN") else "pt"

        tema = temas.get(r["documento"])
        if not tema:
            sem_tema.append(r["documento"])
        r["theme"] = tema

        passagem = r.get("passagem_texto") or ""
        confiavel = len(passagem) <= 1.5 * (r.get("janela") or 0)
        d = separacao(passagem, r["evidencia"]) if confiavel else None
        if not confiavel:
            estouradas.append((r["n"], len(passagem), r.get("janela")))
        elif d is None and r["n_evidencias"] > 1:
            sem_separacao.append(r["n"])
        r["separacao_citacoes"] = d

        if r["n_evidencias"] <= 1:
            r["question_type"] = "factual"
            r["difficulty"] = "easy"
        else:
            r["question_type"] = "multi_hop"
            r["difficulty"] = "hard" if r.get("janela") == 3000 else "medium"

        r["origem"] = "experimento_recorte"

    print(f"perguntas tipadas: {len(perguntas)}")
    print(f"  question_type : {dict(Counter(r['question_type'] for r in perguntas))}")
    print(f"  difficulty    : {dict(Counter(r['difficulty'] for r in perguntas))}")
    print(f"  source_lang   : {dict(Counter(r['source_lang'] for r in perguntas))}")
    print(f"  temas distintos: {len({r['theme'] for r in perguntas if r['theme']})}")
    dist = [r["separacao_citacoes"] for r in perguntas if r["separacao_citacoes"]]
    if dist:
        dist.sort()
        print(f"  separação entre citações: mediana {dist[len(dist)//2]}, "
              f"máx {dist[-1]}, acima de {LONGE}: {sum(1 for d in dist if d > LONGE)}")
    if sem_tema:
        print(f"  ⚠ documento sem theme no corpus: {sorted(set(sem_tema))}")
    if estouradas:
        print(f"\n  passagens alargadas pela reancoragem (separação não medida): "
              f"{len(estouradas)} de {len(perguntas)}")
        for n, t, j in sorted(estouradas, key=lambda x: -x[1])[:5]:
            print(f"      n={n:4} janela {j} -> passagem {t} chars ({t/j:.1f}x)")
    if sem_separacao:
        print(f"  ⚠ citação não localizada na passagem: {sem_separacao}")

    if args.aplicar:
        PERGUNTAS.write_text(json.dumps(perguntas, ensure_ascii=False, indent=1),
                             encoding="utf-8")
        print(f"\n✓ {PERGUNTAS}")
    else:
        print("\n(relatório apenas — use --aplicar para gravar)")


if __name__ == "__main__":
    main()
