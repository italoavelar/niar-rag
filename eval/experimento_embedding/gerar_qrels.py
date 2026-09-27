#!/usr/bin/env python3
"""Deriva o qrels de um corpus a partir das citações literais do acervo.

A INVERSÃO QUE ISTO FAZ. O gabarito antigo GUARDAVA qrels: pares
(pergunta, id de trecho, grau), anotados à mão. Id de trecho é hash do conteúdo,
então qualquer mexida no corpus invalidava o gabarito inteiro — foi o que
obrigou a reancorar 256 linhas a cada limpeza, e o que impediu o piloto de
comparar tamanhos de recorte (um id do corpus de 1.200 não existe no de 400).

Aqui o qrels deixa de ser dado e vira VISTA. O que se guarda é a citação
literal, que é um fato e não depende de recorte nenhum. O qrels de um corpus
qualquer sai deste script, mecanicamente, sem ninguém rejulgar nada.

    acervo.json (citações)  ──┬── corpus de produção  →  qrels.csv
                              ├── corpus T400         →  qrels_T400.csv
                              └── corpus T3500        →  qrels_T3500.csv

GRAU 2 E SÓ. O gabarito antigo tinha grau 1 para "vizinho de página", adicionado
automaticamente e nunca julgado por ninguém; usá-lo numa métrica conjuntiva
derrubava a factual de 17/25 para 4/25. Aqui só existe o trecho que CONTÉM a
citação, e ele é grau 2. Não se inventa relevância parcial.

CITAÇÃO PARTIDA PELA FRONTEIRA é reportada, não escondida. Quando nenhum trecho
contém a citação inteira, a evidência é irrecuperável NAQUELE recorte — é
defeito do corte, e é justamente o que o experimento de recorte mede (o T400
perdia 20% das perguntas assim). O script conta e lista; não emite qrels falso.

Uso:
  python eval/experimento_embedding/gerar_qrels.py
  python eval/experimento_embedding/gerar_qrels.py --aplicar
  python eval/experimento_embedding/gerar_qrels.py \\
      --corpus eval/experimento_embedding/dados/T400.jsonl --saida qrels_T400.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import unicodedata
from collections import Counter
from pathlib import Path

AQUI = Path(__file__).resolve().parent
PROJECT_ROOT = AQUI.parents[1]
DADOS = AQUI / "dados"
ACERVO = DADOS / "acervo.json"
CORPUS_PADRAO = PROJECT_ROOT / "data/processed/documents.jsonl"
# Grava no NOME CANÔNICO. O qrels antigo, anotado à mão e ancorado em id, foi
# apagado em 27/09/2026: manter os dois lado a lado garantiria que alguém
# avaliasse com o errado, e eles não são intercambiáveis (256 linhas contra 490,
# critérios de relevância diferentes). Quem lê `eval/data/qrels.csv` continua
# lendo — só que agora o arquivo é derivado das citações, não anotado.
SAIDA_PADRAO = PROJECT_ROOT / "eval/data/qrels.csv"

for _f in (sys.stdout, sys.stderr):
    try:
        _f.reconfigure(encoding="utf-8")
    except Exception:
        pass


def normalizar(t: str) -> str:
    """Mesma normalização do validador: conteúdo, não bytes."""
    t = (t or "").replace("­", "")
    t = unicodedata.normalize("NFKD", t.lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    return " ".join(t.split())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--corpus", type=Path, default=CORPUS_PADRAO)
    ap.add_argument("--saida", type=Path, default=SAIDA_PADRAO)
    ap.add_argument("--aplicar", action="store_true")
    args = ap.parse_args()

    acervo = json.loads(ACERVO.read_text(encoding="utf-8"))
    trechos = [json.loads(l) for l in args.corpus.open(encoding="utf-8") if l.strip()]
    texto = [(t["id"], normalizar(t["text"])) for t in trechos]
    # DOIS MOTIVOS DIFERENTES para uma citação não ser achada, e confundi-los
    # inverte a leitura: "partida pela fronteira" é defeito do RECORTE e conta
    # contra ele; "documento fora deste corpus" é escopo e não conta contra
    # nada. Os recortes do experimento só têm os 15 documentos da amostra, então
    # sem separar os dois o T400 pareceria perder por fronteira o que na verdade
    # é documento ausente.
    #
    # O TESTE DE ESCOPO É POR DOCUMENTO, não por texto. A primeira versão
    # concatenava os trechos do documento e perguntava se a citação aparecia
    # ali; não funciona, porque a junção insere espaço entre os trechos e a
    # sobreposição duplica texto — uma citação partida some das duas formas, e
    # era classificada como fora de escopo. Aqui cada citação é mapeada ao seu
    # documento pelo CORPUS DE PRODUÇÃO, onde todas existem inteiras, e o escopo
    # passa a ser uma pergunta exata: esse documento está no corpus alvo?
    docs_no_corpus = {t["metadata"]["document_id"] for t in trechos}
    if args.corpus == CORPUS_PADRAO:
        prod = texto
        prod_doc = [t["metadata"]["document_id"] for t in trechos]
    else:
        pt = [json.loads(l) for l in CORPUS_PADRAO.open(encoding="utf-8") if l.strip()]
        prod = [(t["id"], normalizar(t["text"])) for t in pt]   # normaliza UMA vez
        prod_doc = [t["metadata"]["document_id"] for t in pt]
    dono: dict[str, str] = {}
    for r in acervo:
        for e in r["evidencia"]:
            alvo = normalizar(e)
            if alvo in dono:
                continue
            for (_, t), d in zip(prod, prod_doc):
                if alvo in t:
                    dono[alvo] = d
                    break

    linhas: list[tuple[str, str, int]] = []
    partidas, fora_do_escopo, ambiguas = [], [], []
    por_pergunta: Counter[str] = Counter()

    for r in acervo:
        if not r["evidencia"]:
            continue                       # unanswerable: sem trecho a recuperar
        for e in r["evidencia"]:
            alvo = normalizar(e)
            achados = [cid for cid, t in texto if alvo in t]
            if not achados:
                # Está no documento mas em nenhum trecho inteiro → a fronteira
                # do recorte partiu a citação. Não está em documento nenhum →
                # aquele documento simplesmente não faz parte deste corpus.
                if dono.get(alvo) in docs_no_corpus:
                    partidas.append((r["n"], e[:64]))
                else:
                    fora_do_escopo.append((r["n"], e[:64]))
                continue
            if len(achados) > 1:
                # Sobreposição entre trechos faz a citação aparecer em dois. Não
                # é erro: basta um deles vir no top-k. Todos entram como grau 2.
                ambiguas.append((r["n"], len(achados)))
            for cid in achados:
                linhas.append((str(r["n"]), cid, 2))
            por_pergunta[r["n"]] += 1

    # NO ESCOPO = TODAS as citações da pergunta estão em documentos presentes
    # neste corpus. Não "ao menos uma": a métrica é conjuntiva, e uma pergunta
    # cuja segunda citação vive num documento ausente é impossível por escopo,
    # não por recorte. Contá-la no denominador rebaixaria o teto sem que o corte
    # tivesse culpa.
    n_fora = Counter(n for n, _ in fora_do_escopo)
    respondiveis = [r for r in acervo if r["evidencia"]]
    no_escopo = [r for r in respondiveis if n_fora[r["n"]] == 0]
    completas = [r for r in no_escopo if por_pergunta[r["n"]] == r["n_evidencias"]]

    print(f"corpus      : {args.corpus}  ({len(trechos)} trechos)")
    print(f"acervo      : {len(acervo)} perguntas "
          f"({len(respondiveis)} com evidência, "
          f"{len(acervo) - len(respondiveis)} de recusa)")
    print(f"linhas qrels: {len(linhas)}")
    print(f"  perguntas cujo documento está neste corpus  : "
          f"{len(no_escopo)}/{len(respondiveis)}")
    print(f"  dessas, com TODAS as citações localizadas   : "
          f"{len(completas)}/{len(no_escopo)} ({len(completas)/max(len(no_escopo),1):.0%})")
    print(f"  citações partidas pela fronteira do recorte : {len(partidas)}")
    if fora_do_escopo:
        print(f"  citações em documento fora deste corpus     : {len(fora_do_escopo)}"
              "  (escopo, não defeito do recorte)")
    for n, e in partidas[:8]:
        print(f"      {n}: {e!r}")
    if ambiguas:
        print(f"  citações em mais de um trecho (sobreposição): {len(ambiguas)}")

    # O TETO DO RECORTE, que é o número que decide se ele serve. Uma pergunta só
    # é respondível se TODAS as suas citações existem inteiras em algum trecho.
    print(f"\n  TETO deste recorte: {len(completas)/len(respondiveis):.1%} "
          "das perguntas NO ESCOPO são possíveis neste recorte")

    if args.aplicar:
        args.saida.parent.mkdir(parents=True, exist_ok=True)
        with args.saida.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["QueryId", "ChunkId", "Relevance"])
            w.writerows(sorted(set(linhas)))
        print(f"\n✓ {args.saida}")
        print("  Mesmo formato do qrels.csv de sempre — trec_eval, ranx e o "
              "eval/ existente leem sem mudança.")
    else:
        print("\n(relatório apenas — use --aplicar para gravar)")


if __name__ == "__main__":
    main()
