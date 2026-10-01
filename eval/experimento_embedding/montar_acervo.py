#!/usr/bin/env python3
"""Junta as famílias de perguntas num acervo só, no formato único.

AS FAMÍLIAS, e por que ficam em arquivos separados. Cada uma nasceu de um método
diferente, e a procedência importa na hora de ler um resultado por tipo — se uma
categoria for melhor que outra, a primeira hipótese a descartar é quem escreveu,
não o tipo. O campo `origem` carrega isso.

    perguntas_piloto.json        100  gabarito original, convertido de qrels
                                      para citação literal. Único com os quatro
                                      tipos. Janela 1200: nasceu dos trechos.
    perguntas.json                91  experimento de recorte. Janela variada
                                      (600/1400/3000), sem comparativa nem
                                      irrespondível — cada uma saiu de uma
                                      passagem de um documento só.
    perguntas_comparativas.json    9  cruzam dois documentos, montadas a partir
                                      de citações já verificadas no corpus.
    perguntas_irrespondiveis.json 100 sem evidência por construção; a recusa é
                                      a resposta certa.

O FORMATO É O MESMO PARA TODAS, e é o que permite uma máquina só avaliar tudo:
`evidencia` com as citações literais (vazia nas irrespondíveis), `question_type`,
`difficulty`, `source_lang`, `theme`. O qrels sai daqui por `gerar_qrels.py`,
derivado do corpus que estiver em uso.

Uso:
  python eval/experimento_embedding/montar_acervo.py --aplicar
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

AQUI = Path(__file__).resolve().parent
DADOS = AQUI / "dados"
SAIDA = DADOS / "acervo.json"

PARTES = ("perguntas_piloto.json", "perguntas.json",
          "perguntas_comparativas.json", "perguntas_irrespondiveis.json",
          "perguntas_factuais.json", "perguntas_multihop.json")

# Alvo do artigo de dezembro: 75 por tipo. O do KDD era 125, mas a geração pausa
# em 75 porque documentos novos devem entrar para fevereiro — e pergunta escrita
# antes do documento chegar não cobre o documento. Melhor fechar 75 balanceados
# agora e retomar depois com o acervo completo do corpus novo.
ALVO_POR_TIPO = 75

for _f in (sys.stdout, sys.stderr):
    try:
        _f.reconfigure(encoding="utf-8")
    except Exception:
        pass


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--aplicar", action="store_true")
    args = ap.parse_args()

    acervo, vistas, repetidas = [], set(), []
    for nome in PARTES:
        caminho = DADOS / nome
        if not caminho.exists():
            print(f"  ⚠ falta {nome}")
            continue
        parte = json.loads(caminho.read_text(encoding="utf-8"))
        for r in parte:
            chave = " ".join((r.get("question") or "").lower().split())
            if chave in vistas:
                repetidas.append((nome, r["n"], r["question"][:60]))
                continue
            vistas.add(chave)
            acervo.append(r)
        print(f"  {nome:32s} {len(parte):4d}")

    # COERÊNCIA ESTRUTURAL DO TIPO, para TODAS as famílias. O `tipar_perguntas.py`
    # só toca nas 91 do experimento de recorte; o piloto trouxe o tipo atribuído
    # e nunca foi conferido contra a própria estrutura. Duas regras que
    # o projeto já declara em outros lugares e que aqui passam a valer sempre:
    #
    #   uma citação  → `factual`, porque um trecho responde. Rotular de multi_hop
    #                  cria pergunta que não pode exigir dois trechos e ainda assim
    #                  é contada no denominador do tipo.
    #   comparativa  → precisa citar DOIS documentos. Se cita um, a resposta afirma
    #                  algo sobre a outra norma sem lastro — foi o defeito de
    #                  q0081, q0085, q0093 e q0099, achado só em 29/09/2026.
    #
    # A segunda regra só AVISA: consertá-la exige escolher a citação que falta, o
    # que é decisão editorial e não pode acontecer numa fusão automática.
    retipadas, sem_dois_documentos = [], []
    for r in acervo:
        ev = [e for e in (r.get("evidencia") or []) if e]
        if len(ev) == 1 and r["question_type"] in ("multi_hop", "comparative"):
            retipadas.append((r["n"], r["question_type"], r.get("origem")))
            r["question_type"] = "factual"
        elif r["question_type"] == "comparative":
            docs = set(r.get("documentos") or ([r["documento"]] if r.get("documento") else []))
            if len(docs) < 2:
                sem_dois_documentos.append((r["n"], r.get("origem")))
    if retipadas:
        print(f"\n  retipadas para factual (uma citação só): {len(retipadas)}")
        for n, t, o in retipadas:
            print(f"     {n} era {t} ({o})")
    if sem_dois_documentos:
        print(f"\n  ⚠ comparativas com menos de dois documentos: {len(sem_dois_documentos)}")
        for n, o in sem_dois_documentos:
            print(f"     {n} ({o}) — a resposta afirma algo sem citação")

    tipos = Counter(r["question_type"] for r in acervo)
    print(f"\nacervo: {len(acervo)} perguntas")
    print(f"  com evidência (uso em recuperação): "
          f"{sum(1 for r in acervo if r['evidencia'])}")
    print(f"  de recusa                         : "
          f"{sum(1 for r in acervo if not r['evidencia'])}")
    if repetidas:
        print(f"  pergunta repetida, descartada: {len(repetidas)}")
        for n, q, t in repetidas[:5]:
            print(f"      {n} {q}: {t!r}")

    print(f"\n  {'tipo':14s} {'agora':>6s} {'alvo':>6s} {'falta':>6s}")
    for t in ("factual", "multi_hop", "comparative", "unanswerable"):
        n = tipos.get(t, 0)
        print(f"  {t:14s} {n:6d} {ALVO_POR_TIPO:6d} {max(0, ALVO_POR_TIPO - n):6d}")
    total_falta = sum(max(0, ALVO_POR_TIPO - tipos.get(t, 0))
                      for t in ("factual", "multi_hop", "comparative", "unanswerable"))
    print(f"  {'TOTAL':14s} {len(acervo):6d} {ALVO_POR_TIPO*4:6d} {total_falta:6d}")

    print(f"\n  por origem    : {dict(Counter(r.get('origem') for r in acervo))}")
    print(f"  por dificuldade: {dict(Counter(r.get('difficulty') for r in acervo))}")
    docs = set()
    for r in acervo:
        if r.get("documento"):
            docs.add(r["documento"])
        for d in (r.get("documentos") or []):
            docs.add(d)
    print(f"  documentos citados: {len(docs)}")

    if args.aplicar:
        SAIDA.write_text(json.dumps(acervo, ensure_ascii=False, indent=1),
                         encoding="utf-8")
        print(f"\n✓ {SAIDA}")
        print("  Próximo: gerar_qrels.py --aplicar")
    else:
        print("\n(relatório apenas — use --aplicar para gravar)")


if __name__ == "__main__":
    main()
