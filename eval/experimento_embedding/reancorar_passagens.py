#!/usr/bin/env python3
"""Congela dentro de cada pergunta a passagem que a gerou, reancorada no corpus.

O DEFEITO QUE ISTO CORRIGE. A pergunta guardava só `n`, um índice para
`passagens.json`. Mas `passagens.json` guarda `inicio` — um deslocamento em
caracteres dentro do documento reconstruído. Qualquer limpeza do corpus encurta
o documento, todo deslocamento posterior anda, e `n` passa a apontar para outro
texto. Foi o que aconteceu: 3 perguntas boas foram rejeitadas por "citação
ausente da passagem" quando a citação estava no corpus, só que 1.378 caracteres
antes de onde o índice dizia.

Índice por posição é frágil; a citação literal não é. Este script troca um pelo
outro: localiza as citações da pergunta no documento limpo, recorta ao redor
delas uma janela do tamanho declarado, e grava esse texto DENTRO da pergunta.

Depois disso a pergunta é autocontida. Mexer no corpus pode invalidá-la — e deve,
se a evidência sumiu — mas nunca mais por contabilidade de offset.

Uso:
  python eval/experimento_embedding/reancorar_passagens.py            # relatório
  python eval/experimento_embedding/reancorar_passagens.py --aplicar  # grava
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
DADOS = AQUI / "dados"

for _f in (sys.stdout, sys.stderr):
    try:
        _f.reconfigure(encoding="utf-8")
    except Exception:
        pass


def normalizar(t: str) -> str:
    t = (t or "").replace("\u00ad", "")
    t = unicodedata.normalize("NFKD", t.lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    return " ".join(t.split())


def mapa_posicoes(texto: str) -> tuple[str, list[int]]:
    """Texto normalizado + índice que devolve cada caractere à posição original.

    Precisa existir porque a busca é feita no texto normalizado (sem acento, sem
    caixa, espaço colapsado) mas o recorte tem de sair do texto ORIGINAL — é ele
    que vai para a pergunta.
    """
    saida, posicoes, espaco = [], [], True
    for i, c in enumerate(texto):
        d = unicodedata.normalize("NFKD", c.lower())
        d = "".join(x for x in d if not unicodedata.combining(x))
        if c == "\u00ad":
            continue
        if c.isspace():
            if not espaco:
                saida.append(" ")
                posicoes.append(i)
                espaco = True
            continue
        espaco = False
        for x in d:
            saida.append(x)
            posicoes.append(i)
    return "".join(saida), posicoes


def recortar(doc: str, citacoes: list[str], janela: int) -> str | None:
    """Menor trecho do tamanho da janela que contenha todas as citações."""
    norm_doc, posicoes = mapa_posicoes(doc)
    intervalos = []
    for c in citacoes:
        alvo = normalizar(c)
        i = norm_doc.find(alvo)
        if i < 0:
            return None
        intervalos.append((i, i + len(alvo)))

    ini_n, fim_n = min(i for i, _ in intervalos), max(f for _, f in intervalos)
    if fim_n - ini_n > janela:
        # As citações não cabem juntas na janela declarada. Alarga: perder
        # evidência é pior que uma passagem maior que o nominal.
        janela = fim_n - ini_n

    folga = janela - (fim_n - ini_n)
    ini_n = max(0, ini_n - folga // 2)
    fim_n = min(len(norm_doc), ini_n + janela)

    ini, fim = posicoes[ini_n], posicoes[min(fim_n, len(posicoes) - 1)]

    # Fronteira de frase, para a passagem não começar no meio de uma palavra.
    esq = doc.rfind(". ", max(0, ini - 300), ini)
    if esq > 0:
        ini = esq + 2
    dir_ = doc.find(". ", fim, min(len(doc), fim + 300))
    if dir_ > 0:
        fim = dir_ + 1
    return doc[ini:fim].strip()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--aplicar", action="store_true")
    args = ap.parse_args()

    import bench_tamanho as B

    passagens = {p["n"]: p for p in json.loads(
        (DADOS / "passagens.json").read_text(encoding="utf-8"))["passagens"]}
    por_doc = B.carregar_corpus()
    inteiros: dict[str, str] = {}

    lotes = sorted(DADOS.glob("perguntas_lote*.json"))
    total = reparados = perdidos = 0

    for lote in lotes:
        registros = json.loads(lote.read_text(encoding="utf-8"))
        mudou = False
        for r in registros:
            if not r.get("question"):
                continue
            total += 1
            # O que já está congelado na pergunta MANDA. `passagens.json` é só o
            # sorteio, e ele é refeito toda vez que o corpus muda — `n` deixa de
            # apontar para o mesmo documento. Confiar nele aqui reescreveria a
            # pergunta a partir de um documento que não é o dela.
            p = passagens.get(r["n"]) or {}
            doc_id = r.get("documento") or p.get("documento")
            janela = r.get("janela") or p.get("janela")
            forma = r.get("forma") or p.get("forma")
            if not doc_id or not janela:
                print(f"   n={r['n']}: sem documento/janela — passagem inexistente")
                perdidos += 1
                continue
            if doc_id not in inteiros:
                inteiros[doc_id] = B.reconstruir(por_doc[doc_id])

            texto = recortar(inteiros[doc_id], r["evidencia"], janela)
            if texto is None:
                print(f"   n={r['n']}: citação não está no documento limpo — pergunta morre")
                perdidos += 1
                continue

            r["documento"] = doc_id
            r["forma"] = forma
            r["janela"] = janela
            r["passagem_texto"] = texto
            reparados += 1
            mudou = True

        if mudou and args.aplicar:
            lote.write_text(json.dumps(registros, ensure_ascii=False, indent=1),
                            encoding="utf-8")
            print(f"✓ {lote.name}")

    print(f"\nperguntas com texto : {total}")
    print(f"  reancoradas       : {reparados}")
    print(f"  perdidas          : {perdidos}")
    if not args.aplicar:
        print("\n(relatório apenas — use --aplicar para gravar)")


if __name__ == "__main__":
    main()
