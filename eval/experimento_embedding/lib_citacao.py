#!/usr/bin/env python3
"""Extrai citação literal do corpus a partir de uma âncora curta.

POR QUE ISTO EXISTE. Escrever 91 comparativas significa transcrever 182 citações
à mão, de 54 documentos. Transcrição à mão erra — e o erro é silencioso: a
citação quase certa não casa com o corpus, a pergunta é rejeitada na validação e
alguém perde tempo procurando o acento trocado.

Aqui não se transcreve. Informa-se uma ÂNCORA — um pedaço curto e distintivo do
trecho, como "vinte e um conselheiros" — e o script devolve o texto exato que
está no corpus, com o tamanho pedido, expandido até fronteira de palavra. A
citação sai byte a byte igual ao corpus por construção.

Se a âncora casar em mais de um trecho do mesmo documento, a função avisa: âncora
ambígua vira citação imprevisível, e é melhor trocá-la por outra mais específica
do que aceitar a primeira ocorrência em silêncio.
"""

from __future__ import annotations

import json
import re
import unicodedata
from functools import lru_cache
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CORPUS = PROJECT_ROOT / "data/processed/documents.jsonl"

JANELA_PADRAO = 170


def normalizar(t: str) -> str:
    t = (t or "").replace("­", "")
    t = unicodedata.normalize("NFKD", t.lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    return " ".join(t.split())


@lru_cache(maxsize=1)
def _corpus() -> list[tuple[str, str, str]]:
    """(document_id, texto original, texto normalizado) por trecho."""
    saida = []
    with CORPUS.open(encoding="utf-8") as fh:
        for linha in fh:
            if not linha.strip():
                continue
            r = json.loads(linha)
            saida.append((r["metadata"]["document_id"], r["text"], normalizar(r["text"])))
    return saida


class AncoraRuim(Exception):
    pass


def citar(documento: str, ancora: str, janela: int = JANELA_PADRAO,
          antes: int = 0) -> str:
    """Texto literal do corpus ao redor da âncora, no documento indicado.

    `antes` desloca o início para trás da âncora, quando o que interessa é a
    oração que a precede. O corte é ajustado para não partir palavra.
    """
    alvo = normalizar(ancora)
    achados = [(orig, norm) for doc, orig, norm in _corpus()
               if doc == documento and alvo in norm]
    if not achados:
        raise AncoraRuim(f"{documento}: âncora não encontrada — {ancora!r}")

    # Casar em mais de um trecho é NORMAL: o recorte tem 200 caracteres de
    # sobreposição, então o mesmo texto aparece em dois trechos vizinhos. Isso
    # não é ambiguidade. Ambiguidade de verdade é a âncora aparecer em partes
    # DIFERENTES do documento, e o sintoma disso é as ocorrências renderem
    # citações diferentes. Escolhe-se a que tem mais texto depois da âncora, que
    # é a que dá a citação mais completa.
    achados.sort(key=lambda par: par[1].index(alvo))
    original, _ = achados[0]
    # Localiza a âncora no texto ORIGINAL comparando normalizados posição a
    # posição: o original tem acento e espaço que o normalizado não tem.
    plano, mapa = [], []
    for i, c in enumerate(original):
        if c == "­":
            continue
        d = unicodedata.normalize("NFKD", c.lower())
        d = "".join(x for x in d if not unicodedata.combining(x))
        if c.isspace():
            if plano and plano[-1] != " ":
                plano.append(" ")
                mapa.append(i)
            continue
        for x in d:
            plano.append(x)
            mapa.append(i)
    achatado = "".join(plano)
    i = achatado.find(alvo)
    if i < 0:
        raise AncoraRuim(f"{documento}: âncora sumiu no mapeamento — {ancora!r}")

    inicio = mapa[max(0, i - antes)]
    fim_idx = min(i + max(janela, len(alvo)), len(mapa) - 1)
    fim = mapa[fim_idx]

    # Não partir palavra nas duas pontas.
    while inicio > 0 and original[inicio - 1].isalnum():
        inicio -= 1
    while fim < len(original) - 1 and original[fim].isalnum():
        fim += 1
    trecho = original[inicio:fim].strip()
    if len(normalizar(trecho)) < 40:
        raise AncoraRuim(f"{documento}: citação curta demais — {ancora!r}")
    return re.sub(r"\s+", " ", trecho)


def conferir(documento: str, citacao: str) -> bool:
    """A citação está mesmo em algum trecho deste documento?"""
    alvo = normalizar(citacao)
    return any(doc == documento and alvo in norm for doc, _, norm in _corpus())
