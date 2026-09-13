#!/usr/bin/env python3
"""Viabilidade de chunking por unidade normativa (artigo/seção), sem embutir nada.

Pergunta que este script responde, antes de qualquer decisão:
  1. Quantos documentos do acervo têm marcadores regulares de unidade?
  2. Se cortássemos por unidade, que tamanho teriam os trechos? Unidade grande
     demais é tão ruim quanto trecho picado: dilui o sinal e estoura contexto.
  3. Quantas unidades ficariam grandes a ponto de precisar de subdivisão?

Método: o corpus já está chunkado, então o texto original é reconstruído por
documento concatenando os trechos na ordem e removendo a sobreposição de 200
caracteres. É aproximação, suficiente para dimensionar — não é a extração final.

Uso:
  python eval/tools/viabilidade_chunk_normativo.py
"""

from __future__ import annotations

import json
import re
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]   # .../eval/tools/experimentos_descartados/x.py
CORPUS = PROJECT_ROOT / "data/processed/documents.jsonl"

for _f in (sys.stdout, sys.stderr):
    try:
        _f.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Marcadores de unidade normativa, em início de linha ou após pontuação forte.
MARCADORES = re.compile(
    r"(?:(?<=^)|(?<=\n)|(?<=\.\s)|(?<=;\s))\s*("
    r"Art\.\s*\d+[º°]?[-A-Z]*"           # Art. 1º, Art. 12-A
    r"|Artigo\s+\d+[º°]?"
    r"|Article\s+\d+"
    r"|Section\s+\d+(?:\.\d+)*"
    r"|Clause\s+\d+(?:\.\d+)*"
    r"|CAPÍTULO\s+[IVXLC]+"
    r"|CHAPTER\s+[IVXLC0-9]+"
    r"|§\s*\d+[º°]?"
    r"|Parágrafo único"
    r")",
    re.MULTILINE,
)

SOBREPOSICAO = 200
TETO_CONFORTAVEL = 2500     # acima disto, a unidade provavelmente pede subdivisão


def reconstruir(trechos: list[str]) -> str:
    """Concatena removendo a sobreposição aproximada entre trechos consecutivos."""
    if not trechos:
        return ""
    texto = trechos[0]
    for seguinte in trechos[1:]:
        corte = 0
        for tam in range(min(SOBREPOSICAO + 50, len(texto), len(seguinte)), 30, -1):
            if texto[-tam:] == seguinte[:tam]:
                corte = tam
                break
        texto += seguinte[corte:]
    return texto


def main() -> None:
    por_doc: dict[str, list[tuple]] = defaultdict(list)
    with CORPUS.open(encoding="utf-8") as fh:
        for linha in fh:
            d = json.loads(linha)
            md = d["metadata"]
            ordem = (md.get("page") if md.get("page") is not None else 0, md.get("chunk") or 0)
            por_doc[md["document_id"]].append((ordem, d["text"], md.get("source_type")))

    print("── Viabilidade de chunking por unidade normativa ──\n")
    linhas = []
    todas_unidades: list[int] = []
    for doc, itens in sorted(por_doc.items()):
        itens.sort(key=lambda x: x[0])
        texto = reconstruir([t for _, t, _ in itens])
        marcas = list(MARCADORES.finditer(texto))
        n_atual = len(itens)
        if len(marcas) < 2:
            linhas.append((doc, n_atual, 0, None, None, 0, itens[0][2]))
            continue
        cortes = [m.start() for m in marcas] + [len(texto)]
        unidades = [cortes[i + 1] - cortes[i] for i in range(len(cortes) - 1)]
        unidades = [u for u in unidades if u > 80]        # ignora marcador solto
        if not unidades:
            linhas.append((doc, n_atual, 0, None, None, 0, itens[0][2]))
            continue
        todas_unidades.extend(unidades)
        grandes = sum(1 for u in unidades if u > TETO_CONFORTAVEL)
        linhas.append((doc, n_atual, len(unidades), int(st.median(unidades)),
                       max(unidades), grandes, itens[0][2]))

    com = [l for l in linhas if l[2] > 0]
    sem = [l for l in linhas if l[2] == 0]

    print(f"{'documento':<44}{'hoje':>6}{'unid.':>7}{'mediana':>9}{'máx':>8}{'>2500':>7}")
    for doc, n_atual, n_un, mediana, maximo, grandes, _ in sorted(com, key=lambda x: -x[2])[:22]:
        print(f"{doc[:42]:<44}{n_atual:>6}{n_un:>7}{mediana:>9}{maximo:>8}{grandes:>7}")

    print(f"\ndocumentos com marcadores regulares : {len(com)} de {len(linhas)}")
    print(f"documentos sem estrutura detectável  : {len(sem)}")
    if sem:
        print("   " + ", ".join(d[:38] for d, *_ in sem[:8]) + (" …" if len(sem) > 8 else ""))

    if todas_unidades:
        q = st.quantiles(todas_unidades, n=10)
        print(f"\ntamanho das unidades (n={len(todas_unidades)} unidades, em caracteres):")
        print(f"   mediana {int(st.median(todas_unidades))}   "
              f"p10 {int(q[0])}   p90 {int(q[8])}   máximo {max(todas_unidades)}")
        print(f"   abaixo de 300 chars (curtas demais) : {sum(1 for u in todas_unidades if u < 300)}"
              f" ({100*sum(1 for u in todas_unidades if u < 300)/len(todas_unidades):.0f}%)")
        print(f"   acima de {TETO_CONFORTAVEL} chars (pedem subdivisão): "
              f"{sum(1 for u in todas_unidades if u > TETO_CONFORTAVEL)}"
              f" ({100*sum(1 for u in todas_unidades if u > TETO_CONFORTAVEL)/len(todas_unidades):.0f}%)")
        print(f"\ntotal de trechos hoje: {sum(l[1] for l in linhas)}")
        print(f"total se cortasse por unidade (só os documentos com estrutura): "
              f"{len(todas_unidades)} + {sum(l[1] for l in sem)} dos sem estrutura")


if __name__ == "__main__":
    main()
