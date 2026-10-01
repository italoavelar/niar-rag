#!/usr/bin/env python3
"""Extrai o TEXTO LIMPO DE CADA PÁGINA — a fonte comum das estratégias de recorte.

POR QUE ISTO EXISTE. O benchmark de recorte compara C1 (página), C4 (janela
1000/200), C5 (semântico) e C6 (estrutura jurídica). Para que a comparação isole
o recorte, as quatro precisam partir **exatamente do mesmo texto**.

Reconstruir esse texto a partir do `documents.jsonl` não serve: ele já está
recortado em 1.200 com sobreposição de 200, então a reconstrução herda as
fronteiras e a duplicação do recorte atual. Toda estratégia nasceria contaminada
pelo C4, e a comparação mediria outra coisa.

A saída é parar o pipeline no ponto certo. A extração faz, nesta ordem:

    PDF → linhas por página
        → detecta cabeçalho/rodapé repetido e corta
        → corta a seção de referências
        → descarta página não informativa (sumário, índice remissivo, capa)
        → **texto limpo da página**          ← ESTE script para aqui
        → unidades estruturais
        → empacota em trechos                ← daqui para frente é a estratégia
        → funde os pequenos demais

Tudo antes do "texto limpo da página" é extração e vale para todas; tudo depois é
escolha de recorte e é o que se quer comparar.

O script reusa as funções do próprio `src/extract_to_jsonl.py`, e não uma cópia —
se a limpeza mudar lá, muda aqui junto, e as estratégias continuam comparáveis
entre si.

Saída: `data/processed/paginas_limpas.jsonl`, uma linha por página:

    {"document_id", "page", "text", "metadata"}

Uso:
  python eval/tools/extrair_paginas.py
  python eval/tools/extrair_paginas.py --limite 3      # só 3 documentos, para testar
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

for _f in (sys.stdout, sys.stderr):
    try:
        _f.reconfigure(encoding="utf-8")
    except Exception:
        pass

SAIDA = PROJECT_ROOT / "data/processed/paginas_limpas.jsonl"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--limite", type=int, help="processa só os N primeiros documentos")
    ap.add_argument("--saida", default=str(SAIDA))
    args = ap.parse_args()

    import fitz
    import extract_to_jsonl as E

    manifesto = E.load_manifest()
    # INPUT_DIR do extrator é relativo ao repositório; resolve contra PROJECT_ROOT
    pdfs = sorted((PROJECT_ROOT / E.INPUT_DIR).glob("*.pdf"))
    if args.limite:
        pdfs = pdfs[:args.limite]
    print(f"PDFs a processar: {len(pdfs)}")

    saida = Path(args.saida)
    saida.parent.mkdir(parents=True, exist_ok=True)
    n_pag = n_doc = 0

    with saida.open("w", encoding="utf-8") as out:
        for pdf in pdfs:
            doc_id = pdf.stem
            md = manifesto.get(doc_id) or manifesto.get(pdf.name) or {}
            if not md:
                print(f"  ⚠ {doc_id}: sem entrada no manifesto — pulado")
                continue

            with fitz.open(pdf) as documento:
                pages_lines = [E.extract_page_lines(p) for p in documento]

            repetidas = E.detect_repeated_margin_lines(pages_lines)
            inicio_refs = E.find_references_start(pages_lines=pages_lines,
                                                  doc_metadata=md)

            # `cleaning_stats` é contador de diagnóstico do extrator; aqui só
            # precisa existir para as funções escreverem nele.
            from collections import defaultdict
            stats: dict = defaultdict(int)
            limpas = []
            for originais in pages_lines:
                if not originais:
                    limpas.append([])
                    continue
                i = len(limpas)
                linhas = E.apply_references_cut(page_index=i, lines=originais,
                                                references_start=inicio_refs,
                                                cleaning_stats=stats)
                linhas = E.remove_repeated_margin_lines(linhas, repetidas, stats)
                linhas = [l for l in (E.normalize_line(x) for x in linhas) if l]
                linhas = [l for l in linhas if not E.is_noise_line(l)]
                limpas.append(linhas)

            classif = E.classify_noninformative_pages(limpas, md)

            escritas = 0
            for i, linhas in enumerate(limpas, start=1):
                if not linhas or classif[i - 1].discard:
                    continue
                texto = E.clean_text(E.lines_to_text(linhas))
                if not texto:
                    continue
                out.write(json.dumps({
                    "document_id": doc_id,
                    "page": i,
                    "text": texto,
                    "metadata": {**md, "document_id": doc_id, "page": i},
                }, ensure_ascii=False) + "\n")
                escritas += 1

            n_pag += escritas
            n_doc += 1
            print(f"  {doc_id:48s} {escritas:4d} páginas")

    print(f"\n✓ {saida}")
    print(f"  {n_doc} documentos, {n_pag} páginas limpas")
    print("\nEsta é a fonte comum. Cada estratégia de recorte parte daqui — e só daqui.")


if __name__ == "__main__":
    main()
