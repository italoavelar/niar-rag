#!/usr/bin/env python3
"""Prepara a auditoria do acervo: mede o que é mecânico, deixa o julgamento para a pessoa.

Para que serve
──────────────
A Fase 0 pede que alguém classifique os 63 documentos em *íntegro*, *truncado* ou
*não é o documento*. Abrir 63 documentos no escuro é trabalho de dias e o
resultado sai inconsistente entre quem revisa. Este script faz a parte
mecânica — contar, medir, procurar marcador — e entrega uma planilha com os
sinais lado a lado e duas colunas em branco para a decisão humana.

O que ele NÃO faz: decidir. As colunas `sinal_*` são indícios, e a coluna
`sugestao` é um palpite explícito. Quem assina a classificação é a pessoa.

Os três sinais
──────────────
  deontico   verbo de obrigação (deve, deverá, é vedado, shall, must…).
             Norma tem; ficha de produto não tem.
  dispositivo  referência a artigo/seção (Art., §, Section, Clause, Anexo).
             Norma tem; página de catálogo não tem.
  loja       marcador de e-commerce (Add to cart, CHF, Read sample, Shipping
             costs…). É o que denuncia as nove entradas da ISO, que são páginas
             do site e não as normas.

E, para truncamento, um sinal que significa algo: **buraco na sequência de
páginas**. Se o documento tem trechos das páginas 1 a 12 e depois 40 a 60, ou a
extração falhou no meio ou parte do PDF é imagem sem OCR. "Termina no meio de
uma frase" foi testado e descartado: o corpus é chunkado, o último trecho termina
onde o recorte mandou, e o sinal acusava 21 documentos — o GDPR inteiro entre
eles. Sinal que acusa todo mundo não separa nada.

Uso
───
  python eval/tools/auditar_acervo.py                    # relatório na tela
  python eval/tools/auditar_acervo.py --csv auditoria.csv   # planilha para preencher
  python eval/tools/auditar_acervo.py --dump docs/auditoria/  # texto de cada documento
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CORPUS = PROJECT_ROOT / "data/processed/documents.jsonl"
GOLDEN = PROJECT_ROOT / "eval/data/golden_qa.jsonl"

for _fluxo in (sys.stdout, sys.stderr):
    try:
        _fluxo.reconfigure(encoding="utf-8")
    except Exception:
        pass

DEONTICO = re.compile(
    r"\b(dever[áãa]{1,2}o?|devem|é vedado|é obrigat[óo]ri[oa]|fica vedado|"
    r"shall|must|should|may not|is prohibited|is required)\b",
    re.IGNORECASE,
)
DISPOSITIVO = re.compile(
    r"(\bArt\.\s*\d|\bArtigo\s+\d|\bArticle\s+\d|§\s*\d|\bSection\s+\d|"
    r"\bClause\s+\d|\bAnexo\b|\bAnnex\b|\bCAP[ÍI]TULO\b|\bCHAPTER\b)",
)
LOJA = re.compile(
    r"(Add to cart|Read sample|Shipping costs|\bCHF\b|Preview|Buy this standard|"
    r"Life cycle|ICS\s*>|Table of contents\s*$|Abstract Preview)",
    re.IGNORECASE,
)


def buracos_de_pagina(itens: list[dict]) -> tuple[int, str]:
    """Páginas ausentes entre a primeira e a última indexada.

    Extração que falha no meio, ou PDF com páginas em imagem sem OCR, deixa
    buraco. É o único sinal de truncamento que se sustenta sozinho.
    """
    paginas = sorted({r["metadata"]["page"] for r in itens
                      if isinstance(r["metadata"].get("page"), int)})
    if len(paginas) < 2:
        return 0, ""
    faltando = [p for p in range(paginas[0], paginas[-1] + 1) if p not in set(paginas)]
    if not faltando:
        return 0, ""
    trechos, inicio, anterior = [], faltando[0], faltando[0]
    for p in faltando[1:]:
        if p != anterior + 1:
            trechos.append(f"{inicio}-{anterior}" if inicio != anterior else str(inicio))
            inicio = p
        anterior = p
    trechos.append(f"{inicio}-{anterior}" if inicio != anterior else str(inicio))
    return len(faltando), ", ".join(trechos[:6]) + (" …" if len(trechos) > 6 else "")


def carregar():
    docs = defaultdict(list)
    with CORPUS.open(encoding="utf-8") as fh:
        for linha in fh:
            if linha.strip():
                r = json.loads(linha)
                docs[r["metadata"]["document_id"]].append(r)
    for itens in docs.values():
        itens.sort(key=lambda r: (r["metadata"].get("page") or 0,
                                  r["metadata"].get("chunk") or 0))
    return docs


def perguntas_por_documento():
    if not GOLDEN.exists():
        return {}
    contagem = defaultdict(set)
    for linha in GOLDEN.open(encoding="utf-8"):
        if not linha.strip():
            continue
        r = json.loads(linha)
        for cid in (r.get("qrels") or {}):
            contagem[cid.rsplit("_", 1)[0]].add(r["qid"])
    return {d: len(v) for d, v in contagem.items()}


def avaliar(doc_id: str, itens: list[dict]) -> dict:
    texto = "\n".join(r["text"] for r in itens)
    n_chars = len(texto)
    md = itens[0]["metadata"]
    por_mil = lambda n: round(1000 * n / max(n_chars, 1), 2)

    deontico = len(DEONTICO.findall(texto))
    dispositivo = len(DISPOSITIVO.findall(texto))
    loja = len(LOJA.findall(texto))
    n_faltando, faixas = buracos_de_pagina(itens)

    # Palpite, não veredito. A regra é a mesma que qualquer pessoa aplicaria
    # olhando os sinais; existe para ordenar a fila de revisão.
    if loja >= 2 and dispositivo <= 2:
        sugestao = "NAO E O DOCUMENTO (catalogo/loja)"
    elif n_chars < 8000 and dispositivo <= 3:
        sugestao = "SUSPEITO (pequeno demais)"
    elif n_faltando >= 3:
        sugestao = "TRUNCADO? (paginas ausentes)"
    elif deontico == 0 and dispositivo == 0:
        sugestao = "SUSPEITO (sem marca de norma)"
    else:
        sugestao = "parece integro"

    return {
        "document_id": doc_id,
        "trechos": len(itens),
        "caracteres": n_chars,
        "mediana_trecho": int(st.median([len(r["text"]) for r in itens])),
        "tipo": md.get("source_type") or "",
        "fonte": md.get("source") or "",
        "titulo": (md.get("title") or "")[:70],
        "sinal_deontico": deontico,
        "sinal_deontico_por_mil": por_mil(deontico),
        "sinal_dispositivo": dispositivo,
        "sinal_loja": loja,
        "paginas_ausentes": n_faltando,
        "paginas_ausentes_faixas": faixas,
        "sugestao": sugestao,
        "inicio": " ".join(texto[:160].split()),
        "fim": " ".join(texto[-160:].split()),
        # colunas que a PESSOA preenche
        "classificacao": "",
        "destino": "",
        "licenca": "",
        "observacao": "",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepara a auditoria do acervo.")
    ap.add_argument("--csv", type=Path, help="grava a planilha de auditoria")
    ap.add_argument("--dump", type=Path,
                    help="grava o texto indexado de cada documento, um .txt por documento")
    args = ap.parse_args()

    docs = carregar()
    com_pergunta = perguntas_por_documento()
    linhas = [avaliar(d, itens) for d, itens in sorted(docs.items())]
    for l in linhas:
        l["perguntas_hoje"] = com_pergunta.get(l["document_id"], 0)

    def prioridade(rotulo: str) -> int:
        if rotulo.startswith("NAO E O DOCUMENTO"):
            return 0
        if rotulo.startswith("SUSPEITO (pequeno"):
            return 1
        if rotulo.startswith("SUSPEITO (sem marca"):
            return 2
        if rotulo.startswith("TRUNCADO"):
            return 3
        return 4

    linhas.sort(key=lambda l: (prioridade(l["sugestao"]), -l["caracteres"]))

    print(f"── Auditoria do acervo — {len(linhas)} documentos, "
          f"{sum(l['trechos'] for l in linhas)} trechos ──\n")
    cab = (f"{'documento':<46}{'trechos':>8}{'chars':>9}{'deon':>6}{'disp':>6}"
           f"{'loja':>6}{'pag!':>6}{'perg':>6}  sugestao")
    print(cab)
    print("-" * (len(cab) + 26))
    for l in linhas:
        print(f"{l['document_id'][:44]:<46}{l['trechos']:>8}{l['caracteres']:>9}"
              f"{l['sinal_deontico']:>6}{l['sinal_dispositivo']:>6}{l['sinal_loja']:>6}"
              f"{l['paginas_ausentes']:>6}{l['perguntas_hoje']:>6}  {l['sugestao']}")

    print("\n── Fila de revisão, por prioridade ──")
    contagem = Counter(l["sugestao"] for l in linhas)
    for rotulo, n in sorted(contagem.items(), key=lambda x: prioridade(x[0])):
        print(f"  {rotulo:<42}{n:>3} documento(s)")
    revisar = sum(n for r, n in contagem.items() if r != "parece integro")
    print(f"\n  a revisar com atenção: {revisar} de {len(linhas)}   "
          f"o resto é conferência rápida")

    if args.dump:
        args.dump.mkdir(parents=True, exist_ok=True)
        for doc_id, itens in docs.items():
            alvo = args.dump / f"{doc_id}.txt"
            with alvo.open("w", encoding="utf-8") as fh:
                for r in itens:
                    md = r["metadata"]
                    fh.write(f"\n===== pág {md.get('page')} · trecho {md.get('chunk')} "
                             f"· {len(r['text'])} chars · {r['id']}\n")
                    fh.write(r["text"] + "\n")
        print(f"\n✓ texto indexado de {len(docs)} documentos em {args.dump}/")

    if args.csv:
        campos = ["document_id", "titulo", "tipo", "trechos", "caracteres", "mediana_trecho",
                  "perguntas_hoje", "sinal_deontico", "sinal_deontico_por_mil",
                  "sinal_dispositivo", "sinal_loja", "paginas_ausentes",
                  "paginas_ausentes_faixas", "sugestao",
                  "inicio", "fim", "fonte",
                  "classificacao", "destino", "licenca", "observacao"]
        with args.csv.open("w", encoding="utf-8-sig", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=campos)
            w.writeheader()
            w.writerows([{k: l.get(k, "") for k in campos} for l in linhas])
        print(f"\n✓ {args.csv}  ({len(linhas)} linhas)")
        print("  Preencher à mão, uma linha por documento:")
        print("    classificacao : integro | truncado | nao_e_o_documento")
        print("    destino       : manter | recoletar | remover")
        print("    licenca       : pode redistribuir o TEXTO? (sim | nao | so metadados)")
        print("    observacao    : o que você viu que justifica a decisão")


if __name__ == "__main__":
    main()
