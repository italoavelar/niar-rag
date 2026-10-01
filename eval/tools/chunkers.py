#!/usr/bin/env python3
"""As quatro estratégias de recorte do benchmark — C1, C4, C5 e C6.

Todas partem do MESMO texto: `data/processed/paginas_limpas.jsonl`, produzido por
`eval/tools/extrair_paginas.py`. Nenhuma parte do `documents.jsonl`, que já vem
recortado em 1.200/200 e contaminaria a comparação a favor do C4.

    C1  Page              a página é o trecho. Preserva contexto e estrutura do
                          autor, mas mistura assuntos quando a página tem vários.

    C4  Recursive 1000/200  janela de 1.000 com sobreposição de 200, cortando em
                          separador natural quando possível. É a linha de base da
                          literatura, e o mais próximo do recorte atual.

    C5  Semantic          corta onde o ASSUNTO muda: embute sentença a sentença e
                          abre trecho novo quando a similaridade com o anterior
                          cai abaixo do percentil escolhido. Tamanho irregular por
                          construção — por isso o relatório traz a distribuição.

    C6  Legal Structure   corta por unidade normativa, carregando a hierarquia
                          (Capítulo → Seção → Artigo → §/inciso) como contexto.

O QUE C6 **NÃO** É, e por que isto importa
──────────────────────────────────────────
C6 não é aplicável ao acervo inteiro. Medido em 17/set com a hierarquia completa
(inclusive SEÇÃO, TÍTULO, incisos e alíneas, que a primeira medição perdia):

    39 de 63 documentos têm estrutura normativa regular
    24 não têm — e são justamente NIST, WHO, UNESCO, OECD, FDA, ISO, IMDRF, ICO

Esses 24 não são leis: são guidance escrito em prosa com títulos de seção. Não há
artigo para cortar. Então C6 é **híbrido por necessidade**: estrutura onde existe,
C4 onde não existe — e o resultado tem de ser reportado **separado por classe de
documento**, senão mistura o efeito da estratégia com a composição do acervo.

Uso:
  python eval/tools/chunkers.py C1
  python eval/tools/chunkers.py C4
  python eval/tools/chunkers.py C5 --percentil 25
  python eval/tools/chunkers.py C6
  python eval/tools/chunkers.py todas
"""

from __future__ import annotations

import argparse
import json
import re
import statistics as st
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

for _f in (sys.stdout, sys.stderr):
    try:
        _f.reconfigure(encoding="utf-8")
    except Exception:
        pass

PAGINAS = PROJECT_ROOT / "data/processed/paginas_limpas.jsonl"
DESTINO = PROJECT_ROOT / "data/processed/chunking"

MIN_CHUNK = 300          # mesmo piso do pipeline de produção
SEPARADORES = ["\n\n", "\n", ". ", "; ", ", ", " "]

# Hierarquia normativa, do mais alto ao mais baixo. A ordem importa: o contexto
# de um artigo é o capítulo e a seção que o precedem.
NIVEIS = [
    ("titulo",   re.compile(r"^\s*(T[ÍI]TULO\s+[IVXLC]+.*)$", re.M)),
    ("capitulo", re.compile(r"^\s*(CAP[ÍI]TULO\s+[IVXLC]+.*)$", re.M)),
    ("secao",    re.compile(r"^\s*(SE[ÇC][ÃA]O\s+[IVXLC]+.*)$", re.M)),
    ("artigo",   re.compile(r"(Art\.\s*\d+[º°]?[-A-Z]*|Artigo\s+\d+[º°]?)")),
]
ARTIGO = NIVEIS[-1][1]


def carregar_paginas() -> list[dict]:
    if not PAGINAS.exists():
        sys.exit(f"não encontrado: {PAGINAS}\n  rode antes: python eval/tools/extrair_paginas.py")
    return [json.loads(l) for l in PAGINAS.open(encoding="utf-8") if l.strip()]


def chunk_id(doc_id: str, texto: str) -> str:
    """Mesmo esquema do pipeline: id derivado do conteúdo."""
    from chunk_id import chunk_id_por_conteudo
    return chunk_id_por_conteudo(doc_id, texto)


def registro(doc_id: str, texto: str, md: dict, extra: dict | None = None) -> dict:
    meta = {**md, **(extra or {})}
    return {"id": chunk_id(doc_id, texto), "text": texto, "metadata": meta}


def fundir_pequenos(trechos: list[dict], minimo: int = MIN_CHUNK) -> list[dict]:
    """Junta trecho curto demais ao anterior do MESMO documento.

    Sem isto, C5 e C6 produzem muitos fragmentos: 49% das unidades normativas
    ficam abaixo de 300 caracteres. Fragmento dilui o sinal e não responde nada.
    """
    saida: list[dict] = []
    for t in trechos:
        if (saida and len(t["text"]) < minimo
                and saida[-1]["metadata"]["document_id"] == t["metadata"]["document_id"]):
            juntos = saida[-1]["text"] + "\n" + t["text"]
            saida[-1] = registro(t["metadata"]["document_id"], juntos, saida[-1]["metadata"])
        else:
            saida.append(t)
    return saida


# ── C1 — a página é o trecho ─────────────────────────────────────────────────

def c1_page(paginas: list[dict]) -> list[dict]:
    return fundir_pequenos([
        registro(p["document_id"], p["text"], p["metadata"], {"estrategia": "C1_page"})
        for p in paginas if p["text"].strip()
    ])


# ── C4 — janela recursiva 1000/200 ───────────────────────────────────────────

def _cortar_recursivo(texto: str, tamanho: int, sobreposicao: int) -> list[str]:
    if len(texto) <= tamanho:
        return [texto] if texto.strip() else []
    partes, inicio = [], 0
    while inicio < len(texto):
        fim = min(inicio + tamanho, len(texto))
        if fim < len(texto):
            # recua até o separador mais "forte" disponível na cauda da janela
            for sep in SEPARADORES:
                achado = texto.rfind(sep, inicio + tamanho // 2, fim)
                if achado != -1:
                    fim = achado + len(sep)
                    break
        pedaco = texto[inicio:fim].strip()
        if pedaco:
            partes.append(pedaco)
        if fim >= len(texto):
            break
        inicio = max(fim - sobreposicao, inicio + 1)
    return partes


def c4_recursivo(paginas: list[dict], tamanho: int = 1000,
                 sobreposicao: int = 200) -> list[dict]:
    # O documento é reconstruído antes de cortar: a janela de C4 não deve parar
    # na virada de página, senão C4 vira "C1 com subdivisão" e a comparação
    # perde o contraste que ela existe para medir.
    por_doc: dict[str, list[dict]] = {}
    for p in paginas:
        por_doc.setdefault(p["document_id"], []).append(p)

    saida = []
    for doc, pgs in por_doc.items():
        pgs = sorted(pgs, key=lambda x: x["page"])
        inteiro = "\n".join(x["text"] for x in pgs)
        for pedaco in _cortar_recursivo(inteiro, tamanho, sobreposicao):
            saida.append(registro(doc, pedaco, pgs[0]["metadata"],
                                  {"estrategia": "C4_recursivo", "page": None}))
    return fundir_pequenos(saida)


# ── C5 — fronteira semântica ─────────────────────────────────────────────────

def c5_semantico(paginas: list[dict], percentil: int = 25,
                 teto: int = 2500) -> list[dict]:
    """Abre trecho novo onde a similaridade entre sentenças vizinhas despenca."""
    import numpy as np
    from sentence_transformers import SentenceTransformer

    print("  carregando BGE-m3 para medir fronteira semântica...", flush=True)
    mod = SentenceTransformer("BAAI/bge-m3", device="cpu")

    por_doc: dict[str, list[dict]] = {}
    for p in paginas:
        por_doc.setdefault(p["document_id"], []).append(p)

    saida = []
    for n, (doc, pgs) in enumerate(sorted(por_doc.items()), 1):
        pgs = sorted(pgs, key=lambda x: x["page"])
        inteiro = "\n".join(x["text"] for x in pgs)
        sentencas = [s.strip() for s in re.split(r"(?<=[.!?])\s+", inteiro) if s.strip()]
        if len(sentencas) < 3:
            saida.append(registro(doc, inteiro, pgs[0]["metadata"],
                                  {"estrategia": "C5_semantico", "page": None}))
            continue

        V = mod.encode(sentencas, normalize_embeddings=True, batch_size=16,
                       show_progress_bar=False).astype("float32")
        sims = (V[:-1] * V[1:]).sum(axis=1)
        limiar = float(np.percentile(sims, percentil))

        atual = [sentencas[0]]
        for i, s in enumerate(sentencas[1:]):
            quebra = sims[i] < limiar
            grande = len(" ".join(atual)) + len(s) > teto
            if (quebra and len(" ".join(atual)) >= MIN_CHUNK) or grande:
                saida.append(registro(doc, " ".join(atual), pgs[0]["metadata"],
                                      {"estrategia": "C5_semantico", "page": None}))
                atual = [s]
            else:
                atual.append(s)
        if atual:
            saida.append(registro(doc, " ".join(atual), pgs[0]["metadata"],
                                  {"estrategia": "C5_semantico", "page": None}))
        print(f"    [{n:2d}] {doc[:44]:44s} {len(sentencas):5d} sentenças", flush=True)
    return fundir_pequenos(saida)


# ── C6 — estrutura jurídica, híbrido por necessidade ─────────────────────────

def _tem_estrutura(texto: str, minimo: int = 5) -> bool:
    return len(ARTIGO.findall(texto)) >= minimo


def _hierarquia_ate(texto: str, pos: int) -> dict:
    """Capítulo/seção/título vigentes na posição — o contexto do artigo."""
    contexto = {}
    for nome, rx in NIVEIS[:-1]:
        ultimo = None
        for m in rx.finditer(texto, 0, pos):
            ultimo = m.group(1).strip()
        if ultimo:
            contexto[nome] = ultimo
    return contexto


def c6_juridico(paginas: list[dict], teto: int = 2500) -> list[dict]:
    por_doc: dict[str, list[dict]] = {}
    for p in paginas:
        por_doc.setdefault(p["document_id"], []).append(p)

    saida, com_estrutura, sem_estrutura = [], [], []
    for doc, pgs in sorted(por_doc.items()):
        pgs = sorted(pgs, key=lambda x: x["page"])
        inteiro = "\n".join(x["text"] for x in pgs)

        if not _tem_estrutura(inteiro):
            # Guidance internacional: não há artigo para cortar. Cai no C4, e o
            # relatório separa — senão o número mistura estratégia com acervo.
            sem_estrutura.append(doc)
            for pedaco in _cortar_recursivo(inteiro, 1000, 200):
                saida.append(registro(doc, pedaco, pgs[0]["metadata"],
                                      {"estrategia": "C6_juridico",
                                       "classe": "sem_estrutura", "page": None}))
            continue

        com_estrutura.append(doc)
        marcas = list(ARTIGO.finditer(inteiro))
        for i, m in enumerate(marcas):
            ini = m.start()
            fim = marcas[i + 1].start() if i + 1 < len(marcas) else len(inteiro)
            corpo = inteiro[ini:fim].strip()
            if not corpo:
                continue
            ctx = _hierarquia_ate(inteiro, ini)
            cabecalho = " · ".join(v for v in
                                   (ctx.get("titulo"), ctx.get("capitulo"), ctx.get("secao"))
                                   if v)
            # A hierarquia entra NO TEXTO: é ela que dá sentido a "A obrigação
            # será..." quando o trecho é recuperado isolado.
            texto = f"[{cabecalho}]\n{corpo}" if cabecalho else corpo
            for pedaco in (_cortar_recursivo(texto, teto, 200)
                           if len(texto) > teto else [texto]):
                saida.append(registro(doc, pedaco, pgs[0]["metadata"],
                                      {"estrategia": "C6_juridico",
                                       "classe": "com_estrutura",
                                       "section_path": cabecalho or "",
                                       "page": None}))

    print(f"  documentos com estrutura normativa : {len(com_estrutura)}")
    print(f"  sem estrutura (recorte C4 aplicado): {len(sem_estrutura)}")
    if sem_estrutura:
        print(f"    {', '.join(sorted(sem_estrutura)[:6])} …")
    return fundir_pequenos(saida)


# ── main ─────────────────────────────────────────────────────────────────────

ESTRATEGIAS = {
    "C1": ("C1_page", c1_page),
    "C4": ("C4_recursivo_1000_200", c4_recursivo),
    "C5": ("C5_semantico", c5_semantico),
    "C6": ("C6_juridico", c6_juridico),
}


def gravar(nome: str, trechos: list[dict]) -> None:
    DESTINO.mkdir(parents=True, exist_ok=True)
    caminho = DESTINO / f"{nome}.jsonl"
    with caminho.open("w", encoding="utf-8") as f:
        for t in trechos:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    tam = [len(t["text"]) for t in trechos]
    print(f"\n✓ {caminho}")
    print(f"  {len(trechos)} trechos | mediana {st.median(tam):.0f} "
          f"média {st.mean(tam):.0f} min {min(tam)} max {max(tam)}")
    print(f"  abaixo de 300: {sum(1 for x in tam if x < 300)}   "
          f"acima de 2500: {sum(1 for x in tam if x > 2500)}")
    print(f"\n  avaliar: python eval/tools/bench_chunking.py avaliar "
          f"--nome {nome} --corpus data/processed/chunking/{nome}.jsonl")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("estrategia", choices=[*ESTRATEGIAS, "todas"])
    ap.add_argument("--percentil", type=int, default=25, help="C5: corte de similaridade")
    args = ap.parse_args()

    paginas = carregar_paginas()
    docs = {p["document_id"] for p in paginas}
    print(f"fonte comum: {len(paginas)} páginas limpas de {len(docs)} documentos\n")

    alvos = list(ESTRATEGIAS) if args.estrategia == "todas" else [args.estrategia]
    for chave in alvos:
        nome, fn = ESTRATEGIAS[chave]
        print(f"══ {chave} — {nome} ══")
        trechos = fn(paginas, percentil=args.percentil) if chave == "C5" else fn(paginas)
        gravar(nome, trechos)
        print()


if __name__ == "__main__":
    main()
