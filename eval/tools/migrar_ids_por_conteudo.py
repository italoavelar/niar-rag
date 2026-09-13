#!/usr/bin/env python3
"""Migra os identificadores de trecho de POSICIONAIS para DERIVADOS DO CONTEÚDO.

Problema que isto resolve
─────────────────────────
Hoje o id de um trecho é um endereço: `<doc>_p{página}_c{índice}` para PDF
(`src/extract_to_jsonl.py:763-767`) e `<doc>_c{índice}` para HTML
(`src/extract_html_to_jsonl.py:230`), onde o índice é um contador de posição.
Se o recorte muda, o "terceiro pedaço da página 12" passa a ser outro texto e
toda anotação que aponta para ele passa a apontar para outra coisa — em
silêncio. Foi o que aconteceu no re-chunking de 01/09/2026: o texto anotado
deslizou em ~40% das perguntas e a validação não pegou, porque ela confere se o
id EXISTE, e ele existia.

Depois desta migração o id é `<document_id>_<12 hex do sha256 do texto>`:

  • mesmo texto ⇒ mesmo id, independentemente dos vizinhos;
  • documento novo só acrescenta ids, não renumera nada do que existe;
  • reexecutar a extração sem mudar parâmetros reproduz os mesmos ids;
  • se o recorte mudar de verdade, o id some ⇒ falha barulhenta, não silenciosa.

O hash é do texto CRU (`record["text"]`), nunca de `build_embedding_text()`.
É o que garante que enriquecer metadado depois — preencher `section_path`, por
exemplo — NÃO mude os ids: muda só o que se embute, e aí se recalculam os
vetores, mas as anotações continuam válidas.

O que é reescrito
─────────────────
  data/processed/documents.jsonl        ids dos trechos
  eval/data/qrels.csv                   ChunkId
  eval/data/golden_qa.jsonl             chaves de "qrels" + campo NOVO "qrels_text"
  eval/results/indexes/dense_*_context-*.npz
                                        vetor `ids`, `embedding_text_fingerprint`
                                        e o nome do arquivo (o fingerprint inclui o id).
                                        Os VETORES não são recalculados: o texto
                                        embutido não mudou.

O campo `qrels_text` é a apólice de seguro: com o texto da âncora guardado
dentro do gabarito, qualquer mudança futura de recorte é remapeável por
casamento de texto, e nenhuma anotação precisa ser refeita.

NÃO é tocado: `eval/results*/retrieval/rankings*.json` e demais saídas de
experimento, que ficam com os ids antigos de propósito — são registro histórico
de uma execução e devem ser regerados, não reescritos. O script avisa quais são.

Uso
───
  python eval/tools/migrar_ids_por_conteudo.py              # simulação (padrão)
  python eval/tools/migrar_ids_por_conteudo.py --aplicar    # escreve, com backup
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
EVAL = PROJECT_ROOT / "eval"
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# console do Windows costuma vir em cp1252 e quebra em acento/seta; não exigir
# que quem roda saiba de PYTHONIOENCODING
for _fluxo in (sys.stdout, sys.stderr):
    try:
        _fluxo.reconfigure(encoding="utf-8")
    except Exception:
        pass

from chunk_id import chunk_id_por_conteudo, normalizar_texto  # noqa: E402
from embedding_text import (  # noqa: E402
    corpus_embedding_fingerprint, bge_cache_filename,
)

CORPUS = PROJECT_ROOT / "data/processed/documents.jsonl"
QRELS = EVAL / "data/qrels.csv"
GOLDEN = EVAL / "data/golden_qa.jsonl"
INDEXES = EVAL / "results/indexes"
# A geração de id vive em src/chunk_id.py, importada tanto aqui quanto pelos
# dois extratores. Duplicar a função faria a migração e a próxima extração
# divergirem silenciosamente — exatamente o tipo de erro que ela veio corrigir.
normalizar = normalizar_texto
novo_id = chunk_id_por_conteudo


def carregar_corpus() -> list[dict]:
    with CORPUS.open(encoding="utf-8") as fh:
        return [json.loads(linha) for linha in fh if linha.strip()]


def construir_mapa(registros: list[dict]) -> tuple[dict[str, str], list[tuple]]:
    """id antigo -> id novo. Colisão (texto idêntico no mesmo documento) recebe
    sufixo contador, mantendo a propriedade de determinismo."""
    mapa: dict[str, str] = {}
    vistos: dict[str, int] = {}
    colisoes: list[tuple] = []
    for reg in registros:
        doc = reg["metadata"]["document_id"]
        base = novo_id(doc, reg["text"])
        n = vistos.get(base, 0)
        vistos[base] = n + 1
        final = base if n == 0 else f"{base}-{n + 1}"
        if n:
            colisoes.append((reg["id"], final))
        mapa[reg["id"]] = final
    return mapa, colisoes


def backup(caminho: Path, carimbo: str) -> Path:
    destino = caminho.with_suffix(caminho.suffix + f".bak-pre-migracao-{carimbo}")
    shutil.copy2(caminho, destino)
    return destino


def main() -> None:
    ap = argparse.ArgumentParser(description="Migra ids de trecho para hash de conteúdo.")
    ap.add_argument("--aplicar", action="store_true",
                    help="escreve os arquivos (sem isto, só simula)")
    args = ap.parse_args()
    escrever = args.aplicar
    carimbo = datetime.now().strftime("%Y%m%d-%H%M%S")

    print("── Migração de ids posicionais → ids por conteúdo ──")
    print(f"modo: {'APLICAR (escreve com backup)' if escrever else 'SIMULAÇÃO (não escreve nada)'}\n")

    registros = carregar_corpus()
    mapa, colisoes = construir_mapa(registros)
    print(f"corpus: {len(registros)} trechos, {len({r['metadata']['document_id'] for r in registros})} documentos")
    print(f"ids gerados: {len(set(mapa.values()))} distintos   colisões resolvidas por sufixo: {len(colisoes)}")
    for antigo, final in colisoes[:5]:
        print(f"    {antigo} → {final}")

    # ── validação das anotações antes de tocar em nada ──────────────────────
    gold = [json.loads(l) for l in GOLDEN.open(encoding="utf-8") if l.strip()]
    texto_por_id = {r["id"]: r["text"] for r in registros}
    refs = orfas = 0
    for reg in gold:
        for cid in (reg.get("qrels") or {}):
            refs += 1
            if cid not in mapa:
                orfas += 1
                print(f"    ÓRFÃ: {reg['qid']} aponta para {cid}, que não existe no corpus")
    print(f"\nanotações no gabarito: {refs} referências, {orfas} órfãs")
    if orfas:
        print("  ⚠ resolva as órfãs antes de aplicar — elas não sobrevivem à migração")

    # ── o que muda em cada arquivo ──────────────────────────────────────────
    npz_alvos = sorted(INDEXES.glob("dense_*_context-*.npz"))
    print(f"\narquivos a reescrever:")
    print(f"  {CORPUS.relative_to(PROJECT_ROOT)}  ({len(registros)} ids)")
    print(f"  {QRELS.relative_to(PROJECT_ROOT)}")
    print(f"  {GOLDEN.relative_to(PROJECT_ROOT)}  (+ campo qrels_text)")
    for p in npz_alvos:
        print(f"  {p.relative_to(PROJECT_ROOT)}  (ids + fingerprint + nome do arquivo)")

    historicos = sorted(EVAL.glob("results*/retrieval/rankings*.json"))
    if historicos:
        print(f"\nNÃO serão tocados ({len(historicos)} arquivos de resultado histórico, ids antigos):")
        for p in historicos[:4]:
            print(f"  {p.relative_to(PROJECT_ROOT)}")
        if len(historicos) > 4:
            print(f"  … e mais {len(historicos) - 4}")
        print("  Regere-os depois; reescrever saída de experimento falsifica o registro.")

    if not escrever:
        print("\n(simulação — rode com --aplicar para escrever)")
        return

    # ── 1. corpus ───────────────────────────────────────────────────────────
    backup(CORPUS, carimbo)
    with CORPUS.open("w", encoding="utf-8") as fh:
        for reg in registros:
            reg["id"] = mapa[reg["id"]]
            fh.write(json.dumps(reg, ensure_ascii=False) + "\n")
    print(f"\n✓ {CORPUS.name}")

    # ── 2. qrels.csv ────────────────────────────────────────────────────────
    backup(QRELS, carimbo)
    with QRELS.open(encoding="utf-8", newline="") as fh:
        linhas = list(csv.DictReader(fh))
    for linha in linhas:
        linha["ChunkId"] = mapa.get(linha["ChunkId"], linha["ChunkId"])
    with QRELS.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["QueryId", "ChunkId", "Relevance"])
        w.writeheader()
        w.writerows(linhas)
    print(f"✓ {QRELS.name}  ({len(linhas)} linhas)")

    # ── 3. golden_qa.jsonl, agora carregando o texto da âncora ──────────────
    backup(GOLDEN, carimbo)
    with GOLDEN.open("w", encoding="utf-8") as fh:
        for reg in gold:
            antigos = reg.get("qrels") or {}
            reg["qrels"] = {mapa.get(c, c): g for c, g in antigos.items()}
            reg["qrels_text"] = {mapa.get(c, c): texto_por_id.get(c, "") for c in antigos}
            fh.write(json.dumps(reg, ensure_ascii=False) + "\n")
    print(f"✓ {GOLDEN.name}  ({len(gold)} perguntas, com qrels_text)")

    # ── 4. caches de embedding: relabel, sem recalcular vetores ─────────────
    fp_novo = corpus_embedding_fingerprint(registros)
    for caminho in npz_alvos:
        with np.load(caminho, allow_pickle=True) as z:
            # materializa tudo e FECHA antes de renomear: o .npz é lido de forma
            # preguiçosa e o Windows bloqueia rename de arquivo ainda aberto
            dados = {k: z[k] for k in z.keys()}
        dados["ids"] = np.array([mapa.get(str(i), str(i)) for i in dados["ids"]], dtype=object)
        if "embedding_text_fingerprint" in dados:
            dados["embedding_text_fingerprint"] = np.array(fp_novo)
        modelo = caminho.name.split("_context-")[0].replace("dense_", "")
        destino = INDEXES / bge_cache_filename(modelo, fp_novo)
        np.savez_compressed(destino, **dados)
        caminho.rename(caminho.with_suffix(caminho.suffix + f".bak-pre-migracao-{carimbo}"))
        print(f"✓ {destino.name}  (vetores preservados, {len(dados['ids'])} ids)")

    # ── 5. verificação pós-escrita ──────────────────────────────────────────
    print("\n── verificação ──")
    novos = carregar_corpus()
    ids_novos = {r["id"] for r in novos}
    texto_novo = {r["id"]: r["text"] for r in novos}
    gold2 = [json.loads(l) for l in GOLDEN.open(encoding="utf-8") if l.strip()]
    faltando = divergente = conferidas = 0
    for reg in gold2:
        for cid, texto in (reg.get("qrels_text") or {}).items():
            conferidas += 1
            if cid not in ids_novos:
                faltando += 1
            elif normalizar(texto_novo[cid]) != normalizar(texto):
                divergente += 1
    print(f"âncoras conferidas: {conferidas}")
    print(f"  ids ausentes no corpus novo : {faltando}")
    print(f"  texto divergente do gravado : {divergente}")
    estavel = all(novo_id(r["metadata"]["document_id"], r["text"]) in ids_novos for r in novos[:200])
    print(f"  ids reproduzíveis a partir do texto: {'sim' if estavel else 'NÃO — investigar'}")
    print(f"\nfingerprint novo do corpus: {fp_novo}")
    print("Próximo passo: reindexar o Qdrant de produção (LEME_gemini), que também guarda ids.")


if __name__ == "__main__":
    main()
