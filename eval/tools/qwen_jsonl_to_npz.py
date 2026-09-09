#!/usr/bin/env python3
"""Converte o backup .jsonl do Qwen no cache .npz que o DenseRetriever lê.

O Colab devolve os vetores do Qwen em .jsonl — o formato que
src/build_qwen_vectorstore.py exige para subir ao Qdrant. Já o pipeline de
avaliação (eval/lib/retrievers.py:DenseRetriever) lê um .npz nomeado por
embedding_text.bge_cache_filename(). Este script faz a ponte, sem tocar no
.jsonl original: ele continua sendo a fonte do upload ao Qdrant.

O .npz gerado carrega os mesmos metadados que o cache do BGE, porque
DenseRetriever chama validate_embedding_cache_metadata() e rejeita cache de
outro perfil, outro corpus ou outro modelo.

Uso:
    python eval/tools/qwen_jsonl_to_npz.py [--force]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from embedding_text import (                       # noqa: E402
    EMBEDDING_TEXT_PROFILE,
    bge_cache_filename,
    corpus_embedding_fingerprint,
    validate_embedding_cache_metadata,
)

CORPUS_FILE = PROJECT_ROOT / "data/processed/documents.jsonl"
QWEN_BACKUP = PROJECT_ROOT / "data/processed/embeddings_qwen3_0_6b_context_v1_backup.jsonl"
INDEXES_DIR = PROJECT_ROOT / "eval/results/indexes"

EMBEDDER_NAME = "qwen3_0_6b"          # precisa casar com a chave em eval/config.yaml
MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"
EMBED_DIM = 1024


def load_corpus() -> "OrderedDict[str, dict]":
    corpus: "OrderedDict[str, dict]" = OrderedDict()
    with CORPUS_FILE.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                doc = json.loads(line)
                corpus[str(doc["id"])] = doc
    return corpus


def read_backup(corpus) -> np.ndarray:
    """Lê o .jsonl validando modelo, perfil, dimensão e ordem dos ids."""
    if not QWEN_BACKUP.exists():
        raise FileNotFoundError(f"Backup Qwen ausente: {QWEN_BACKUP}")

    ids, vectors = [], []
    with QWEN_BACKUP.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("embedding_model") != MODEL_ID:
                raise ValueError(f"linha {lineno}: modelo {rec.get('embedding_model')!r}.")
            if rec.get("embedding_text_profile") != EMBEDDING_TEXT_PROFILE:
                raise ValueError(f"linha {lineno}: perfil textual diferente.")
            if rec.get("embedding_dimension") != EMBED_DIM:
                raise ValueError(f"linha {lineno}: dimensão diferente.")
            ids.append(str(rec["document"]["id"]))
            vectors.append(rec["vector"])

    if ids != list(corpus.keys()):
        raise ValueError(
            "Ordem dos ids do backup não corresponde ao corpus. O DenseRetriever "
            "compara posição a posição e rejeitaria o cache."
        )

    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.shape != (len(corpus), EMBED_DIM):
        raise ValueError(f"shape inesperado: {matrix.shape}")
    return matrix


def main() -> None:
    parser = argparse.ArgumentParser(description="Backup .jsonl do Qwen -> cache .npz.")
    parser.add_argument("--force", action="store_true",
                        help="Sobrescreve o .npz se ele já existir.")
    args = parser.parse_args()

    corpus = load_corpus()
    fingerprint = corpus_embedding_fingerprint(corpus.values())
    out = INDEXES_DIR / bge_cache_filename(EMBEDDER_NAME, fingerprint)

    if out.exists() and not args.force:
        print(f"Já existe: {out.name}\nUse --force para sobrescrever.")
        return

    print(f"corpus      : {len(corpus)} chunks")
    print(f"fingerprint : {fingerprint[:16]}")
    matrix = read_backup(corpus)
    print(f"vetores     : {matrix.shape}")

    norms = np.linalg.norm(matrix, axis=1)
    print(f"norma L2    : min={norms.min():.6f} max={norms.max():.6f}")

    # Os vetores do Colab sairam em bfloat16 (8 bits de mantissa), entao a
    # normalizacao do modelo nao sobreviveu ao arredondamento: as normas
    # desviam ate ~0,4% de 1. O DenseRetriever e o Qdrant tratam a busca como
    # cosseno; renormalizar aqui torna o produto interno cosseno exato.
    # ATENCAO: isto NAO desfaz a perda de precisao do bfloat16 — so corrige a
    # escala. Ver a nota sobre dtype no cabecalho do e1_qwen_vs_bge.py.
    if not np.allclose(norms, 1.0, atol=1e-5):
        matrix = (matrix / norms[:, None]).astype(np.float32)
        print("norma L2    : renormalizada para 1.0 (cosseno exato)")

    INDEXES_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        matrix=matrix,
        ids=np.array(list(corpus.keys()), dtype=object),
        embedding_text_profile=EMBEDDING_TEXT_PROFILE,
        embedding_text_fingerprint=fingerprint,
        embedding_model=MODEL_ID,
    )

    # Relê pelo mesmo caminho do DenseRetriever, para o script falhar aqui e
    # não no meio da avaliação.
    check = np.load(out, allow_pickle=True)
    validate_embedding_cache_metadata(
        {k: check[k] for k in ("embedding_text_profile", "embedding_text_fingerprint",
                               "embedding_model") if k in check.files},
        fingerprint,
        MODEL_ID,
    )
    if [str(v) for v in check["ids"]] != list(corpus.keys()):
        raise ValueError("Releitura falhou: ordem dos ids divergiu.")
    if check["matrix"].shape != matrix.shape:
        raise ValueError("Releitura falhou: shape divergiu.")

    size_mb = out.stat().st_size / 1e6
    print(f"\nescrito e revalidado: {out.relative_to(PROJECT_ROOT)} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
