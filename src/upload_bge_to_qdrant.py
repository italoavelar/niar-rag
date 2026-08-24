import json
import os
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from tqdm import tqdm

from embedding_text import (
    EMBEDDING_TEXT_PROFILE,
    bge_cache_filename,
    corpus_embedding_fingerprint,
    validate_embedding_cache_metadata,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CACHE_DIR = PROJECT_ROOT / "eval/results/indexes"
MODEL_KEY = "bge_m3"
MODEL_ID = "BAAI/bge-m3"

CORPUS_FILE = (
    PROJECT_ROOT
    / "data/processed/documents.jsonl"
)

EMBED_DIM = 1024
BATCH_SIZE = 64


def load_corpus() -> dict:
    corpus = {}

    with CORPUS_FILE.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                document = json.loads(line)
                corpus[str(document["id"])] = document

    return corpus


def build_payload(document: dict) -> dict:
    metadata = document.get("metadata", {})

    return {
        "id_original": str(document.get("id", "")),
        "document_id": metadata.get("document_id", ""),
        "texto": document.get("text", ""),
        "fonte": metadata.get("source", ""),
        "source_type": metadata.get("source_type", ""),
        "title": metadata.get("title", ""),
        "page": metadata.get("page"),
        "chunk": metadata.get("chunk", ""),
        "document_type": metadata.get("document_type", ""),
        "author": metadata.get("author", ""),
        "issuer": metadata.get("issuer", ""),
        "year": metadata.get("year", ""),
        "theme": metadata.get("theme", ""),
        "ria_dimensions": metadata.get("ria_dimensions", []),
        "source_url": metadata.get("source_url", ""),
        "section_path": metadata.get("section_path", ""),
        "embedding_model": MODEL_ID,
        "embedding_text_profile": EMBEDDING_TEXT_PROFILE,
    }


def validate_contextual_bge_cache(cached, fingerprint: str) -> None:
    """Não permite subir vetores sem prova do perfil contextual atual."""
    validate_embedding_cache_metadata(
        {
            key: cached[key]
            for key in (
                "embedding_text_profile",
                "embedding_text_fingerprint",
                "embedding_model",
            )
            if key in cached.files
        },
        fingerprint,
        MODEL_ID,
    )


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")

    corpus = load_corpus()
    corpus_fingerprint = corpus_embedding_fingerprint(corpus.values())
    cache_file = CACHE_DIR / bge_cache_filename(
        MODEL_KEY,
        corpus_fingerprint,
    )

    if not cache_file.exists():
        raise FileNotFoundError(
            "Cache BGE contextual não encontrado: "
            f"{cache_file}. Caches antigos de texto bruto não são compatíveis."
        )

    print("Carregando cache contextual do BGE-M3...")
    cached = np.load(cache_file, allow_pickle=True)
    validate_contextual_bge_cache(cached, corpus_fingerprint)

    matrix = cached["matrix"].astype(np.float32)
    chunk_ids = [str(value) for value in cached["ids"]]

    if matrix.shape != (len(chunk_ids), EMBED_DIM):
        raise ValueError(
            "Formato inesperado do cache: "
            f"matrix={matrix.shape}, ids={len(chunk_ids)}"
        )

    if chunk_ids != list(corpus.keys()):
        raise ValueError(
            "Cache incompatível: IDs não correspondem à ordem do corpus."
        )

    print(f"Vetores carregados: {matrix.shape}")

    qdrant_url = os.getenv("QDRANT_BGE_URL")
    qdrant_api_key = os.getenv("QDRANT_BGE_API_KEY")
    collection = os.getenv(
        "QDRANT_BGE_COLLECTION",
        "niar_rag_documents_bge_m3",
    )

    if not qdrant_url:
        raise RuntimeError("QDRANT_BGE_URL não foi definido no .env.")

    if not qdrant_api_key:
        raise RuntimeError(
            "QDRANT_BGE_API_KEY não foi definido no .env."
        )

    print("Conectando ao Qdrant...")
    client = QdrantClient(
        url=qdrant_url,
        api_key=qdrant_api_key,
        timeout=120,
    )

    existing = {
        item.name
        for item in client.get_collections().collections
    }

    if collection not in existing:
        print(f"Criando coleção '{collection}'...")

        client.create_collection(
            collection_name=collection,
            vectors_config=models.VectorParams(
                size=EMBED_DIM,
                distance=models.Distance.COSINE,
            ),
        )
    else:
        print(f"Coleção '{collection}' já existe.")

    print("Enviando vetores para o Qdrant...")

    for start in tqdm(
        range(0, len(chunk_ids), BATCH_SIZE),
        desc="Upload",
    ):
        end = min(start + BATCH_SIZE, len(chunk_ids))
        points = []

        for index in range(start, end):
            chunk_id = chunk_ids[index]
            document = corpus[chunk_id]

            points.append(
                models.PointStruct(
                    id=index,
                    vector=matrix[index].tolist(),
                    payload=build_payload(document),
                )
            )

        client.upsert(
            collection_name=collection,
            points=points,
            wait=True,
        )

    count = client.count(
        collection_name=collection,
        exact=True,
    )

    print("\nUpload concluído.")
    print(f"Coleção: {collection}")
    print(f"Pontos no Qdrant: {count.count}")

    if count.count != len(chunk_ids):
        raise RuntimeError(
            "A quantidade de pontos no Qdrant não corresponde "
            "à quantidade de embeddings."
        )


if __name__ == "__main__":
    main()
