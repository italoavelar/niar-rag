import json
import os
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parent.parent

CACHE_FILE = (
    PROJECT_ROOT
    / "eval/results/indexes/dense_bge_m3_5791.npz"
)

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
        "year": metadata.get("year", ""),
        "theme": metadata.get("theme", ""),
        "ria_dimensions": metadata.get("ria_dimensions", []),
        "source_url": metadata.get("source_url", ""),
        "embedding_model": "BAAI/bge-m3",
    }


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")

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

    if not CACHE_FILE.exists():
        raise FileNotFoundError(
            f"Cache do BGE não encontrado: {CACHE_FILE}"
        )

    print("Carregando cache do BGE-M3...")
    cached = np.load(CACHE_FILE, allow_pickle=True)

    matrix = cached["matrix"].astype(np.float32)
    chunk_ids = [str(value) for value in cached["ids"]]

    if matrix.shape != (len(chunk_ids), EMBED_DIM):
        raise ValueError(
            "Formato inesperado do cache: "
            f"matrix={matrix.shape}, ids={len(chunk_ids)}"
        )

    print(f"Vetores carregados: {matrix.shape}")
    corpus = load_corpus()

    missing_ids = [
        chunk_id
        for chunk_id in chunk_ids
        if chunk_id not in corpus
    ]

    if missing_ids:
        raise ValueError(
            f"{len(missing_ids)} IDs do cache não existem no corpus."
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
