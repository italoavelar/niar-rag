from __future__ import annotations

import json
import time
from pathlib import Path
from tqdm import tqdm
import os
import numpy as np

from google import genai
from google.genai import types
from qdrant_client import QdrantClient
from qdrant_client.http import models
from dotenv import load_dotenv

from embedding_text import (
    EMBEDDING_TEXT_PROFILE,
    build_embedding_text,
    corpus_embedding_fingerprint,
)

COLLECTION_NAME = "niar_rag_documents"

load_dotenv()

# Configurações e constantes
JSONL_FILE = Path("data/processed/documents.jsonl")
LEGACY_BACKUP_FILE = Path("data/processed/embeddings_backup.jsonl")
BACKUP_FILE = Path(
    os.getenv(
        "GEMINI_CONTEXTUAL_BACKUP_FILE",
        "data/processed/embeddings_context_v1_backup.jsonl",
    )
)
if BACKUP_FILE == LEGACY_BACKUP_FILE:
    raise ValueError(
        "GEMINI_CONTEXTUAL_BACKUP_FILE não pode apontar para o backup legado."
    )

# Variáveis de ambiente para conexões
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
GOOGLE_GENAI_API_KEY = os.getenv("GOOGLE_GENAI_API_KEY")

EMBED_DIM = 3072

BATCH_SIZE_EMBEDDINGS = 20
BATCH_SIZE_QDRANT = 64

# Normaliza um vetor para ter norma 1 (unitário)
def normalize(vec):
    v = np.array(vec)
    norm = np.linalg.norm(v)

    if norm == 0:
        return v.tolist()

    return (v / norm).tolist()

# Carrega os documentos processados do arquivo JSONL
def load_documents():
    documents = []

    with open(JSONL_FILE, "r", encoding="utf-8") as file:
        for line in file:
            documents.append(json.loads(line))

    return documents


def build_embedding_inputs(documents: list[dict]) -> list[str]:
    """Monta as entradas contextuais sem alterar os registros originais."""
    return [build_embedding_text(document) for document in documents]

# Carrega o backup de embeddings já processados para evitar retrabalho em caso de falhas
def load_backup(documents: list[dict]):
    processed_data = []

    if BACKUP_FILE.exists():
        with open(BACKUP_FILE, "r", encoding="utf-8") as file:
            for line in file:
                processed_data.append(json.loads(line))

        if len(processed_data) > len(documents):
            raise ValueError(
                "Backup contextual possui mais registros que o corpus atual."
            )

        for index, item in enumerate(processed_data):
            expected_fingerprint = corpus_embedding_fingerprint(
                [documents[index]]
            )
            if item.get("embedding_text_profile") != EMBEDDING_TEXT_PROFILE:
                raise ValueError(
                    "Backup incompatível: perfil de embedding ausente ou diferente "
                    f"de {EMBEDDING_TEXT_PROFILE!r}."
                )
            if item.get("embedding_text_fingerprint") != expected_fingerprint:
                raise ValueError(
                    "Backup incompatível: fingerprint do chunk contextual não confere."
                )

        print(
            f"Retomando backup {EMBEDDING_TEXT_PROFILE}: "
            f"{len(processed_data)} embeddings já salvos."
        )
    else:
        print(
            f"Nenhum backup {EMBEDDING_TEXT_PROFILE} encontrado. "
            "Iniciando do zero."
        )

    return processed_data

# Gera embeddings para os documentos usando a API Gemini e salva em backup
def generate_embeddings(documents, processed_data):
    documents_to_process = documents[len(processed_data):]

    if not documents_to_process:
        print("Todos os embeddings já foram gerados.")
        return processed_data

    print("Inicializando cliente Gemini...")
    client = genai.Client(api_key=GOOGLE_GENAI_API_KEY)

    BACKUP_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(BACKUP_FILE, "a", encoding="utf-8") as backup:
        for i in tqdm(
            range(0, len(documents_to_process), BATCH_SIZE_EMBEDDINGS),
            desc="Gerando embeddings",
        ):
            batch = documents_to_process[i:i + BATCH_SIZE_EMBEDDINGS]
            texts = build_embedding_inputs(batch)

            try:
                response = client.models.embed_content(
                    model="gemini-embedding-001",
                    contents=texts,
                    config=types.EmbedContentConfig(
                        task_type="RETRIEVAL_DOCUMENT",
                        title="Base de Conhecimento NIAR Saúde",
                    ),
                )

                for doc, emb in zip(batch, response.embeddings):
                    record = {
                        "document": doc,
                        "vector": normalize(emb.values),
                        "embedding_text_profile": EMBEDDING_TEXT_PROFILE,
                        "embedding_text_fingerprint": (
                            corpus_embedding_fingerprint([doc])
                        ),
                    }

                    backup.write(json.dumps(record, ensure_ascii=False) + "\n")
                    processed_data.append(record)

                time.sleep(15)

            except Exception as error:
                print("Erro ao gerar embeddings. Progresso salvo no backup.")
                print(f"Detalhe: {error}")
                raise

    return processed_data

# Constrói o payload para cada ponto a ser inserido no Qdrant com os metadados
def build_payload(doc: dict) -> dict:
    metadata = doc.get("metadata", {})

    return {
        "id_original": doc.get("id", ""),
        "document_id": metadata.get("document_id", ""),
        "texto": doc.get("text", ""),
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
        "embedding_text_profile": EMBEDDING_TEXT_PROFILE,
    }

def recreate_collection(qdrant):
    if not COLLECTION_NAME:
        raise ValueError(
            "COLLECTION_NAME está vazio. Defina o nome da collection em agent/utils/tools.py"
        )

    print(f"Recriando coleção '{COLLECTION_NAME}' com {EMBED_DIM} dimensões...")

    qdrant.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=EMBED_DIM,
            distance=models.Distance.COSINE,
        ),
    )


def upload_to_qdrant(processed_data):
    print("Conectando ao Qdrant...")
    qdrant = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
    )

    recreate_collection(qdrant)

    print("Inserindo pontos no Qdrant...")

    buffer_points = []

    for index, item in enumerate(tqdm(processed_data, desc="Enviando para Qdrant")):
        doc = item["document"]
        vector = item["vector"]

        point = models.PointStruct(
            id=index,
            vector=vector,
            payload=build_payload(doc),
        )

        buffer_points.append(point)

        if len(buffer_points) >= BATCH_SIZE_QDRANT:
            qdrant.upsert(
                collection_name=COLLECTION_NAME,
                points=buffer_points,
            )
            buffer_points = []

    if buffer_points:
        qdrant.upsert(
            collection_name=COLLECTION_NAME,
            points=buffer_points,
        )

# Função principal para construir o vectorstore: carrega os documentos, gera embeddings e envia para o Qdrant
def build_vectorstore():
    print("Carregando chunks...")
    documents = load_documents()
    print(f"{len(documents)} chunks encontrados.")

    processed_data = load_backup(documents)
    processed_data = generate_embeddings(documents, processed_data)

    upload_to_qdrant(processed_data)

    print("Indexação finalizada com sucesso.")
    print(f"Total de chunks indexados: {len(processed_data)}")


if __name__ == "__main__":
    build_vectorstore()
