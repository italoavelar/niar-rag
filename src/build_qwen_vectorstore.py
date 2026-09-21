"""Indexação isolada com Qwen3-Embedding e contexto documental.

Este entrypoint não reutiliza nem altera o pipeline Gemini, seus backups ou a
coleção Qdrant legada. Ele só executa operações externas quando chamado como
script; importar este módulo é seguro para testes.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable
from urllib.parse import urlsplit, urlunsplit

from embedding_text import (
    EMBEDDING_TEXT_PROFILE,
    build_embedding_text,
    corpus_embedding_fingerprint,
)


QWEN_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
QWEN_EMBEDDING_DIM = 1024

JSONL_FILE = Path("data/processed/documents.jsonl")
DEFAULT_BACKUP_FILE = Path(
    "data/processed/embeddings_qwen3_0_6b_context_v1_backup.jsonl"
)
PROTECTED_BACKUP_FILES = frozenset(
    {
        Path("data/processed/embeddings_backup.jsonl"),
        Path("data/processed/embeddings_context_v1_backup.jsonl"),
    }
)

LEGACY_COLLECTION_NAME = "niar_rag_documents"
COLLECTION_NAME_PREFIX = "niar_rag_documents_qwen3_0_6b_context_v1"

BATCH_SIZE_EMBEDDINGS = 2
BATCH_SIZE_QDRANT = 64


def _resolved(path: Path) -> Path:
    return path.expanduser().resolve()


def resolve_backup_file(configured_path: str | None = None) -> Path:
    """Resolve um backup Qwen e bloqueia explicitamente os dois backups Gemini."""
    value = configured_path or os.getenv("QWEN_EMBEDDING_BACKUP_FILE")
    backup_file = Path(value) if value else DEFAULT_BACKUP_FILE

    protected = {_resolved(path) for path in PROTECTED_BACKUP_FILES}
    if _resolved(backup_file) in protected:
        raise ValueError(
            "QWEN_EMBEDDING_BACKUP_FILE não pode apontar para um backup legado "
            "Gemini. Use um arquivo exclusivo do Qwen3."
        )

    return backup_file


def collection_name_for_fingerprint(corpus_fingerprint: str) -> str:
    """Nome imutável da coleção associado ao corpus contextual."""
    if not corpus_fingerprint:
        raise ValueError("Fingerprint global do corpus não pode ser vazio.")
    return f"{COLLECTION_NAME_PREFIX}_{corpus_fingerprint[:16]}"


def validate_collection_name(
    configured_name: str | None = None,
    *,
    corpus_fingerprint: str | None = None,
) -> str:
    """Impede que o pipeline Qwen escreva na coleção Gemini legada."""
    collection = (
        configured_name
        or os.getenv("QDRANT_QWEN_COLLECTION")
        or (
            collection_name_for_fingerprint(corpus_fingerprint)
            if corpus_fingerprint
            else COLLECTION_NAME_PREFIX
        )
    ).strip()

    if not collection:
        raise ValueError("QDRANT_QWEN_COLLECTION não pode ser vazio.")
    if collection == LEGACY_COLLECTION_NAME:
        raise ValueError(
            f"A coleção legada {LEGACY_COLLECTION_NAME!r} não pode ser usada pelo Qwen."
        )

    return collection


def normalize_qdrant_url(url: str) -> str:
    """Remove diferenças irrelevantes na URL antes de comparar ambientes."""
    candidate = url.strip()
    parts = urlsplit(candidate)
    if parts.scheme and parts.netloc:
        return urlunsplit(
            (
                parts.scheme.lower(),
                parts.netloc.lower(),
                parts.path.rstrip("/"),
                parts.query,
                parts.fragment,
            )
        )
    return candidate.rstrip("/")


def validate_qdrant_environment_urls(
    qwen_url: str,
    legacy_url: str | None = None,
) -> None:
    """Garante que a indexação Qwen não aponte para o Qdrant legado do WFA."""
    legacy_url = legacy_url if legacy_url is not None else os.getenv("QDRANT_URL")
    if legacy_url and normalize_qdrant_url(qwen_url) == normalize_qdrant_url(
        legacy_url
    ):
        raise ValueError(
            "QDRANT_QWEN_URL deve apontar para um ambiente separado do "
            "Qdrant legado do WFA (QDRANT_URL)."
        )


def load_documents() -> list[dict]:
    """Carrega os chunks sem alterar o corpus de origem."""
    documents = []
    with JSONL_FILE.open("r", encoding="utf-8") as source:
        for line in source:
            if line.strip():
                documents.append(json.loads(line))
    return documents


def document_fingerprint(document: dict) -> str:
    """Fingerprint por chunk, usado para retomar um backup com segurança."""
    return corpus_embedding_fingerprint([document])


def build_document_inputs(documents: Iterable[dict]) -> list[str]:
    """Monta somente as entradas contextuais enviadas ao modelo de documentos."""
    return [build_embedding_text(document) for document in documents]


def load_qwen_model():
    """Carrega o modelo somente durante uma geração explicitamente solicitada."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(QWEN_EMBEDDING_MODEL)


def embed_documents(model, documents: list[dict]):
    """Gera vetores documentais normalizados a partir do helper compartilhado."""
    return model.encode(
        build_document_inputs(documents),
        normalize_embeddings=True,
    )


def embed_query(model, query: str):
    """Gera vetor de query sem contexto documental e com o prompt oficial Qwen."""
    return model.encode(
        [query],
        prompt_name="query",
        normalize_embeddings=True,
    )[0]


def _vector_as_list(vector) -> list[float]:
    values = vector.tolist() if hasattr(vector, "tolist") else list(vector)
    if len(values) != QWEN_EMBEDDING_DIM:
        raise ValueError(
            "Dimensão inesperada do embedding Qwen: "
            f"{len(values)} (esperado {QWEN_EMBEDDING_DIM})."
        )
    return values


def _validate_backup_item(item: dict, expected_document: dict) -> None:
    """Valida modelo, perfil, ordem, conteúdo e vetor antes de uma retomada."""
    if item.get("embedding_model") != QWEN_EMBEDDING_MODEL:
        raise ValueError("Backup Qwen incompatível: modelo de embedding diferente.")
    if item.get("embedding_dimension") != QWEN_EMBEDDING_DIM:
        raise ValueError("Backup Qwen incompatível: dimensão de embedding diferente.")
    if item.get("embedding_text_profile") != EMBEDDING_TEXT_PROFILE:
        raise ValueError("Backup Qwen incompatível: perfil textual diferente.")
    if item.get("document") != expected_document:
        raise ValueError("Backup Qwen incompatível: ordem ou documento não confere.")
    if item.get("embedding_text_fingerprint") != document_fingerprint(
        expected_document
    ):
        raise ValueError("Backup Qwen incompatível: fingerprint do chunk não confere.")
    _vector_as_list(item.get("vector", []))


def _validate_backup_global_fingerprint(
    item: dict,
    corpus_fingerprint: str,
) -> None:
    if item.get("corpus_embedding_fingerprint") != corpus_fingerprint:
        raise ValueError(
            "Backup Qwen incompatível: fingerprint global do corpus não confere."
        )


def load_backup(
    documents: list[dict],
    backup_file: Path,
    corpus_fingerprint: str,
    *,
    require_complete: bool = False,
) -> list[dict]:
    """Lê somente um backup Qwen compatível; nunca tenta o backup Gemini."""
    backup_file = resolve_backup_file(str(backup_file))
    if not backup_file.exists():
        return []

    records = []
    with backup_file.open("r", encoding="utf-8") as source:
        for line in source:
            if line.strip():
                records.append(json.loads(line))

    if len(records) > len(documents):
        raise ValueError("Backup Qwen possui mais registros que o corpus atual.")

    for index, item in enumerate(records):
        _validate_backup_item(item, documents[index])
        _validate_backup_global_fingerprint(item, corpus_fingerprint)

    if require_complete and len(records) != len(documents):
        raise ValueError(
            "Backup Qwen incompleto para upload: "
            f"{len(records)} de {len(documents)} registros."
        )

    return records


def generate_embeddings(
    documents: list[dict],
    processed_data: list[dict],
    backup_file: Path,
    corpus_fingerprint: str,
    model=None,
) -> list[dict]:
    """Gera somente os chunks pendentes e os salva no backup Qwen exclusivo."""
    backup_file = resolve_backup_file(str(backup_file))
    documents_to_process = documents[len(processed_data):]
    if not documents_to_process:
        return processed_data

    model = model or load_qwen_model()
    backup_file.parent.mkdir(parents=True, exist_ok=True)

    from tqdm import tqdm

    with backup_file.open("a", encoding="utf-8") as backup:
        for start in tqdm(
            range(0, len(documents_to_process), BATCH_SIZE_EMBEDDINGS),
            desc="Gerando embeddings Qwen3",
        ):
            batch = documents_to_process[start:start + BATCH_SIZE_EMBEDDINGS]
            vectors = embed_documents(model, batch)

            if len(vectors) != len(batch):
                raise ValueError("O modelo Qwen retornou quantidade inesperada de vetores.")

            for document, vector in zip(batch, vectors):
                record = {
                    "embedding_model": QWEN_EMBEDDING_MODEL,
                    "embedding_dimension": QWEN_EMBEDDING_DIM,
                    "embedding_text_profile": EMBEDDING_TEXT_PROFILE,
                    "embedding_text_fingerprint": document_fingerprint(document),
                    "corpus_embedding_fingerprint": corpus_fingerprint,
                    "document": document,
                    "vector": _vector_as_list(vector),
                }
                backup.write(json.dumps(record, ensure_ascii=False) + "\n")
                processed_data.append(record)

    return processed_data


def build_payload(document: dict, corpus_fingerprint: str) -> dict:
    """Mantém texto original no payload e registra o perfil do vetor Qwen."""
    metadata = document.get("metadata", {})
    return {
        "id_original": document.get("id", ""),
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
        "embedding_model": QWEN_EMBEDDING_MODEL,
        "embedding_dimension": QWEN_EMBEDDING_DIM,
        "embedding_text_profile": EMBEDDING_TEXT_PROFILE,
        "corpus_embedding_fingerprint": corpus_fingerprint,
    }


def _collection_names(client) -> set[str]:
    return {item.name for item in client.get_collections().collections}


def upload_to_qdrant(
    processed_data: list[dict],
    *,
    corpus_fingerprint: str,
    recreate: bool = False,
    collection_name: str | None = None,
) -> None:
    """Cria ou popula exclusivamente a coleção Qwen configurada.

    Por padrão falha se a coleção já existir. A recriação é possível apenas por
    uma opção explícita, nunca pela execução normal.
    """
    collection = validate_collection_name(collection_name)
    qdrant_url = os.getenv("QDRANT_QWEN_URL")
    qdrant_api_key = os.getenv("QDRANT_QWEN_API_KEY")
    if not qdrant_url or not qdrant_api_key:
        raise RuntimeError(
            "Defina QDRANT_QWEN_URL e QDRANT_QWEN_API_KEY para indexar Qwen."
        )
    validate_qdrant_environment_urls(qdrant_url)

    from qdrant_client import QdrantClient, models
    from tqdm import tqdm

    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
    exists = collection in _collection_names(client)
    if exists and not recreate:
        raise ValueError(
            f"A coleção Qwen {collection!r} já existe. "
            "Use --recreate explicitamente se desejar recriá-la."
        )

    vectors_config = models.VectorParams(
        size=QWEN_EMBEDDING_DIM,
        distance=models.Distance.COSINE,
    )
    if exists:
        client.recreate_collection(
            collection_name=collection,
            vectors_config=vectors_config,
        )
    else:
        client.create_collection(
            collection_name=collection,
            vectors_config=vectors_config,
        )

    for start in tqdm(
        range(0, len(processed_data), BATCH_SIZE_QDRANT),
        desc="Enviando vetores Qwen3",
    ):
        batch = processed_data[start:start + BATCH_SIZE_QDRANT]
        points = [
            models.PointStruct(
                id=start + offset,
                vector=item["vector"],
                payload=build_payload(
                    item["document"], corpus_fingerprint
                ),
            )
            for offset, item in enumerate(batch)
        ]
        client.upsert(collection_name=collection, points=points, wait=True)


def verify_qdrant_collection(
    documents: list[dict],
    corpus_fingerprint: str,
    collection_name: str,
) -> None:
    """Compara o corpus local à coleção sem criar ou alterar recursos remotos."""
    qdrant_url = os.getenv("QDRANT_QWEN_URL")
    qdrant_api_key = os.getenv("QDRANT_QWEN_API_KEY")
    if not qdrant_url or not qdrant_api_key:
        raise RuntimeError(
            "Defina QDRANT_QWEN_URL e QDRANT_QWEN_API_KEY para verificar Qwen."
        )
    validate_qdrant_environment_urls(qdrant_url)

    from qdrant_client import QdrantClient

    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
    if collection_name not in _collection_names(client):
        raise ValueError(f"Coleção Qwen inexistente: {collection_name!r}.")

    info = client.get_collection(collection_name)
    vectors = info.config.params.vectors
    dimension = vectors.size if hasattr(vectors, "size") else vectors["size"]
    if dimension != QWEN_EMBEDDING_DIM:
        raise ValueError(
            f"Dimensão Qdrant incompatível: {dimension}; "
            f"esperado {QWEN_EMBEDDING_DIM}."
        )

    expected_ids = [str(document["id"]) for document in documents]
    expected_set = set(expected_ids)
    if len(expected_set) != len(expected_ids):
        raise ValueError("Corpus local contém IDs de chunk duplicados.")

    count = client.count(collection_name, exact=True).count
    if count != len(expected_ids):
        raise ValueError(
            f"Quantidade de pontos incompatível: {count}; "
            f"esperado {len(expected_ids)}."
        )

    found_ids: list[str] = []
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name,
            limit=BATCH_SIZE_QDRANT,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for point in points:
            payload = point.payload or {}
            if payload.get("embedding_model") != QWEN_EMBEDDING_MODEL:
                raise ValueError("Modelo Qwen incompatível no payload.")
            if payload.get("embedding_dimension") != QWEN_EMBEDDING_DIM:
                raise ValueError("Dimensão Qwen incompatível no payload.")
            if payload.get("embedding_text_profile") != EMBEDDING_TEXT_PROFILE:
                raise ValueError("Perfil textual incompatível no payload.")
            if payload.get("corpus_embedding_fingerprint") != corpus_fingerprint:
                raise ValueError("Fingerprint global incompatível no payload.")
            original_id = str(payload.get("id_original") or "")
            if not original_id:
                raise ValueError("Ponto Qdrant sem id_original.")
            found_ids.append(original_id)
        if offset is None:
            break

    found_set = set(found_ids)
    if len(found_ids) != len(found_set):
        raise ValueError("Coleção Qdrant contém id_original duplicado.")
    missing = expected_set - found_set
    extra = found_set - expected_set
    if missing or extra:
        raise ValueError(
            "IDs Qdrant não correspondem ao corpus: "
            f"ausentes={len(missing)}, extras={len(extra)}."
        )

    print("Verificação Qwen concluída com sucesso.")
    print(f"Coleção: {collection_name}")
    print(f"Pontos/IDs: {len(found_ids)}")
    print(f"Fingerprint: {corpus_fingerprint}")


def _print_run_summary(
    documents: list[dict],
    corpus_fingerprint: str,
) -> None:
    print(f"Modelo: {QWEN_EMBEDDING_MODEL}")
    print(f"Dimensão: {QWEN_EMBEDDING_DIM}")
    print(f"Perfil: {EMBEDDING_TEXT_PROFILE}")
    print(f"Fingerprint: {corpus_fingerprint}")
    print(f"Chunks: {len(documents)}")


def build_qwen_vectorstore(
    mode: str,
    *,
    recreate: bool = False,
    collection_name: str | None = None,
) -> None:
    """Executa uma fase explícita da indexação Qwen."""
    from dotenv import load_dotenv

    load_dotenv()
    documents = load_documents()
    corpus_fingerprint = corpus_embedding_fingerprint(documents)
    collection = validate_collection_name(
        collection_name or collection_name_for_fingerprint(corpus_fingerprint),
        corpus_fingerprint=corpus_fingerprint,
    )
    backup_file = resolve_backup_file()

    if mode == "generate-only":
        processed_data = load_backup(
            documents, backup_file, corpus_fingerprint
        )
        generate_embeddings(
            documents, processed_data, backup_file, corpus_fingerprint
        )
        load_backup(
            documents, backup_file, corpus_fingerprint, require_complete=True
        )
        _print_run_summary(documents, corpus_fingerprint)
        return

    if mode == "upload-only":
        processed_data = load_backup(
            documents, backup_file, corpus_fingerprint, require_complete=True
        )
        upload_to_qdrant(
            processed_data,
            corpus_fingerprint=corpus_fingerprint,
            recreate=recreate,
            collection_name=collection,
        )
        return

    if mode == "verify":
        verify_qdrant_collection(documents, corpus_fingerprint, collection)
        return

    raise ValueError(f"Modo Qwen desconhecido: {mode!r}.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Indexa o corpus com Qwen3-Embedding.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--generate-only",
        action="store_true",
        help="Gera e valida apenas o backup Qwen; não acessa Qdrant.",
    )
    mode.add_argument(
        "--upload-only",
        action="store_true",
        help="Envia somente um backup Qwen completo e validado.",
    )
    mode.add_argument(
        "--verify",
        action="store_true",
        help="Compara a coleção Qwen ao corpus local sem alterá-la.",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Recria a coleção Qwen se ela já existir (operação destrutiva).",
    )
    parser.add_argument(
        "--collection",
        help="Coleção Qwen de destino; omita para usar o nome versionado.",
    )
    args = parser.parse_args()
    if args.recreate and not args.upload_only:
        parser.error("--recreate só pode ser usado com --upload-only.")
    selected_mode = (
        "generate-only" if args.generate_only else
        "upload-only" if args.upload_only else
        "verify"
    )
    build_qwen_vectorstore(
        selected_mode,
        recreate=args.recreate,
        collection_name=args.collection,
    )


if __name__ == "__main__":
    main()
