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
    embedding_text_hash,
)

load_dotenv()

# Colecao de destino da indexacao Gemini. Vem do .env para nao colidir com a
# colecao que o agente serve (agent/utils/tools.py). Precisa casar com
# eval/config.yaml -> embedders.gemini.qdrant_collection.
COLLECTION_NAME = os.getenv("QDRANT_GEMINI_COLLECTION", "LEME_gemini")

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

# Pausa entre lotes. O padrao 15s foi calibrado para o limite por minuto do
# free tier; em plano pago da para reduzir. Um 429 nao perde trabalho: o
# backup e gravado a cada lote e a execucao retoma de onde parou.
SLEEP_BETWEEN_BATCHES = float(os.getenv("GEMINI_SLEEP_BETWEEN_BATCHES", "15"))

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
    # O backup é indexado pelo TEXTO DE EMBEDDING, não por posição nem por
    # fingerprint.
    #
    # Antes, `load_backup` assumia que o backup era o prefixo do corpus atual, na
    # mesma ordem, e conferia `documents[index]` contra `processed_data[index]`.
    # Isso quebra sempre que o corpus é reprocessado: remover um documento ou
    # mudar o recorte desloca tudo a partir do primeiro trecho alterado, e o
    # script aborta com "fingerprint não confere" mesmo tendo em mãos milhares de
    # vetores reaproveitáveis.
    #
    # Por que NÃO dá para casar por `corpus_embedding_fingerprint`: ele hasheia o
    # **id** junto com o texto (`src/embedding_text.py:61-75`). Como os ids
    # deixaram de ser posicionais e passaram a vir do conteúdo, todo fingerprint
    # mudou — inclusive o dos trechos cujo texto é idêntico. Medido: casando por
    # fingerprint, 0 de 4.897 seriam reaproveitados.
    #
    # A chave certa é o texto que de fato produziu o vetor: `build_embedding_text`,
    # com o prefixo `[título · emissor · seção]` incluído. Se o prefixo mudou, o
    # vetor não serve — e é justo recalcular.
    por_texto = {}

    if BACKUP_FILE.exists():
        with open(BACKUP_FILE, "r", encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue
                item = json.loads(line)
                if item.get("embedding_text_profile") != EMBEDDING_TEXT_PROFILE:
                    # Falha alta, de propósito: um backup de outro perfil não é
                    # reaproveitável, e seguir em silêncio significaria recalcular
                    # o acervo inteiro com a API sem ninguém perceber a conta.
                    raise ValueError(
                        "Backup incompatível: perfil de embedding ausente ou diferente "
                        f"de {EMBEDDING_TEXT_PROFILE!r}."
                    )
                doc_backup = item.get("document") or {}
                chave = build_embedding_text(doc_backup)
                if chave:
                    por_texto[chave] = item

        aproveitaveis = sum(
            1 for doc in documents if build_embedding_text(doc) in por_texto
        )
        print(
            f"Backup {EMBEDDING_TEXT_PROFILE}: {len(por_texto)} vetores em disco, "
            f"{aproveitaveis} reaproveitáveis para os {len(documents)} trechos atuais "
            f"({len(documents) - aproveitaveis} a calcular)."
        )
    else:
        print(
            f"Nenhum backup {EMBEDDING_TEXT_PROFILE} encontrado. "
            "Iniciando do zero."
        )

    return por_texto

# Gera embeddings para os documentos usando a API Gemini e salva em backup
def generate_embeddings(documents, por_texto):
    """Devolve um registro por documento, NA ORDEM DO CORPUS.

    Reaproveita do backup todo trecho cujo texto contextual não mudou e calcula
    só o resto. A saída segue `documents`, então o upload para o Qdrant fica
    alinhado com o corpus mesmo que o backup esteja em outra ordem.
    """
    processed_data = []
    documents_to_process = []
    for doc in documents:
        achado = por_texto.get(build_embedding_text(doc))
        if achado is not None:
            # O vetor é reaproveitado, mas o documento vem do corpus atual e o
            # fingerprint é recalculado: ele carrega o id, que mudou.
            processed_data.append({
                **achado,
                "document": doc,
                "embedding_text_fingerprint": corpus_embedding_fingerprint([doc]),
            })
        else:
            processed_data.append(None)
            documents_to_process.append(doc)

    if not documents_to_process:
        print("Todos os embeddings já estavam no backup.")
        return processed_data

    print(f"A calcular: {len(documents_to_process)} de {len(documents)} trechos.")

    print("Inicializando cliente Gemini...")
    client = genai.Client(api_key=GOOGLE_GENAI_API_KEY)

    BACKUP_FILE.parent.mkdir(parents=True, exist_ok=True)
    novos = {}

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
                        # hash SÓ do texto: é por ele que se reaproveita vetor
                        # depois de uma troca de id. Ver embedding_text_hash.
                        "embedding_text_hash": embedding_text_hash(doc),
                    }

                    backup.write(json.dumps(record, ensure_ascii=False) + "\n")
                    novos[doc["id"]] = record

                time.sleep(SLEEP_BETWEEN_BATCHES)

            except Exception as error:
                print("Erro ao gerar embeddings. Progresso salvo no backup.")
                print(f"Detalhe: {error}")
                raise

    # encaixa os recém-calculados nos buracos, preservando a ordem do corpus
    for i, doc in enumerate(documents):
        if processed_data[i] is None:
            processed_data[i] = novos[doc["id"]]

    faltando = [i for i, r in enumerate(processed_data) if r is None]
    if faltando:
        raise RuntimeError(
            f"{len(faltando)} trechos ficaram sem vetor — não enviar ao Qdrant."
        )
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
            "COLLECTION_NAME esta vazio. Defina QDRANT_GEMINI_COLLECTION no .env."
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

    por_texto = load_backup(documents)
    processed_data = generate_embeddings(documents, por_texto)

    upload_to_qdrant(processed_data)

    print("Indexação finalizada com sucesso.")
    print(f"Total de chunks indexados: {len(processed_data)}")


if __name__ == "__main__":
    build_vectorstore()
