import os

import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types
from qdrant_client import QdrantClient


COLLECTION_NAME = "niar_rag_documents"
DEFAULT_LIMIT = 4

load_dotenv(dotenv_path=".env")

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
GOOGLE_GENAI_API_KEY = os.getenv("GOOGLE_GENAI_API_KEY")


def validate_environment() -> None:
    """
    Confirma se as variáveis necessárias estão disponíveis.
    """
    missing_variables = [
        name
        for name, value in {
            "QDRANT_URL": QDRANT_URL,
            "QDRANT_API_KEY": QDRANT_API_KEY,
            "GOOGLE_GENAI_API_KEY": GOOGLE_GENAI_API_KEY,
        }.items()
        if not value
    ]

    if missing_variables:
        raise EnvironmentError(
            "Variáveis de ambiente ausentes: "
            + ", ".join(missing_variables)
        )


def normalize(vector: list[float]) -> list[float]:
    """
    Normaliza um vetor para norma 1.
    """
    values = np.array(vector, dtype=float)
    norm = np.linalg.norm(values)

    if norm == 0:
        return values.tolist()

    return (values / norm).tolist()


def embed_query(query: str) -> list[float]:
    """
    Gera o embedding da consulta usando o mesmo modelo
    utilizado na indexação dos documentos.
    """
    client = genai.Client(
        api_key=GOOGLE_GENAI_API_KEY
    )

    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=query,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY"
        ),
    )

    return normalize(
        response.embeddings[0].values
    )


def format_source(payload: dict) -> tuple[str, str]:
    """
    Monta a referência da fonte conforme PDF ou HTML.
    """
    source_type = (
        payload.get("source_type") or ""
    ).strip().upper()

    source_url = (
        payload.get("source_url") or ""
    ).strip()

    source = (
        payload.get("fonte") or ""
    ).strip()

    if source_type == "PDF":
        page = payload.get("page")

        location = (
            f"Página {page}"
            if page not in (None, "")
            else "Página não informada"
        )

    elif source_type == "HTML":
        location = "Página web"

    else:
        location = "Localização não informada"

    reference = source_url or source or "[Fonte não disponível]"

    return location, reference


def test_retrieval(
    query: str,
    limit: int = DEFAULT_LIMIT,
) -> None:
    """
    Consulta o Qdrant e exibe os resultados recuperados
    com os metadados de PDF e HTML.
    """
    if not query.strip():
        raise ValueError(
            "A consulta não pode estar vazia."
        )

    qdrant = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
    )

    query_vector = embed_query(query)

    results = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=limit,
    )

    print(f"\nPergunta: {query}")
    print("=" * 80)

    if not results.points:
        print("Nenhum resultado encontrado.")
        return

    for index, point in enumerate(
        results.points,
        start=1,
    ):
        payload = point.payload or {}

        source_type = (
            payload.get("source_type") or ""
        ).strip().upper()

        location, source_reference = (
            format_source(payload)
        )

        print(f"\nResultado {index}")
        print(f"Score: {point.score}")
        print(
            f"Título: "
            f"{payload.get('title', '[Título não disponível]')}"
        )
        print(
            f"Documento: "
            f"{payload.get('document_id', '[ID não disponível]')}"
        )
        print(
            f"Formato: "
            f"{source_type or '[Formato não disponível]'}"
        )
        print(
            f"Fonte original: "
            f"{payload.get('fonte', '[Fonte não disponível]')}"
        )
        print(f"URL: {payload.get('source_url', '')}")
        print(f"Localização: {location}")
        print(
            f"Tipo: "
            f"{payload.get('document_type', '')}"
        )
        print(f"Autor: {payload.get('author', '')}")
        print(f"Ano: {payload.get('year', '')}")
        print(f"Tema: {payload.get('theme', '')}")
        print(
            "Dimensões de IA responsável: "
            f"{payload.get('ria_dimensions', [])}"
        )
        print(f"Referência final: {source_reference}")

        print("\nTrecho:")
        print(
            str(payload.get("texto", ""))[:1200]
        )
        print("-" * 80)


def run_default_tests() -> None:
    """
    Executa perguntas para validar recuperação
    de documentos PDF e HTML.
    """
    queries = [
        "O que é telemedicina segundo a resolução do CFM?",
        "Quais são os princípios de IA responsável?",
        "O PL 2338 fala sobre sistemas de alto risco?",
        "O que são dados pessoais sensíveis segundo a LGPD?",
        "O que é a Rede Nacional de Dados em Saúde?",
        "O que é uma avaliação de impacto à proteção de dados DPIA?",
        "Quais são os princípios de inteligência artificial da OCDE?",
    ]

    for query in queries:
        test_retrieval(query)


if __name__ == "__main__":
    validate_environment()
    run_default_tests()