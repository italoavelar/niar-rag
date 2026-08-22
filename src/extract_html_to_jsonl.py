import csv
import json
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from extract_to_jsonl import chunk_text, parse_ria_dimensions


MANIFEST_FILE = Path("corpus_manifest.csv")

HTML_INPUT_FILE = Path(
    "data/raw/html/conteudo_coletado_limpo.json"
)

OUTPUT_FILE = Path(
    "data/processed/html_chunks.jsonl"
)

SELECTED_DOCUMENTS_FILE = Path(
    "data/processed/html_documents.json"
)


def normalize_url(url: str) -> str:
    """
    Normaliza URLs para permitir a comparação entre
    o corpus_manifest.csv e o JSON coletado.
    """
    url = (url or "").strip()

    if not url:
        return ""

    parsed = urlsplit(url)

    scheme = parsed.scheme.lower()
    netloc = parsed.netloc.lower()

    path = parsed.path.rstrip("/")

    if not path:
        path = "/"

    return urlunsplit(
        (
            scheme,
            netloc,
            path,
            parsed.query,
            "",
        )
    )


def load_html_manifest() -> dict[str, dict]:
    """
    Carrega somente os documentos HTML ativos
    presentes no corpus_manifest.csv.

    Retorna um dicionário usando a URL normalizada
    como chave.
    """
    if not MANIFEST_FILE.exists():
        raise FileNotFoundError(
            f"Manifesto não encontrado: {MANIFEST_FILE}"
        )

    metadata_by_url = {}

    with open(
        MANIFEST_FILE,
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        required_columns = {
            "filename",
            "title",
            "document_type",
            "year",
            "theme",
            "source_url",
            "source_type",
        }

        available_columns = set(reader.fieldnames or [])

        missing_columns = required_columns - available_columns

        if missing_columns:
            raise ValueError(
                "Colunas obrigatórias ausentes no manifesto: "
                + ", ".join(sorted(missing_columns))
            )

        for row_number, row in enumerate(
            reader,
            start=2,
        ):
            source_type = (
                row.get("source_type") or ""
            ).strip().upper()

            if source_type != "HTML":
                continue

            document_id = (
                row.get("filename") or ""
            ).strip()

            source_url = (
                row.get("source_url") or ""
            ).strip()

            if not document_id:
                print(
                    f"[AVISO] Linha {row_number} ignorada: "
                    "filename vazio."
                )
                continue

            if not source_url:
                print(
                    f"[AVISO] Linha {row_number} ignorada: "
                    "source_url vazia."
                )
                continue

            normalized_url = normalize_url(source_url)

            if normalized_url in metadata_by_url:
                raise ValueError(
                    "URL HTML duplicada no manifesto: "
                    f"{source_url}"
                )

            row["filename"] = document_id
            row["source_url"] = source_url

            row["ria_dimensions"] = parse_ria_dimensions(
                row.get("ria_dimension", "")
            )

            metadata_by_url[normalized_url] = row

    return metadata_by_url


def load_collected_html() -> dict[str, dict]:
    """
    Carrega o JSON limpo produzido pelo scraper.

    Em caso de URLs duplicadas, dá preferência
    ao registro marcado como raiz.
    """
    if not HTML_INPUT_FILE.exists():
        raise FileNotFoundError(
            f"JSON HTML não encontrado: {HTML_INPUT_FILE}"
        )

    with open(
        HTML_INPUT_FILE,
        "r",
        encoding="utf-8",
    ) as file:
        records = json.load(file)

    if not isinstance(records, list):
        raise ValueError(
            "O JSON HTML deve conter uma lista de documentos."
        )

    records_by_url = {}

    for record in records:
        if not isinstance(record, dict):
            continue

        normalized_url = normalize_url(
            record.get("url", "")
        )

        if not normalized_url:
            continue

        previous_record = records_by_url.get(
            normalized_url
        )

        if previous_record is None:
            records_by_url[normalized_url] = record
            continue

        previous_relation = (
            previous_record.get("relacao") or ""
        ).lower()

        current_relation = (
            record.get("relacao") or ""
        ).lower()

        if (
            previous_relation != "raiz"
            and current_relation == "raiz"
        ):
            records_by_url[normalized_url] = record

    return records_by_url


def build_chunk_record(
    metadata: dict,
    collected_record: dict,
    chunk: str,
    chunk_index: int,
) -> dict:
    """
    Cria um chunk HTML no mesmo formato geral
    utilizado pelos chunks PDF.
    """
    document_id = metadata["filename"]
    source_url = metadata["source_url"]

    return {
        "id": f"{document_id}_c{chunk_index}",
        "text": chunk,
        "metadata": {
            "document_id": document_id,
            "source": source_url,
            "source_type": "HTML",
            "title": (
                metadata.get("title")
                or collected_record.get("titulo")
                or document_id
            ),
            "page": None,
            "chunk": chunk_index,
            "document_type": metadata.get(
                "document_type",
                "",
            ),
            "author": metadata.get(
                "author",
                "",
            ),
            "year": metadata.get(
                "year",
                "",
            ),
            "theme": metadata.get(
                "theme",
                "",
            ),
            "ria_dimensions": metadata.get(
                "ria_dimensions",
                [],
            ),
            "source_url": source_url,
            "origin": collected_record.get(
                "origem",
                "",
            ),
            "relation": collected_record.get(
                "relacao",
                "",
            ),
            "parent_url": collected_record.get(
                "pai",
                "",
            ),
            "depth": collected_record.get(
                "profundidade",
                0,
            ),
        },
    }


def process_html() -> None:
    """
    Seleciona os HTMLs que pertencem ao manifesto,
    cria os chunks e salva o resultado em JSONL.
    """
    manifest = load_html_manifest()
    collected_records = load_collected_html()

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    selected_documents = []
    missing_documents = []
    documents_without_text = []
    matched_child_documents = []

    total_chunks = 0

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
    ) as output_file:
        for normalized_url, metadata in manifest.items():
            collected_record = collected_records.get(
                normalized_url
            )

            if collected_record is None:
                missing_documents.append(
                    {
                        "document_id": metadata["filename"],
                        "source_url": metadata["source_url"],
                    }
                )
                continue

            text = (
                collected_record.get("texto") or ""
            ).strip()

            if not text:
                documents_without_text.append(
                    {
                        "document_id": metadata["filename"],
                        "source_url": metadata["source_url"],
                    }
                )
                continue

            relation = (
                collected_record.get("relacao") or ""
            ).lower()

            if relation == "filho":
                matched_child_documents.append(
                    metadata["filename"]
                )

            chunks = chunk_text(text)

            if not chunks:
                documents_without_text.append(
                    {
                        "document_id": metadata["filename"],
                        "source_url": metadata["source_url"],
                    }
                )
                continue

            selected_documents.append(
                {
                    "document_id": metadata["filename"],
                    "title": (
                        metadata.get("title")
                        or collected_record.get("titulo")
                        or metadata["filename"]
                    ),
                    "source_url": metadata["source_url"],
                    "source_type": "HTML",
                    "origin": collected_record.get(
                        "origem",
                        "",
                    ),
                    "relation": collected_record.get(
                        "relacao",
                        "",
                    ),
                    "text_length": len(text),
                    "chunks": len(chunks),
                }
            )

            for chunk_index, chunk in enumerate(chunks):
                record = build_chunk_record(
                    metadata=metadata,
                    collected_record=collected_record,
                    chunk=chunk,
                    chunk_index=chunk_index,
                )

                output_file.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                total_chunks += 1

    with open(
        SELECTED_DOCUMENTS_FILE,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            selected_documents,
            file,
            ensure_ascii=False,
            indent=2,
        )

    ignored_records = (
        len(collected_records)
        - len(selected_documents)
    )

    print("\n" + "=" * 72)
    print("PROCESSAMENTO HTML FINALIZADO")
    print("=" * 72)

    print(
        f"HTMLs ativos no manifesto: {len(manifest)}"
    )

    print(
        "HTMLs encontrados e processados: "
        f"{len(selected_documents)}"
    )

    print(
        "Registros extras do crawler ignorados: "
        f"{ignored_records}"
    )

    print(
        f"Chunks HTML gerados: {total_chunks}"
    )

    print(
        f"Chunks salvos em: {OUTPUT_FILE}"
    )

    print(
        "Documentos selecionados salvos em: "
        f"{SELECTED_DOCUMENTS_FILE}"
    )

    if missing_documents:
        print("\nHTMLs não encontrados no JSON:")

        for document in missing_documents:
            print(
                f"- {document['document_id']}: "
                f"{document['source_url']}"
            )

    if documents_without_text:
        print("\nHTMLs encontrados, mas sem texto útil:")

        for document in documents_without_text:
            print(
                f"- {document['document_id']}: "
                f"{document['source_url']}"
            )

    if matched_child_documents:
        print(
            "\n[AVISO] Documentos do manifesto encontrados "
            "como página-filha:"
        )

        for document_id in matched_child_documents:
            print(f"- {document_id}")


if __name__ == "__main__":
    process_html()