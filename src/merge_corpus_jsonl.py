import json
from collections import Counter
from pathlib import Path


INPUT_FILES = [
    Path("data/processed/pdf_chunks.jsonl"),
    Path("data/processed/html_chunks.jsonl"),
]

OUTPUT_FILE = Path(
    "data/processed/documents.jsonl"
)


def validate_record(
    record: dict,
    input_file: Path,
    line_number: int,
) -> None:
    """
    Valida a estrutura mínima de cada chunk.
    """
    required_fields = {
        "id",
        "text",
        "metadata",
    }

    missing_fields = (
        required_fields - set(record)
    )

    if missing_fields:
        raise ValueError(
            f"Campos ausentes em "
            f"{input_file}:{line_number}: "
            + ", ".join(sorted(missing_fields))
        )

    if not isinstance(record["metadata"], dict):
        raise ValueError(
            f"metadata inválido em "
            f"{input_file}:{line_number}"
        )

    if not str(record["id"]).strip():
        raise ValueError(
            f"ID vazio em "
            f"{input_file}:{line_number}"
        )

    if not str(record["text"]).strip():
        raise ValueError(
            f"Texto vazio em "
            f"{input_file}:{line_number}"
        )

    source_type = (
        record["metadata"].get("source_type")
        or ""
    ).strip().upper()

    if source_type not in {"PDF", "HTML"}:
        raise ValueError(
            f"source_type inválido em "
            f"{input_file}:{line_number}: "
            f"{source_type}"
        )

    if not record["metadata"].get(
        "document_id"
    ):
        raise ValueError(
            f"document_id ausente em "
            f"{input_file}:{line_number}"
        )


def merge_jsonl() -> None:
    """
    Une os chunks PDF e HTML em um único
    documents.jsonl.
    """
    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    seen_ids = set()
    document_ids = set()

    chunks_by_type = Counter()
    documents_by_type = {
        "PDF": set(),
        "HTML": set(),
    }

    chunks_by_file = Counter()

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
    ) as output_file:
        for input_file in INPUT_FILES:
            if not input_file.exists():
                raise FileNotFoundError(
                    f"Arquivo não encontrado: "
                    f"{input_file}"
                )

            with open(
                input_file,
                "r",
                encoding="utf-8",
            ) as source_file:
                for line_number, line in enumerate(
                    source_file,
                    start=1,
                ):
                    if not line.strip():
                        continue

                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as error:
                        raise ValueError(
                            f"JSON inválido em "
                            f"{input_file}:"
                            f"{line_number}: {error}"
                        ) from error

                    validate_record(
                        record=record,
                        input_file=input_file,
                        line_number=line_number,
                    )

                    record_id = record["id"]

                    if record_id in seen_ids:
                        raise ValueError(
                            f"ID de chunk duplicado: "
                            f"{record_id}"
                        )

                    seen_ids.add(record_id)

                    metadata = record["metadata"]

                    source_type = (
                        metadata.get("source_type")
                        or ""
                    ).strip().upper()

                    document_id = metadata[
                        "document_id"
                    ]

                    document_ids.add(document_id)

                    documents_by_type[
                        source_type
                    ].add(document_id)

                    chunks_by_type[source_type] += 1
                    chunks_by_file[
                        input_file.name
                    ] += 1

                    output_file.write(
                        json.dumps(
                            record,
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

    print("\n" + "=" * 72)
    print("CORPUS PDF + HTML UNIFICADO")
    print("=" * 72)

    for filename, quantity in (
        chunks_by_file.items()
    ):
        print(
            f"{filename}: {quantity} chunks"
        )

    print(
        f"\nChunks PDF: "
        f"{chunks_by_type['PDF']}"
    )

    print(
        f"Chunks HTML: "
        f"{chunks_by_type['HTML']}"
    )

    print(
        f"Total de chunks: "
        f"{len(seen_ids)}"
    )

    print(
        f"\nDocumentos PDF: "
        f"{len(documents_by_type['PDF'])}"
    )

    print(
        f"Documentos HTML: "
        f"{len(documents_by_type['HTML'])}"
    )

    print(
        f"Total de documentos: "
        f"{len(document_ids)}"
    )

    print(
        f"\nArquivo final salvo em: "
        f"{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    merge_jsonl()