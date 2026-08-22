import json
from pathlib import Path


JSONL_FILE = Path("data/processed/documents.jsonl")

REQUIRED_FIELDS = ["id", "text", "metadata"]

REQUIRED_METADATA_FIELDS = [
    "source",
    "title",
    "page",
    "chunk",
    "document_type",
    "year",
    "theme",
    "source_url",
]


def validate_jsonl():
    if not JSONL_FILE.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {JSONL_FILE}")

    total_lines = 0

    with open(JSONL_FILE, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            total_lines += 1

            try:
                data = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Linha {line_number}: JSON inválido. Erro: {error}")

            for field in REQUIRED_FIELDS:
                if field not in data:
                    raise ValueError(f"Linha {line_number}: campo ausente: {field}")

            if not data["text"].strip():
                raise ValueError(f"Linha {line_number}: campo text está vazio")

            metadata = data["metadata"]

            for field in REQUIRED_METADATA_FIELDS:
                if field not in metadata:
                    raise ValueError(
                        f"Linha {line_number}: campo metadata.{field} ausente"
                    )

    print("VALIDAÇÃO COM SUCESSO")
    print(f"Total de linhas validadas: {total_lines}")


if __name__ == "__main__":
    validate_jsonl()