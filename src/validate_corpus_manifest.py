import csv
import json
from collections import Counter
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit


PDF_DIR = Path("docs/raw")
MANIFEST_FILE = Path("corpus_manifest.csv")
HTML_FILE = Path(
    "data/raw/html/conteudo_coletado_limpo.json"
)


def normalize_url(url: str) -> str:
    """
    Normaliza URLs para comparar o manifesto
    com o resultado do scraper.
    """
    url = (url or "").strip()

    if not url:
        return ""

    parsed = urlsplit(url)

    path = parsed.path.rstrip("/") or "/"

    return urlunsplit(
        (
            parsed.scheme.lower(),
            parsed.netloc.lower(),
            path,
            parsed.query,
            "",
        )
    )


def load_manifest() -> tuple[list[dict], list[dict]]:
    """
    Carrega o manifesto e separa documentos
    PDF e HTML.
    """
    if not MANIFEST_FILE.exists():
        raise FileNotFoundError(
            f"Manifesto não encontrado: {MANIFEST_FILE}"
        )

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
            "source_url",
            "source_type",
        }

        available_columns = set(reader.fieldnames or [])

        missing_columns = (
            required_columns - available_columns
        )

        if missing_columns:
            raise ValueError(
                "Colunas obrigatórias ausentes: "
                + ", ".join(sorted(missing_columns))
            )

        pdf_entries = []
        html_entries = []
        unsupported_entries = []

        for row_number, row in enumerate(
            reader,
            start=2,
        ):
            filename = (
                row.get("filename") or ""
            ).strip()

            source_type = (
                row.get("source_type") or ""
            ).strip().upper()

            source_url = (
                row.get("source_url") or ""
            ).strip()

            row["filename"] = filename
            row["source_type"] = source_type
            row["source_url"] = source_url
            row["row_number"] = row_number

            if source_type == "PDF":
                pdf_entries.append(row)

            elif source_type == "HTML":
                html_entries.append(row)

            else:
                unsupported_entries.append(row)

        if unsupported_entries:
            print(
                "\nTipos de fonte inválidos no manifesto:"
            )

            for row in unsupported_entries:
                print(
                    f"- Linha {row['row_number']}: "
                    f"{row['filename']} "
                    f"({row['source_type']})"
                )

            raise ValueError(
                "O manifesto possui source_type inválido."
            )

    return pdf_entries, html_entries


def load_collected_html() -> dict[str, dict]:
    """
    Carrega os HTMLs coletados e usa a URL
    normalizada como chave.
    """
    if not HTML_FILE.exists():
        raise FileNotFoundError(
            f"JSON HTML não encontrado: {HTML_FILE}"
        )

    with open(
        HTML_FILE,
        "r",
        encoding="utf-8",
    ) as file:
        records = json.load(file)

    if not isinstance(records, list):
        raise ValueError(
            "O JSON HTML deve conter uma lista."
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

        previous = records_by_url.get(
            normalized_url
        )

        if previous is None:
            records_by_url[normalized_url] = record
            continue

        previous_relation = (
            previous.get("relacao") or ""
        ).strip().lower()

        current_relation = (
            record.get("relacao") or ""
        ).strip().lower()

        if (
            previous_relation != "raiz"
            and current_relation == "raiz"
        ):
            records_by_url[normalized_url] = record

    return records_by_url


def validate_duplicates(
    pdf_entries: list[dict],
    html_entries: list[dict],
) -> tuple[list[str], list[str]]:
    """
    Identifica IDs e URLs duplicados no manifesto.
    """
    all_filenames = [
        row["filename"]
        for row in pdf_entries + html_entries
        if row["filename"]
    ]

    duplicate_filenames = sorted(
        filename
        for filename, count in Counter(
            all_filenames
        ).items()
        if count > 1
    )

    html_urls = [
        normalize_url(row["source_url"])
        for row in html_entries
        if row["source_url"]
    ]

    duplicate_html_urls = sorted(
        url
        for url, count in Counter(
            html_urls
        ).items()
        if count > 1
    )

    return (
        duplicate_filenames,
        duplicate_html_urls,
    )


def validate_corpus() -> None:
    """
    Valida separadamente os documentos PDF
    e HTML do corpus.
    """
    if not PDF_DIR.exists():
        raise FileNotFoundError(
            f"Pasta não encontrada: {PDF_DIR}"
        )

    pdf_entries, html_entries = load_manifest()
    collected_html = load_collected_html()

    duplicate_filenames, duplicate_html_urls = (
        validate_duplicates(
            pdf_entries=pdf_entries,
            html_entries=html_entries,
        )
    )

    pdf_files = sorted(
        path.name
        for path in PDF_DIR.glob("*.pdf")
        if path.is_file()
    )

    pdf_manifest_filenames = sorted(
        row["filename"]
        for row in pdf_entries
        if row["filename"]
    )

    pdf_files_set = set(pdf_files)
    pdf_manifest_set = set(
        pdf_manifest_filenames
    )

    pdfs_without_metadata = sorted(
        pdf_files_set - pdf_manifest_set
    )

    metadata_without_pdf = sorted(
        pdf_manifest_set - pdf_files_set
    )

    html_manifest_by_url = {
        normalize_url(row["source_url"]): row
        for row in html_entries
        if row["source_url"]
    }

    html_manifest_urls = set(
        html_manifest_by_url
    )

    collected_html_urls = set(
        collected_html
    )

    html_not_collected = sorted(
        html_manifest_urls - collected_html_urls
    )

    html_without_text = []

    for url in sorted(
        html_manifest_urls & collected_html_urls
    ):
        record = collected_html[url]

        text = (
            record.get("texto") or ""
        ).strip()

        if not text:
            html_without_text.append(url)

    extra_crawler_urls = sorted(
        collected_html_urls - html_manifest_urls
    )

    print("=" * 72)
    print("VALIDAÇÃO DO CORPUS PDF + HTML")
    print("=" * 72)

    print(
        f"PDFs no manifesto: "
        f"{len(pdf_manifest_filenames)}"
    )

    print(
        f"PDFs em docs/raw: "
        f"{len(pdf_files)}"
    )

    print(
        f"HTMLs no manifesto: "
        f"{len(html_entries)}"
    )

    print(
        f"URLs no JSON do scraper: "
        f"{len(collected_html_urls)}"
    )

    print(
        f"PDFs sem metadados: "
        f"{len(pdfs_without_metadata)}"
    )

    print(
        f"Metadados PDF sem arquivo: "
        f"{len(metadata_without_pdf)}"
    )

    print(
        f"HTMLs não encontrados no JSON: "
        f"{len(html_not_collected)}"
    )

    print(
        f"HTMLs sem texto útil: "
        f"{len(html_without_text)}"
    )

    print(
        f"IDs duplicados no manifesto: "
        f"{len(duplicate_filenames)}"
    )

    print(
        f"URLs HTML duplicadas no manifesto: "
        f"{len(duplicate_html_urls)}"
    )

    print(
        "URLs extras coletadas pelo scraper "
        f"e ignoradas: {len(extra_crawler_urls)}"
    )

    if pdfs_without_metadata:
        print("\nPDFs sem metadados:")

        for filename in pdfs_without_metadata:
            print(f"- {filename}")

    if metadata_without_pdf:
        print(
            "\nRegistros PDF sem arquivo correspondente:"
        )

        for filename in metadata_without_pdf:
            print(f"- {filename}")

    if html_not_collected:
        print("\nHTMLs não encontrados no JSON:")

        for url in html_not_collected:
            row = html_manifest_by_url[url]

            print(
                f"- {row['filename']}: "
                f"{row['source_url']}"
            )

    if html_without_text:
        print("\nHTMLs sem texto útil:")

        for url in html_without_text:
            row = html_manifest_by_url[url]

            print(
                f"- {row['filename']}: "
                f"{row['source_url']}"
            )

    if duplicate_filenames:
        print("\nIDs duplicados no manifesto:")

        for filename in duplicate_filenames:
            print(f"- {filename}")

    if duplicate_html_urls:
        print("\nURLs HTML duplicadas:")

        for url in duplicate_html_urls:
            print(f"- {url}")

    has_errors = any(
        [
            pdfs_without_metadata,
            metadata_without_pdf,
            html_not_collected,
            html_without_text,
            duplicate_filenames,
            duplicate_html_urls,
        ]
    )

    if has_errors:
        raise ValueError(
            "Foram encontrados problemas no corpus."
        )

    print("\nCORPUS VALIDADO COM SUCESSO")


if __name__ == "__main__":
    validate_corpus()