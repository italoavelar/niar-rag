from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path

import fitz
from tqdm import tqdm

from chunking import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    MIN_CHUNK_SIZE,
    choose_split_point,
    chunk_text,
    clean_text,
)


INPUT_DIR = Path("docs/raw")
MANIFEST_FILE = Path("corpus_manifest.csv")
OUTPUT_FILE = Path("data/processed/pdf_chunks.jsonl")

HEADER_FOOTER_SCAN_LINES = 4
REPEATED_LINE_MIN_PAGES = 3
REPEATED_LINE_MIN_RATIO = 0.25


def parse_ria_dimensions(value: str) -> list[str]:
    """
    Converte o campo ria_dimension do CSV em uma lista.
    """
    if not value:
        return []

    return [
        dimension.strip()
        for dimension in value.split(";")
        if dimension.strip()
    ]


def load_manifest() -> dict[str, dict]:
    """
    Carrega os metadados dos documentos PDF presentes no corpus_manifest.csv.

    O dicionário retornado utiliza o nome do arquivo PDF como chave.
    Registros HTML são ignorados por este extrator.
    """
    if not MANIFEST_FILE.exists():
        raise FileNotFoundError(
            f"Manifesto não encontrado: {MANIFEST_FILE}"
        )

    metadata_by_file = {}

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

        for row_number, row in enumerate(reader, start=2):
            source_type = (
                row.get("source_type") or ""
            ).strip().upper()

            if source_type != "PDF":
                continue

            filename = (
                row.get("filename") or ""
            ).strip()

            if not filename:
                print(
                    f"[AVISO] Linha {row_number} ignorada: "
                    "filename vazio."
                )
                continue

            if filename in metadata_by_file:
                raise ValueError(
                    f"Filename duplicado no manifesto: {filename}"
                )

            row["filename"] = filename
            row["source_type"] = source_type
            row["ria_dimensions"] = parse_ria_dimensions(
                row.get("ria_dimension", "")
            )

            metadata_by_file[filename] = row

    return metadata_by_file

def is_legislation_document(doc_metadata: dict) -> bool:
    """
    Identifica documentos jurídicos/normativos.

    Para esses documentos, a limpeza é mais conservadora,
    porque referências normativas podem ser conteúdo importante.
    """
    document_type = doc_metadata.get("document_type", "").lower()
    title = doc_metadata.get("title", "").lower()

    legal_terms = [
        "legislação",
        "lei",
        "resolução",
        "regulation",
        "regulamento",
        "rdc",
        "projeto de lei",
    ]

    return any(term in document_type for term in legal_terms) or any(
        term in title for term in legal_terms
    )


def normalize_line(line: str) -> str:
    """
    Normaliza uma linha mantendo seu conteúdo.
    """
    line = line.replace("\u00a0", " ")
    line = line.replace("\t", " ")
    line = re.sub(r"\s+", " ", line)
    return line.strip()


def normalize_line_for_matching(line: str) -> str:
    """
    Normaliza uma linha para comparação de repetição.
    """
    line = normalize_line(line).lower()
    line = re.sub(r"\s+", " ", line)
    return line.strip(" -–—•·.:;|")


def is_page_number_line(line: str) -> bool:
    """
    Detecta linhas que são apenas número de página.
    """
    line = normalize_line(line)

    patterns = [
        r"^\d+$",
        r"^[-–—]\s*\d+\s*[-–—]$",
        r"^\d+\s*/\s*\d+$",
        r"^(page|página|pagina)\s+\d+(\s+(of|de)\s+\d+)?$",
        r"^\d+\s+(of|de)\s+\d+$",
    ]

    return any(re.match(pattern, line, flags=re.IGNORECASE) for pattern in patterns)


def is_noise_line(line: str) -> bool:
    """
    Remove linhas sem conteúdo semântico.
    """
    line = normalize_line(line)

    if not line:
        return True

    if is_page_number_line(line):
        return True

    if len(line) <= 2 and not line.isalpha():
        return True

    if re.fullmatch(r"[-–—_=*•·. ]+", line):
        return True

    return False


def extract_page_lines(page) -> list[str]:
    """
    Extrai texto da página preservando linhas.
    """
    try:
        raw_text = page.get_text("text", sort=True)
    except TypeError:
        raw_text = page.get_text("text")

    lines = []

    for raw_line in raw_text.splitlines():
        line = normalize_line(raw_line)

        if line:
            lines.append(line)

    return lines


def detect_repeated_margin_lines(
    pages_lines: list[list[str]],
) -> set[str]:
    """
    Detecta cabeçalhos e rodapés repetidos.

    A detecção considera apenas as primeiras e últimas linhas
    de cada página para evitar remover conteúdo normal do corpo.
    """
    page_count = len(pages_lines)

    if page_count < REPEATED_LINE_MIN_PAGES:
        return set()

    line_counter = Counter()

    for lines in pages_lines:
        margin_lines = (
            lines[:HEADER_FOOTER_SCAN_LINES]
            + lines[-HEADER_FOOTER_SCAN_LINES:]
        )

        normalized_lines_on_page = {
            normalize_line_for_matching(line)
            for line in margin_lines
            if line and not is_page_number_line(line)
        }

        for normalized_line in normalized_lines_on_page:
            if 4 <= len(normalized_line) <= 180:
                line_counter[normalized_line] += 1

    min_repetitions = max(
        REPEATED_LINE_MIN_PAGES,
        int(page_count * REPEATED_LINE_MIN_RATIO),
    )

    return {
        line
        for line, count in line_counter.items()
        if count >= min_repetitions
    }


def remove_repeated_margin_lines(
    lines: list[str],
    repeated_margin_lines: set[str],
    cleaning_stats: dict,
) -> list[str]:
    """
    Remove cabeçalhos/rodapés repetidos apenas nas margens da página.
    """
    cleaned_lines = []

    last_index = len(lines) - 1

    for index, line in enumerate(lines):
        normalized = normalize_line_for_matching(line)

        is_margin_position = (
            index < HEADER_FOOTER_SCAN_LINES
            or index > last_index - HEADER_FOOTER_SCAN_LINES
        )

        if is_margin_position and normalized in repeated_margin_lines:
            cleaning_stats["repeated_margin_lines_removed"] += 1
            continue

        cleaned_lines.append(line)

    return cleaned_lines


def clean_page_lines(
    lines: list[str],
    repeated_margin_lines: set[str],
    cleaning_stats: dict,
) -> list[str]:
    """
    Limpa linhas da página sem juntar tudo de imediato.
    """
    cleaned_lines = remove_repeated_margin_lines(
        lines=lines,
        repeated_margin_lines=repeated_margin_lines,
        cleaning_stats=cleaning_stats,
    )

    final_lines = []

    for line in cleaned_lines:
        line = normalize_line(line)

        if is_noise_line(line):
            cleaning_stats["noise_lines_removed"] += 1
            continue

        final_lines.append(line)

    return final_lines


def is_toc_like_line(line: str) -> bool:
    """
    Detecta linhas típicas de sumário.
    """
    line = normalize_line(line)

    return bool(
        re.search(r"\.{3,}\s*\d+$", line)
        or re.search(r"\s{2,}\d+$", line)
    )


def should_discard_noninformative_page(
    lines: list[str],
    doc_metadata: dict,
) -> bool:
    """
    Remove páginas geralmente não informativas:
    sumário, listas de figuras/tabelas e agradecimentos.

    A regra é conservadora para documentos legislativos.
    """
    if not lines:
        return True

    is_legislation = is_legislation_document(doc_metadata)

    first_lines_text = " ".join(lines[:8]).lower()
    page_text = " ".join(lines).lower()

    toc_headings = [
        "sumário",
        "índice",
        "indice",
        "table of contents",
        "contents",
    ]

    list_headings = [
        "lista de figuras",
        "lista de tabelas",
        "list of figures",
        "list of tables",
    ]

    acknowledgement_headings = [
        "agradecimentos",
        "acknowledgements",
        "acknowledgments",
    ]

    if any(heading in first_lines_text for heading in toc_headings):
        toc_like_lines = sum(1 for line in lines if is_toc_like_line(line))

        if toc_like_lines >= 2 or len(page_text) < 2500:
            return True

    if any(heading in first_lines_text for heading in list_headings):
        return True

    if (
        not is_legislation
        and any(heading in first_lines_text for heading in acknowledgement_headings)
        and len(page_text) < 3500
    ):
        return True

    return False


def find_references_start(
    pages_lines: list[list[str]],
    doc_metadata: dict,
) -> tuple[int, int] | None:
    """
    Localiza o início de uma seção bibliográfica final.

    Não remove referências em documentos legislativos,
    pois referências normativas podem ser conteúdo essencial.
    """
    if is_legislation_document(doc_metadata):
        return None

    if not pages_lines:
        return None

    reference_headings = {
        "references",
        "bibliography",
        "referências",
        "referencias",
        "referências bibliográficas",
        "referencias bibliograficas",
        "bibliografia",
        "works cited",
    }

    start_page = max(0, int(len(pages_lines) * 0.55))

    for page_index in range(start_page, len(pages_lines)):
        lines = pages_lines[page_index]

        for line_index, line in enumerate(lines):
            normalized = normalize_line_for_matching(line)

            if normalized in reference_headings:
                return page_index, line_index

            if (
                len(normalized) <= 35
                and normalized.startswith("references")
            ):
                return page_index, line_index

    return None


def apply_references_cut(
    page_index: int,
    lines: list[str],
    references_start: tuple[int, int] | None,
    cleaning_stats: dict,
) -> list[str]:
    """
    Remove a seção de referências bibliográficas finais.
    """
    if references_start is None:
        return lines

    reference_page_index, reference_line_index = references_start

    if page_index < reference_page_index:
        return lines

    if page_index > reference_page_index:
        cleaning_stats["reference_pages_removed"] += 1
        return []

    cleaning_stats["reference_section_starts_removed"] += 1
    return lines[:reference_line_index]


def lines_to_text(lines: list[str]) -> str:
    """
    Junta linhas corrigindo hifenização de fim de linha.
    """
    if not lines:
        return ""

    text_parts = []

    for line in lines:
        line = normalize_line(line)

        if not line:
            continue

        if text_parts and text_parts[-1].endswith("-"):
            previous = text_parts.pop()[:-1]
            text_parts.append(previous + line)
        else:
            text_parts.append(line)

    text = " ".join(text_parts)

    text = re.sub(r"(\w)-\s+(\w)", r"\1\2", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def build_chunk_record(
    pdf_file: Path,
    doc_metadata: dict,
    page_index: int,
    chunk_index: int,
    chunk: str,
) -> dict:
    """
    Cria o registro JSONL de um chunk com seus metadados.
    """
    return {
        "id": (
            f"{pdf_file.stem}_"
            f"p{page_index}_"
            f"c{chunk_index}"
        ),
        "text": chunk,
        "metadata": {
            "document_id": pdf_file.stem,
            "source": pdf_file.name,
            "source_type": doc_metadata.get(
                "source_type",
                "PDF",
            ),
            "title": doc_metadata.get(
                "title",
                pdf_file.stem,
            ),
            "page": page_index,
            "chunk": chunk_index,
            "document_type": doc_metadata.get(
                "document_type",
                "",
            ),
            "author": doc_metadata.get(
                "author",
                "",
            ),
            "year": doc_metadata.get(
                "year",
                "",
            ),
            "theme": doc_metadata.get(
                "theme",
                "",
            ),
            "ria_dimensions": doc_metadata.get(
                "ria_dimensions",
                [],
            ),
            "source_url": doc_metadata.get(
                "source_url",
                "",
            ),
        },
    }


def process_document(
    pdf_file: Path,
    doc_metadata: dict,
    output_file,
    global_stats: dict,
) -> bool:
    """
    Processa um PDF completo.

    Primeiro extrai todas as páginas para conseguir detectar
    cabeçalhos e rodapés repetidos. Depois limpa e chunkifica.
    """
    pages_lines = []

    with fitz.open(pdf_file) as document:
        for page in document:
            page_lines = extract_page_lines(page)
            pages_lines.append(page_lines)

    global_stats["total_pages"] += len(pages_lines)

    repeated_margin_lines = detect_repeated_margin_lines(pages_lines)
    references_start = find_references_start(
        pages_lines=pages_lines,
        doc_metadata=doc_metadata,
    )

    document_text_found = False

    for page_index, original_lines in enumerate(pages_lines, start=1):
        if not original_lines:
            global_stats["raw_empty_pages"] += 1
            continue

        page_lines = apply_references_cut(
            page_index=page_index - 1,
            lines=original_lines,
            references_start=references_start,
            cleaning_stats=global_stats,
        )

        page_lines = clean_page_lines(
            lines=page_lines,
            repeated_margin_lines=repeated_margin_lines,
            cleaning_stats=global_stats,
        )

        if should_discard_noninformative_page(
            lines=page_lines,
            doc_metadata=doc_metadata,
        ):
            global_stats["noninformative_pages_removed"] += 1
            continue

        text = clean_text(lines_to_text(page_lines))

        if not text:
            global_stats["empty_pages_after_cleaning"] += 1
            continue

        chunks = chunk_text(text)

        if not chunks:
            global_stats["pages_without_chunks"] += 1
            continue

        document_text_found = True

        for chunk_index, chunk in enumerate(chunks):
            data = build_chunk_record(
                pdf_file=pdf_file,
                doc_metadata=doc_metadata,
                page_index=page_index,
                chunk_index=chunk_index,
                chunk=chunk,
            )

            output_file.write(
                json.dumps(
                    data,
                    ensure_ascii=False,
                )
                + "\n"
            )

            global_stats["total_chunks"] += 1

    return document_text_found


def process_pdfs() -> None:
    """
    Processa os PDFs do corpus, aplica limpeza avançada,
    cria chunks e salva o resultado em JSONL.
    """
    if not INPUT_DIR.exists():
        raise FileNotFoundError(
            f"Pasta de PDFs não encontrada: {INPUT_DIR}"
        )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest = load_manifest()

    pdf_files = sorted(
        INPUT_DIR.glob("*.pdf"),
        key=lambda path: path.name.lower(),
    )

    if not pdf_files:
        raise FileNotFoundError(
            f"Nenhum PDF encontrado em: {INPUT_DIR}"
        )

    stats = {
        "total_documents": len(pdf_files),
        "processed_documents": 0,
        "failed_documents": 0,
        "total_pages": 0,
        "raw_empty_pages": 0,
        "empty_pages_after_cleaning": 0,
        "noninformative_pages_removed": 0,
        "repeated_margin_lines_removed": 0,
        "noise_lines_removed": 0,
        "reference_section_starts_removed": 0,
        "reference_pages_removed": 0,
        "pages_without_chunks": 0,
        "total_chunks": 0,
    }

    documents_without_metadata = []
    documents_without_text = []
    processing_errors = []

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
    ) as output_file:
        for pdf_file in tqdm(
            pdf_files,
            desc="Processando PDFs",
        ):
            doc_metadata = manifest.get(pdf_file.name)

            if doc_metadata is None:
                documents_without_metadata.append(
                    pdf_file.name
                )

                print(
                    f"\n[ERRO] PDF sem metadados: "
                    f"{pdf_file.name}"
                )

                stats["failed_documents"] += 1
                continue

            try:
                document_text_found = process_document(
                    pdf_file=pdf_file,
                    doc_metadata=doc_metadata,
                    output_file=output_file,
                    global_stats=stats,
                )

                if document_text_found:
                    stats["processed_documents"] += 1
                else:
                    documents_without_text.append(
                        pdf_file.name
                    )
                    stats["failed_documents"] += 1

            except Exception as error:
                stats["failed_documents"] += 1

                processing_errors.append(
                    {
                        "filename": pdf_file.name,
                        "error": str(error),
                    }
                )

                print(
                    f"\n[ERRO] Falha ao processar "
                    f"{pdf_file.name}: {error}"
                )

    print("\n" + "=" * 72)
    print("PROCESSAMENTO FINALIZADO")
    print("=" * 72)
    print(f"PDFs encontrados: {stats['total_documents']}")
    print(f"PDFs processados: {stats['processed_documents']}")
    print(f"PDFs com falha: {stats['failed_documents']}")
    print(f"Páginas analisadas: {stats['total_pages']}")
    print(f"Páginas originalmente sem texto: {stats['raw_empty_pages']}")
    print(
        "Páginas vazias após limpeza: "
        f"{stats['empty_pages_after_cleaning']}"
    )
    print(
        "Páginas não informativas removidas: "
        f"{stats['noninformative_pages_removed']}"
    )
    print(
        "Linhas repetidas de cabeçalho/rodapé removidas: "
        f"{stats['repeated_margin_lines_removed']}"
    )
    print(
        "Linhas ruidosas removidas: "
        f"{stats['noise_lines_removed']}"
    )
    print(
        "Inícios de seção de referências removidos: "
        f"{stats['reference_section_starts_removed']}"
    )
    print(
        "Páginas de referências removidas: "
        f"{stats['reference_pages_removed']}"
    )
    print(
        "Páginas sem chunks após limpeza: "
        f"{stats['pages_without_chunks']}"
    )
    print(f"Chunks gerados: {stats['total_chunks']}")
    print(f"Arquivo salvo em: {OUTPUT_FILE}")

    if documents_without_metadata:
        print("\nPDFs sem metadados:")
        for filename in documents_without_metadata:
            print(f"- {filename}")

    if documents_without_text:
        print("\nPDFs sem texto extraível:")
        for filename in documents_without_text:
            print(f"- {filename}")
        print(
            "Esses documentos podem ser escaneados "
            "e precisar de OCR."
        )

    if processing_errors:
        print("\nErros encontrados:")
        for item in processing_errors:
            print(
                f"- {item['filename']}: "
                f"{item['error']}"
            )


if __name__ == "__main__":
    process_pdfs()
