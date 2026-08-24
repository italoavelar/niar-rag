from __future__ import annotations

import csv
import json
import re
from collections import Counter
from dataclasses import dataclass
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
from structured_units import PackedChunk, extract_pdf_page_units, pack_structured_units


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


@dataclass(frozen=True)
class NoninformativePageClassification:
    """Decisão de descarte e estado estrutural de uma página."""

    discard: bool
    state: str
    reason: str = ""
    confidence: float = 0.0


def _page_noninformative_signals(lines: list[str]) -> dict[str, object]:
    """Calcula sinais locais, sem tomar decisão a partir da posição da página."""
    first_lines_text = " ".join(lines[:8]).lower()
    page_text = " ".join(lines)
    page_text_lower = page_text.lower()

    toc_heading = any(
        heading in first_lines_text
        for heading in ("sumário", "table of contents", "contents")
    )
    subject_index_heading = any(
        heading in first_lines_text
        for heading in (
            "índice remissivo",
            "indice remissivo",
            "subject index",
            "alphabetical index",
        )
    )
    generic_index_heading = (
        "índice" in first_lines_text or "indice" in first_lines_text
    )
    list_heading = any(
        heading in first_lines_text
        for heading in (
            "lista de figuras",
            "lista de tabelas",
            "list of figures",
            "list of tables",
        )
    )
    acknowledgement_heading = any(
        heading in first_lines_text
        for heading in (
            "agradecimentos",
            "acknowledgements",
            "acknowledgments",
        )
    )

    toc_like_lines = sum(is_toc_like_line(line) for line in lines)
    short_lines = sum(len(line) <= 110 for line in lines)
    short_line_ratio = short_lines / len(lines) if lines else 0.0

    # Entradas de índice costumam combinar um termo com localizadores como
    # capítulo, artigo, parágrafo ou uma lista de números. O padrão requer um
    # localizador explícito para não confundir uma tabela numérica comum.
    reference_pattern = re.compile(
        r"\b(?:cap(?:[íi]tulo)?\.?|art(?:igo)?s?\.?|§|"
        r"par[aá]grafo|inciso|p(?:p)?\.)\s*"
        r"(?:[IVXLCDM]+|\d+)",
        flags=re.IGNORECASE,
    )
    index_reference_lines = sum(
        bool(reference_pattern.search(line)) for line in lines
    )
    term_reference_entries = sum(
        bool(
            re.match(r"^[A-ZÁÉÍÓÚÀÂÊÔÃÕÇ].{2,90}", line)
            and reference_pattern.search(line)
        )
        for line in lines
    )
    index_reference_ratio = (
        index_reference_lines / len(lines) if lines else 0.0
    )

    # Uma página normativa deve prevalecer sobre sinais incidentais de
    # referências: o artigo/dispositivo no início e linguagem prescritiva são
    # estruturalmente incompatíveis com uma continuação de índice.
    normative_lines = sum(
        bool(
            re.match(
                r"^(?:(?:art\.?|artigo|par[aá]grafo|inciso)\b|"
                r"§\s*(?:\d+\s*[ºo]?|[úu]nico)(?=\s|\.|$))",
                line,
                flags=re.IGNORECASE,
            )
            and re.search(
                r"\b(?:deve|dever[aá]|[ée] vedado|fica vedado|"
                r"proibido|obrigat[oó]rio)\b",
                line,
                flags=re.IGNORECASE,
            )
        )
        for line in lines
    )
    has_normative_content = normative_lines > 0

    glossary_definition_lines = sum(
        bool(re.match(r"^[^:]{2,70}:\s+.{25,}$", line)) for line in lines
    )
    has_glossary_shape = glossary_definition_lines >= 2

    prose_lines = sum(
        len(line) >= 100 and bool(re.search(r"[.!?]$", line))
        for line in lines
    )

    return {
        "first_lines_text": first_lines_text,
        "page_text": page_text_lower,
        "toc_heading": toc_heading,
        "subject_index_heading": subject_index_heading,
        "generic_index_heading": generic_index_heading,
        "list_heading": list_heading,
        "acknowledgement_heading": acknowledgement_heading,
        "toc_like_lines": toc_like_lines,
        "short_line_ratio": short_line_ratio,
        "index_reference_lines": index_reference_lines,
        "index_reference_ratio": index_reference_ratio,
        "term_reference_entries": term_reference_entries,
        "has_normative_content": has_normative_content,
        "has_glossary_shape": has_glossary_shape,
        "prose_lines": prose_lines,
    }


def _static_noninformative_classification(
    lines: list[str],
    doc_metadata: dict,
) -> NoninformativePageClassification | None:
    """Preserva os filtros estáticos existentes numa forma auditável."""
    if not lines:
        return NoninformativePageClassification(True, "normal", "empty", 1.0)

    signals = _page_noninformative_signals(lines)
    is_legislation = is_legislation_document(doc_metadata)
    page_length = len(str(signals["page_text"]))

    # Mantém a compatibilidade com a regra anterior: o termo genérico
    # "índice" ainda é considerado sumário quando apresenta leaders ou uma
    # página curta. A classificação sequencial abaixo o reinterpreta como
    # índice remissivo quando os sinais específicos forem fortes.
    if (
        (
            signals["toc_heading"] or signals["generic_index_heading"]
        )
        and (
            int(signals["toc_like_lines"]) >= 2 or page_length < 2500
        )
    ):
        return NoninformativePageClassification(True, "normal", "toc_start", 0.9)

    if signals["list_heading"]:
        return NoninformativePageClassification(True, "normal", "list", 1.0)

    if (
        not is_legislation
        and signals["acknowledgement_heading"]
        and page_length < 3500
    ):
        return NoninformativePageClassification(
            True, "normal", "acknowledgements", 0.8
        )

    return None


def should_discard_noninformative_page(
    lines: list[str],
    doc_metadata: dict,
) -> bool:
    """
    Remove páginas geralmente não informativas:
    sumário, listas de figuras/tabelas e agradecimentos.

    A regra é conservadora para documentos legislativos.
    """
    return _static_noninformative_classification(lines, doc_metadata) is not None


def _is_confirmed_toc_start(signals: dict[str, object]) -> bool:
    """Exige título explícito e estrutura de leaders para iniciar um sumário."""
    return bool(
        signals["toc_heading"]
        and int(signals["toc_like_lines"]) >= 2
    ) or bool(
        signals["generic_index_heading"]
        and int(signals["toc_like_lines"]) >= 2
        and not signals["subject_index_heading"]
    )


def _is_toc_continuation(signals: dict[str, object]) -> bool:
    """Continuação exige vários leaders; texto corrido encerra o estado.

    Linhas de sumário em PDFs podem ser muito longas por causa dos leaders
    pontilhados. Portanto, comprimento não é requisito: leaders repetidos e
    ausência de prosa são os sinais estruturais relevantes.
    """
    return bool(
        int(signals["toc_like_lines"]) >= 2
        and int(signals["prose_lines"]) == 0
    )


def _is_confirmed_subject_index(
    signals: dict[str, object],
    *,
    require_heading: bool,
) -> bool:
    """Reconhece índice remissivo sem usar posição no documento como regra."""
    if signals["has_normative_content"] or signals["has_glossary_shape"]:
        return False

    if require_heading and not (
        signals["subject_index_heading"]
        or signals["generic_index_heading"]
    ):
        return False

    return bool(
        int(signals["index_reference_lines"]) >= 4
        and float(signals["index_reference_ratio"]) >= 0.45
        and int(signals["term_reference_entries"]) >= 2
        and float(signals["short_line_ratio"]) >= 0.65
        and int(signals["prose_lines"]) == 0
    )


def classify_noninformative_pages(
    pages_lines: list[list[str]],
    doc_metadata: dict,
) -> list[NoninformativePageClassification]:
    """Classifica páginas em ordem, incluindo continuações de sumário/índice.

    A máquina de estados é intencionalmente conservadora: uma página que não
    repete a estrutura esperada encerra imediatamente o estado. A posição no
    documento não participa da decisão, apenas os sinais textuais locais e o
    contexto da página anterior já confirmada.
    """
    state = "normal"
    classifications = []

    for lines in pages_lines:
        signals = _page_noninformative_signals(lines)
        static = _static_noninformative_classification(lines, doc_metadata)

        if _is_confirmed_subject_index(signals, require_heading=True):
            state = "subject_index"
            classifications.append(
                NoninformativePageClassification(
                    True, state, "subject_index_start", 0.95
                )
            )
            continue

        if _is_confirmed_toc_start(signals):
            state = "toc"
            classifications.append(
                NoninformativePageClassification(True, state, "toc_start", 0.95)
            )
            continue

        if state == "toc" and _is_toc_continuation(signals):
            classifications.append(
                NoninformativePageClassification(
                    True, state, "toc_continuation", 0.9
                )
            )
            continue

        if state == "subject_index" and _is_confirmed_subject_index(
            signals, require_heading=False
        ):
            classifications.append(
                NoninformativePageClassification(
                    True, state, "subject_index_continuation", 0.9
                )
            )
            continue

        state = "normal"
        if static is not None:
            classifications.append(static)
        else:
            classifications.append(NoninformativePageClassification(False, state))

    return classifications



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
    section_path: str | None = None,
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
            "section_path": section_path or "",
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

    # As decisões que dependem da página anterior (continuações de sumário e
    # índice remissivo) precisam enxergar o documento já limpo por inteiro,
    # antes de descartar fisicamente qualquer página ou extrair sua estrutura.
    cleaned_pages = []
    for original_lines in pages_lines:
        if not original_lines:
            cleaned_pages.append([])
            continue

        page_index = len(cleaned_pages)
        page_lines = apply_references_cut(
            page_index=page_index,
            lines=original_lines,
            references_start=references_start,
            cleaning_stats=global_stats,
        )
        cleaned_pages.append(
            clean_page_lines(
                lines=page_lines,
                repeated_margin_lines=repeated_margin_lines,
                cleaning_stats=global_stats,
            )
        )

    page_classifications = classify_noninformative_pages(
        cleaned_pages,
        doc_metadata,
    )

    document_text_found = False

    # A segunda abertura preserva o comportamento de leitura inicial usado
    # para margens/referências e disponibiliza a geometria das páginas para a
    # extração de blocos e tabelas, sem materializar resultados em disco.
    with fitz.open(pdf_file) as document:
        for page_index, original_lines in enumerate(pages_lines, start=1):
            if not original_lines:
                global_stats["raw_empty_pages"] += 1
                continue

            page_lines = cleaned_pages[page_index - 1]
            if page_classifications[page_index - 1].discard:
                global_stats["noninformative_pages_removed"] += 1
                continue

            text = clean_text(lines_to_text(page_lines))

            if not text:
                global_stats["empty_pages_after_cleaning"] += 1
                continue

            units = extract_pdf_page_units(
                page=document[page_index - 1],
                page_number=page_index,
                cleaned_lines=page_lines,
            )
            packed_chunks = pack_structured_units(units)

            # Fallback conservador: nenhum conteúdo útil é descartado quando
            # a estrutura do PDF não pode ser extraída com segurança.
            if not packed_chunks:
                packed_chunks = [PackedChunk(chunk) for chunk in chunk_text(text)]

            if not packed_chunks:
                global_stats["pages_without_chunks"] += 1
                continue

            document_text_found = True

            for chunk_index, packed_chunk in enumerate(packed_chunks):
                data = build_chunk_record(
                    pdf_file=pdf_file,
                    doc_metadata=doc_metadata,
                    page_index=page_index,
                    chunk_index=chunk_index,
                    chunk=packed_chunk.text,
                    section_path=packed_chunk.section_path,
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
