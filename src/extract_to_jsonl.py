from __future__ import annotations

import csv
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import fitz
from tqdm import tqdm

from chunk_id import chunk_id_por_conteudo
from chunking import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    MIN_CHUNK_SIZE,
    choose_split_point,
    chunk_text,
    clean_text,
)
from structured_units import (
    PackedChunk,
    extract_pdf_page_units,
    merge_undersized_chunks,
    pack_structured_units,
)


INPUT_DIR = Path("docs/raw")
MANIFEST_FILE = Path("corpus_manifest.csv")
OUTPUT_FILE = Path("data/processed/pdf_chunks.jsonl")
AUDIT_OUTPUT_FILE = Path("data/processed/pdf_extraction_audit.csv")

AUDIT_FIELDNAMES = [
    "document_id",
    "page",
    "total_pages",
    "status",
    "reason",
    "raw_chars",
    "clean_chars",
    "chunks_generated",
]

INTENTIONALLY_DISCARDED_AUDIT_STATUSES = {
    "NONINFORMATIVE_REMOVED",
    "REFERENCES_REMOVED",
    "EMPTY_AFTER_CLEANING",
}
PROBLEMATIC_AUDIT_STATUSES = {
    "RAW_EMPTY",
    "NO_CHUNKS",
    "ERROR",
}

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


# Numeral romano estrito: aceita i, iv, xiv, mcmxc, mas recusa palavras como
# "civil", "dim" ou "mild", que um casamento solto de [ivxlcdm] deixaria passar.
_ROMAN_NUMERAL_RE = re.compile(
    "^(?=[ivxlcdm])m*(c[md]|d?c{0,3})(x[cl]|l?x{0,3})(i[xv]|v?i{0,3})$",
    flags=re.IGNORECASE,
)


def is_roman_numeral_line(line: str) -> bool:
    """Numeração romana do pré-textual (i, ii, iv, xiv), isolada na linha."""
    return bool(_ROMAN_NUMERAL_RE.match(line.strip("-–— ").strip()))

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

    if any(re.match(p, line, flags=re.IGNORECASE) for p in patterns):
        return True

    return is_roman_numeral_line(line)


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


def extract_page_text(page) -> str:
    """Extrai o texto bruto de uma página com a compatibilidade legada."""
    try:
        return page.get_text("text", sort=True)
    except TypeError:
        return page.get_text("text")


def page_text_to_lines(raw_text: str) -> list[str]:
    """Converte texto bruto em linhas normalizadas, sem descartar texto útil."""
    lines = []

    for raw_line in raw_text.splitlines():
        line = normalize_line(raw_line)

        if line:
            lines.append(line)

    return lines


def extract_page_lines(page) -> list[str]:
    """Extrai texto da página preservando linhas."""
    return page_text_to_lines(extract_page_text(page))


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

    # Títulos pré-textuais precisam ocupar uma linha própria. Procurá-los como
    # simples substring confundia, por exemplo, "sumário de alta" no texto
    # normativo com o título de uma página de sumário.
    def heading_candidate(line: str) -> str:
        candidate = normalize_line_for_matching(line)
        # PDFs de relatórios frequentemente deixam o número físico antes do
        # título ("61 Table of Contents", "53 Glossary"). O número não deve
        # impedir o reconhecimento estrutural do título.
        candidate = re.sub(r"^\d+\s+(?=[a-záà-ÿ])", "", candidate)
        return candidate

    first_line_headings = {
        heading_candidate(line) for line in lines[:8]
    }

    toc_heading = any(
        heading == "contents" or heading.startswith("sumário")
        or heading.startswith("table of contents")
        for heading in first_line_headings
    )
    subject_index_heading = any(
        heading.startswith(prefix)
        for heading in first_line_headings
        for prefix in (
            "índice remissivo",
            "indice remissivo",
            "subject index",
            "alphabetical index",
        )
    )
    generic_index_heading = any(
        heading.startswith(("índice", "indice"))
        for heading in first_line_headings
    )
    list_heading = any(
        heading.startswith(prefix)
        for heading in first_line_headings
        for prefix in (
            "lista de figuras",
            "lista de tabelas",
            "list of figures",
            "list of tables",
        )
    )
    acknowledgement_heading = any(
        heading.startswith(prefix)
        for heading in first_line_headings
        for prefix in (
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
    normative_structure_lines = sum(
        bool(
            re.match(
                r"^(?:(?:art\.?|artigo|par[aá]grafo|inciso)\b|§\s*\d+)",
                line,
                flags=re.IGNORECASE,
            )
        )
        for line in lines
    )
    # Nem toda página normativa repete a fórmula "é vedado". Dois ou mais
    # dispositivos são sinal forte e não se parecem com entradas de índice.
    has_normative_content = (
        normative_lines > 0 or normative_structure_lines >= 2
    )

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

        # Dispositivos normativos têm precedência sobre qualquer heurística de
        # sumário/índice, inclusive quando uma palavra incidental coincide com
        # um cabeçalho (como "sumário de alta").
        if signals["has_normative_content"]:
            state = "normal"
            classifications.append(
                NoninformativePageClassification(False, state)
            )
            continue

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
    Localiza o início de uma seção bibliográfica posterior ao conteúdo inicial.

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


_SUBSTANTIVE_SECTION_HEADING_RE = re.compile(
    r"^(?:annex|appendix|glossary|definitions?|recommendations?)\b",
    flags=re.IGNORECASE,
)
_REFERENCE_ENTRY_RE = re.compile(
    r"^(?:\[\d{1,4}\]|\d{1,4}[.)])\s+",
)


def find_substantive_section_start(lines: list[str]) -> int | None:
    """Localiza a retomada de conteúdo útil após uma bibliografia.

    A leitura é deliberadamente conservadora: só títulos no início da página
    ou dispositivos normativos inequívocos encerram a faixa de referências.
    Na dúvida, preservamos a página em vez de descartar um possível anexo.
    """
    for line_index, line in enumerate(lines[:8]):
        normalized = normalize_line_for_matching(line)
        normalized = re.sub(r"^\d+\s+(?=[a-záà-ÿ])", "", normalized)
        if (
            len(normalized) <= 180
            and _SUBSTANTIVE_SECTION_HEADING_RE.match(normalized)
        ):
            return line_index

    signals = _page_noninformative_signals(lines)
    if signals["has_normative_content"]:
        return 0

    return None


def is_reference_like_page(lines: list[str]) -> bool:
    """Exige evidência forte antes de continuar removendo bibliografia.

    Nem todos os PDFs numeram referências no início da linha. Alguns colocam
    ``[n]`` no fim da entrada, ou extraem o marcador de página antes do autor.
    DOI/URL e padrões de publicação são usados em conjunto para não confundir
    prosa técnica com uma bibliografia.
    """
    meaningful = [
        normalize_line(line)
        for line in lines
        if line and not is_page_number_line(line)
    ]
    if not meaningful:
        return False

    numbered_entries = sum(
        bool(_REFERENCE_ENTRY_RE.match(line))
        or bool(re.search(r"\[\d{1,4}\]\s*[.,;)]?$", line))
        for line in meaningful
    )
    url_or_doi = sum(
        bool(re.search(r"(?:https?://|doi\s*:\s*|doi\.org/)", line, re.I))
        for line in meaningful
    )
    publication_markers = sum(
        bool(
            re.search(
                r"\b(?:et\s+al\.?|vol\.?|pp?\.?\s*\d|retrieved|accessed|"
                r"proceedings|publisher|journal|report)\b",
                line,
                re.I,
            )
        )
        for line in meaningful
    )
    year_entries = sum(
        bool(re.search(r"\b(?:19|20)\d{2}\b", line))
        and bool(re.search(r"[,.(]", line))
        for line in meaningful
    )

    # Duas evidências independentes ou pelo menos três marcadores explícitos
    # são necessárias. Isso mantém anexos, glossários e prosa metodológica.
    evidence = numbered_entries + url_or_doi + publication_markers
    if numbered_entries >= 2 and url_or_doi == 0 and year_entries == 0:
        # A forma clássica ``1. Autor. Título`` continua sendo inequívoca,
        # mesmo quando o extrator não recupera ano/editora.
        return True
    return (
        (numbered_entries >= 2 and (url_or_doi >= 1 or year_entries >= 2))
        or (url_or_doi >= 3 and year_entries >= 2)
        or (publication_markers >= 3 and year_entries >= 2)
    )


def _substantive_section_heading(lines: list[str]) -> bool:
    """Retorna se as primeiras linhas anunciam uma seção semântica."""
    for line in lines[:8]:
        normalized = normalize_line_for_matching(line)
        normalized = re.sub(r"^\d+\s+(?=[a-záà-ÿ])", "", normalized)
        if _SUBSTANTIVE_SECTION_HEADING_RE.match(normalized):
            return True
    return False


def short_page_noninformative_reason(
    lines: list[str],
    previous_lines: list[str] | None = None,
) -> str | None:
    """Classifica apenas ruído de alta confiança em uma página já limpa.

    O tamanho sozinho nunca decide. A função também é aplicada a algumas
    páginas longas quando a forma é inequivocamente copyright/contato/TOC;
    conteúdo normativo, definições, headings substantivos, prosa e estruturas
    tabulares sempre têm precedência.
    """
    if not lines:
        return None

    signals = _page_noninformative_signals(lines)
    text = lines_to_text(lines)
    lower = text.lower()
    page_length = len(text)
    non_page_lines = [line for line in lines if not is_page_number_line(line)]
    if not non_page_lines:
        return "page_number_only"

    has_prose = int(signals["prose_lines"]) > 0
    has_complete_sentence = any(
        len(line) >= 30
        and len(re.findall(r"[A-Za-zÀ-ÿ]{3,}", line)) >= 5
        and bool(re.search(r"[.!?]$", line))
        for line in non_page_lines
    )
    has_structure = bool(
        signals["has_normative_content"]
        or signals["has_glossary_shape"]
        or _substantive_section_heading(lines)
        or has_complete_sentence
        or len(non_page_lines) >= 4
        and sum(bool(re.search(r"\b[A-Z][A-Z0-9/-]{1,12}\b", line)) for line in non_page_lines) >= 3
    )

    # Listagens de fontes dentro de um anexo têm alta densidade de DOI/URL,
    # mas não anunciam glossário, definições ou outra seção semântica. Elas
    # continuam sendo bibliografia mesmo depois de uma retomada substantiva.
    source_markers = sum(
        bool(re.search(r"(?:https?://|doi\s*:\s*|doi\.org/)", line, re.I))
        for line in non_page_lines
    )
    source_bullets = sum(
        line.lstrip().startswith(("•", "-", "*"))
        for line in non_page_lines
    )
    first_normalized = normalize_line_for_matching(non_page_lines[0])
    first_is_source_bullet = non_page_lines[0].lstrip().startswith(
        ("•", "-", "*")
    )
    source_heading = bool(
        re.match(
            r"^(?:references?|bibliograph(?:y|ies)|how[- ]to articles?)\b",
            first_normalized,
            re.I,
        )
    )
    previous_source_shape = False
    if previous_lines:
        previous_meaningful = [
            line for line in previous_lines if not is_page_number_line(line)
        ]
        if previous_meaningful:
            previous_signals = _page_noninformative_signals(previous_lines)
            previous_source_markers = sum(
                bool(
                    re.search(
                        r"(?:https?://|doi\s*:\s*|doi\.org/)",
                        line,
                        re.I,
                    )
                )
                for line in previous_meaningful
            )
            previous_source_shape = bool(
                previous_source_markers >= 2
                and int(previous_signals["prose_lines"]) == 0
                and float(previous_signals["short_line_ratio"]) >= 0.9
                and not _substantive_section_heading(previous_lines)
                and (
                    previous_meaningful[0].lstrip().startswith(
                        ("•", "-", "*")
                    )
                    or re.match(
                        r"^(?:references?|bibliograph(?:y|ies)|how[- ]to articles?)\b",
                        normalize_line_for_matching(previous_meaningful[0]),
                        re.I,
                    )
                )
            )
    if (
        source_markers >= 2
        and (is_reference_like_page(lines) or previous_source_shape)
        and (
            (first_is_source_bullet and source_bullets >= 2)
            or source_heading
            or previous_source_shape
        )
        and not _substantive_section_heading(lines)
        and not signals["has_normative_content"]
        and not re.search(
            r"\b(?:glossary|definitions?|acronyms?|symbols?)\b",
            lower,
        )
    ):
        return "bibliographic_continuation"

    # Sumários podem ter uma linha inicial precedida pelo número físico.
    if _is_confirmed_toc_start(signals) or _is_toc_continuation(signals):
        return "toc_short_or_continuation"

    if re.search(r"\bphoto\s+credit\b|\bcr[ée]dito\s+da\s+foto\b", lower):
        if not has_prose and not has_complete_sentence and not has_structure:
            return "photo_credit_only"

    contact_markers = sum(
        bool(
            re.search(
                r"\b(?:world health organization|organization|department|"
                r"division|avenue|street|road|geneva|switzerland|secretariat|"
                r"website|www\.)\b",
                line,
                re.I,
            )
        )
        for line in non_page_lines
    )
    if contact_markers >= 3 and not signals["has_normative_content"] and not has_structure:
        return "institutional_contact_only"

    copyright_markers = sum(
        bool(
            re.search(
                r"\b(?:copyright|disclaimer|terms\s+and\s+conditions|"
                r"visit\s+our\s+website|all\s+other\s+rights\s+are\s+reserved)\b",
                line,
                re.I,
            )
        )
        for line in non_page_lines
    )
    if copyright_markers >= 2 and not signals["has_normative_content"]:
        return "copyright_or_disclaimer"

    if (
        re.fullmatch(
            r"(?:additional\s+resources?\s*:\s*see\s+.+|"
            r"(?:figure|fig\.?|photo)\s+.+)",
            text,
            flags=re.IGNORECASE,
        )
        or (
            len(non_page_lines) <= 3
            and all("additional resources" in line.lower() for line in non_page_lines)
        )
    ) and not has_structure:
        return "editorial_footer_or_figure_caption"

    # URL/DOI ou aviso editorial isolado, sem uma frase/proposição, é ruído.
    stripped = re.sub(r"https?://\S+|www\.\S+|doi\.org/\S+", "", text, flags=re.I)
    stripped = re.sub(r"\b(?:available\s+free\s+of\s+charge|publication|keywords?)\b", "", stripped, flags=re.I)
    stripped = re.sub(r"[\d\W_]+", " ", stripped).strip()
    if page_length <= 350 and len(stripped.split()) <= 8 and not has_structure and not has_prose and not has_complete_sentence:
        return "boilerplate_or_url_only"

    # Capas/títulos e headings soltos sem uma proposição semântica.
    if page_length <= 300 and not has_structure and not has_prose and not has_complete_sentence:
        if len(non_page_lines) <= 6:
            return "cover_or_heading_only"

    return None


def find_references_end(
    pages_lines: list[list[str]],
    references_start: tuple[int, int] | None,
    doc_metadata: dict,
) -> tuple[int, int] | None:
    """Encontra a primeira seção substantiva posterior às referências.

    ``None`` mantém o comportamento correto para bibliografia que de fato vai
    até o fim do documento. ``(página, linha)`` marca onde o pipeline deve
    voltar a preservar conteúdo.
    """
    if references_start is None or is_legislation_document(doc_metadata):
        return None

    reference_page, reference_line = references_start
    for page_index in range(reference_page, len(pages_lines)):
        lines = pages_lines[page_index]
        offset = reference_line + 1 if page_index == reference_page else 0
        restart_line = find_substantive_section_start(lines[offset:])
        if restart_line is not None:
            return page_index, offset + restart_line

        # A página de abertura já é bibliografia por definição. Nas páginas
        # seguintes, só continuamos cortando com estrutura bibliográfica
        # inequívoca; sem ela, preservamos o possível conteúdo substantivo.
        if page_index > reference_page and not is_reference_like_page(lines):
            return page_index, 0

    return None


def apply_references_cut(
    page_index: int,
    lines: list[str],
    references_start: tuple[int, int] | None,
    cleaning_stats: dict,
    references_end: tuple[int, int] | None = None,
) -> list[str]:
    """
    Remove a seção de referências bibliográficas finais.
    """
    if references_start is None:
        return lines

    reference_page_index, reference_line_index = references_start

    if references_end is not None:
        end_page_index, end_line_index = references_end
        if page_index > end_page_index:
            return lines
        if page_index == end_page_index:
            if end_line_index:
                cleaning_stats["reference_pages_removed"] += 1
            return lines[end_line_index:]

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
        # id derivado do conteúdo, não da posição — ver src/chunk_id.py.
        # Página e índice continuam no metadado, como informação de ordem.
        "id": chunk_id_por_conteudo(pdf_file.stem, chunk),
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


def build_audit_record(
    pdf_file: Path,
    page: int,
    total_pages: int,
    status: str,
    reason: str,
    raw_chars: int,
    clean_chars: int,
    chunks_generated: int,
) -> dict[str, object]:
    """Cria o registro de auditoria de uma página física do PDF."""
    return {
        "document_id": pdf_file.stem,
        "page": page,
        "total_pages": total_pages,
        "status": status,
        "reason": reason,
        "raw_chars": raw_chars,
        "clean_chars": clean_chars,
        "chunks_generated": chunks_generated,
    }


def validate_document_audit_records(
    pdf_file: Path,
    total_pages: int,
    records: list[dict[str, object]],
) -> None:
    """Garante que toda página física recebeu exatamente uma classificação."""
    pages = [record["page"] for record in records]
    expected_pages = list(range(1, total_pages + 1))

    if pages != expected_pages:
        raise ValueError(
            "Auditoria de extração incompleta ou duplicada para "
            f"{pdf_file.name}: esperado={expected_pages}, obtido={pages}"
        )

    unclassified = [
        record["page"]
        for record in records
        if not record["status"]
    ]
    if unclassified:
        raise ValueError(
            "Páginas sem classificação na auditoria de "
            f"{pdf_file.name}: {unclassified}"
        )


def _format_page_ranges(pages: list[int]) -> str:
    """Formata [1, 2, 3, 6] como '1-3, 6' para o resumo operacional."""
    if not pages:
        return "-"

    ranges = []
    start = previous = pages[0]
    for page in pages[1:]:
        if page == previous + 1:
            previous = page
            continue
        ranges.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = page
    ranges.append(str(start) if start == previous else f"{start}-{previous}")
    return ", ".join(ranges)


def add_document_audit_summary(
    global_stats: dict,
    pdf_file: Path,
    total_pages: int,
    records: list[dict[str, object]],
) -> None:
    """Armazena o resumo por documento para exibição ao final da execução."""
    status_pages: dict[str, list[int]] = {}
    for record in records:
        status_pages.setdefault(str(record["status"]), []).append(
            int(record["page"])
        )

    intentional_pages = sorted(
        page
        for status, pages in status_pages.items()
        if status in INTENTIONALLY_DISCARDED_AUDIT_STATUSES
        for page in pages
    )
    problematic_by_status = {
        status: pages
        for status, pages in status_pages.items()
        if status in PROBLEMATIC_AUDIT_STATUSES
    }

    global_stats.setdefault("document_audit_summaries", []).append(
        {
            "document_id": pdf_file.stem,
            "total_pages": total_pages,
            "indexed_pages": len(status_pages.get("INDEXED", [])),
            "intentionally_discarded_pages": intentional_pages,
            "problematic_by_status": problematic_by_status,
        }
    )


def print_document_audit_summaries(global_stats: dict) -> None:
    """Exibe a cobertura e as faixas de páginas por documento."""
    summaries = global_stats.get("document_audit_summaries", [])
    if not summaries:
        return

    print("\nResumo da auditoria por documento:")
    for summary in summaries:
        intentional_pages = summary["intentionally_discarded_pages"]
        problematic_by_status = summary["problematic_by_status"]
        problematic_count = sum(
            len(pages) for pages in problematic_by_status.values()
        )
        problematic_ranges = "; ".join(
            f"{status}: {_format_page_ranges(pages)}"
            for status, pages in sorted(problematic_by_status.items())
        ) or "-"

        print(
            f"- {summary['document_id']}: "
            f"PDF={summary['total_pages']}; "
            f"indexadas={summary['indexed_pages']}; "
            "descartadas intencionalmente="
            f"{len(intentional_pages)} "
            f"({_format_page_ranges(intentional_pages)}); "
            f"problemáticas={problematic_count} "
            f"({problematic_ranges})"
        )


def process_document(
    pdf_file: Path,
    doc_metadata: dict,
    output_file,
    audit_writer: csv.DictWriter,
    global_stats: dict,
) -> bool:
    """
    Processa um PDF completo.

    Primeiro extrai todas as páginas para conseguir detectar
    cabeçalhos e rodapés repetidos. Depois limpa e chunkifica.
    """
    pages_lines = []
    raw_texts = []
    page_extraction_errors: dict[int, str] = {}

    with fitz.open(pdf_file) as document:
        total_pages = len(document)
        for page_index, page in enumerate(document, start=1):
            try:
                raw_text = extract_page_text(page)
                raw_texts.append(raw_text)
                pages_lines.append(page_text_to_lines(raw_text))
            except Exception as error:
                # A página ainda ganha registro no relatório; não pode sumir
                # porque a extração textual de uma página falhou.
                raw_texts.append("")
                pages_lines.append([])
                page_extraction_errors[page_index] = str(error)

    global_stats["total_pages"] += total_pages

    repeated_margin_lines = detect_repeated_margin_lines(pages_lines)
    references_start = find_references_start(
        pages_lines=pages_lines,
        doc_metadata=doc_metadata,
    )
    references_end = find_references_end(
        pages_lines=pages_lines,
        references_start=references_start,
        doc_metadata=doc_metadata,
    )

    # As decisões que dependem da página anterior (continuações de sumário e
    # índice remissivo) precisam enxergar o documento já limpo por inteiro,
    # antes de descartar fisicamente qualquer página ou extrair sua estrutura.
    cleaned_pages = []
    for page_index, original_lines in enumerate(pages_lines):
        if page_index + 1 in page_extraction_errors:
            cleaned_pages.append([])
            continue

        if not original_lines:
            cleaned_pages.append([])
            continue

        page_lines = apply_references_cut(
            page_index=page_index,
            lines=original_lines,
            references_start=references_start,
            cleaning_stats=global_stats,
            references_end=references_end,
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
    audit_records: list[dict[str, object]] = []

    # A segunda abertura preserva o comportamento de leitura inicial usado
    # para margens/referências e disponibiliza a geometria das páginas para a
    # extração de blocos e tabelas, sem materializar resultados em disco.
    with fitz.open(pdf_file) as document:
        for page_index, original_lines in enumerate(pages_lines, start=1):
            raw_chars = len(raw_texts[page_index - 1])
            page_lines = cleaned_pages[page_index - 1]
            clean_text_for_audit = clean_text(lines_to_text(page_lines))
            clean_chars = len(clean_text_for_audit)

            if page_index in page_extraction_errors:
                global_stats["pages_with_errors"] += 1
                audit_records.append(
                    build_audit_record(
                        pdf_file, page_index, total_pages, "ERROR",
                        "raw_text_extraction_error: "
                        + page_extraction_errors[page_index],
                        raw_chars, clean_chars, 0,
                    )
                )
                continue

            if not original_lines:
                global_stats["raw_empty_pages"] += 1
                audit_records.append(
                    build_audit_record(
                        pdf_file, page_index, total_pages, "RAW_EMPTY",
                        "raw_text_empty", raw_chars, clean_chars, 0,
                    )
                )
                continue

            classification = page_classifications[page_index - 1]
            references_removed = (
                references_start is not None
                and page_index - 1 >= references_start[0]
                and (
                    references_end is None
                    or page_index - 1 < references_end[0]
                )
                and not page_lines
            )
            if references_removed:
                global_stats["references_pages_audited_removed"] += 1
                reason = (
                    "reference_section_starts_removed"
                    if page_index - 1 == references_start[0]
                    else "reference_pages_removed"
                )
                audit_records.append(
                    build_audit_record(
                        pdf_file, page_index, total_pages, "REFERENCES_REMOVED",
                        reason, raw_chars, clean_chars, 0,
                    )
                )
                continue

            if classification.discard:
                global_stats["noninformative_pages_removed"] += 1
                status = "NONINFORMATIVE_REMOVED"
                if classification.reason == "empty":
                    status = "EMPTY_AFTER_CLEANING"
                    global_stats["empty_pages_after_cleaning"] += 1
                audit_records.append(
                    build_audit_record(
                        pdf_file, page_index, total_pages, status,
                        classification.reason, raw_chars, clean_chars, 0,
                    )
                )
                continue

            text = clean_text_for_audit

            # A correção de páginas curtas preserva qualquer conteúdo sem
            # evidência forte de ruído. Depois da limpeza estrutural, porém,
            # algumas páginas são inequivocamente capa, crédito, contato,
            # copyright ou rodapé. Elas não devem voltar ao corpus só porque
            # ``chunk_text`` passou a preservar fragmentos isolados.
            previous_page_lines = (
                cleaned_pages[page_index - 2]
                if page_index > 1
                else None
            )
            short_reason = short_page_noninformative_reason(
                page_lines,
                previous_lines=previous_page_lines,
            )
            if short_reason:
                global_stats["noninformative_pages_removed"] += 1
                audit_records.append(
                    build_audit_record(
                        pdf_file, page_index, total_pages,
                        "NONINFORMATIVE_REMOVED", short_reason,
                        raw_chars, clean_chars, 0,
                    )
                )
                continue

            if not text:
                global_stats["empty_pages_after_cleaning"] += 1
                audit_records.append(
                    build_audit_record(
                        pdf_file, page_index, total_pages,
                        "EMPTY_AFTER_CLEANING", "clean_text_empty",
                        raw_chars, clean_chars, 0,
                    )
                )
                continue

            try:
                units = extract_pdf_page_units(
                    page=document[page_index - 1],
                    page_number=page_index,
                    cleaned_lines=page_lines,
                )
                packed_chunks = pack_structured_units(units)

                # Fallback conservador: nenhum conteúdo útil é descartado quando
                # a estrutura do PDF não pode ser extraída com segurança.
                if not packed_chunks:
                    packed_chunks = [
                        PackedChunk(chunk) for chunk in chunk_text(text)
                    ]

                packed_chunks = merge_undersized_chunks(packed_chunks)
            except Exception as error:
                global_stats["pages_with_errors"] += 1
                audit_records.append(
                    build_audit_record(
                        pdf_file, page_index, total_pages, "ERROR",
                        "page_processing_error: " + str(error),
                        raw_chars, clean_chars, 0,
                    )
                )
                continue

            if not packed_chunks:
                global_stats["pages_without_chunks"] += 1
                audit_records.append(
                    build_audit_record(
                        pdf_file, page_index, total_pages, "NO_CHUNKS",
                        "no_chunks_after_packing", raw_chars, clean_chars, 0,
                    )
                )
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

            audit_records.append(
                build_audit_record(
                    pdf_file, page_index, total_pages, "INDEXED", "",
                    raw_chars, clean_chars, len(packed_chunks),
                )
            )

    validate_document_audit_records(pdf_file, total_pages, audit_records)
    for record in audit_records:
        audit_writer.writerow(record)
    add_document_audit_summary(
        global_stats, pdf_file, total_pages, audit_records
    )

    return document_text_found


def audit_document_without_metadata(
    pdf_file: Path,
    audit_writer: csv.DictWriter,
    global_stats: dict,
) -> None:
    """Mantém a cobertura da auditoria mesmo para PDFs fora do manifesto."""
    with fitz.open(pdf_file) as document:
        total_pages = len(document)

    records = [
        build_audit_record(
            pdf_file=pdf_file,
            page=page,
            total_pages=total_pages,
            status="ERROR",
            reason="document_metadata_missing",
            raw_chars=0,
            clean_chars=0,
            chunks_generated=0,
        )
        for page in range(1, total_pages + 1)
    ]
    validate_document_audit_records(pdf_file, total_pages, records)
    for record in records:
        audit_writer.writerow(record)
    global_stats["total_pages"] += total_pages
    global_stats["pages_with_errors"] += total_pages
    add_document_audit_summary(global_stats, pdf_file, total_pages, records)


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
        "references_pages_audited_removed": 0,
        "pages_without_chunks": 0,
        "pages_with_errors": 0,
        "total_chunks": 0,
        "document_audit_summaries": [],
    }

    documents_without_metadata = []
    documents_without_text = []
    processing_errors = []

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
    ) as output_file, open(
        AUDIT_OUTPUT_FILE,
        "w",
        encoding="utf-8",
        newline="",
    ) as audit_file:
        audit_writer = csv.DictWriter(
            audit_file,
            fieldnames=AUDIT_FIELDNAMES,
        )
        audit_writer.writeheader()

        for pdf_file in tqdm(pdf_files, desc="Processando PDFs"):
            doc_metadata = manifest.get(pdf_file.name)

            if doc_metadata is None:
                documents_without_metadata.append(pdf_file.name)

                print(
                    f"\n[ERRO] PDF sem metadados: {pdf_file.name}"
                )
                try:
                    audit_document_without_metadata(
                        pdf_file, audit_writer, stats
                    )
                except Exception as error:
                    processing_errors.append(
                        {"filename": pdf_file.name, "error": str(error)}
                    )
                stats["failed_documents"] += 1
                continue

            try:
                document_text_found = process_document(
                    pdf_file=pdf_file,
                    doc_metadata=doc_metadata,
                    output_file=output_file,
                    audit_writer=audit_writer,
                    global_stats=stats,
                )

                if document_text_found:
                    stats["processed_documents"] += 1
                else:
                    documents_without_text.append(pdf_file.name)
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
    print(f"Páginas com erro: {stats['pages_with_errors']}")
    print(f"Chunks gerados: {stats['total_chunks']}")
    print(f"Arquivo salvo em: {OUTPUT_FILE}")
    print(f"Auditoria salva em: {AUDIT_OUTPUT_FILE}")
    print_document_audit_summaries(stats)

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
