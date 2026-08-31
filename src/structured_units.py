"""Unidades estruturais simples e agrupamento semântico para PDFs.

O módulo preserva tabelas, headings e marcadores normativos quando eles podem
ser reconhecidos com segurança. Texto sem estrutura continua usando o chunker
existente como fallback.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

from chunking import CHUNK_OVERLAP, CHUNK_SIZE, MIN_CHUNK_SIZE, chunk_text


_ARTICLE_RE = re.compile(
    r"^\s*(?:art\.|artigo)\s+\d+(?:\s*[º°oª])?",
    flags=re.IGNORECASE,
)
_PARAGRAPH_RE = re.compile(
    r"^\s*(?:§\s*(?:\d+\s*[º°oª]?|[úu]nico)|par[aá]grafo\s+[úu]nico)",
    flags=re.IGNORECASE,
)
_HEADING_RE = re.compile(
    r"^\s*(?:cap[ií]tulo|t[ií]tulo|subse[cç][aã]o|se[cç][aã]o)\b",
    flags=re.IGNORECASE,
)
_INCISO_RE = re.compile(r"^\s*(?:[IVXLCDM]+|[a-z])\s*[-–)]\s+", re.IGNORECASE)


@dataclass
class StructuralUnit:
    """Menor unidade semântica preservada antes do agrupamento."""

    kind: str
    text: str
    source_order: int
    page: int
    bbox: tuple[float, float, float, float] | None = None
    section_path: str | None = None
    table_id: str | None = None
    row_index: int | None = None
    table_headers: tuple[str, ...] = ()


@dataclass
class PackedChunk:
    """Resultado do agrupamento, com contexto de seção opcional."""

    text: str
    section_path: str | None = None


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").replace("\u00a0", " ")).strip()


def _bbox_intersects(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> bool:
    return not (
        left[2] <= right[0]
        or right[2] <= left[0]
        or left[3] <= right[1]
        or right[3] <= left[1]
    )


def _line_is_in_table(
    bbox: tuple[float, float, float, float],
    table_bboxes: list[tuple[float, float, float, float]],
) -> bool:
    return any(_bbox_intersects(bbox, table_bbox) for table_bbox in table_bboxes)


def _classify_text_kind(
    text: str,
    is_bold: bool,
    article_active: bool = False,
) -> str:
    """Classifica somente sinais explícitos ou de alta confiança."""
    normalized = _normalize_text(text)

    if _ARTICLE_RE.match(normalized):
        return "article"

    if _HEADING_RE.match(normalized):
        return "heading"

    words = normalized.split()
    is_short_label = (
        0 < len(normalized) <= 90
        and len(words) <= 10
        and not normalized.endswith((".", ";", ":", "?", "!"))
    )

    if is_bold and is_short_label:
        return "heading"

    # Incisos só são reconhecidos como estrutura subordinada depois de um
    # artigo. Eles permanecem ``text`` para conservar o modelo enxuto.
    if article_active and _INCISO_RE.match(normalized):
        return "text"

    if _PARAGRAPH_RE.match(normalized):
        return "text"

    return "text"


def _heading_level(text: str) -> int:
    normalized = _normalize_text(text).lower()
    if normalized.startswith("título") or normalized.startswith("titulo"):
        return 1
    if normalized.startswith("capítulo") or normalized.startswith("capitulo"):
        return 2
    if normalized.startswith("seção") or normalized.startswith("secao"):
        return 3
    if normalized.startswith("subseção") or normalized.startswith("subsecao"):
        return 4
    return 5


def _article_label(text: str) -> str | None:
    match = _ARTICLE_RE.match(_normalize_text(text))
    return match.group(0).strip() if match else None


def _apply_structure_context(units: list[StructuralUnit]) -> None:
    """Associa caminho de seção e artigo às unidades posteriores."""
    sections: list[tuple[int, str]] = []
    active_article: str | None = None

    for unit in units:
        if unit.kind == "heading":
            label = _normalize_text(unit.text)
            level = _heading_level(label)
            sections = [item for item in sections if item[0] < level]
            sections.append((level, label))
            active_article = None
            unit.section_path = " > ".join(value for _, value in sections)
            continue

        if unit.kind == "article":
            active_article = _article_label(unit.text)

        path_parts = [value for _, value in sections]
        if active_article:
            path_parts.append(active_article)
        unit.section_path = " > ".join(path_parts) or None


def _table_headers(table) -> tuple[str, ...]:
    names = getattr(getattr(table, "header", None), "names", None) or []
    return tuple(
        _normalize_text(name) or f"Coluna {index + 1}"
        for index, name in enumerate(names)
    )


def _table_column_count(rows: list[list[str | None]], headers: tuple[str, ...]) -> int:
    return max([len(headers), *(len(row) for row in rows)], default=0)


def _has_usable_table_header(
    table,
    rows: list[list[str | None]],
    headers: tuple[str, ...],
) -> bool:
    """Evita promover uma linha de conteúdo longa a cabeçalho da tabela.

    Em alguns PDFs, ``find_tables()`` informa uma primeira linha interna como
    ``header`` embora ela seja conteúdo. O caso confiavelmente problemático é
    uma tabela multicoluna com só uma célula de cabeçalho preenchida e extensa:
    repetir esse texto como rótulo em cada fragmento cria chunks redundantes e
    ainda remove a primeira linha de dados. Cabeçalhos externos e tabelas de
    uma única coluna permanecem inalterados.
    """
    header = getattr(table, "header", None)
    if bool(getattr(header, "external", False)):
        return bool(headers)

    raw_names = getattr(header, "names", None) or []
    nonempty_names = [_normalize_text(name) for name in raw_names if _normalize_text(name)]
    column_count = _table_column_count(rows, headers)

    if not nonempty_names:
        return False

    return not (
        column_count > 1
        and len(nonempty_names) == 1
        and len(nonempty_names[0]) > 240
    )


def _generic_table_headers(column_count: int) -> tuple[str, ...]:
    return tuple(f"Coluna {index + 1}" for index in range(column_count))


def _table_signature(
    headers: tuple[str, ...],
    rows: list[list[str | None]],
) -> tuple[tuple[str, ...], tuple[tuple[str, ...], ...]]:
    """Representação estrutural estável para comparar detecções de tabela."""
    return (
        tuple(_normalize_text(header) for header in headers),
        tuple(
            tuple(_normalize_text(cell or "") for cell in row)
            for row in rows
        ),
    )


def _bbox_overlap_of_smaller(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    """Fração da menor caixa coberta pela interseção das duas caixas."""
    intersection_width = max(0.0, min(left[2], right[2]) - max(left[0], right[0]))
    intersection_height = max(0.0, min(left[3], right[3]) - max(left[1], right[1]))
    intersection = intersection_width * intersection_height
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    smaller_area = min(left_area, right_area)

    if smaller_area == 0:
        return 1.0 if left == right else 0.0
    return intersection / smaller_area


def _is_duplicate_table(
    signature: tuple[tuple[str, ...], tuple[tuple[str, ...], ...]],
    bbox: tuple[float, float, float, float],
    seen_tables: list[
        tuple[
            tuple[tuple[str, ...], tuple[tuple[str, ...], ...]],
            tuple[float, float, float, float],
        ]
    ],
) -> bool:
    """Identifica a mesma tabela detectada repetidamente na mesma região."""
    return any(
        signature == seen_signature
        and _bbox_overlap_of_smaller(bbox, seen_bbox) >= 0.9
        for seen_signature, seen_bbox in seen_tables
    )


def _serialize_table_row(headers: tuple[str, ...], row: list[str | None]) -> str:
    return "\n".join(
        f"{header}: {_normalize_text(cell or '')}"
        for header, cell in zip(headers, row)
    )


def _page_dict(page) -> dict:
    try:
        return page.get_text("dict", sort=True)
    except TypeError:
        return page.get_text("dict")


def _page_tables(page) -> list:
    find_tables = getattr(page, "find_tables", None)
    if find_tables is None:
        return []

    try:
        result = find_tables()
    except Exception:
        return []

    return list(getattr(result, "tables", result))


def extract_pdf_page_units(
    page,
    page_number: int,
    cleaned_lines: list[str],
) -> list[StructuralUnit]:
    """Extrai blocos e tabelas sem repetir células como texto comum.

    ``cleaned_lines`` vem do pipeline existente; ele mantém as decisões atuais
    sobre margens, ruído e referências antes de se criar as unidades.
    """
    tables = _page_tables(page)
    allowed_lines = Counter(_normalize_text(line) for line in cleaned_lines)
    table_content = []
    entries = []
    prepared_tables = []
    seen_tables = []

    for table in tables:
        headers = _table_headers(table)
        rows = table.extract()
        if not rows:
            continue

        # O primeiro registro é o cabeçalho somente quando a API o identificou
        # dentro da área da tabela. Em cabeçalhos externos, ``extract()`` já
        # começa na primeira linha de dados.
        header = getattr(table, "header", None)
        header_is_external = bool(getattr(header, "external", False))
        if _has_usable_table_header(table, rows, headers):
            data_rows = rows[1:] if not header_is_external else rows
        else:
            headers = _generic_table_headers(_table_column_count(rows, headers))
            data_rows = rows

        table_bbox = tuple(table.bbox)
        signature = _table_signature(headers, data_rows)
        if _is_duplicate_table(signature, table_bbox, seen_tables):
            continue
        seen_tables.append((signature, table_bbox))
        prepared_tables.append((table_bbox, headers, data_rows))

    table_bboxes = [table_bbox for table_bbox, _, _ in prepared_tables]

    for table_index, (table_bbox, headers, data_rows) in enumerate(prepared_tables):
        table_content.append(_normalize_text(" ".join(headers)))
        table_content.extend(
            _normalize_text(" ".join(cell or "" for cell in row))
            for row in data_rows
        )

        table_id = f"p{page_number}-t{table_index}"
        entries.append(
            (
                table_bbox[1],
                table_bbox[0],
                "table",
                {
                    "table_id": table_id,
                    "bbox": table_bbox,
                    "headers": headers,
                    "rows": data_rows,
                },
            )
        )

    for block in _page_dict(page).get("blocks", []):
        if block.get("type") != 0:
            continue

        kept_lines: list[tuple[str, bool]] = []
        for line in block.get("lines", []):
            line_bbox = tuple(line["bbox"])
            if _line_is_in_table(line_bbox, table_bboxes):
                continue

            spans = line.get("spans", [])
            line_text = _normalize_text("".join(span.get("text", "") for span in spans))
            if not line_text or allowed_lines[line_text] <= 0:
                continue

            allowed_lines[line_text] -= 1
            line_is_bold = any(
                "bold" in span.get("font", "").lower()
                or bool(span.get("flags", 0) & 16)
                for span in spans
            )
            kept_lines.append((line_text, line_is_bold))

        if kept_lines:
            bbox = tuple(block["bbox"])
            segments: list[tuple[str, bool]] = []
            current_lines: list[str] = []
            current_is_bold = False

            for line_text, line_is_bold in kept_lines:
                starts_structure = bool(
                    _ARTICLE_RE.match(line_text)
                    or _HEADING_RE.match(line_text)
                    or _PARAGRAPH_RE.match(line_text)
                    or _INCISO_RE.match(line_text)
                )
                if starts_structure and current_lines:
                    segments.append((" ".join(current_lines), current_is_bold))
                    current_lines = []
                    current_is_bold = False

                current_lines.append(line_text)
                current_is_bold = current_is_bold or line_is_bold

            if current_lines:
                segments.append((" ".join(current_lines), current_is_bold))

            entries.append(
                (
                    bbox[1],
                    bbox[0],
                    "text",
                    {"segments": segments, "bbox": bbox},
                )
            )

    entries.sort(key=lambda entry: (entry[0], entry[1]))
    units: list[StructuralUnit] = []
    article_active = False

    for source_order, (_, _, entry_type, value) in enumerate(entries):
        if entry_type == "table":
            headers = value["headers"]
            units.append(
                StructuralUnit(
                    kind="table",
                    text="[TABELA]\nColunas: " + " | ".join(headers),
                    source_order=source_order,
                    page=page_number,
                    bbox=value["bbox"],
                    table_id=value["table_id"],
                    table_headers=headers,
                )
            )
            for row_index, row in enumerate(value["rows"]):
                units.append(
                    StructuralUnit(
                        kind="table_row",
                        text=_serialize_table_row(headers, row),
                        source_order=source_order,
                        page=page_number,
                        bbox=value["bbox"],
                        table_id=value["table_id"],
                        row_index=row_index,
                        table_headers=headers,
                    )
                )
            continue

        for text, is_bold in value["segments"]:
            kind = _classify_text_kind(text, is_bold, article_active)
            units.append(
                StructuralUnit(
                    kind=kind,
                    text=text,
                    source_order=source_order,
                    page=page_number,
                    bbox=value["bbox"],
                )
            )
            if kind == "heading":
                article_active = False
            elif kind == "article":
                article_active = True

    # Quando o modo ``dict`` não reproduz uma linha normal selecionada pela
    # limpeza existente, volta-se ao fallback plano da página. Assim, uma
    # detecção estrutural parcial nunca pode remover conteúdo do corpus.
    normalized_table_content = " ".join(table_content)
    unmatched_non_table_lines = [
        line
        for line, count in allowed_lines.items()
        if count > 0 and line not in normalized_table_content
    ]
    if unmatched_non_table_lines:
        return []

    for source_order, unit in enumerate(units):
        unit.source_order = source_order

    _apply_structure_context(units)
    return units


def _article_context(section_path: str | None) -> str | None:
    if not section_path:
        return None
    label = section_path.split(" > ")[-1]
    return label if _ARTICLE_RE.match(label) else None


def _safe_chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    if chunk_size < MIN_CHUNK_SIZE:
        return [text]
    safe_overlap = min(overlap, chunk_size - 1)
    return chunk_text(text, chunk_size=chunk_size, overlap=safe_overlap)


def _split_prose_unit(
    unit: StructuralUnit,
    chunk_size: int,
    overlap: int,
) -> list[PackedChunk]:
    context = _article_context(unit.section_path)
    prefix = ""
    if context and not _normalize_text(unit.text).startswith(context):
        prefix = context + "\n\n"

    available = max(MIN_CHUNK_SIZE, chunk_size - len(prefix))
    pieces = _safe_chunk_text(unit.text, available, overlap)
    return [
        PackedChunk(prefix + piece, unit.section_path)
        for piece in pieces
    ]


def _table_chunk_text(header: str, rows: list[str]) -> str:
    body = "\n\n".join(rows)
    if body:
        return f"{header}\n\n{body}\n[/TABELA]"
    return f"{header}\n[/TABELA]"


def _split_table_row(
    header: str,
    row: StructuralUnit,
    chunk_size: int,
    overlap: int,
    section_path: str | None,
) -> list[PackedChunk]:
    identity, separator, body = row.text.partition("\n")
    row_prefix = identity + ("\n" if separator else "")
    wrapper_size = len(header) + len(row_prefix) + len("\n\n\n[/TABELA]")
    available = max(MIN_CHUNK_SIZE, chunk_size - wrapper_size)
    pieces = _safe_chunk_text(body or row.text, available, overlap)
    return [
        PackedChunk(
            _table_chunk_text(header, [row_prefix + piece]),
            section_path,
        )
        for piece in pieces
    ]


def _pack_table(
    table: StructuralUnit,
    rows: list[StructuralUnit],
    chunk_size: int,
    overlap: int,
    prefix: str = "",
    caption: str = "",
) -> list[PackedChunk]:
    header = table.text
    if prefix:
        header = prefix + "\n\n" + header

    packed: list[PackedChunk] = []
    current_rows: list[str] = []

    for row in rows:
        candidate = _table_chunk_text(header, current_rows + [row.text])
        if len(candidate) <= chunk_size:
            current_rows.append(row.text)
            continue

        if current_rows:
            packed.append(PackedChunk(_table_chunk_text(header, current_rows), table.section_path))
            current_rows = []

        single = _table_chunk_text(header, [row.text])
        if len(single) <= chunk_size:
            current_rows.append(row.text)
        else:
            packed.extend(_split_table_row(header, row, chunk_size, overlap, table.section_path))

    if current_rows or not packed:
        packed.append(PackedChunk(_table_chunk_text(header, current_rows), table.section_path))

    if caption:
        last_chunk = packed[-1]
        last_chunk.text = last_chunk.text.replace(
            "\n[/TABELA]",
            f"\nNota: {caption}\n[/TABELA]",
        )

    return packed


def pack_structured_units(
    units: list[StructuralUnit],
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[PackedChunk]:
    """Agrupa unidades sem cruzar linhas de tabela ou fronteiras explícitas."""
    if not units:
        return []

    packed: list[PackedChunk] = []
    current_parts: list[str] = []
    current_path: str | None = None
    pending_heading: StructuralUnit | None = None

    def flush() -> None:
        nonlocal current_parts, current_path
        if current_parts:
            packed.append(PackedChunk("\n\n".join(current_parts), current_path))
        current_parts = []
        current_path = None

    index = 0
    while index < len(units):
        unit = units[index]

        if unit.kind == "heading":
            flush()
            pending_heading = unit
            index += 1
            continue

        if unit.kind == "table":
            flush()
            rows: list[StructuralUnit] = []
            next_index = index + 1
            while (
                next_index < len(units)
                and units[next_index].kind == "table_row"
                and units[next_index].table_id == unit.table_id
            ):
                rows.append(units[next_index])
                next_index += 1

            caption = ""
            if (
                next_index < len(units)
                and units[next_index].kind == "text"
                and re.match(r"^\s*\(?table\b", units[next_index].text, re.IGNORECASE)
            ):
                caption = units[next_index].text
                next_index += 1

            table_prefix = pending_heading.text if pending_heading else ""
            pending_heading = None
            packed.extend(
                _pack_table(
                    unit,
                    rows,
                    chunk_size,
                    overlap,
                    table_prefix,
                    caption,
                )
            )
            index = next_index
            continue

        content = unit.text
        path = unit.section_path
        if pending_heading:
            content = pending_heading.text + "\n\n" + content
            path = path or pending_heading.section_path
            pending_heading = None

        if current_parts and path != current_path:
            flush()

        if not current_parts:
            article_context = _article_context(path)
            if (
                article_context
                and unit.kind != "article"
                and not _normalize_text(content).startswith(article_context)
            ):
                content = article_context + "\n\n" + content
            current_path = path

        candidate = "\n\n".join(current_parts + [content])
        if len(candidate) <= chunk_size:
            current_parts.append(content)
            index += 1
            continue

        if current_parts:
            flush()
            continue

        split_unit = StructuralUnit(
            kind=unit.kind,
            text=content,
            source_order=unit.source_order,
            page=unit.page,
            section_path=path,
        )
        packed.extend(_split_prose_unit(split_unit, chunk_size, overlap))
        index += 1

    if pending_heading:
        packed.append(PackedChunk(pending_heading.text, pending_heading.section_path))
    flush()
    return packed
