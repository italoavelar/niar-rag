"""Chunking compartilhado pelos pipelines de extração PDF e HTML."""

from __future__ import annotations

import re


CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
MIN_CHUNK_SIZE = 120

# Limita cortes naturais ao fim da janela nominal. Com os valores padrão,
# chunks normais terão entre 1000 e 1200 caracteres antes do overlap.
SPLIT_SEARCH_BACKTRACK = CHUNK_OVERLAP


def clean_text(text: str) -> str:
    """Limpeza final do texto já estruturado."""
    text = text.replace("\u00a0", " ")
    text = text.replace("\t", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _validate_chunk_config(
    chunk_size: int,
    overlap: int,
) -> None:
    if chunk_size <= 0:
        raise ValueError("chunk_size deve ser maior que zero.")

    if overlap < 0:
        raise ValueError("overlap não pode ser negativo.")

    if overlap >= chunk_size:
        raise ValueError("overlap deve ser menor que chunk_size.")


def choose_split_point(
    text: str,
    start: int,
    raw_end: int,
    min_size: int,
    search_backtrack: int = SPLIT_SEARCH_BACKTRACK,
) -> int:
    """Escolhe uma quebra natural no fim da janela nominal do chunk.

    A busca não percorre toda a janela: uma pontuação pouco depois de
    ``min_size`` não pode encurtar um chunk de 1200 caracteres para cerca de
    200 e desencadear uma sequência de overlaps quase completos.
    """
    if raw_end >= len(text):
        return len(text)

    min_end = start + min_size
    preferred_start = max(
        min_end,
        raw_end - search_backtrack,
    )

    punctuation_candidates = [
        text.rfind(". ", preferred_start, raw_end),
        text.rfind("; ", preferred_start, raw_end),
        text.rfind(": ", preferred_start, raw_end),
        text.rfind("? ", preferred_start, raw_end),
        text.rfind("! ", preferred_start, raw_end),
    ]

    best_punctuation = max(punctuation_candidates)

    if best_punctuation >= preferred_start:
        return best_punctuation + 1

    whitespace = text.rfind(" ", preferred_start, raw_end)

    if whitespace >= preferred_start:
        return whitespace

    # Uma palavra excepcionalmente longa pode não conter espaço na faixa
    # preferida. Cortar no limite nominal preserva a cobertura e o progresso.
    return raw_end


def _chunk_spans(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
    min_chunk_size: int = MIN_CHUNK_SIZE,
) -> list[tuple[int, int]]:
    """Retorna spans contínuos sobre texto já normalizado.

    Esta função interna permite testar diretamente cobertura, overlap e
    progresso sem alterar a API pública de ``chunk_text``.
    """
    _validate_chunk_config(chunk_size, overlap)

    if not text:
        return []

    if len(text) <= chunk_size:
        return [(0, len(text))] if len(text) >= min_chunk_size else []

    spans = []
    start = 0
    minimum_progress = max(
        1,
        min_chunk_size,
        chunk_size - (2 * overlap),
    )

    while start < len(text):
        raw_end = min(start + chunk_size, len(text))
        end = choose_split_point(
            text=text,
            start=start,
            raw_end=raw_end,
            min_size=min_chunk_size,
            search_backtrack=overlap,
        )

        if end <= start:
            raise RuntimeError("O chunker não conseguiu avançar no texto.")

        spans.append((start, end))

        if end >= len(text):
            break

        # Em condições normais, o corte ocorre nos últimos 200 caracteres da
        # janela: o avanço fica entre 800 e 1000 e o overlap permanece 200.
        # O limite defensivo impede que qualquer futura mudança no seletor de
        # corte reintroduza strides de 1--3 caracteres. O clamp em ``end``
        # garante que essa defesa jamais salte conteúdo.
        new_start = max(
            end - overlap,
            start + minimum_progress,
        )
        new_start = min(new_start, end)

        if new_start <= start:
            raise RuntimeError("O chunker entrou em progresso não positivo.")

        start = new_start

    return spans


def _merge_small_final_chunk(
    chunks: list[str],
    min_chunk_size: int,
) -> list[str]:
    """Anexa um fragmento final muito pequeno ao chunk precedente."""
    if len(chunks) < 2 or len(chunks[-1]) >= min_chunk_size:
        return chunks

    merged = list(chunks)
    merged[-2] = f"{merged[-2]} {merged[-1]}".strip()
    merged.pop()
    return merged


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
    min_chunk_size: int = MIN_CHUNK_SIZE,
) -> list[str]:
    """Divide texto em chunks sobrepostos, sem perder cobertura."""
    text = clean_text(text)
    spans = _chunk_spans(
        text=text,
        chunk_size=chunk_size,
        overlap=overlap,
        min_chunk_size=min_chunk_size,
    )

    chunks = [text[start:end].strip() for start, end in spans]
    return _merge_small_final_chunk(chunks, min_chunk_size)
