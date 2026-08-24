"""Representação contextual determinística usada exclusivamente em embeddings."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Mapping


EMBEDDING_TEXT_PROFILE = "context-v1"


def _nonempty_text(value: object) -> str:
    """Converte valores textuais válidos sem produzir placeholders."""
    if value is None:
        return ""
    return str(value).strip()


def build_embedding_text(record: dict) -> str:
    """Prefixa o texto original com o contexto disponível, sem mutar o record."""
    metadata = record.get("metadata") or {}
    text = record.get("text", "")
    if text is None:
        text = ""
    elif not isinstance(text, str):
        text = str(text)

    document = next(
        (
            value
            for value in (
                _nonempty_text(metadata.get("title")),
                _nonempty_text(metadata.get("document_id")),
                _nonempty_text(metadata.get("source")),
            )
            if value
        ),
        "",
    )
    issuer = next(
        (
            value
            for value in (
                _nonempty_text(metadata.get("issuer")),
                _nonempty_text(metadata.get("author")),
            )
            if value
        ),
        "",
    )
    section = _nonempty_text(metadata.get("section_path"))
    context_parts = [part for part in (document, issuer, section) if part]

    if not context_parts:
        return text

    return f"[{' · '.join(context_parts)}]\n{text}"


def corpus_embedding_fingerprint(records: Iterable[dict]) -> str:
    """Hash estável do perfil, IDs e entradas contextuais na ordem fornecida."""
    digest = hashlib.sha256()
    digest.update(EMBEDDING_TEXT_PROFILE.encode("utf-8"))
    digest.update(b"\0")

    for record in records:
        chunk_id = _nonempty_text(record.get("id"))
        embedding_text = build_embedding_text(record)
        for value in (chunk_id, embedding_text):
            encoded = value.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)

    return digest.hexdigest()


def bge_cache_filename(model_key: str, fingerprint: str) -> str:
    """Nome versionado para evitar colisão com caches de texto bruto."""
    safe_model_key = re.sub(r"[^a-zA-Z0-9_.-]+", "_", model_key).strip("_")
    return (
        f"dense_{safe_model_key}_{EMBEDDING_TEXT_PROFILE}_"
        f"{fingerprint[:16]}.npz"
    )


def _cache_value(metadata: Mapping[str, object], key: str) -> str:
    """Lê valores escalares de dicts normais ou arrays escalares do NumPy."""
    value = metadata.get(key)
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return _nonempty_text(value)


def validate_embedding_cache_metadata(
    metadata: Mapping[str, object],
    expected_fingerprint: str,
    expected_model: str | None = None,
) -> None:
    """Rejeita explicitamente cache de outro perfil ou de outro corpus."""
    profile = _cache_value(metadata, "embedding_text_profile")
    if profile != EMBEDDING_TEXT_PROFILE:
        raise ValueError(
            "Cache incompatível: perfil de embedding ausente ou diferente de "
            f"{EMBEDDING_TEXT_PROFILE!r}."
        )

    fingerprint = _cache_value(metadata, "embedding_text_fingerprint")
    if fingerprint != expected_fingerprint:
        raise ValueError(
            "Cache incompatível: fingerprint do conteúdo contextual não confere."
        )

    if expected_model is not None:
        model = _cache_value(metadata, "embedding_model")
        if model != expected_model:
            raise ValueError(
                "Cache incompatível: modelo de embedding ausente ou diferente de "
                f"{expected_model!r}."
            )
