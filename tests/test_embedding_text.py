"""Testes da representação contextual usada somente para embeddings."""

from __future__ import annotations

import importlib
import importlib.util
import sys
import tempfile
import types
import unittest
from copy import deepcopy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _record(text: str = "Conteúdo original.", **metadata: object) -> dict:
    return {"id": "doc_p1_c0", "text": text, "metadata": metadata}


class EmbeddingTextTests(unittest.TestCase):
    def _module(self):
        return importlib.import_module("embedding_text")

    def test_contextual_header_uses_title_author_and_section(self) -> None:
        record = _record(
            title="Código de Ética Médica",
            author="CFM",
            section_path="CAPÍTULO IX",
        )

        self.assertEqual(
            self._module().build_embedding_text(record),
            "[Código de Ética Médica · CFM · CAPÍTULO IX]\n"
            "Conteúdo original.",
        )

    def test_fallbacks_omit_absent_context_values(self) -> None:
        module = self._module()

        self.assertEqual(
            module.build_embedding_text(_record(title="Título", author="CFM")),
            "[Título · CFM]\nConteúdo original.",
        )
        self.assertEqual(
            module.build_embedding_text(_record(section_path="CAPÍTULO IX")),
            "[CAPÍTULO IX]\nConteúdo original.",
        )
        self.assertEqual(
            module.build_embedding_text(_record()),
            "Conteúdo original.",
        )

    def test_issuer_has_priority_and_text_is_not_mutated(self) -> None:
        record = _record(
            title="Título",
            issuer="Emissor oficial",
            author="Autor secundário",
        )
        original = deepcopy(record)

        result = self._module().build_embedding_text(record)

        self.assertEqual(result, "[Título · Emissor oficial]\nConteúdo original.")
        self.assertEqual(record, original)
        self.assertNotIn("None", result)
        self.assertNotIn("· ]", result)

    def test_context_fingerprint_changes_with_all_relevant_inputs(self) -> None:
        module = self._module()
        base = _record(title="Título", author="CFM", section_path="I")
        same = deepcopy(base)
        changed_text = deepcopy(base); changed_text["text"] = "Outro conteúdo."
        changed_title = deepcopy(base); changed_title["metadata"]["title"] = "Outro"
        changed_author = deepcopy(base); changed_author["metadata"]["author"] = "OMS"
        changed_section = deepcopy(base); changed_section["metadata"]["section_path"] = "II"
        issuer_base = _record(title="Título", issuer="CFM")
        issuer_changed = deepcopy(issuer_base)
        issuer_changed["metadata"]["issuer"] = "OMS"

        fingerprint = module.corpus_embedding_fingerprint([base])
        self.assertEqual(fingerprint, module.corpus_embedding_fingerprint([same]))
        for changed in (
            changed_text,
            changed_title,
            changed_author,
            changed_section,
        ):
            self.assertNotEqual(
                fingerprint,
                module.corpus_embedding_fingerprint([changed]),
            )
        self.assertNotEqual(
            module.corpus_embedding_fingerprint([issuer_base]),
            module.corpus_embedding_fingerprint([issuer_changed]),
        )

    def test_cache_metadata_requires_context_profile_and_fingerprint(self) -> None:
        module = self._module()
        fingerprint = module.corpus_embedding_fingerprint([_record()])

        with self.assertRaisesRegex(ValueError, "perfil"):
            module.validate_embedding_cache_metadata({}, fingerprint)
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            module.validate_embedding_cache_metadata(
                {"embedding_text_profile": module.EMBEDDING_TEXT_PROFILE},
                fingerprint,
            )

        module.validate_embedding_cache_metadata(
            {
                "embedding_text_profile": module.EMBEDDING_TEXT_PROFILE,
                "embedding_text_fingerprint": fingerprint,
            },
            fingerprint,
        )
        with self.assertRaisesRegex(ValueError, "modelo"):
            module.validate_embedding_cache_metadata(
                {
                    "embedding_text_profile": module.EMBEDDING_TEXT_PROFILE,
                    "embedding_text_fingerprint": fingerprint,
                },
                fingerprint,
                "BAAI/bge-m3",
            )


class EmbeddingIntegrationTests(unittest.TestCase):
    @staticmethod
    def _install_build_stubs() -> None:
        numpy = sys.modules.get("numpy")
        if numpy is None:
            numpy = types.ModuleType("numpy")
            sys.modules["numpy"] = numpy
        numpy.ndarray = getattr(numpy, "ndarray", object)
        numpy.array = getattr(numpy, "array", lambda value: value)
        numpy.linalg = getattr(
            numpy,
            "linalg",
            types.SimpleNamespace(norm=lambda _value: 1),
        )
        if "tqdm" not in sys.modules:
            tqdm = types.ModuleType("tqdm")
            tqdm.tqdm = lambda values, **_kwargs: values
            sys.modules["tqdm"] = tqdm
        if "dotenv" not in sys.modules:
            dotenv = types.ModuleType("dotenv")
            dotenv.load_dotenv = lambda *_args, **_kwargs: None
            sys.modules["dotenv"] = dotenv
        if "google" not in sys.modules:
            google = types.ModuleType("google")
            google.genai = types.ModuleType("google.genai")
            google.genai.types = types.ModuleType("google.genai.types")
            sys.modules["google"] = google
            sys.modules["google.genai"] = google.genai
            sys.modules["google.genai.types"] = google.genai.types
        if "qdrant_client" not in sys.modules:
            qdrant = types.ModuleType("qdrant_client")
            qdrant.QdrantClient = object
            qdrant_http = types.ModuleType("qdrant_client.http")
            qdrant_http.models = types.SimpleNamespace()
            sys.modules["qdrant_client"] = qdrant
            sys.modules["qdrant_client.http"] = qdrant_http

    def test_gemini_input_and_payload_keep_separate_texts(self) -> None:
        self._install_build_stubs()
        vectorstore = importlib.import_module("build_vectorstore")
        record = _record(
            title="Título",
            author="CFM",
            section_path="CAPÍTULO IX",
        )

        self.assertEqual(
            vectorstore.build_embedding_inputs([record]),
            ["[Título · CFM · CAPÍTULO IX]\nConteúdo original."],
        )
        payload = vectorstore.build_payload(record)
        self.assertEqual(payload["texto"], "Conteúdo original.")
        self.assertEqual(payload["section_path"], "CAPÍTULO IX")
        self.assertEqual(
            payload["embedding_text_profile"],
            "context-v1",
        )
        self.assertNotIn("embedding_text", payload)

    def test_contextual_backup_is_distinct_and_rejects_legacy_records(self) -> None:
        self._install_build_stubs()
        vectorstore = importlib.import_module("build_vectorstore")
        self.assertNotEqual(
            vectorstore.BACKUP_FILE,
            vectorstore.LEGACY_BACKUP_FILE,
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            previous = vectorstore.BACKUP_FILE
            vectorstore.BACKUP_FILE = Path(temporary_directory) / "backup.jsonl"
            vectorstore.BACKUP_FILE.write_text(
                '{"document": {"id": "doc_p1_c0"}, "vector": [0.0]}\n',
                encoding="utf-8",
            )
            try:
                with self.assertRaisesRegex(ValueError, "perfil"):
                    vectorstore.load_backup([_record()])
            finally:
                vectorstore.BACKUP_FILE = previous

    def test_bge_document_input_uses_shared_helper_and_query_is_unchanged(self) -> None:
        self._install_build_stubs()
        if "tqdm" not in sys.modules:
            tqdm = types.ModuleType("tqdm")
            tqdm.tqdm = lambda values, **_kwargs: values
            sys.modules["tqdm"] = tqdm

        retrievers = importlib.import_module("eval.lib.retrievers")
        embedders = importlib.import_module("eval.lib.embedders")
        record = _record(title="Título", author="CFM")
        corpus = {record["id"]: record}

        self.assertEqual(
            retrievers.contextual_document_texts(corpus, [record["id"]]),
            ["[Título · CFM]\nConteúdo original."],
        )

        captured = []
        embedder = object.__new__(embedders.SentenceTransformerEmbedder)
        embedder.query_instruction = ""
        embedder._encode = lambda texts: captured.append(texts) or ["vector"]
        self.assertEqual(embedder.embed_query("Consulta original"), "vector")
        self.assertEqual(captured, [["Consulta original"]])


if __name__ == "__main__":
    unittest.main()
