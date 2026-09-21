"""Testes do pipeline isolado de indexação Qwen3-Embedding."""

from __future__ import annotations

import importlib
import json
import os
import sys
import tempfile
import types
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _record(text: str = "Conteúdo original.", **metadata: object) -> dict:
    return {
        "id": "documento_p1_c0",
        "text": text,
        "metadata": {
            "title": "Documento de teste",
            "author": "Instituição de teste",
            **metadata,
        },
    }


class FakeQwenModel:
    def __init__(self) -> None:
        self.calls = []

    def encode(self, texts, **kwargs):
        self.calls.append((list(texts), kwargs))
        return [[0.0] * 1024 for _ in texts]


class QwenVectorstoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = importlib.import_module("build_qwen_vectorstore")

    def _backup_record(
        self,
        source_document: dict,
        corpus_fingerprint: str | None = None,
        **overrides: object,
    ) -> dict:
        module = self.module
        corpus_fingerprint = (
            corpus_fingerprint
            or module.corpus_embedding_fingerprint([source_document])
        )
        record = {
            "embedding_model": module.QWEN_EMBEDDING_MODEL,
            "embedding_dimension": module.QWEN_EMBEDDING_DIM,
            "embedding_text_profile": module.EMBEDDING_TEXT_PROFILE,
            "embedding_text_fingerprint": module.corpus_embedding_fingerprint(
                [source_document]
            ),
            "corpus_embedding_fingerprint": corpus_fingerprint,
            "document": source_document,
            "vector": [0.0] * module.QWEN_EMBEDDING_DIM,
        }
        record.update(overrides)
        return record

    def test_default_qwen_profile_uses_0_6b_isolated_resources(self) -> None:
        module = self.module

        self.assertEqual(module.QWEN_EMBEDDING_MODEL, "Qwen/Qwen3-Embedding-0.6B")
        self.assertEqual(module.QWEN_EMBEDDING_DIM, 1024)
        self.assertEqual(module.BATCH_SIZE_EMBEDDINGS, 2)
        self.assertEqual(
            module.collection_name_for_fingerprint("a" * 64),
            "niar_rag_documents_qwen3_0_6b_context_v1_aaaaaaaaaaaaaaaa",
        )
        self.assertEqual(
            module.DEFAULT_BACKUP_FILE,
            Path("data/processed/embeddings_qwen3_0_6b_context_v1_backup.jsonl"),
        )

    def test_documents_use_contextual_text_without_mutating_record(self) -> None:
        document = _record(section_path="CAPÍTULO IX")
        original = deepcopy(document)
        model = FakeQwenModel()

        vectors = self.module.embed_documents(model, [document])

        self.assertEqual(len(vectors), 1)
        self.assertEqual(document, original)
        self.assertEqual(
            model.calls,
            [(
                [
                    "[Documento de teste · Instituição de teste · CAPÍTULO IX]\n"
                    "Conteúdo original."
                ],
                {"normalize_embeddings": True},
            )],
        )

    def test_query_uses_official_prompt_without_contextual_header(self) -> None:
        model = FakeQwenModel()

        vector = self.module.embed_query(model, "Qual é a regra aplicável?")

        self.assertEqual(len(vector), self.module.QWEN_EMBEDDING_DIM)
        self.assertEqual(
            model.calls,
            [(
                ["Qual é a regra aplicável?"],
                {"prompt_name": "query", "normalize_embeddings": True},
            )],
        )

    def test_payload_preserves_original_text_and_qwen_metadata(self) -> None:
        document = _record(section_path="CAPÍTULO IX", issuer="Emissor oficial")

        payload = self.module.build_payload(document, "f" * 64)

        self.assertEqual(payload["texto"], "Conteúdo original.")
        self.assertEqual(payload["issuer"], "Emissor oficial")
        self.assertEqual(payload["section_path"], "CAPÍTULO IX")
        self.assertEqual(payload["embedding_model"], "Qwen/Qwen3-Embedding-0.6B")
        self.assertEqual(payload["embedding_dimension"], 1024)
        self.assertEqual(payload["embedding_text_profile"], "context-v1")
        self.assertEqual(payload["corpus_embedding_fingerprint"], "f" * 64)
        self.assertNotIn("embedding_text", payload)

    def test_backup_path_is_new_and_legacy_paths_are_rejected(self) -> None:
        module = self.module
        self.assertNotIn(
            module.DEFAULT_BACKUP_FILE.name,
            {path.name for path in module.PROTECTED_BACKUP_FILES},
        )

        for protected in module.PROTECTED_BACKUP_FILES:
            with self.subTest(path=protected):
                with self.assertRaisesRegex(ValueError, "legado"):
                    module.resolve_backup_file(str(protected))

    def test_legacy_collection_is_rejected_without_connecting(self) -> None:
        module = self.module

        self.assertEqual(
            module.validate_collection_name(None),
            module.COLLECTION_NAME_PREFIX,
        )
        with self.assertRaisesRegex(ValueError, "niar_rag_documents"):
            module.validate_collection_name("niar_rag_documents")

    def test_same_legacy_and_qwen_urls_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "ambiente separado"):
            self.module.validate_qdrant_environment_urls(
                "https://cluster.qdrant.io",
                "https://cluster.qdrant.io",
            )

    def test_urls_that_differ_only_by_trailing_slash_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "ambiente separado"):
            self.module.validate_qdrant_environment_urls(
                "https://cluster.qdrant.io/",
                "https://cluster.qdrant.io",
            )

    def test_different_qdrant_urls_are_accepted(self) -> None:
        self.module.validate_qdrant_environment_urls(
            "https://qwen.qdrant.io/",
            "https://legacy.qdrant.io",
        )

    def test_absent_legacy_qdrant_url_does_not_block_qwen(self) -> None:
        previous = os.environ.pop("QDRANT_URL", None)
        try:
            self.module.validate_qdrant_environment_urls("https://qwen.qdrant.io")
        finally:
            if previous is not None:
                os.environ["QDRANT_URL"] = previous

    def test_existing_qwen_collection_fails_without_recreation(self) -> None:
        module = self.module

        class FakeQdrantClient:
            recreated = False

            def __init__(self, **_kwargs) -> None:
                pass

            def get_collections(self):
                return types.SimpleNamespace(
                    collections=[types.SimpleNamespace(name=module.COLLECTION_NAME_PREFIX)]
                )

        qdrant = types.ModuleType("qdrant_client")
        qdrant.QdrantClient = FakeQdrantClient
        qdrant.models = types.SimpleNamespace()
        tqdm = types.ModuleType("tqdm")
        tqdm.tqdm = lambda values, **_kwargs: values
        previous_qdrant = sys.modules.get("qdrant_client")
        previous_tqdm = sys.modules.get("tqdm")
        old_url = os.environ.get("QDRANT_QWEN_URL")
        old_key = os.environ.get("QDRANT_QWEN_API_KEY")
        sys.modules["qdrant_client"] = qdrant
        sys.modules["tqdm"] = tqdm
        os.environ["QDRANT_QWEN_URL"] = "https://test.invalid"
        os.environ["QDRANT_QWEN_API_KEY"] = "test-key"
        try:
            with self.assertRaisesRegex(ValueError, "já existe"):
                module.upload_to_qdrant(
                    [], corpus_fingerprint="f" * 64
                )
        finally:
            if previous_qdrant is None:
                del sys.modules["qdrant_client"]
            else:
                sys.modules["qdrant_client"] = previous_qdrant
            if previous_tqdm is None:
                del sys.modules["tqdm"]
            else:
                sys.modules["tqdm"] = previous_tqdm
            if old_url is None:
                del os.environ["QDRANT_QWEN_URL"]
            else:
                os.environ["QDRANT_QWEN_URL"] = old_url
            if old_key is None:
                del os.environ["QDRANT_QWEN_API_KEY"]
            else:
                os.environ["QDRANT_QWEN_API_KEY"] = old_key

    def test_backup_rejects_incompatible_fingerprint_model_dimension_and_document(self) -> None:
        module = self.module
        document = _record()

        invalid_records = [
            self._backup_record(document, embedding_text_fingerprint="incompatível"),
            self._backup_record(document, embedding_model="outro-modelo"),
            self._backup_record(
                document,
                embedding_dimension=module.QWEN_EMBEDDING_DIM + 1,
            ),
            self._backup_record(
                document,
                document={**document, "id": "outro_p1_c0"},
            ),
        ]

        for invalid in invalid_records:
            with self.subTest(invalid=invalid["embedding_model"]):
                with tempfile.TemporaryDirectory() as temporary_directory:
                    backup = Path(temporary_directory) / "qwen-backup.jsonl"
                    backup.write_text(json.dumps(invalid) + "\n", encoding="utf-8")
                    with self.assertRaises(ValueError):
                        module.load_backup(
                            [document],
                            backup,
                            module.corpus_embedding_fingerprint([document]),
                        )

    def test_backup_validates_context_profile(self) -> None:
        module = self.module
        document = _record()
        invalid = self._backup_record(document, embedding_text_profile="raw-text")

        with tempfile.TemporaryDirectory() as temporary_directory:
            backup = Path(temporary_directory) / "qwen-backup.jsonl"
            backup.write_text(json.dumps(invalid) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "perfil"):
                module.load_backup(
                    [document],
                    backup,
                    module.corpus_embedding_fingerprint([document]),
                )

    def test_backup_rejects_other_global_corpus_fingerprint(self) -> None:
        module = self.module
        document = _record()
        backup_record = self._backup_record(
            document, corpus_fingerprint="outro-corpus"
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            backup = Path(temporary_directory) / "qwen-backup.jsonl"
            backup.write_text(json.dumps(backup_record) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "fingerprint global"):
                module.load_backup(
                    [document],
                    backup,
                    module.corpus_embedding_fingerprint([document]),
                )

    def test_generate_embeddings_records_global_fingerprint(self) -> None:
        module = self.module
        document = _record()
        fingerprint = module.corpus_embedding_fingerprint([document])

        with tempfile.TemporaryDirectory() as temporary_directory:
            backup = Path(temporary_directory) / "qwen-backup.jsonl"
            module.generate_embeddings(
                [document], [], backup, fingerprint, model=FakeQwenModel()
            )
            saved = json.loads(backup.read_text(encoding="utf-8"))
            self.assertEqual(saved["corpus_embedding_fingerprint"], fingerprint)
            self.assertEqual(saved["document"], document)

    def test_upload_only_rejects_incomplete_backup_before_connecting(self) -> None:
        module = self.module
        documents = [_record(), {**_record(), "id": "documento_p2_c0"}]
        fingerprint = module.corpus_embedding_fingerprint(documents)
        incomplete = self._backup_record(documents[0], fingerprint)

        with tempfile.TemporaryDirectory() as temporary_directory:
            backup = Path(temporary_directory) / "qwen-backup.jsonl"
            backup.write_text(json.dumps(incomplete) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "incompleto"):
                module.load_backup(
                    documents, backup, fingerprint, require_complete=True
                )

    def test_generate_only_does_not_call_qdrant(self) -> None:
        module = self.module
        document = _record()
        fingerprint = module.corpus_embedding_fingerprint([document])
        valid = self._backup_record(document, fingerprint)
        dotenv = types.ModuleType("dotenv")
        dotenv.load_dotenv = lambda: None

        with patch.dict(sys.modules, {"dotenv": dotenv}), patch.object(
            module, "load_documents", return_value=[document]
        ), patch.object(module, "resolve_backup_file", return_value=Path("fake")), patch.object(
            module, "load_backup", side_effect=[[], [valid]]
        ), patch.object(module, "generate_embeddings") as generate, patch.object(
            module, "upload_to_qdrant"
        ) as upload:
            module.build_qwen_vectorstore("generate-only")

        generate.assert_called_once()
        upload.assert_not_called()

    def test_upload_only_does_not_load_model(self) -> None:
        module = self.module
        document = _record()
        fingerprint = module.corpus_embedding_fingerprint([document])
        valid = self._backup_record(document, fingerprint)
        dotenv = types.ModuleType("dotenv")
        dotenv.load_dotenv = lambda: None

        with patch.dict(sys.modules, {"dotenv": dotenv}), patch.object(
            module, "load_documents", return_value=[document]
        ), patch.object(module, "resolve_backup_file", return_value=Path("fake")), patch.object(
            module, "load_backup", return_value=[valid]
        ), patch.object(module, "load_qwen_model", side_effect=AssertionError), patch.object(
            module, "upload_to_qdrant"
        ) as upload:
            module.build_qwen_vectorstore("upload-only")

        upload.assert_called_once()

    def _verify_with_points(self, documents: list[dict], points: list[dict]) -> None:
        module = self.module

        class FakeQdrantClient:
            calls = []

            def __init__(self, **_kwargs) -> None:
                pass

            def get_collections(self):
                return types.SimpleNamespace(
                    collections=[types.SimpleNamespace(name="qwen-test")]
                )

            def get_collection(self, _name):
                return types.SimpleNamespace(
                    config=types.SimpleNamespace(
                        params=types.SimpleNamespace(
                            vectors=types.SimpleNamespace(size=1024)
                        )
                    )
                )

            def count(self, _name, exact):
                self.calls.append(("count", exact))
                return types.SimpleNamespace(count=len(points))

            def scroll(self, _name, **_kwargs):
                self.calls.append(("scroll",))
                return [types.SimpleNamespace(payload=point) for point in points], None

            def create_collection(self, *_args, **_kwargs):
                raise AssertionError("verify não pode criar coleção")

            def recreate_collection(self, *_args, **_kwargs):
                raise AssertionError("verify não pode recriar coleção")

            def upsert(self, *_args, **_kwargs):
                raise AssertionError("verify não pode fazer upload")

        qdrant = types.ModuleType("qdrant_client")
        qdrant.QdrantClient = FakeQdrantClient
        old_url = os.environ.pop("QDRANT_URL", None)
        old_qwen_url = os.environ.get("QDRANT_QWEN_URL")
        old_qwen_key = os.environ.get("QDRANT_QWEN_API_KEY")
        os.environ["QDRANT_QWEN_URL"] = "https://qwen.invalid"
        os.environ["QDRANT_QWEN_API_KEY"] = "test-key"
        try:
            with patch.dict(sys.modules, {"qdrant_client": qdrant}):
                module.verify_qdrant_collection(
                    documents,
                    module.corpus_embedding_fingerprint(documents),
                    "qwen-test",
                )
        finally:
            if old_url is not None:
                os.environ["QDRANT_URL"] = old_url
            if old_qwen_url is None:
                del os.environ["QDRANT_QWEN_URL"]
            else:
                os.environ["QDRANT_QWEN_URL"] = old_qwen_url
            if old_qwen_key is None:
                del os.environ["QDRANT_QWEN_API_KEY"]
            else:
                os.environ["QDRANT_QWEN_API_KEY"] = old_qwen_key

    def test_verify_is_read_only_and_accepts_exact_ids(self) -> None:
        documents = [_record(), {**_record(), "id": "documento_p2_c0"}]
        fingerprint = self.module.corpus_embedding_fingerprint(documents)
        points = [
            self.module.build_payload(document, fingerprint)
            for document in documents
        ]
        self._verify_with_points(documents, points)

    def test_verify_detects_missing_extra_and_duplicate_ids(self) -> None:
        documents = [_record(), {**_record(), "id": "documento_p2_c0"}]
        fingerprint = self.module.corpus_embedding_fingerprint(documents)
        valid = self.module.build_payload(documents[0], fingerprint)

        cases = {
            "ausentes": [valid],
            "extras": [
                valid,
                {**self.module.build_payload(documents[1], fingerprint), "id_original": "extra"},
            ],
            "duplicado": [valid, valid],
        }
        for label, points in cases.items():
            with self.subTest(label=label):
                with self.assertRaises(ValueError):
                    self._verify_with_points(documents, points)


if __name__ == "__main__":
    unittest.main()
