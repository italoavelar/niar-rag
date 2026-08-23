"""Regressões do chunker compartilhado por PDF e HTML.

Os testes usam ``unittest`` para não introduzir uma dependência de execução
adicional no repositório.
"""

from __future__ import annotations

import importlib
import sys
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _install_optional_dependency_stubs() -> None:
    """Permite importar os entrypoints sem exigir extração de PDF nos testes."""
    if "fitz" not in sys.modules and importlib.util.find_spec("fitz") is None:
        sys.modules["fitz"] = types.ModuleType("fitz")

    if "tqdm" not in sys.modules:
        tqdm_module = types.ModuleType("tqdm")
        tqdm_module.tqdm = lambda values, **_kwargs: values
        sys.modules["tqdm"] = tqdm_module


def _early_punctuation_text() -> str:
    """Pontuação válida em ~200 chars, seguida de texto longo sem pontuação."""
    return ("a " * 100) + ". " + ("b " * 1_200)


class ChunkerRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _install_optional_dependency_stubs()

    def _chunking(self):
        return importlib.import_module("chunking")

    def _assert_full_coverage(self, text: str, spans: list[tuple[int, int]]) -> None:
        normalized = self._chunking().clean_text(text)

        self.assertTrue(spans)
        self.assertEqual(spans[0][0], 0)
        self.assertEqual(spans[-1][1], len(normalized))

        for (previous_start, previous_end), (start, end) in zip(
            spans,
            spans[1:],
        ):
            self.assertGreater(start, previous_start)
            self.assertLessEqual(start, previous_end)
            self.assertGreater(end, start)

    def test_early_punctuation_does_not_create_tiny_stride_or_explosion(self) -> None:
        """Regressão: antes da correção gera ~80 chunks por stride de 1 char."""
        _install_optional_dependency_stubs()
        legacy_module = importlib.import_module("extract_to_jsonl")

        chunks = legacy_module.chunk_text(_early_punctuation_text())

        self.assertLessEqual(len(chunks), 4)
        self.assertTrue(all(len(chunk) >= 1_000 for chunk in chunks[:-1]))

    def test_early_punctuation_uses_normal_overlap_and_preserves_coverage(self) -> None:
        text = _early_punctuation_text()
        chunking = self._chunking()
        normalized = chunking.clean_text(text)
        spans = chunking._chunk_spans(
            normalized,
            chunk_size=1_200,
            overlap=200,
            min_chunk_size=120,
        )

        self._assert_full_coverage(text, spans)
        self.assertTrue(
            all(end - start >= 1_000 for start, end in spans[:-1])
        )
        self.assertTrue(
            all(
                previous_end - start == 200
                for (_, previous_end), (start, _) in zip(spans, spans[1:])
            )
        )

    def test_text_without_punctuation_uses_nominal_windows(self) -> None:
        text = " ".join(["word"] * 1_100)
        chunking = self._chunking()
        normalized = chunking.clean_text(text)
        spans = chunking._chunk_spans(normalized, 1_200, 200, 120)

        self._assert_full_coverage(text, spans)
        self.assertTrue(
            all(1_000 <= end - start <= 1_200 for start, end in spans[:-1])
        )
        self.assertTrue(
            all(
                previous_end - start == 200
                for (_, previous_end), (start, _) in zip(spans, spans[1:])
            )
        )

    def test_very_long_word_neither_loops_nor_loses_content(self) -> None:
        long_word = "x" * 1_500
        text = ("intro " * 220) + long_word + (" outro" * 220)
        chunking = self._chunking()
        normalized = chunking.clean_text(text)
        spans = chunking._chunk_spans(normalized, 1_200, 200, 120)

        self._assert_full_coverage(text, spans)
        self.assertLessEqual(len(spans), 6)

        long_word_start = normalized.index(long_word)
        long_word_end = long_word_start + len(long_word)
        self.assertTrue(any(start <= long_word_start < end for start, end in spans))
        self.assertTrue(any(start < long_word_end <= end for start, end in spans))

    def test_small_final_fragment_merge_is_preserved(self) -> None:
        chunking = self._chunking()
        merged = chunking._merge_small_final_chunk(
            ["primeiro chunk", "fragmento final"],
            min_chunk_size=120,
        )

        self.assertEqual(merged, ["primeiro chunk fragmento final"])
        self.assertEqual(chunking.chunk_text("x" * 119), [])
        self.assertEqual(chunking.chunk_text("x" * 120), ["x" * 120])

    def test_pdf_and_html_entrypoints_share_the_same_chunker(self) -> None:
        _install_optional_dependency_stubs()
        chunking = self._chunking()
        pdf_module = importlib.import_module("extract_to_jsonl")
        html_module = importlib.import_module("extract_html_to_jsonl")

        self.assertIs(pdf_module.chunk_text, chunking.chunk_text)
        self.assertIs(html_module.chunk_text, chunking.chunk_text)


if __name__ == "__main__":
    unittest.main()
