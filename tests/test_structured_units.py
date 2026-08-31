"""Testes do agrupamento de unidades estruturais de PDF."""

from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


from structured_units import (
    StructuralUnit,
    _apply_structure_context,
    _classify_text_kind,
    extract_pdf_page_units,
    pack_structured_units,
)


def _table_units(rows: list[tuple[str, str]]) -> list[StructuralUnit]:
    units = [
        StructuralUnit(
            kind="table",
            text="[TABELA]\nColunas: Level | Description",
            source_order=0,
            page=1,
            table_id="table-1",
            table_headers=("Level", "Description"),
        )
    ]
    units.extend(
        StructuralUnit(
            kind="table_row",
            text=f"Level: {level}\nDescription: {description}",
            source_order=index,
            page=1,
            table_id="table-1",
            row_index=index - 1,
            table_headers=("Level", "Description"),
        )
        for index, (level, description) in enumerate(rows, start=1)
    )
    return units


class _FakeTable:
    def __init__(
        self,
        bbox: tuple[float, float, float, float],
        headers: list[str],
        rows: list[list[str | None]],
        *,
        header_external: bool = False,
    ) -> None:
        self.bbox = bbox
        self.header = types.SimpleNamespace(
            names=headers,
            external=header_external,
        )
        self._rows = rows

    def extract(self) -> list[list[str | None]]:
        return self._rows


class _FakeTablePage:
    def __init__(
        self,
        tables: list[_FakeTable],
        text_lines: list[str] | None = None,
    ) -> None:
        self._tables = tables
        self._text_lines = text_lines or []

    def find_tables(self):
        return types.SimpleNamespace(tables=self._tables)

    def get_text(self, mode: str, **_kwargs):
        if mode != "dict":
            raise AssertionError(f"modo inesperado: {mode}")
        return {
            "blocks": [
                {
                    "type": 0,
                    "bbox": (0, 200, 300, 260),
                    "lines": [
                        {
                            "bbox": (0, 200 + index * 12, 300, 210 + index * 12),
                            "spans": [{"text": text, "font": "", "flags": 0}],
                        }
                        for index, text in enumerate(self._text_lines)
                    ],
                }
            ] if self._text_lines else []
        }


class StructuredUnitPackingTests(unittest.TestCase):
    def test_fully_empty_table_does_not_generate_units_or_chunks(self) -> None:
        table = _FakeTable(
            (10, 10, 200, 100),
            ["", ""],
            [["", ""], [" ", None]],
        )

        units = extract_pdf_page_units(_FakeTablePage([table]), 1, [])

        self.assertFalse(units)
        self.assertFalse(pack_structured_units(units))

    def test_generic_headers_and_empty_rows_do_not_generate_a_table(self) -> None:
        table = _FakeTable(
            (10, 10, 200, 100),
            ["", "", ""],
            [[None, " ", ""], ["", None, ""]],
        )

        units = extract_pdf_page_units(_FakeTablePage([table]), 1, [])

        self.assertFalse(units)

    def test_empty_rows_are_removed_from_an_otherwise_useful_table(self) -> None:
        table = _FakeTable(
            (10, 10, 200, 100),
            ["", ""],
            [["", ""], ["", "Resultado preservado"], ["", ""]],
        )

        units = extract_pdf_page_units(
            _FakeTablePage([table]),
            1,
            ["Resultado preservado"],
        )

        rows = [unit for unit in units if unit.kind == "table_row"]
        self.assertEqual(len(rows), 1)
        self.assertIn("Resultado preservado", rows[0].text)
        self.assertEqual(len(pack_structured_units(units)), 1)

    def test_removed_margin_content_does_not_reappear_as_a_table(self) -> None:
        table = _FakeTable(
            (10, 10, 200, 100),
            ["", ""],
            [["", ""], ["", "Conteúdo de rodapé removido"]],
        )
        page = _FakeTablePage([table], ["Conteúdo principal preservado"])

        units = extract_pdf_page_units(page, 1, ["Conteúdo principal preservado"])

        self.assertFalse(any(unit.kind.startswith("table") for unit in units))
        self.assertEqual([unit.text for unit in units], ["Conteúdo principal preservado"])

    def test_table_in_similar_position_is_kept_when_its_text_survives_cleaning(self) -> None:
        table = _FakeTable(
            (10, 10, 200, 100),
            ["", ""],
            [["", ""], ["", "Conteúdo de tabela preservado"]],
        )

        units = extract_pdf_page_units(
            _FakeTablePage([table]),
            1,
            ["Conteúdo de tabela preservado"],
        )

        self.assertEqual(sum(unit.kind == "table" for unit in units), 1)
        self.assertEqual(sum(unit.kind == "table_row" for unit in units), 1)

    def test_duplicate_table_detections_are_emitted_once(self) -> None:
        rows = [["Level", "Description"], ["Low", "Small impact."]]
        first = _FakeTable((10, 10, 200, 100), ["Level", "Description"], rows)
        duplicate = _FakeTable((10, 10, 200, 100), ["Level", "Description"], rows)

        units = extract_pdf_page_units(_FakeTablePage([first, duplicate]), 1, [])

        self.assertEqual(sum(unit.kind == "table" for unit in units), 1)
        self.assertEqual(sum(unit.kind == "table_row" for unit in units), 1)
        packed = pack_structured_units(units)
        self.assertEqual(len(packed), 1)
        self.assertEqual(packed[0].text.count("Level: Low"), 1)

    def test_different_tables_on_the_same_page_are_preserved(self) -> None:
        first = _FakeTable(
            (10, 10, 200, 100),
            ["Level", "Description"],
            [["Level", "Description"], ["Low", "Small impact."]],
        )
        second = _FakeTable(
            (10, 120, 200, 210),
            ["Level", "Description"],
            [["Level", "Description"], ["High", "Major impact."]],
        )

        units = extract_pdf_page_units(_FakeTablePage([first, second]), 1, [])

        self.assertEqual(sum(unit.kind == "table" for unit in units), 2)
        self.assertEqual(sum("Low" in unit.text for unit in units), 1)
        self.assertEqual(sum("High" in unit.text for unit in units), 1)

    def test_same_headers_with_different_content_are_not_deduplicated(self) -> None:
        first = _FakeTable(
            (10, 10, 200, 100),
            ["Level", "Description"],
            [["Level", "Description"], ["Low", "Small impact."]],
        )
        second = _FakeTable(
            (10, 10, 200, 100),
            ["Level", "Description"],
            [["Level", "Description"], ["High", "Major impact."]],
        )

        units = extract_pdf_page_units(_FakeTablePage([first, second]), 1, [])

        self.assertEqual(sum(unit.kind == "table" for unit in units), 2)
        self.assertEqual(sum("Low" in unit.text for unit in units), 1)
        self.assertEqual(sum("High" in unit.text for unit in units), 1)

    def test_long_single_cell_pseudo_header_is_kept_as_table_content(self) -> None:
        long_text = "Pergunta extensa " * 45
        table = _FakeTable(
            (10, 10, 200, 100),
            ["", long_text, ""],
            [["", long_text, ""], ["", "Resposta", ""]],
        )

        units = extract_pdf_page_units(_FakeTablePage([table]), 1, [])

        table = next(unit for unit in units if unit.kind == "table")
        rows = [unit for unit in units if unit.kind == "table_row"]
        self.assertEqual(table.table_headers, ("Coluna 1", "Coluna 2", "Coluna 3"))
        self.assertEqual(len(rows), 2)
        self.assertIn("Pergunta extensa", rows[0].text)

    def test_small_table_is_kept_in_one_chunk(self) -> None:
        packed = pack_structured_units(
            _table_units([("Low", "Small impact."), ("High", "Major impact.")])
        )

        self.assertEqual(len(packed), 1)
        self.assertIn("Colunas: Level | Description", packed[0].text)
        self.assertIn("Level: Low", packed[0].text)
        self.assertIn("Level: High", packed[0].text)
        self.assertIn("[/TABELA]", packed[0].text)

    def test_table_caption_is_kept_with_the_table(self) -> None:
        units = _table_units([("Low", "Small impact.")])
        units.append(StructuralUnit("text", "(Table adapted from source)", 2, 1))
        packed = pack_structured_units(units)

        self.assertEqual(len(packed), 1)
        self.assertIn("Nota: (Table adapted from source)", packed[0].text)

    def test_large_table_splits_only_between_rows_and_repeats_header(self) -> None:
        rows = [
            (f"L{index}", f"descricao-{index} " + ("x" * 390))
            for index in range(5)
        ]
        packed = pack_structured_units(_table_units(rows))

        self.assertGreater(len(packed), 1)
        self.assertTrue(
            all("Colunas: Level | Description" in chunk.text for chunk in packed)
        )
        combined = "\n".join(chunk.text for chunk in packed)
        for index in range(5):
            self.assertEqual(combined.count(f"Level: L{index}"), 1)

    def test_oversized_table_row_uses_fallback_with_table_context(self) -> None:
        row_text = "palavra " * 420
        packed = pack_structured_units(_table_units([("Huge", row_text)]))

        self.assertGreater(len(packed), 1)
        self.assertTrue(
            all("Colunas: Level | Description" in chunk.text for chunk in packed)
        )
        self.assertTrue(all(len(chunk.text) <= 1_260 for chunk in packed))

    def test_article_paragraph_unique_and_incisos_receive_article_context(self) -> None:
        units = [
            StructuralUnit("heading", "CAPÍTULO I", 0, 1),
            StructuralUnit("article", "Art. 1º Esta norma estabelece regras.", 1, 1),
            StructuralUnit("text", "§ 1º O dever é aplicável.", 2, 1),
            StructuralUnit("text", "Parágrafo único. A exceção deve ser motivada.", 3, 1),
            StructuralUnit("text", "I - primeiro inciso.", 4, 1),
            StructuralUnit("text", "II - segundo inciso.", 5, 1),
        ]
        _apply_structure_context(units)
        packed = pack_structured_units(units, chunk_size=120)

        self.assertTrue(all(unit.section_path for unit in units[1:]))
        self.assertTrue(any("Art. 1º" in chunk.text for chunk in packed[1:]))

    def test_generic_legal_and_section_markers_are_conservative(self) -> None:
        self.assertEqual(_classify_text_kind("Artigo 1 - Do objeto", False), "article")
        self.assertEqual(_classify_text_kind("§ único. Regra especial.", False), "text")
        self.assertEqual(_classify_text_kind("CAPÍTULO II", False), "heading")
        self.assertEqual(_classify_text_kind("SEÇÃO 1", False), "heading")
        self.assertEqual(_classify_text_kind("I - inciso", False, article_active=True), "text")
        self.assertEqual(_classify_text_kind("I - texto comum", False), "text")

    def test_heading_stays_with_following_text(self) -> None:
        units = [
            StructuralUnit("heading", "Escopo", 0, 1),
            StructuralUnit("text", "Texto relacionado ao escopo.", 1, 1),
        ]
        _apply_structure_context(units)
        packed = pack_structured_units(units)

        self.assertEqual(len(packed), 1)
        self.assertLess(packed[0].text.index("Escopo"), packed[0].text.index("Texto relacionado"))

    def test_prose_units_have_complete_coverage_without_cross_unit_duplication(self) -> None:
        units = [
            StructuralUnit("text", "primeira unidade exclusiva", 0, 1),
            StructuralUnit("text", "segunda unidade exclusiva", 1, 1),
            StructuralUnit("text", "terceira unidade exclusiva", 2, 1),
        ]
        packed = pack_structured_units(units, chunk_size=50, overlap=10)
        combined = "\n".join(chunk.text for chunk in packed)

        for unit in units:
            self.assertEqual(combined.count(unit.text), 1)

    def test_plain_text_and_oversized_text_use_chunker_fallback(self) -> None:
        units = [StructuralUnit("text", "palavra " * 450, 0, 1)]
        packed = pack_structured_units(units)

        self.assertGreater(len(packed), 1)
        self.assertTrue(all(chunk.text for chunk in packed))
        self.assertTrue(all(len(chunk.text) <= 1_200 for chunk in packed))


@unittest.skipUnless(importlib.util.find_spec("fitz"), "PyMuPDF não instalado")
class UnescoTableRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if "tqdm" not in sys.modules:
            tqdm_module = types.ModuleType("tqdm")
            tqdm_module.tqdm = lambda values, **_kwargs: values
            sys.modules["tqdm"] = tqdm_module

        import fitz
        import extract_to_jsonl as extractor
        from structured_units import extract_pdf_page_units

        cls.fitz = fitz
        cls.extractor = extractor
        cls.extract_pdf_page_units = staticmethod(extract_pdf_page_units)
        cls.pdf_path = ROOT / "docs/raw/ethical_impact_assessment_UNESCO_2023.pdf"

    def _clean_page_lines(self, document, page_index: int, metadata: dict) -> list[str]:
        pages_lines = [self.extractor.extract_page_lines(page) for page in document]
        repeated = self.extractor.detect_repeated_margin_lines(pages_lines)
        references = self.extractor.find_references_start(pages_lines, metadata)
        stats = Counter()
        lines = self.extractor.apply_references_cut(
            page_index,
            pages_lines[page_index],
            references,
            stats,
        )
        return self.extractor.clean_page_lines(lines, repeated, stats)

    def test_unesco_page_30_does_not_repeat_a_content_row_as_table_header(self) -> None:
        metadata = self.extractor.load_manifest()[self.pdf_path.name]
        with self.fitz.open(self.pdf_path) as document:
            page = document[29]
            detected = page.find_tables().tables
            self.assertEqual(len(detected), 1)
            self.assertEqual(detected[0].row_count, 5)
            self.assertEqual(detected[0].col_count, 3)

            units = self.extract_pdf_page_units(
                page,
                30,
                self._clean_page_lines(document, 29, metadata),
            )

        table = next(unit for unit in units if unit.kind == "table")
        rows = [unit for unit in units if unit.kind == "table_row"]
        packed = pack_structured_units(units)
        combined = "\n".join(chunk.text for chunk in packed)

        self.assertEqual(table.table_headers, ("Coluna 1", "Coluna 2", "Coluna 3"))
        self.assertEqual(len(rows), 3)
        self.assertEqual(len(packed), 2)
        self.assertEqual(combined.count("8.2.1.5. Is the data being stored"), 1)
        self.assertEqual(combined.count("8.2.2.1. Has a privacy impact"), 1)

    def test_unesco_page_45_table_is_atomic_and_not_duplicated(self) -> None:
        metadata = self.extractor.load_manifest()[self.pdf_path.name]
        with self.fitz.open(self.pdf_path) as document:
            page = document[44]
            detected = page.find_tables().tables
            self.assertEqual(len(detected), 1)
            self.assertEqual(detected[0].row_count, 5)
            self.assertEqual(detected[0].col_count, 2)

            units = self.extract_pdf_page_units(
                page,
                45,
                self._clean_page_lines(document, 44, metadata),
            )

        table_rows = [unit for unit in units if unit.kind == "table_row"]
        common_text = [unit.text for unit in units if unit.kind == "text"]
        packed = pack_structured_units(units)
        combined = "\n".join(chunk.text for chunk in packed)
        table_chunks = [chunk for chunk in packed if "[TABELA]" in chunk.text]

        self.assertEqual(len(table_rows), 4)
        self.assertGreater(len(table_chunks), 1)
        for label in ("Catastrophic", "Critical", "Serious", "Moderate/ minor"):
            self.assertEqual(sum(label in row.text for row in table_rows), 1)
        for row in table_rows:
            self.assertFalse(any(row.text in text for text in common_text))
        self.assertTrue(all("Colunas: GRAVITY LEVEL | DESCRIPTION" in chunk.text for chunk in table_chunks))
        self.assertEqual(combined.count("GRAVITY LEVEL: Catastrophic"), 1)
        self.assertEqual(combined.count("GRAVITY LEVEL: Critical"), 1)
        self.assertEqual(combined.count("GRAVITY LEVEL: Serious"), 1)
        self.assertEqual(combined.count("GRAVITY LEVEL: Moderate/ minor"), 1)
        descriptions = {
            "Catastrophic": "deprivation of the right to life",
            "Critical": "significant and enduring degradation",
            "Serious": "temporary degradation",
            "Moderate/ minor": "do not lead to any significant",
        }
        for label, description in descriptions.items():
            row = next(row for row in table_rows if label in row.text)
            self.assertIn(description, row.text)


if __name__ == "__main__":
    unittest.main()
