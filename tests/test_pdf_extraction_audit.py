"""Cobertura da auditoria página a página da extração de PDFs."""

from __future__ import annotations

import csv
import io
import sys
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import extract_to_jsonl as extractor  # noqa: E402


class _FakePage:
    def __init__(self, text: str) -> None:
        self.text = text

    def get_text(self, _mode: str, *, sort: bool = False) -> str:
        return self.text


class _FakeDocument:
    def __init__(self, pages: list[_FakePage]) -> None:
        self.pages = pages

    def __enter__(self):
        return self

    def __exit__(self, *_args) -> None:
        return None

    def __iter__(self):
        return iter(self.pages)

    def __len__(self) -> int:
        return len(self.pages)

    def __getitem__(self, index: int) -> _FakePage:
        return self.pages[index]


def _informative_page(label: str) -> str:
    paragraph = (
        f"{label}. "
        "Este texto explica detalhadamente os requisitos de governança, "
        "responsabilidade, segurança e monitoramento de sistemas de saúde. "
        "A organização deve documentar decisões, controles, evidências e "
        "resultados durante todo o ciclo de vida."
    )
    return " ".join([paragraph] * 6)


class PdfExtractionAuditTests(unittest.TestCase):
    def _run_document(
        self, pages: list[_FakePage] | None = None
    ) -> list[dict[str, str]]:
        pages = pages or [
            _FakePage(_informative_page("Página inicial")),
            _FakePage(
                "SUMÁRIO\n"
                "Introdução ........ 3\n"
                "Método ........ 7\n"
                "Resultados ........ 12"
            ),
            _FakePage(""),
            _FakePage(
                "A organização deve documentar decisões e controles relevantes."
            ),
            _FakePage(_informative_page("Página final")),
        ]
        document = _FakeDocument(pages)
        output = io.StringIO()
        audit_output = io.StringIO(newline="")
        audit_writer = csv.DictWriter(
            audit_output,
            fieldnames=extractor.AUDIT_FIELDNAMES,
        )
        audit_writer.writeheader()

        with patch.object(
            extractor.fitz, "open", return_value=document
        ), patch.object(
            extractor, "extract_pdf_page_units", return_value=[]
        ):
            found_text = extractor.process_document(
                pdf_file=Path("documento_teste.pdf"),
                doc_metadata={"document_type": "relatório", "title": "Teste"},
                output_file=output,
                audit_writer=audit_writer,
                global_stats=Counter(),
            )

        self.assertTrue(found_text)
        return list(csv.DictReader(io.StringIO(audit_output.getvalue())))

    def test_every_physical_page_receives_one_audit_record(self) -> None:
        records = self._run_document()

        # PDF de cinco páginas: inclusive a página final.
        self.assertEqual(len(records), 5)
        self.assertEqual([record["page"] for record in records], ["1", "2", "3", "4", "5"])
        self.assertTrue(all(record["total_pages"] == "5" for record in records))
        self.assertEqual(records[-1]["status"], "INDEXED")

    def test_toc_raw_empty_and_short_content_are_explicitly_classified(self) -> None:
        records = self._run_document()

        self.assertEqual(records[1]["status"], "NONINFORMATIVE_REMOVED")
        self.assertEqual(records[1]["reason"], "toc_start")
        self.assertEqual(records[2]["status"], "RAW_EMPTY")
        self.assertEqual(records[3]["status"], "INDEXED")
        self.assertEqual(records[3]["chunks_generated"], "1")
        self.assertTrue(all(record["status"] for record in records))

    def test_references_are_audited_with_pipeline_reasons(self) -> None:
        records = self._run_document(
            [
                _FakePage(_informative_page("Página um")),
                _FakePage(_informative_page("Página dois")),
                _FakePage(
                    "REFERENCES\n"
                    "1. Autor. Obra. Editora, 2024.\n"
                    "2. Outro autor. Outra obra. Editora, 2023."
                ),
                _FakePage(
                    "3. Terceiro autor. Última obra. Editora, 2022.\n"
                    "4. Quarto autor. Outra obra. Editora, 2021."
                ),
                _FakePage(
                    "5. Quinto autor. Última obra. Editora, 2020.\n"
                    "6. Sexto autor. Outra obra. Editora, 2019."
                ),
            ]
        )

        self.assertEqual(records[2]["status"], "REFERENCES_REMOVED")
        self.assertEqual(
            records[2]["reason"], "reference_section_starts_removed"
        )
        self.assertEqual(records[3]["status"], "REFERENCES_REMOVED")
        self.assertEqual(
            records[3]["reason"], "reference_pages_removed"
        )

    def test_short_content_policy_preserves_substantive_and_discards_boilerplate(self) -> None:
        records = self._run_document(
            [
                _FakePage(
                    "Art. 40. É vedado ao médico revelar fato protegido.\n"
                    "§ 1º O dever permanece após o término da relação."
                ),
                _FakePage(
                    "ANNEX. METHODS\n"
                    "The method establishes controls for validation and monitoring."
                ),
                _FakePage("pg. 1\nwww.fda.gov"),
                _FakePage(
                    "Additional resources: see Section 2.7 in GHTF SG3 N18:2010[6]\n"
                    "Page 4 of 5"
                ),
                _FakePage(
                    "Disclaimer\nCopyright © International Medical Device Regulators Forum.\n"
                    "All other rights are reserved."
                ),
            ]
        )

        self.assertEqual(records[0]["status"], "INDEXED")
        self.assertEqual(records[1]["status"], "INDEXED")
        self.assertEqual(records[2]["status"], "NONINFORMATIVE_REMOVED")
        self.assertEqual(records[2]["reason"], "boilerplate_or_url_only")
        self.assertEqual(records[3]["status"], "NONINFORMATIVE_REMOVED")
        self.assertEqual(records[3]["reason"], "editorial_footer_or_figure_caption")
        self.assertEqual(records[4]["status"], "NONINFORMATIVE_REMOVED")
        self.assertEqual(records[4]["reason"], "copyright_or_disclaimer")


if __name__ == "__main__":
    unittest.main()
