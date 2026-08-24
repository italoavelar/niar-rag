"""Regressões do classificador sequencial de páginas não informativas."""

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


def _install_optional_dependency_stubs() -> None:
    if "fitz" not in sys.modules and importlib.util.find_spec("fitz") is None:
        sys.modules["fitz"] = types.ModuleType("fitz")

    if "tqdm" not in sys.modules:
        tqdm_module = types.ModuleType("tqdm")
        tqdm_module.tqdm = lambda values, **_kwargs: values
        sys.modules["tqdm"] = tqdm_module


_install_optional_dependency_stubs()

from extract_to_jsonl import (  # noqa: E402
    classify_noninformative_pages,
    clean_page_lines,
    detect_repeated_margin_lines,
    extract_page_lines,
    find_references_start,
)


NORMAL_METADATA = {"document_type": "relatório", "title": "Documento"}
LEGAL_METADATA = {"document_type": "legislação", "title": "Código"}


class NoninformativePageClassifierTests(unittest.TestCase):
    def test_toc_continuation_requires_structure_and_exits_on_prose(self) -> None:
        pages = [
            [
                "SUMÁRIO",
                "Introdução ........ 3",
                "Método ........ 6",
                "Resultados ........ 12",
            ],
            [
                "Discussão sobre a governança, os direitos e a implementação "
                "de sistemas de inteligência artificial em serviços de saúde "
                "................................................................ 22",
                "Conclusões ........ 29",
                "Referências ........ 31",
            ],
            [
                "APRESENTAÇÃO",
                "Este documento apresenta os objetivos, o método e os resultados ",
                "da iniciativa em linguagem corrido e com explicações detalhadas.",
            ],
        ]

        result = classify_noninformative_pages(pages, NORMAL_METADATA)

        self.assertEqual([item.discard for item in result], [True, True, False])
        self.assertEqual(result[0].state, "toc")
        self.assertEqual(result[1].state, "toc")
        self.assertEqual(result[2].state, "normal")

    def test_subject_index_continuation_is_discarded_but_normative_page_is_not(self) -> None:
        pages = [
            [
                "ÍNDICE REMISSIVO",
                "Abandono Cap. III – art. 36",
                "Autonomia Cap. I – arts. 22, 24, 31",
                "Consentimento arts. 22, 31, 34",
                "Sigilo Cap. IX – arts. 73, 75, 79",
            ],
            [
                "Médico Cap. II – arts. 1, 3, 5",
                "Paciente Cap. III – arts. 31, 34, 36",
                "Prontuário Cap. X – arts. 87, 88",
                "Publicidade Cap. XIII – arts. 111, 112",
            ],
            [
                "CAPÍTULO III",
                "Art. 36. É vedado ao médico abandonar paciente sob seus cuidados.",
                "§ 1º O médico deverá assegurar a continuidade da assistência.",
            ],
        ]

        result = classify_noninformative_pages(pages, LEGAL_METADATA)

        self.assertEqual([item.discard for item in result], [True, True, False])
        self.assertEqual(result[0].state, "subject_index")
        self.assertEqual(result[1].state, "subject_index")
        self.assertEqual(result[2].state, "normal")

    def test_paragraph_markers_end_subject_index_state(self) -> None:
        pages = [
            [
                "ÍNDICE REMISSIVO",
                "Abandono Cap. III – art. 36",
                "Autonomia Cap. I – arts. 22, 24, 31",
                "Consentimento arts. 22, 31, 34",
                "Sigilo Cap. IX – arts. 73, 75, 79",
            ],
            [
                "§ 1º O médico deverá assegurar a continuidade da assistência.",
                "§ 2º É vedado ao médico abandonar paciente sob seus cuidados.",
                "§ único O dever previsto neste artigo aplica-se imediatamente.",
            ],
        ]

        result = classify_noninformative_pages(pages, LEGAL_METADATA)

        self.assertTrue(result[0].discard)
        self.assertFalse(result[1].discard)
        self.assertEqual(result[1].state, "normal")

    def test_glossary_and_informative_table_are_not_discarded(self) -> None:
        pages = [
            [
                "GLOSSÁRIO",
                "Autonomia: capacidade de uma pessoa tomar decisões informadas.",
                "Beneficência: dever de promover o melhor interesse do paciente.",
                "Equidade: distribuição justa de benefícios e riscos.",
            ],
            [
                "Tabela 4 – Indicadores de segurança",
                "Indicador Valor Meta",
                "Cobertura vacinal 91% 95%",
                "Eventos adversos 4 0",
                "A tabela apresenta resultados consolidados do período avaliado.",
            ],
        ]

        result = classify_noninformative_pages(pages, NORMAL_METADATA)

        self.assertTrue(all(not item.discard for item in result))

    def test_bibliographic_references_detection_is_unchanged(self) -> None:
        pages = [
            ["Capa"],
            ["Introdução", "Texto explicativo."],
            ["REFERÊNCIAS", "Autor. Título. Editora, 2024."],
            ["Outro autor. Outro título. Editora, 2023."],
        ]

        self.assertEqual(
            find_references_start(pages, NORMAL_METADATA),
            (2, 0),
        )

    @unittest.skipUnless(
        importlib.util.find_spec("fitz") is not None,
        "PyMuPDF não está disponível neste ambiente",
    )
    def test_cfm_subject_index_pages_and_exit(self) -> None:
        import fitz

        from extract_to_jsonl import lines_to_text

        pdf_path = ROOT / "docs/raw/codigo_etica_medica_CFM_2019.pdf"
        with fitz.open(pdf_path) as document:
            raw_pages = [extract_page_lines(page) for page in document]

        stats = Counter()
        repeated = detect_repeated_margin_lines(raw_pages)
        cleaned_pages = [
            clean_page_lines(lines, repeated, stats) for lines in raw_pages
        ]
        result = classify_noninformative_pages(cleaned_pages, LEGAL_METADATA)

        for page_number in (55, 56, 58, 82):
            with self.subTest(page=page_number):
                self.assertTrue(result[page_number - 1].discard)
                self.assertEqual(result[page_number - 1].state, "subject_index")

        self.assertFalse(result[82].discard)  # página 83: composição do CFM
        self.assertEqual(result[82].state, "normal")
        self.assertIn("COMPOSIÇÃO", lines_to_text(cleaned_pages[82]).upper())


if __name__ == "__main__":
    unittest.main()
