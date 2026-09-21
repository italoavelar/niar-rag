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
    apply_references_cut,
    is_page_number_line,
    classify_noninformative_pages,
    clean_page_lines,
    detect_repeated_margin_lines,
    extract_page_lines,
    find_references_end,
    find_references_start,
    is_reference_like_page,
    short_page_noninformative_reason,
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

    def test_references_stop_before_a_substantive_annex(self) -> None:
        pages = [
            ["Capa"],
            ["Introdução", "Texto substantivo."],
            ["Mais conteúdo", "Ainda substantivo."],
            ["REFERENCES", "[1] Autor. Obra. 2024."],
            [
                "[2] Outro autor. Outra obra. 2023.",
                "[3] Terceiro autor. Terceira obra. 2022.",
            ],
            ["ANNEX. METHODS", "Método substantivo do anexo."],
            ["Definições", "Termo: explicação substantiva."],
        ]
        references_start = find_references_start(pages, NORMAL_METADATA)

        self.assertEqual(references_start, (3, 0))
        self.assertEqual(
            find_references_end(pages, references_start, NORMAL_METADATA),
            (5, 0),
        )

        stats = Counter()
        references_end = find_references_end(
            pages, references_start, NORMAL_METADATA
        )
        self.assertEqual(
            apply_references_cut(
                3, pages[3], references_start, stats, references_end
            ),
            [],
        )
        self.assertEqual(
            apply_references_cut(
                5, pages[5], references_start, stats, references_end
            ),
            pages[5],
        )
        self.assertEqual(
            apply_references_cut(
                6, pages[6], references_start, stats, references_end
            ),
            pages[6],
        )

    def test_references_do_not_remove_unknown_following_content(self) -> None:
        pages = [
            ["Capa"],
            ["Introdução", "Texto substantivo."],
            ["Mais conteúdo", "Ainda substantivo."],
            ["REFERENCES", "[1] Autor. Obra. 2024."],
            [
                "Material metodológico complementar sem cabeçalho padronizado.",
                "Este conteúdo não tem a forma de uma entrada bibliográfica.",
            ],
        ]
        references_start = find_references_start(pages, NORMAL_METADATA)

        self.assertEqual(
            find_references_end(pages, references_start, NORMAL_METADATA),
            (4, 0),
        )

    def test_normative_content_overrides_incidental_toc_word(self) -> None:
        pages = [
            [
                "§ 2º O prontuário estará sob a guarda do médico.",
                "§ 3º O médico deve entregar o sumário de alta ao paciente.",
                "Art. 88. Negar ao paciente acesso ao prontuário é vedado.",
                "Art. 89. Liberar cópias depende de autorização legal.",
            ]
        ]

        result = classify_noninformative_pages(pages, LEGAL_METADATA)

        self.assertFalse(result[0].discard)
        self.assertEqual(result[0].state, "normal")

    def test_short_normative_content_is_preserved(self) -> None:
        lines = [
            "Art. 40. É vedado ao médico revelar fato de que tenha conhecimento.",
            "§ 1º O dever permanece após o término da relação profissional.",
        ]

        self.assertIsNone(short_page_noninformative_reason(lines))

    def test_short_glossary_and_annex_content_are_preserved(self) -> None:
        self.assertIsNone(
            short_page_noninformative_reason(
                [
                    "53 Glossary",
                    "API: mecanismo para comunicação entre sistemas.",
                    "Data: informação registrada em formato digital.",
                ]
            )
        )
        self.assertIsNone(
            short_page_noninformative_reason(
                [
                    "ANNEX. METHODS",
                    "The method establishes controls for validation and monitoring.",
                ]
            )
        )

    def test_high_confidence_short_boilerplate_is_discarded(self) -> None:
        self.assertEqual(
            short_page_noninformative_reason(["pg. 1", "www.fda.gov"]),
            "boilerplate_or_url_only",
        )
        self.assertEqual(
            short_page_noninformative_reason(
                ["Additional resources: see Section 2.7 in GHTF SG3 N18:2010[6]"]
            ),
            "editorial_footer_or_figure_caption",
        )
        self.assertEqual(
            short_page_noninformative_reason(
                [
                    "Health Ethics and Governance Unit",
                    "Research for Health Department",
                    "Digital Health and Innovation Department",
                    "World Health Organization",
                    "Avenue Appia 20",
                    "1121 Geneva 27",
                    "Switzerland",
                ]
            ),
            "institutional_contact_only",
        )

    def test_toc_with_physical_page_prefix_is_discarded(self) -> None:
        result = classify_noninformative_pages(
            [
                [
                    "61 Table of Contents",
                    "62 I. Introduction ................................ 1",
                    "63 II. Scope ....................................... 2",
                ]
            ],
            NORMAL_METADATA,
        )
        self.assertTrue(result[0].discard)

    def test_reference_page_without_leading_number_is_still_detected(self) -> None:
        self.assertTrue(
            is_reference_like_page(
                [
                    "48 Chapman, Peter et al. (2021), Grasping the Justice Gap, [3]",
                    "https://worldjusticeproject.org/example.pdf",
                    "CMA (2016), Retail banking market investigation, [62]",
                ]
            )
        )

    def test_bulleted_source_listing_is_removed_but_prose_with_links_is_kept(self) -> None:
        self.assertEqual(
            short_page_noninformative_reason(
                [
                    "• Author et al. (2020). Technical report. DOI: 10.1234/example.",
                    "• Another source (2021). https://example.org/report.pdf",
                    "• Third source (2022). https://example.org/third.pdf",
                ]
            ),
            "bibliographic_continuation",
        )
        self.assertIsNone(
            short_page_noninformative_reason(
                [
                    "The policy describes a governance model for public institutions.",
                    "The model is evaluated against evidence and implementation risks.",
                    "Further details are available at https://example.org/method.",
                ]
            )
        )

    def test_source_listing_continuation_uses_previous_page_shape(self) -> None:
        previous = [
            "• First source (2020). DOI: 10.1234/first.",
            "• Second source (2021). https://example.org/second.pdf",
        ]
        continuation = [
            "November 26, 2012. https://example.org/third.pdf",
            "• Fourth source (2013). https://example.org/fourth.pdf",
        ]
        self.assertEqual(
            short_page_noninformative_reason(
                continuation,
                previous_lines=previous,
            ),
            "bibliographic_continuation",
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

        # Página 40: "sumário de alta" é conteúdo normativo, não um sumário.
        self.assertFalse(result[39].discard)
        self.assertEqual(result[39].state, "normal")

class RomanPageNumberTests(unittest.TestCase):
    """A numeração romana do pré-textual escapava e virava chunk de 1 caractere."""

    def test_roman_page_numbers_are_recognized(self) -> None:
        for linha in ("i", "v", "vi", "vii", "viii", "xi", "xiii", "xiv", "- iv -"):
            with self.subTest(linha=linha):
                self.assertTrue(is_page_number_line(linha))

    def test_words_made_of_roman_letters_are_not_page_numbers(self) -> None:
        for linha in ("civil", "dim", "mild", "Article", "The"):
            with self.subTest(linha=linha):
                self.assertFalse(is_page_number_line(linha))

    def test_arabic_page_numbers_keep_working(self) -> None:
        self.assertTrue(is_page_number_line("42"))
        self.assertTrue(is_page_number_line("3 / 10"))
        self.assertFalse(is_page_number_line("42 paginas"))

if __name__ == "__main__":
    unittest.main()
