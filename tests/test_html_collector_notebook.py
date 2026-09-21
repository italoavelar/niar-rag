"""Regressões do isolamento de conteúdo no coletor HTML do notebook."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from bs4 import BeautifulSoup


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "web_scraper_estatico_profundo_v6.ipynb"


def _collector_namespace() -> dict[str, object]:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = "".join(notebook["cells"][14]["source"])
    namespace: dict[str, object] = {"BeautifulSoup": BeautifulSoup}
    exec(source, namespace)
    return namespace


def _text_from(html: str) -> str:
    namespace = _collector_namespace()
    isolate = namespace["isolar_conteudo_central"]
    central = isolate(BeautifulSoup(html, "html.parser"))
    return " ".join(central.get_text(" ", strip=True).split())


class HtmlCollectorNotebookTests(unittest.TestCase):
    def test_substantive_modal_content_is_preserved(self) -> None:
        text = _text_from(
            """
            <html><body><main>
              <h1>AI terms &amp; concepts</h1>
              <p>An AI system is a machine-based system.</p>
              <div class="modal"><div class="modal-content">
                <h2>Governments that have committed to the AI Principles</h2>
                <p>Australia, Brazil, Canada and other adherents commit to the Principles.</p>
                <p>This section is substantive documentation, not a transient dialog. It records
                the public commitment of governments and contains the official list of adherents
                to the recommendation, which is relevant to policy research and must be retained.</p>
              </div></div>
            </main></body></html>
            """
        )

        self.assertIn("An AI system", text)
        self.assertIn("Governments that have committed", text)

    def test_interactive_modal_and_layout_noise_are_removed(self) -> None:
        text = _text_from(
            """
            <html><body>
              <nav>Main menu</nav><header>Banner</header>
              <main><h1>Document</h1><p>Substantive legal content.</p></main>
              <div class="cookie-modal"><h2>Cookies</h2><p>Accept all cookies.</p></div>
              <footer>Newsletter and copyright</footer><script>tracking_code()</script>
            </body></html>
            """
        )

        self.assertIn("Substantive legal content", text)
        for noise in ("Main menu", "Banner", "Accept all cookies", "Newsletter", "tracking_code"):
            self.assertNotIn(noise, text)

    def test_malformed_planalto_like_document_keeps_content_after_body(self) -> None:
        text = _text_from(
            """
            <html><body><p>Art. 24. Competência concorrente.</p></body></html>
            <p>Art. 25. Conteúdo posterior à competência concorrente.</p>
            <h2>ATO DAS DISPOSIÇÕES CONSTITUCIONAIS TRANSITÓRIAS</h2>
            <p>Art. 250. Disposição transitória.</p><p>Brasília, 5 de outubro de 1988.</p>
            """
        )

        for marker in ("Art. 25", "ATO DAS DISPOSIÇÕES", "Art. 250", "Brasília, 5 de outubro de 1988"):
            self.assertIn(marker, text)
        self.assertEqual(text.count("Art. 24."), 1)
        self.assertEqual(text.count("Art. 250"), 1)

    def test_malformed_eca_like_document_reaches_final_articles_once(self) -> None:
        text = _text_from(
            """
            <html><body><p>Art. 8º-A. Semana Nacional de Prevenção.</p></body></html>
            <p>Art. 9º. Conteúdo posterior.</p><p>Art. 267. Revogam-se as leis anteriores.</p>
            <p>Brasília, 13 de julho de 1990.</p>
            """
        )

        for marker in ("Art. 9º", "Art. 267", "Brasília, 13 de julho de 1990"):
            self.assertIn(marker, text)
            self.assertEqual(text.count(marker), 1)
