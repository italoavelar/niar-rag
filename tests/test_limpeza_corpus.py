"""Testes das regras que removem ruído de extração do corpus já extraído.

POR QUE ESTE ARQUIVO EXISTE. Toda regra aqui é um compromisso entre remover
ruído e destruir texto real, e as duas primeiras versões erraram para o lado
caro: a do rodapé transformou "Revisão do Código de Ética Médica do CFM" em
"Revisão do do CFM", e a da sigla "PL" comeu o "PL" de "PL nº 2338". Nos dois
casos o estrago só apareceu porque havia um caso de teste cobrindo o texto
legítimo que a regra não podia tocar.

Cada regra tem portanto DOIS testes: um que prova que o ruído sai, outro que
prova que o texto legítimo parecido com ele fica. Regra nova sem o par não
entra.

Rodar:
  python -m pytest tests/test_limpeza_corpus.py
  python -m unittest tests.test_limpeza_corpus
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTO = ROOT / "eval/experimento_embedding"

if str(EXPERIMENTO) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTO))

from limpar import (documento_com_numeracao_de_linha, eh_materia_editorial,
                    legenda_repetida, limpar, remover_numeracao_de_linha)


def limpo(texto: str) -> str:
    """Saída da limpeza com espaço colapsado, para comparar conteúdo e não bytes."""
    return " ".join(limpar(texto)[0].split())


class MarcaDagua(unittest.TestCase):
    """`*CD252287612600*` da Câmara, injetada no meio da frase."""

    def test_sai_e_costura_a_frase(self):
        saida = limpo("cooperação internacional para o atendimento a "
                      "*CD252287612600*padrões técnicos")
        self.assertIn("atendimento a padrões técnicos", saida)
        self.assertNotIn("*CD", saida)


class BarraLateralDaCamara(unittest.TestCase):
    """Texto vertical da capa que o extrator lê como texto corrido."""

    def test_apresentacao_antes_de_minuscula_sai(self):
        saida = limpo("uso ético e responsável da inteligência artificial com "
                      "base na Apresentação: centralidade da pessoa humana")
        self.assertIn("com base na centralidade", saida)

    def test_apresentacao_antes_de_maiuscula_sai(self):
        saida = limpo("tutela individual, coletiva e difusa. Apresentação: "
                      "CAPÍTULO III DA CATEGORIZAÇÃO")
        self.assertIn("difusa. CAPÍTULO III", saida)

    def test_apresentacao_como_titulo_de_linha_fica(self):
        """Título de verdade começa linha. É o que separa um do outro."""
        saida = limpar("Apresentação:\nEste documento descreve o programa")[0]
        self.assertIn("Apresentação:", saida)


class SiglaPL(unittest.TestCase):
    """"PL" da barra lateral, solto ou colado — sem comer a sigla legítima."""

    def test_pl_solto_antes_de_minuscula_sai(self):
        saida = limpo("o uso ético e responsável da PL inteligência artificial")
        self.assertIn("responsável da inteligência artificial", saida)

    def test_pl_colado_em_palavra_capitalizada_sai(self):
        saida = limpo("preceitos da Lei de Defesa da PLConcorrência, vedada")
        self.assertIn("Defesa da Concorrência", saida)

    def test_pl_colado_em_minuscula_sai(self):
        saida = limpo("pode ser integrado em PLdiversos sistemas ou aplicações")
        self.assertIn("integrado em diversos sistemas", saida)

    def test_pl_seguido_de_numero_fica(self):
        self.assertIn("PL 2338", limpo("o PL 2338 de 2023 tramita na Câmara"))

    def test_pl_com_numero_abreviado_fica(self):
        """A versão anterior comia este: 'nº' começa com minúscula."""
        self.assertIn("PL nº 2338", limpo("conforme o PL nº 2338, de 2023"))

    def test_sigla_pln_fica(self):
        self.assertIn("PLN", limpo("técnicas de PLN aplicadas ao direito"))

    def test_revista_plos_fica(self):
        self.assertIn("PLoS Medicine", limpo("publicado na PLoS Medicine em 2019"))

    def test_pl_antes_de_marcador_de_estrutura_sai(self):
        """Projeto de lei nenhum escreve "PL CAPÍTULO III" — ali é barra lateral."""
        for junk, esperado in (
            ("tutela individual, coletiva e difusa. PL CAPÍTULO III DA CATEGORIZAÇÃO",
             "difusa. CAPÍTULO III"),
            ("Governança de Inteligência Artificial PL Art. 45. O Poder Executivo",
             "Artificial Art. 45."),
            ("desenvolvimento dos sistemas de alto risco. PL Seção IV Do Conselho",
             "risco. Seção IV"),
        ):
            self.assertIn(esperado, limpo(junk))

    def test_pl_antes_de_nome_proprio_fica(self):
        """Alargar para qualquer maiúscula comeria isto. Por isso o gatilho é
        marcador de estrutura, não caixa alta."""
        self.assertIn("PL Brasileiro", limpo("debate sobre o PL Brasileiro de IA"))


class LegendaDeCaixa(unittest.TestCase):
    """Caixa de texto que `find_tables()` enquadra como tabela de uma coluna."""

    CAIXA = ("Box 1. Eviction data \n\n Box 1. Eviction data: In the US, landlords are\n\n"
             " Box 1. Eviction data: courts screen tenants\n\n"
             " Box 1. Eviction data: before making it open\n")

    TABELA = ("Colunas: SIGNIFICANCE LEVEL | DESCRIPTION\n\n"
              "SIGNIFICANCE LEVEL: Very high\nDESCRIPTION: Extremely transformative\n"
              "SIGNIFICANCE LEVEL: High\nDESCRIPTION: Very positive\n"
              "SIGNIFICANCE LEVEL: Low\nDESCRIPTION: Small\n")

    def test_legenda_repetida_e_detectada(self):
        self.assertEqual(legenda_repetida(self.CAIXA), "Box 1. Eviction data")

    def test_prefixo_sai_e_a_prosa_se_reconstitui(self):
        saida = limpo(self.CAIXA)
        self.assertIn("In the US, landlords are courts screen tenants", saida)
        self.assertNotIn("Eviction data: In the US", saida)

    def test_tabela_de_duas_colunas_nao_e_legenda(self):
        """Dois cabeçalhos ALTERNANDO: o prefixo é a única coisa que diz a coluna."""
        self.assertIsNone(legenda_repetida(self.TABELA))

    def test_tabela_de_duas_colunas_mantem_o_prefixo(self):
        self.assertIn("SIGNIFICANCE LEVEL: Very high", limpo(self.TABELA))

    def test_legenda_em_camadas_sai_inteira(self):
        """A legenda da OCDE tem dois níveis de dois-pontos.

        "Box 8. Colombia: Using Legal Needs Surveys...: <conteúdo>". Uma passada
        só tira "Box 8. Colombia" e deixa a segunda metade repetindo. Por isso a
        remoção é em laço.
        """
        camadas = (
            "Box 8. Colombia: Using Legal Needs Surveys \n\n"
            "Box 8. Colombia: Using Legal Needs Surveys: Colombia has implemented one\n\n"
            "Box 8. Colombia: Using Legal Needs Surveys: of the most comprehensive\n\n"
            "Box 8. Colombia: Using Legal Needs Surveys: systems in the region\n")
        saida = limpo(camadas)
        self.assertIn("Colombia has implemented one of the most comprehensive", saida)
        self.assertNotIn("Box 8. Colombia:", saida)


class RodapeDoCodigoDeEtica(unittest.TestCase):
    """Exige CAIXA ALTA com acento **e** número de página. As duas, sempre."""

    def test_rodape_colado_no_artigo_sai(self):
        saida = limpo("sem o consentimento do paciente. CÓDIGO DE ÉTICA MÉDICA "
                      "35Art. 78. Deixar de informar")
        self.assertIn("Art. 78. Deixar de informar", saida)
        self.assertNotIn("MÉDICA 35", saida)

    def test_mencao_legitima_fica(self):
        """A primeira versão fazia disto 'Revisão do do CFM'."""
        saida = limpo("Comissão Nacional de Revisão do Código de Ética Médica do CFM")
        self.assertIn("Revisão do Código de Ética Médica do CFM", saida)

    def test_titulo_em_caixa_alta_sem_numero_fica(self):
        saida = limpo("CÓDIGO DE ÉTICA MÉDICA\nResolução CFM nº 2.217, de 2018")
        self.assertIn("CÓDIGO DE ÉTICA MÉDICA", saida)


class AndaimeDeTabela(unittest.TestCase):
    """Cabeçalho inventado por `_generic_table_headers` quando o PDF não tem um."""

    def test_rotulo_inventado_sai_e_a_celula_fica(self):
        saida = limpo("Coluna 1: 6.2.1. Preventing discriminatory outcomes\nColuna 3:")
        self.assertIn("Preventing discriminatory outcomes", saida)
        self.assertNotIn("Coluna 1", saida)
        self.assertNotIn("Coluna 3", saida)

    def test_marcador_de_fronteira_sai(self):
        self.assertNotIn("[TABELA]", limpo("[TABELA] dados do exercício [/TABELA]"))


class MateriaEditorial(unittest.TestCase):
    """Expediente e lista de nomes — a embalagem do documento, não o documento."""

    def test_lista_de_conselheiros_e_editorial(self):
        texto = ("COMPOSIÇÃO DO CONSELHO Federal de Medicina Conselheiros titulares "
                 "Mauro Luiz de Britto Ribeiro Dilza Teresinha Ambros Ribeiro "
                 "Emmanuel Fortes Silveira Cavalcanti José Hiran da Silva Gallo "
                 "Jeancarlo Fernandes Cavalcante Donizetti Dimer Giamberardino")
        self.assertTrue(eh_materia_editorial(texto))

    def test_norma_que_regula_conselheiros_nao_e_editorial(self):
        """O decreto REGULA a eleição de suplentes — é norma, não lista."""
        texto = ("Os conselheiros suplentes deverão ser eleitos na mesma ocasião "
                 "dos efetivos, cabendo-lhes entrar em exercício em caso de "
                 "impedimento ou falta do titular, na forma que for estabelecida.")
        self.assertFalse(eh_materia_editorial(texto))


class NumeracaoDeLinha(unittest.TestCase):
    """Minuta em consulta pública, com o PDF numerando cada linha."""

    MINUTA = (
        "1154 As the device moves along this spectrum, the nature of the clinical study "
        "1155 or other studies that would be appropriate to support performance evaluation "
        "1156 of an AI-based medical device will vary according to the intended use. "
        "1157 For some devices, more emphasis may be placed on standalone performance. ")

    # Sumário do NIST: parece cadeia crescente, é número de página.
    SUMARIO = ("4. Technical Steps for Data De-Identifcation . . . . . 46 "
               "4.1. Determine the Privacy and Access Objectives . . . . . 46 "
               "4.2. Conducting a Data Survey . . . . . 47 "
               "4.3. De-Identifcation by Removing Identifers . . . . . 49 ")

    def test_minuta_e_detectada_pelo_documento_inteiro(self):
        self.assertTrue(documento_com_numeracao_de_linha(self.MINUTA * 30))

    def test_documento_normal_nao_e_detectado(self):
        prosa = ("Art. 5º São direitos do titular a confirmação da existência de "
                 "tratamento e o acesso aos dados. ")
        self.assertFalse(documento_com_numeracao_de_linha(prosa * 60))

    def test_sumario_com_paginas_nao_e_detectado(self):
        """Cadeia crescente existe, mas a densidade é baixa demais."""
        self.assertFalse(documento_com_numeracao_de_linha(self.SUMARIO * 30))

    def test_numeros_saem_e_a_frase_se_costura(self):
        saida, n = remover_numeracao_de_linha(self.MINUTA)
        saida = " ".join(saida.split())
        self.assertEqual(n, 4)
        self.assertIn("the nature of the clinical study or other studies", saida)
        self.assertNotIn("1155", saida)

    def test_o_numero_que_abre_o_trecho_tambem_sai(self):
        """Sem o `(?<![^\\s])` ele escapava — e abre quase todo trecho da minuta."""
        self.assertNotIn("1154", remover_numeracao_de_linha(self.MINUTA)[0])

    def test_numero_fora_da_cadeia_fica(self):
        texto = ("10 primeiro 11 segundo 12 terceiro 13 quarto, conforme a "
                 "Lei 8080 de 1990 e o artigo 196 . ")
        saida, _ = remover_numeracao_de_linha(texto)
        self.assertIn("8080", saida)
        self.assertIn("196", saida)
        self.assertNotIn(" 11 ", saida)

    def test_cadeia_curta_nao_e_numeracao(self):
        """Menos de quatro números seguidos não é corrida — é coincidência."""
        texto = "os incisos 1 e 2 e 3 do artigo tratam de definições gerais "
        self.assertEqual(remover_numeracao_de_linha(texto)[1], 0)

    # Caso real do FDA: o "5630" de um endereço parte a numeração em duas
    # corridas. A primeira versão removia só a mais longa e deixava a outra.
    DUAS_CADEIAS = ("comments within 90 days of 17 publication in the Federal Register "
                    "18 of the notice announcing the availability of the draft 19 guidance. "
                    "Submit written 20 comments to the Dockets Management Staff, "
                    "5630 Fishers Lane, 21 Room 1061, Rockville, MD 22 20852-1740. "
                    "Identify all 23 comments with the docket 24 number listed. ")

    def test_duas_cadeias_partidas_por_numero_legitimo(self):
        saida, _ = remover_numeracao_de_linha(self.DUAS_CADEIAS)
        for n in ("17", "18", "19", "20", "21", "23", "24"):
            self.assertNotIn(f" {n} ", saida, f"sobrou o número de linha {n}")

    def test_numero_legitimo_sobrevive_as_duas_passadas(self):
        saida, _ = remover_numeracao_de_linha(self.DUAS_CADEIAS)
        self.assertIn("5630 Fishers Lane", saida)
        self.assertIn("90 days", saida)

    def test_e_idempotente(self):
        """Rodar de novo não pode achar mais nada — senão a limpeza mente."""
        uma, _ = remover_numeracao_de_linha(self.DUAS_CADEIAS)
        duas, n = remover_numeracao_de_linha(uma)
        self.assertEqual(n, 0)
        self.assertEqual(uma, duas)


class RodapeDePaginaIMDRF(unittest.TestCase):

    def test_rodape_com_data_sai_inteiro(self):
        saida = limpo("Regulatory Pathway is described below. 21 September 2017 "
                      "Page 6 of 30 The following section")
        self.assertNotIn("Page 6 of 30", saida)
        self.assertNotIn("21 September 2017", saida)
        self.assertIn("described below. The following section", saida)

    def test_data_citada_no_texto_fica(self):
        """Só sai quando colada ao contador de página."""
        saida = limpo("O documento foi adotado em 21 September 2017 pelo IMDRF.")
        self.assertIn("21 September 2017", saida)


class TextoNormativoIntacto(unittest.TestCase):
    """A trava geral: nada que seja norma pode ser tocado por regra nenhuma."""

    def test_artigo_passa_sem_alteracao(self):
        texto = "Art. 5º São direitos do titular: I - confirmação da existência de tratamento;"
        self.assertEqual(limpar(texto)[0], texto)


if __name__ == "__main__":
    unittest.main()
