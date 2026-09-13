"""Identidade de um trecho — fonte única da verdade.

O id de um trecho é derivado do CONTEÚDO, não da posição:

    <document_id>_<12 primeiros hex do sha256 do texto normalizado>

Por que não a posição: até 09/2026 o id era um endereço
(`<doc>_p{página}_c{índice}`). Quando o recorte mudava, o "terceiro pedaço da
página 12" virava outro texto e as anotações passavam a apontar para outra
coisa sem avisar — aconteceu no re-chunking de 01/09/2026 e contaminou ~40% das
âncoras do gabarito sem que nada falhasse.

Propriedades que este esquema garante:
  • mesmo texto ⇒ mesmo id, independentemente dos vizinhos;
  • acrescentar documento não renumera nada do que já existe;
  • reexecutar a extração com os mesmos parâmetros reproduz os mesmos ids;
  • se o recorte mudar de verdade, o id some — falha barulhenta, não silenciosa.

IMPORTANTE: o hash é do texto CRU do trecho, nunca de `build_embedding_text()`.
É isso que permite enriquecer metadado depois (preencher `section_path`, por
exemplo) sem invalidar uma única anotação: muda o que se embute, recalculam-se
os vetores, e os ids ficam de pé.

Quem usa: src/extract_to_jsonl.py, src/extract_html_to_jsonl.py e
eval/tools/migrar_ids_por_conteudo.py. Não duplique a função — importe daqui,
senão os ids divergem entre a migração e a próxima extração.
"""

from __future__ import annotations

import hashlib
import re

HASH_LEN = 12

_ESPACOS = re.compile(r"\s+")


def normalizar_texto(texto: str) -> str:
    """Colapsa espaços em branco. Diferença de formatação não pode gerar id novo."""
    if texto is None:
        return ""
    if not isinstance(texto, str):
        texto = str(texto)
    return _ESPACOS.sub(" ", texto).strip()


def chunk_id_por_conteudo(document_id: str, texto: str) -> str:
    """Id estável de um trecho. Colisão (texto idêntico repetido no mesmo
    documento) é resolvida por quem chama, com sufixo contador — ver
    `eval/tools/migrar_ids_por_conteudo.py`."""
    digest = hashlib.sha256(normalizar_texto(texto).encode("utf-8")).hexdigest()
    return f"{document_id}_{digest[:HASH_LEN]}"
