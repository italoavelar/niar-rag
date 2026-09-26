#!/usr/bin/env python3
"""Remove do corpus o ruído de extração, sem reextrair.

POR QUE ISTO EXISTE, e por que não basta consertar o extrator. A causa do ruído
foi corrigida em `src/structured_units.py` (`_serialize_table_row` deixou de
prefixar células com cabeçalho inventado), mas essa correção só vale da próxima
extração em diante. O corpus em `documents.jsonl` foi extraído antes e carrega o
andaime dentro do texto — e é ele que o LEME serve hoje.

Reextrair custaria caro e mexeria em tudo. Como o ruído é mecanicamente
identificável, dá para removê-lo do texto já extraído: as duas pontas ficam
consertadas, a de hoje e a de amanhã.

⚠️ Limpar o texto MUDA o id do trecho (o id vem do conteúdo). Os ids alterados
precisam de reancoragem do gabarito — que é mecânica, por `qrels_text`. Medido:
150 trechos mudam, 14 âncoras são atingidas, 13 delas reancoram sozinhas.

O QUE É REMOVIDO, e o que NÃO é:

    Coluna 3:                  rótulo inventado quando o PDF não tem cabeçalho.
    Colunas: Coluna 1 | ...    O conteúdo das células FICA; some só o rótulo.

    [TABELA] / [/TABELA]       marcador de fronteira, não conteúdo. Some.

    CÓDIGO DE ÉTICA MÉDICA 35  rodapé institucional repetido centenas de vezes.
                               Remove só quando isolado com número de página.

Nada que seja conteúdo normativo é tocado. A regra é conservadora de propósito:
na dúvida, mantém — perder texto real é pior que carregar ruído.

Medido em 22/set, antes da limpeza: 5,3% dos trechos com andaime de tabela,
2,6% com rodapé do CEM. Concentrados no UNESCO e no Código de Ética.

Uso:
  python eval/experimento_embedding/limpar.py            # relatório, não grava
  python eval/experimento_embedding/limpar.py --aplicar  # grava
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
PROJECT_ROOT = AQUI.parents[1]

for _f in (sys.stdout, sys.stderr):
    try:
        _f.reconfigure(encoding="utf-8")
    except Exception:
        pass

CORPUS = PROJECT_ROOT / "data/processed/documents.jsonl"
SAIDA = AQUI / "dados/corpus_limpo.jsonl"

REGRAS = [
    # "Colunas: Coluna 1 | Coluna 2 | Coluna 3" — só rótulos inventados
    ("cabecalho_inventado",
     re.compile(r"Colunas:\s*(?:Coluna\s*\d+\s*\|?\s*)+\n?")),
    # "Coluna 3:" isolado, com ou sem conteúdo na mesma linha
    ("rotulo_inventado",
     re.compile(r"^[ \t]*Coluna\s*\d+:[ \t]*$\n?", re.M)),
    ("rotulo_inventado_inline",
     re.compile(r"(?m)^[ \t]*Coluna\s*\d+:[ \t]*")),
    # marcadores de fronteira de tabela
    ("marcador_tabela", re.compile(r"\[/?TABELA\]\n?")),
    # Rodapé do Código de Ética. Exige CAIXA ALTA **e** número de página — as
    # duas marcas que o rodapé tem e a menção legítima não tem.
    #
    # A primeira versão casava "Código de Ética Médica" em qualquer lugar e
    # destruiu frase real: "Comissão Nacional de Revisão do Código de Ética
    # Médica do CFM" virou "Revisão do do CFM", e "Aprova o Código de Ética
    # Médica." virou "Aprova o .". Remover conteúdo é pior que carregar ruído.
    # Sem `re.I`: a CAIXA ALTA é metade do critério. O número de página é a outra
    # metade. Âncora de linha não serve — o rodapé vem colado no texto seguinte
    # ("CÓDIGO DE ÉTICA MÉDICA 35Art. 78."), sem quebra.
    ("rodape_cem",
     re.compile(r"\s*\d{1,3}\s*CÓDIGO\s*(?:DE\s*)?ÉTICA\s*MÉDICA\s*(?:DE)?\s*\d{0,3}\s*"
                r"|\s*CÓDIGO\s*(?:DE\s*)?ÉTICA\s*MÉDICA\s*(?:DE)?\s*\d{1,3}\s*")),
    # Sobra do cabeçalho inventado que não vem precedido de "Colunas:"
    ("coluna_solta", re.compile(r"\|\s*Coluna\s*\d+\b")),
    # Marca d'água da Câmara dos Deputados, injetada no meio da frase:
    # "cooperação internacional para o atendimento a *CD252287612600*padrões
    # técnicos". Uma string só, 36 trechos, nada legítimo casa com ela.
    ("marca_dagua_camara", re.compile(r"\*CD\d{10,}\*")),
    # Barra lateral vertical da mesma capa, que o extrator lê como texto corrido:
    # "uso ético e responsável da PL inteligência artificial com base na
    # Apresentação: centralidade da pessoa humana". São três fragmentos —
    # "Apresentação:", "PL" e a data — costurados em posições arbitrárias.
    #
    # "PL" só é removido quando NÃO é a sigla usada de verdade. A primeira
    # versão exigia só vizinho minúsculo e comia "PL nº 2338" — o "nº" começa
    # com minúscula. Destruir uma citação de norma é pior que carregar ruído,
    # então a regra recusa qualquer "PL" seguido de número, com ou sem "nº".
    # Sobra o que é barra lateral mesmo: "da PL inteligência", "PLConcorrência".
    # "Apresentação:" cai tanto antes de minúscula ("na Apresentação:
    # centralidade") quanto antes de maiúscula ("essencial; Apresentação: II –
    # sistemas"). O que nunca acontece é ele ser um título de verdade — título
    # começa linha. `(?<!^)` em modo multilinha é o que preserva esse caso.
    ("barra_lateral_camara", re.compile(r"(?m)(?<!^)Apresenta[çc][ãa]o:")),
    # "PL" aparece solto ("da PL inteligência") e colado ("em PLdiversos",
    # "PLConcorrência"). Segue sendo recusado diante de número: "PL 2338" e
    # "PL nº 2338" ficam. Siglas como PLN e PLoS também, por exigirem minúscula
    # logo depois da segunda letra.
    ("sigla_pl_solta",
     re.compile(r"(?<=[a-zà-ÿ,]\s)PL(?![\s.]*(?:n[ºo°]?\.?\s*)?\d)"
                r"(?=\s+[a-zà-ÿ]|[A-ZÀ-Ýa-zà-ÿ][a-zà-ÿ])")),
    # O mesmo "PL" antes de MAIÚSCULA. Alargar a regra acima para qualquer
    # maiúscula comeria "o PL Brasileiro de IA" em algum documento futuro, então
    # o gatilho é outro: marcador de estrutura legal. Um projeto de lei nunca
    # escreve "PL CAPÍTULO III" nem "PL Art. 45" — quando isso aparece, o "PL"
    # é a barra lateral da Câmara caindo entre o fim de uma frase e o título
    # seguinte. Foram os 5 casos que sobraram no corpus, todos assim.
    # O `\b` fica FORA da alternativa "Art\." de propósito: entre o ponto e o
    # espaço não há fronteira de palavra (os dois são não-palavra), então
    # `Art\.\b` nunca casa — e eram justamente 3 dos 5 casos.
    ("sigla_pl_antes_de_estrutura",
     re.compile(r"(?<=[a-zà-ÿ,.]\s)PL\s+(?=(?:CAPÍTULO|TÍTULO|SEÇÃO|Seção|"
                r"Subseção|Artigo|Parágrafo|ANEXO)\b|Art\.)")),
    # Rodapé de página dos documentos do IMDRF: "21 September 2017 Page 3 of 30".
    # A data sai junto — sozinha ela vira lixo pendurado no fim da frase. Só sai
    # quando colada ao contador de página, então data citada no texto fica.
    ("rodape_pagina_imdrf",
     re.compile(r"\s*\d{1,2}\s+\w+\s+\d{4}\s+Page\s+\d+\s+of\s+\d+")),
]


# Inteiro isolado entre espaços (ou no começo do trecho). O `(?<![^\s])` vale
# "precedido de espaço OU início da string" — sem ele, o número que abre o
# trecho escapa, e ele abre quase todos num documento numerado por linha.
_NUMERO_SOLTO = re.compile(r"(?<![^\s])(\d{1,4})(?=\s)")


def documento_com_numeracao_de_linha(texto_inteiro: str) -> bool:
    """O PDF numera cada linha? (minuta em consulta pública)

    POR QUE A DECISÃO É POR DOCUMENTO E NÃO POR TRECHO. Uma cadeia de inteiros
    quase consecutivos parece numeração de linha mas quase nunca é: o sumário do
    NIST ("4.1 ... 46  4.2 ... 47"), as notas de rodapé da EBIA ("6 THE WORLD
    BANK ... 7 OECD"), os artigos da convenção do Conselho da Europa ("Article
    18 ... 19 ... 20"), "GRUPO 1 GRUPO 2 GRUPO 3" no CFM. Aplicar a regra por
    trecho estragaria conteúdo legítimo em cinco documentos.

    O que só a minuta tem são as DUAS coisas ao mesmo tempo, e com folga:

        densidade   1 número solto a cada 85 caracteres  (2º lugar: 234)
        monotonia   88% dos números maiores que o anterior  (2º lugar: 67%)

    Medido em 22/set sobre os 54 documentos: só `ai_device_software_guidance_
    FDA_2025` passa, e nada chega perto dos dois limiares juntos.
    """
    numeros = [int(m.group(1)) for m in _NUMERO_SOLTO.finditer(texto_inteiro)]
    if len(numeros) < 50:
        return False
    densidade = len(texto_inteiro) / len(numeros)
    crescentes = sum(1 for a, b in zip(numeros, numeros[1:]) if b > a)
    return densidade < 150 and crescentes / (len(numeros) - 1) > 0.75


def remover_numeracao_de_linha(texto: str, _passadas: int = 6) -> tuple[str, int]:
    """Tira os números que formam corrida crescente no trecho.

    Mesmo dentro da minuta há número legítimo — marcador de nota, ano, número de
    seção. Remover todo inteiro solto levaria esses junto. O que se remove é a
    CADEIA: pelo menos quatro números em que cada um vem 1 a 15 acima do
    anterior. Número fora da cadeia fica.

    EM LAÇO, porque um número legítimo no meio PARTE a numeração em duas cadeias:

        ... comments within 90 days of 17 publication ... draft 18 guidance ...
        Dockets Management Staff, 5630 Fishers Lane, 20 Room 1061 ... 21 ... 23

    O "5630" do endereço quebra a corrida, e sobram [17,18,19] de um lado e
    [20,21,23,24] do outro. A primeira versão removia só a cadeia mais longa e
    deixava a outra — não era idempotente, e 17 trechos do FDA ficaram com
    numeração pela metade depois de uma limpeza que se dizia concluída.
    """
    total = 0
    for _ in range(_passadas):
        texto, n = _uma_cadeia(texto)
        total += n
        if not n:
            break
    return texto, total


def _uma_cadeia(texto: str) -> tuple[str, int]:
    """Remove a corrida mais longa do trecho, se tiver 4 ou mais números."""
    achados = [(m.start(1), m.end(1), int(m.group(1)))
               for m in _NUMERO_SOLTO.finditer(texto)]
    melhor: list[tuple[int, int, int]] = []
    atual: list[tuple[int, int, int]] = []
    for item in achados:
        if atual and 1 <= item[2] - atual[-1][2] <= 15:
            atual.append(item)
        else:
            if len(atual) > len(melhor):
                melhor = atual
            atual = [item]
    if len(atual) > len(melhor):
        melhor = atual
    if len(melhor) < 4:
        return texto, 0
    for inicio, fim, _ in reversed(melhor):
        texto = texto[:inicio] + texto[fim:]
    return texto, len(melhor)


def legenda_repetida(texto: str) -> str | None:
    """Legenda de caixa de texto repetida em cada linha, ou None.

    `find_tables()` enquadra caixa de texto como tabela de uma coluna só, e aí o
    "cabeçalho" é a legenda da caixa — que passa a prefixar toda linha e corta a
    frase ao meio a cada quebra:

        Box 1. Courts' case management systems...: In the US, landlords are
        Box 1. Courts' case management systems...: courts' case management

    O QUE SEPARA ISTO DE UMA TABELA DE VERDADE. Tabela real tem dois ou mais
    cabeçalhos ALTERNANDO ("SIGNIFICANCE LEVEL:" e depois "DESCRIPTION:"), e aí
    o prefixo é a única coisa que diz de que coluna a célula veio. A legenda de
    caixa é um prefixo só, repetido. Exigir exatamente um prefixo repetido é o
    que deixa o UNESCO intacto e limpa a OCDE.

    A correção de origem está em `src/structured_units.py`: tabela de uma coluna
    não leva prefixo. Isto aqui é para o corpus já extraído.
    """
    contagem: dict[str, int] = {}
    for m in re.finditer(r"(?m)^[ \t]*([^\n:]{6,120}):[ \t]", texto):
        chave = m.group(1).strip()
        contagem[chave] = contagem.get(chave, 0) + 1
    repetidos = [k for k, v in contagem.items() if v >= 3]
    return repetidos[0] if len(repetidos) == 1 else None


MATERIA_EDITORIAL = re.compile(
    r"Supervis[ãa]o editorial|Copidesque|Diagrama[çc][ãa]o|Projeto [Gg]r[áa]fico"
    r"|\bISBN\b|Dados Internacionais de Cataloga|\b\d+\s*p\.\s*\d+\s*cm"
    r"|COMPOSI[ÇC][ÃA]O DO CONSELHO|Conselheiros? (?:titulares|suplentes)"
    r"|COMISS[ÃA]O NACIONAL DE REVIS[ÃA]O|COMISS[ÕO]ES ESTADUAIS")

VERBO = re.compile(
    r"\b(dever[áã]o?|deve|devem|ser[áã]o?|[ée]|s[ãa]o|fica|ficam|cabe|cabendo"
    r"|compete|poder[áã]|caber[-\s]?lhes|entrar|exercer|aplicar)\b", re.I)


def eh_materia_editorial(texto: str) -> bool:
    """Front/back matter: expediente, créditos, ficha catalográfica, lista de nomes.

    Isto não é filtro semântico — não julga se o conteúdo é "relevante". É
    estrutural: capa, créditos e lista de conselheiros não são o documento, são
    a embalagem dele.

    A TRAVA CONTRA FALSO POSITIVO. O `decreto_44045_BR_1958` fala de "conselheiros
    suplentes" porque REGULA a eleição deles — "deverão ser eleitos na mesma
    ocasião dos efetivos, cabendo-lhes entrar em exercício em caso de
    impedimento". É norma, não lista. O que separa:

        lista de nomes     muitas palavras capitalizadas, quase nenhum verbo
        norma sobre nomes  verbos deônticos e de ligação em densidade normal

    Exige-se as duas condições juntas — marcador editorial E cara de lista — para
    que texto normativo que mencione os mesmos termos sobreviva.
    """
    if not MATERIA_EDITORIAL.search(texto):
        return False
    palavras = re.findall(r"\b[A-Za-zÀ-ÿ]{2,}\b", texto)
    if len(palavras) < 10:
        return False
    capitalizadas = sum(1 for p in palavras if p[0].isupper()) / len(palavras)
    verbos = len(VERBO.findall(texto)) / len(palavras)
    return capitalizadas > 0.45 and verbos < 0.04


def limpar(texto: str, numerado: bool = False) -> tuple[str, dict[str, int]]:
    """`numerado`: o documento deste trecho numera cada linha do PDF.

    Vem de fora porque a decisao e do DOCUMENTO, nao do trecho — ver
    `documento_com_numeracao_de_linha`.
    """
    contagem: dict[str, int] = {}
    if numerado:
        texto, n = remover_numeracao_de_linha(texto)
        if n:
            contagem["numero_de_linha"] = n
    # Em laço porque a legenda vem em camadas: "Box 8. Colombia: Using Legal
    # Needs Surveys to address unmet needs: <conteúdo>". Uma passada só tira
    # "Box 8. Colombia" — o que sobra ainda é legenda repetida. O limite de 4
    # existe só para não haver laço infinito se alguma regra virar idempotente
    # pela metade; medido, nenhum trecho precisa de mais que 2.
    for _ in range(4):
        legenda = legenda_repetida(texto)
        if not legenda:
            break
        texto, n = re.subn(r"(?m)^[ \t]*" + re.escape(legenda) + r":[ \t]*", "", texto)
        if not n:
            break
        contagem["legenda_de_caixa"] = contagem.get("legenda_de_caixa", 0) + n
    for nome, rx in REGRAS:
        texto, n = rx.subn(" ", texto)
        if n:
            contagem[nome] = contagem.get(nome, 0) + n
    # espaço colapsado só onde a remoção deixou buraco
    texto = re.sub(r"[ \t]{2,}", " ", texto)
    texto = re.sub(r"\n{3,}", "\n\n", texto)
    return texto.strip(), contagem


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--aplicar", action="store_true", help="grava corpus_limpo.jsonl")
    ap.add_argument("--producao", action="store_true",
                    help="grava em data/processed/documents.jsonl, com backup")
    args = ap.parse_args()

    registros = [json.loads(l) for l in CORPUS.open(encoding="utf-8") if l.strip()]

    # Quais documentos sao minuta numerada por linha. Precisa do documento
    # INTEIRO: e uma propriedade de densidade, invisivel num trecho so.
    inteiros: dict[str, list[str]] = {}
    for r in registros:
        inteiros.setdefault(r["metadata"]["document_id"], []).append(r["text"])
    numerados = {d for d, ts in inteiros.items()
                 if documento_com_numeracao_de_linha(" ".join(ts))}
    total: dict[str, int] = {}
    tocados = 0
    antes = sum(len(r["text"]) for r in registros)
    saida = []

    removidos = [r for r in registros if eh_materia_editorial(r["text"])]
    registros = [r for r in registros if not eh_materia_editorial(r["text"])]

    # O id vem do CONTEÚDO (`src/chunk_id.py`). Mudar o texto sem recalcular o id
    # quebra a invariante do projeto: ficaria um id que não é o hash do próprio
    # texto, e a próxima extração geraria outro id para o mesmo conteúdo sem que
    # ninguém entendesse por quê. Recalcular é obrigatório — e é o que obriga a
    # reancorar o gabarito depois.
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from chunk_id import chunk_id_por_conteudo

    for r in registros:
        novo, cont = limpar(r["text"],
                            numerado=r["metadata"]["document_id"] in numerados)
        if cont:
            tocados += 1
            for k, v in cont.items():
                total[k] = total.get(k, 0) + v
        registro = {**r, "text": novo}
        if novo != r["text"]:
            registro["id"] = chunk_id_por_conteudo(r["metadata"]["document_id"], novo)
        saida.append(registro)

    depois = sum(len(r["text"]) for r in saida)
    vazios = [r for r in saida if len(r["text"]) < 50]

    if numerados:
        print(f"minuta numerada por linha  : {', '.join(sorted(numerados))}")
    print(f"trechos            : {len(registros)}")
    print(f"trechos alterados  : {tocados}  ({tocados/len(registros):.1%})")
    print(f"caracteres         : {antes:,} → {depois:,}  ({(antes-depois)/antes:.2%} removido)"
          .replace(",", "."))
    print("\nremoções por regra:")
    for k, v in sorted(total.items(), key=lambda x: -x[1]):
        print(f"   {k:26s} {v:6d}")
    print(f"\ntrechos removidos por serem matéria editorial: {len(removidos)}")
    for r in removidos:
        print(f"   {r['metadata']['document_id'][:34]:34s} {r['text'][:58]!r}")
    print(f"\ntrechos que ficaram com < 50 chars: {len(vazios)}")
    for r in vazios[:5]:
        print(f"   {r['metadata']['document_id'][:40]}: {r['text'][:60]!r}")

    if args.producao:
        import shutil
        from datetime import datetime
        carimbo = datetime.now().strftime("%Y%m%d-%H%M%S")
        shutil.copy2(CORPUS, f"{CORPUS}.bak-pre-limpeza-{carimbo}")
        with CORPUS.open("w", encoding="utf-8") as f:
            for r in saida:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\n\u2713 {CORPUS}  ({len(saida)} trechos)")
        print(f"\u2713 backup: {CORPUS.name}.bak-pre-limpeza-{carimbo}")
        print("  \u26a0 o id dos trechos alterados MUDOU \u2014 reancore o gabarito:")
        print("    python eval/tools/reancorar_por_texto.py --aplicar")
        return

    if args.aplicar:
        with SAIDA.open("w", encoding="utf-8") as f:
            for r in saida:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\n✓ {SAIDA}")
        print("  Recorte agora parte deste arquivo: bench_tamanho.py usa corpus_limpo.jsonl")
    else:
        print("\n(relatório apenas — use --aplicar para gravar)")


if __name__ == "__main__":
    main()
