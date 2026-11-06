#!/usr/bin/env python3
"""Planilhas da auditoria de recuperação para o 2º e o 3º anotador.

PARA QUE SERVE. O primeiro especialista já julgou as 78 perguntas inteiras.
Dois anotadores a mais não servem para cobrir mais — servem para medir
CONCORDÂNCIA ENTRE ESPECIALISTAS, e isso exige que eles vejam as MESMAS
perguntas. Medido por simulação em 06/11 (bootstrap de cluster, α de
Krippendorff de intervalo, quatro juízes LLM como substitutos de anotadores):

  · os dois nos MESMOS 30 casos -> α com IC 95% de ±0,151, viés -0,008
  · cada um em 30 DIFERENTES    -> IC 95% de ±0,117, mas viés +0,031

O desenho disjunto dá intervalo mais estreito e ERRADO. O motivo é estrutural:
se cada anotador vê perguntas diferentes, o par ENTRE OS DOIS NOVOS nunca é
observado, e o α se calcula só com os pares que existem. Pior, o viés não é
limitável — o quanto ele infla depende de quanto o par não observado difere, e o
desenho não dá como saber. Com as mesmas perguntas os três pares aparecem e o
viés desaparece.

AS 30 SÃO O LOTE 1, e não um sorteio novo. Três razões: já é amostra
estratificada válida sozinha (10 de cada tipo, 5 por célula de tipo × acerto);
contém TODAS as 5 factuais que o sistema erra, que é a célula mais rara do
acervo inteiro; e é o mesmo recorte que o primeiro especialista recebeu como
lote 1, então a comparação é sobre as mesmas perguntas.

A AMOSTRA NÃO É RE-SORTEADA. Este script lê `auditoria/auditoria_trechos.csv` e
a CHAVE que já estão no repositório, em vez de recalcular o top-5. Assim as 78
perguntas, os 390 trechos e a divisão em lotes são, por construção, os mesmos
que o primeiro especialista recebeu. O que muda de anotador para anotador é só a
ORDEM DE EXIBIÇÃO.

A ORDEM É DIFERENTE PARA CADA UM, e nenhuma igual à do especialista 1. Semente
própria por anotador. Se os três vissem os 5 trechos na mesma ordem, um erro de
fadiga ("a essa altura eu já estava chutando") apareceria nos três no mesmo
lugar e passaria por concordância. A ordem verdadeira do ranking continua só na
chave.

AS 48 RESTANTES VÃO DEPOIS, marcadas como extra. A ordem de trabalho pedida é:
as 30 primeiro, depois a auditoria da GERAÇÃO, e só então as 48 restantes se
sobrar tempo. As 30 são o que a concordância precisa; as 48 são ganho de
cobertura e podem ficar incompletas sem prejuízo para a medida.

Uso:
  python eval/experimento_embedding/montar_auditoria_anotador.py
  python eval/experimento_embedding/montar_auditoria_anotador.py --anotadores 3
"""

from __future__ import annotations

import argparse
import csv
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parents[1]

ORIGEM = AQUI / "auditoria"
TRECHOS = ORIGEM / "auditoria_trechos.csv"
CHAVE = ORIGEM / "auditoria_CHAVE_nao_abrir_antes.csv"
SAIDA = AQUI / "auditoria_anotadores"

LOTE_OBRIGATORIO = "1"
# Semente por anotador. A 2026 e do especialista 1 (montar_auditoria.py) e NAO
# pode ser reusada aqui, senao a ordem dos trechos sairia igual a dele.
SEMENTE_BASE = 2027

for _f in (sys.stdout, sys.stderr):
    try:
        _f.reconfigure(encoding="utf-8")
    except Exception:
        pass


def limpar(t: str) -> str:
    return re.sub(r"\s+", " ", (t or "").strip())


def escrever_csv(caminho: Path, cabecalho: list[str], linhas: list[list]) -> None:
    # utf-8-sig e ';': o Excel em portugues abre com as colunas ja separadas.
    with caminho.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh, delimiter=";", quoting=csv.QUOTE_ALL)
        w.writerow(cabecalho)
        w.writerows(linhas)
    print(f"    ok {caminho.relative_to(RAIZ)}  ({len(linhas)} linhas)")


CAB_XLSX = [
    ("fazer", 17), ("qid", 8), ("pergunta", 44), ("trecho", 7),
    ("documento", 26), ("título", 26), ("pág.", 6),
    ("texto do trecho", 88),
    ("JUIZ: relevância 0-3", 13), ("JUIZ: observação do trecho", 26),
    ("JUIZ: os 5 bastam? S/PARCIAL/N", 15), ("JUIZ: o que falta", 30),
]
COLS_DO_BLOCO = (1, 2, 3, 11, 12)
COLS_AMARELAS_TRECHO = (9, 10)
COLS_AMARELAS_BLOCO = (11, 12)


def escrever_xlsx(caminho: Path, blocos: list[dict], nome: str) -> None:
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    from openpyxl.utils import get_column_letter
    from openpyxl.worksheet.datavalidation import DataValidation

    wb = Workbook()
    ws = wb.active
    ws.title = "auditoria"

    cinza = PatternFill("solid", fgColor="EFEFEF")
    azulado = PatternFill("solid", fgColor="EAF1F8")      # bloco extra
    amarelo = PatternFill("solid", fgColor="FFF4CC")
    topo = Border(top=Side(style="medium", color="999999"))
    corte = Border(top=Side(style="double", color="C00000"))
    alto = Alignment(vertical="top", wrap_text=True)
    centro = Alignment(vertical="center", horizontal="center", wrap_text=True)

    for i, (nome_col, largura) in enumerate(CAB_XLSX, start=1):
        c = ws.cell(row=1, column=i, value=nome_col)
        c.font = Font(bold=True, color="FFFFFF")
        c.fill = PatternFill("solid", fgColor="444444")
        c.alignment = centro
        ws.column_dimensions[get_column_letter(i)].width = largura
    ws.freeze_panes = "D2"
    ws.row_dimensions[1].height = 32

    linha = 2
    primeiro_extra = None
    for n, b in enumerate(blocos):
        ini = linha
        # A borda dupla vermelha marca onde as 30 acabam. Sem ela a fronteira
        # some no meio de 390 linhas e o anotador passa direto.
        inicio_do_extra = b["extra"] and primeiro_extra is None
        if inicio_do_extra:
            primeiro_extra = ini
        for pos, t in enumerate(b["trechos"], start=1):
            vals = [b["fazer"] if pos == 1 else None,
                    b["qid"] if pos == 1 else None,
                    b["pergunta"] if pos == 1 else None,
                    pos, t["documento"], t["titulo"], t["pagina"], t["texto"],
                    None, None, None, None]
            for col, v in enumerate(vals, start=1):
                c = ws.cell(row=linha, column=col, value=v)
                c.alignment = alto
                if col in COLS_AMARELAS_TRECHO:
                    c.fill = amarelo
                elif b["extra"]:
                    if n % 2 == 0:
                        c.fill = azulado
                elif n % 2 == 0:
                    c.fill = cinza
                if linha == ini:
                    c.border = corte if inicio_do_extra else topo
            linha += 1
        fim = linha - 1
        for col in COLS_DO_BLOCO:
            ws.merge_cells(start_row=ini, start_column=col,
                           end_row=fim, end_column=col)
            c = ws.cell(row=ini, column=col)
            c.alignment = centro if col in (1, 2) else alto
            if col in COLS_AMARELAS_BLOCO:
                c.fill = amarelo
            if col == 1:
                c.font = Font(bold=not b["extra"], size=10,
                              color="C00000" if not b["extra"] else "808080")

    rel = DataValidation(type="list", formula1='"0,1,2,3"', allow_blank=True,
                         showErrorMessage=True, error="Use 0, 1, 2 ou 3.",
                         errorTitle="Nota inválida")
    suf = DataValidation(type="list", formula1='"S,PARCIAL,N"', allow_blank=True,
                         showErrorMessage=True,
                         error="Use S, PARCIAL ou N.", errorTitle="Valor inválido")
    ws.add_data_validation(rel)
    ws.add_data_validation(suf)
    rel.add(f"I2:I{linha - 1}")
    suf.add(f"K2:K{linha - 1}")

    guia = wb.create_sheet("como preencher")
    guia.column_dimensions["A"].width = 112
    for i, (texto, negrito) in enumerate(guia_do_anotador(nome, primeiro_extra),
                                         start=1):
        c = guia.cell(row=i, column=1, value=texto)
        c.alignment = Alignment(vertical="top", wrap_text=True)
        c.font = Font(bold=negrito, size=13 if negrito else 11)

    wb.save(caminho)
    obrig = sum(1 for b in blocos if not b["extra"])
    print(f"    ok {caminho.relative_to(RAIZ)}  ({obrig} obrigatórias + "
          f"{len(blocos) - obrig} extra, {linha - 2} linhas de trecho)")


def guia_do_anotador(nome: str, linha_do_corte: int | None) -> list[tuple]:
    corte = (f"A borda dupla vermelha na linha {linha_do_corte} marca o fim das "
             f"30." if linha_do_corte else "")
    return [
        ("Auditoria da recuperação — como preencher", True),
        ("", False),
        ("Uma pergunta entra; o sistema busca em 54 normas (brasileiras e "
         "internacionais, em português e inglês) e devolve os 5 trechos que "
         "considerou mais relevantes. A resposta ao usuário é gerada SÓ com "
         "esses 5 trechos. Estamos auditando a busca.", False),
        ("", False),
        ("A ORDEM DE TRABALHO, e ela importa", True),
        ("", False),
        ("1) Faça as 30 primeiras perguntas desta planilha. Elas estão no topo, "
         "numeradas na coluna 'fazer' de '1 de 30' até '30 de 30'. "
         + corte, True),
        ("2) Ao terminar as 30, PARE AQUI e vá para a planilha "
         "AUDITORIA_GERACAO.xlsx. Faça o lote 1 dela.", True),
        ("3) Só depois disso, se sobrar tempo, volte a esta planilha e siga "
         "pelas perguntas marcadas 'extra' na coluna 'fazer'.", True),
        ("", False),
        ("Por que nessa ordem: as 30 primeiras são as mesmas que o outro "
         "avaliador está julgando. A medida de concordância entre avaliadores "
         "só existe nas perguntas que vocês dois fizeram, então essas 30 são o "
         "que não pode ficar pela metade. As 'extra' acrescentam cobertura e "
         "podem ficar incompletas sem prejuízo.", False),
        ("Não precisa avisar quando terminar as 30 — a coluna 'fazer' já "
         "separa. Se parar no meio das extra, devolva assim mesmo.", False),
        ("", False),
        ("Na aba 'auditoria', cada pergunta ocupa um bloco de 5 linhas. As "
         "colunas amarelas são suas.", False),
        ("", False),
        ("1) Para cada trecho, preencha 'relevância 0-3'", True),
        ("0 = não tem relação com a pergunta", False),
        ("1 = mesmo assunto, mas não contribui para a resposta", False),
        ("2 = contém PARTE do que a resposta precisa", False),
        ("3 = contém, sozinho, o suficiente para responder", False),
        ("", False),
        ("Julgue cada trecho CONTRA A PERGUNTA, não contra os outros trechos. "
         "Não há cota: os 5 podem ser 0, ou os 5 podem ser 3.", False),
        ("A ORDEM DOS TRECHOS É ALEATÓRIA. A coluna 'trecho' é posição na "
         "planilha, não a posição que o sistema deu — não tente inferir qual "
         "ele achou melhor. E a ordem é diferente da que os outros avaliadores "
         "receberam, de propósito.", True),
        ("", False),
        ("2) Depois dos 5, preencha 'os 5 bastam?' na primeira linha do bloco", True),
        ("S = sim, dá para responder a pergunta de forma completa e correta", False),
        ("PARCIAL = dá para responder parte, ou responder com ressalva", False),
        ("N = não dá", False),
        ("", False),
        ("Se marcar PARCIAL ou N, diga em 'o que falta' o que faltou: o "
         "dispositivo, a norma, o lado da comparação, o número, o prazo.", False),
        ("", False),
        ("Use 'observação do trecho' quando a nota não contar a história — por "
         "exemplo \"é o artigo certo mas a frase que responde está no trecho "
         "anterior\", que é o defeito mais comum que encontramos.", False),
        ("", False),
        ("Duas coisas que evitam mal-entendido", True),
        ("TODAS as perguntas aqui têm resposta no acervo. Você não precisa "
         "decidir se a resposta existe em algum lugar — só se ela está nestes "
         "5 trechos. Marcar N significa 'a busca não trouxe', nunca 'não "
         "existe'.", False),
        ("Metade das perguntas sorteadas são casos em que o sistema, pelo "
         "gabarito automático, ERROU. Isso é de propósito: auditar só os "
         "acertos mediria satisfação, não qualidade. Não se surpreenda se "
         "vários conjuntos parecerem ruins.", False),
        ("", False),
        ("Os enunciados estão em português; os trechos podem estar em "
         "português ou inglês, porque o acervo é bilíngue. Isso é normal, não "
         "é defeito.", False),
        ("", False),
        ("Não compare suas notas com as de ninguém enquanto preenche. A medida "
         "de concordância só vale se os julgamentos forem independentes.", True),
    ]


INSTRUCOES = """# Auditoria da recuperação — {nome}

Obrigado por auditar. São duas planilhas e a ordem entre elas importa.

## A ordem de trabalho

1. **As 30 primeiras perguntas** de `AUDITORIA_RECUPERACAO_{slug}.xlsx`. Estão
   no topo, numeradas na coluna `fazer` de `1 de 30` até `30 de 30`, e uma borda
   dupla vermelha marca onde elas acabam.
2. **Pare e vá para `AUDITORIA_GERACAO.xlsx`**, lote 1.
3. **Se sobrar tempo**, volte a esta planilha e siga pelas perguntas marcadas
   `extra`.

As 30 primeiras são as mesmas que o outro avaliador está julgando. A medida de
concordância entre avaliadores só existe nas perguntas que vocês dois fizeram,
então são essas que não podem ficar pela metade. As `extra` acrescentam
cobertura e podem ficar incompletas.

## O que o sistema faz

Uma pergunta entra; o sistema busca em 54 normas (brasileiras e internacionais,
em português e inglês) e devolve os **5 trechos** que considerou mais
relevantes. Cada trecho tem cerca de 1.200 caracteres. A pergunta é respondida
**só** com esses 5 trechos.

A pergunta de fundo é simples: **esses 5 trechos bastam?**

## Por trecho — `relevância 0-3`

| nota | quando |
|---|---|
| **0** | não tem relação com a pergunta |
| **1** | mesmo assunto, mas não contribui para a resposta |
| **2** | contém **parte** do que a resposta precisa |
| **3** | contém, sozinho, o suficiente para responder |

- Julgue o trecho **contra a pergunta**, não contra os outros trechos. Não há
  cota: os 5 podem ser 0, ou os 5 podem ser 3.
- **A ordem dos trechos é aleatória**, e é diferente da que os outros
  avaliadores receberam. A coluna `trecho` é posição de exibição, não a posição
  que o sistema deu.

## Por pergunta — `os 5 bastam?`

| valor | quando |
|---|---|
| **S** | sim, completa e correta |
| **PARCIAL** | dá para responder parte, ou responder com ressalva |
| **N** | não dá |

Se for `PARCIAL` ou `N`, diga em `o que falta` o que faltou.

**Todas as perguntas desta planilha têm resposta no acervo.** Quando marcar `N`,
está dizendo "a busca não trouxe", não "não existe".

Metade das perguntas sorteadas são casos em que o sistema, pelo gabarito
automático, **errou**. É de propósito.

## Uma regra

**Não compare suas notas com as de ninguém enquanto preenche.** A medida de
concordância só vale se os julgamentos forem independentes.
"""


def carregar_amostra() -> dict:
    """As 78 perguntas e os 390 trechos que o especialista 1 recebeu.

    Lidos do repositorio, nao recalculados: assim a amostra e os lotes sao por
    construcao os mesmos. A CHAVE entra para o rank real e o `esta_no_gabarito`
    sobreviverem na chave de cada anotador.
    """
    faltam = [p for p in (TRECHOS, CHAVE) if not p.exists()]
    if faltam:
        sys.exit("ERRO: faltam arquivos da auditoria do especialista 1:\n  "
                 + "\n  ".join(str(p.relative_to(RAIZ)) for p in faltam)
                 + "\nRode montar_auditoria.py primeiro.")
    csv.field_size_limit(10 ** 7)
    with TRECHOS.open(encoding="utf-8-sig") as fh:
        trechos = list(csv.DictReader(fh, delimiter=";"))
    with CHAVE.open(encoding="utf-8-sig") as fh:
        chave = {(r["qid"], r["trecho_id"]): r
                 for r in csv.DictReader(fh, delimiter=";")
                 if r["trecho_id"] and r["trecho_n_exibido"]}

    perguntas: dict = {}
    for r in trechos:
        k = (r["qid"], r["trecho_id"])
        if k not in chave:
            sys.exit(f"ERRO: {r['qid']} / {r['trecho_id']} nao esta na CHAVE; "
                     f"os dois arquivos nao sao da mesma rodada.")
        c = chave[k]
        p = perguntas.setdefault(r["qid"], {
            "qid": r["qid"], "lote": r["lote"],
            "pergunta": limpar(r["pergunta"]), "trechos": [],
        })
        p["trechos"].append({
            "trecho_id": r["trecho_id"],
            "documento": r["documento"], "titulo": limpar(r["titulo"]),
            "pagina": r["pagina"], "secao": limpar(r.get("secao") or ""),
            "texto": limpar(r["texto_do_trecho"]),
            "rank_real": c["rank_real_no_ranking"],
            "nota_reranker": c["nota_reranker"],
            "no_gabarito": c["esta_no_gabarito"],
        })
    ruins = [q for q, p in perguntas.items() if len(p["trechos"]) != 5]
    if ruins:
        sys.exit(f"ERRO: perguntas sem 5 trechos: {ruins[:5]}")
    return perguntas


def montar_para(nome: str, semente: int, perguntas: dict) -> None:
    rnd = random.Random(semente)
    # ordena por qid ANTES de embaralhar, para a semente ser o unico sorteio:
    # a ordem de insercao no dict depende da ordem do CSV e nao e garantia.
    obrig = sorted((p for p in perguntas.values()
                    if p["lote"] == LOTE_OBRIGATORIO),
                   key=lambda p: p["qid"])
    extra = sorted((p for p in perguntas.values()
                    if p["lote"] != LOTE_OBRIGATORIO),
                   key=lambda p: p["qid"])
    rnd.shuffle(obrig)
    rnd.shuffle(extra)

    slug = re.sub(r"[^a-z0-9]+", "_", nome.lower()).strip("_")
    blocos, linhas_t, linhas_p, linhas_c = [], [], [], []
    for grupo, e_extra in ((obrig, False), (extra, True)):
        for i, p in enumerate(grupo, start=1):
            ordem = list(p["trechos"])
            rnd.shuffle(ordem)
            rotulo = (f"extra {i} de {len(extra)}" if e_extra
                      else f"{i} de {len(obrig)}")
            blocos.append({
                "fazer": rotulo, "qid": p["qid"], "pergunta": p["pergunta"],
                "extra": e_extra,
                "trechos": [{"documento": t["documento"], "titulo": t["titulo"],
                             "pagina": t["pagina"], "texto": t["texto"]}
                            for t in ordem],
            })
            for pos, t in enumerate(ordem, start=1):
                linhas_t.append([
                    rotulo, p["qid"], p["pergunta"], pos, t["trecho_id"],
                    t["documento"], t["titulo"], t["pagina"], t["secao"],
                    t["texto"], "", "",
                ])
                linhas_c.append([
                    nome, rotulo, p["lote"], p["qid"], pos, t["rank_real"],
                    t["trecho_id"], t["nota_reranker"], t["no_gabarito"],
                ])
            linhas_p.append([rotulo, p["qid"], p["pergunta"], "", "", ""])

    SAIDA.mkdir(exist_ok=True)
    print(f"  {nome} (semente {semente}):")
    escrever_xlsx(SAIDA / f"AUDITORIA_RECUPERACAO_{slug}.xlsx", blocos, nome)
    escrever_csv(SAIDA / f"auditoria_trechos_{slug}.csv", [
        "fazer", "qid", "pergunta", "trecho_n", "trecho_id", "documento",
        "titulo", "pagina", "secao", "texto_do_trecho",
        "JUIZ_relevancia_0a3", "JUIZ_observacao",
    ], linhas_t)
    escrever_csv(SAIDA / f"auditoria_perguntas_{slug}.csv", [
        "fazer", "qid", "pergunta", "JUIZ_suficiente_S_PARCIAL_N",
        "JUIZ_o_que_falta", "JUIZ_observacao",
    ], linhas_p)
    escrever_csv(SAIDA / f"CHAVE_nao_abrir_antes_{slug}.csv", [
        "anotador", "fazer", "lote_original", "qid", "trecho_n_exibido",
        "rank_real_no_ranking", "trecho_id", "nota_reranker",
        "esta_no_gabarito",
    ], linhas_c)
    (SAIDA / f"INSTRUCOES_{slug}.md").write_text(
        INSTRUCOES.format(nome=nome, slug=slug), encoding="utf-8")
    print(f"    ok {(SAIDA / f'INSTRUCOES_{slug}.md').relative_to(RAIZ)}")
    copiar_geracao(nome, slug)


def copiar_geracao(nome: str, slug: str) -> None:
    """A planilha da GERACAO, uma copia por avaliador, com o nome dele.

    O CONTEUDO E IDENTICO para os dois, de proposito, e isso nao e preguica:
    as letras A e B tem de significar a mesma coisa nos dois arquivos. Se um
    visse o `rag` como A e o outro como B na mesma pergunta, a coluna "qual voce
    preferiria receber?" deixaria de ser comparavel entre eles sem passar por
    duas chaves diferentes. A ordem das perguntas tambem fica igual: assim, se
    um parar na pergunta 20, os dois pararam nas MESMAS 20 e a sobreposicao
    sobrevive ao trabalho incompleto.

    O que muda e so o nome no cabecalho da aba de instrucoes e duas linhas
    ligando as duas planilhas, para ninguem comecar por esta.
    """
    from openpyxl import load_workbook
    from openpyxl.styles import Alignment, Font

    origem = AQUI / "auditoria_geracao/AUDITORIA_GERACAO.xlsx"
    if not origem.exists():
        print(f"    ! nao achei {origem.relative_to(RAIZ)}; rode "
              f"montar_auditoria_geracao.py primeiro. A planilha da geracao "
              f"deste avaliador NAO foi gerada.")
        return
    wb = load_workbook(origem)
    guia = wb["como preencher"]
    guia.cell(row=1, column=1,
              value=f"Auditoria da geração — como preencher ({nome})")
    guia.insert_rows(2, amount=4)
    linhas = [
        ("", False),
        ("ANTES DESTA, faça as 30 primeiras perguntas de "
         f"AUDITORIA_RECUPERACAO_{slug}.xlsx. Esta planilha é o passo 2.", True),
        ("Aqui, faça o LOTE 1 inteiro. Se sobrar tempo depois dele, volte à "
         "planilha da recuperação e siga pelas perguntas marcadas 'extra' — "
         "os lotes 2 e 3 desta são os últimos da fila.", True),
        ("", False),
    ]
    for i, (texto, negrito) in enumerate(linhas, start=2):
        c = guia.cell(row=i, column=1, value=texto)
        c.alignment = Alignment(vertical="top", wrap_text=True)
        c.font = Font(bold=negrito, size=13 if negrito else 11)
    destino = SAIDA / f"AUDITORIA_GERACAO_{slug}.xlsx"
    wb.save(destino)
    ws = wb["auditoria"]
    print(f"    ok {destino.relative_to(RAIZ)}  "
          f"({ws.max_row - 1} linhas de resposta, conteudo identico ao original)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--anotadores", type=int, default=2,
                    help="quantas planilhas gerar; padrao 2 (o 2o e o 3o "
                         "avaliador, ja que o 1o tem as 78 inteiras)")
    ap.add_argument("--nomes", nargs="*",
                    help="nomes para os arquivos; padrao anotador_2, "
                         "anotador_3, ...")
    args = ap.parse_args()

    nomes = args.nomes or [f"anotador_{i + 2}" for i in range(args.anotadores)]
    if len(nomes) != args.anotadores:
        sys.exit(f"ERRO: --anotadores {args.anotadores} e {len(nomes)} nomes.")

    perguntas = carregar_amostra()
    n_obrig = sum(1 for p in perguntas.values() if p["lote"] == LOTE_OBRIGATORIO)
    print(f"amostra do especialista 1: {len(perguntas)} perguntas, "
          f"{sum(len(p['trechos']) for p in perguntas.values())} trechos")
    print(f"  obrigatorias (lote {LOTE_OBRIGATORIO}): {n_obrig}   "
          f"extra: {len(perguntas) - n_obrig}")
    print(f"  por anotador: {n_obrig} perguntas e {n_obrig * 5} trechos "
          f"obrigatorios, {(len(perguntas) - n_obrig) * 5} trechos de extra\n")

    for i, nome in enumerate(nomes):
        montar_para(nome, SEMENTE_BASE + i, perguntas)
        print()

    print("MANDE para cada avaliador os TRES arquivos com o nome dele:")
    print("  AUDITORIA_RECUPERACAO_<nome>.xlsx, AUDITORIA_GERACAO_<nome>.xlsx")
    print("  e INSTRUCOES_<nome>.md")
    print("As CHAVES NAO vao: elas tem o rank real e o gabarito.")
    print("\nAs 30 obrigatorias sao AS MESMAS nas duas planilhas; so a ordem de")
    print("exibicao difere, e nenhuma das duas repete a ordem do especialista 1.")


if __name__ == "__main__":
    main()
