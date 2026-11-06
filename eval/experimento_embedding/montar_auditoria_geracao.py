#!/usr/bin/env python3
"""Planilha de auditoria da GERAÇÃO: a mesma pergunta, a resposta com e sem RAG.

PARA QUE SERVE. A comparação `rag` × `norag` hoje é julgada por um painel de
quatro LLMs (rubrica `mista-v2`). Esse painel precisa de uma âncora humana: em
05/11 dois dos quatro se recusaram a pontuar 809 notas de
`correctness`/`completeness` alegando motivos que o próprio prompt contradizia,
e isso só apareceu porque alguém leu as justificativas. Um especialista julgando
as mesmas respostas dá (1) concordância especialista × painel, por critério, e
(2) um veredito independente sobre a pergunta do artigo: a recuperação melhora a
resposta, ou o modelo já sabia?

OS QUATRO TIPOS, INCLUSIVE AS IRRESPONDÍVEIS. A primeira versão deste script
herdava as 78 perguntas da auditoria de recuperação, que são todas respondíveis
— e isso estava errado para a geração. Na recuperação a exclusão se justifica:
para dizer "estes 5 trechos não respondem porque o acervo não tem a resposta", o
juiz teria de garantir que a resposta não está em nenhuma das 54 normas, o que
não se verifica lendo 5 trechos. Mas aqui o objeto é a RESPOSTA, e a abstenção é
metade da comparação: medido em 05/11 no qwen3_27b, das 75 irrespondíveis o
`norag` afirmou em 15 e o `rag` em 7, e em 2 os dois afirmaram. Deixar essas
fora era medir meia pergunta — a parte em que o RAG ajuda, sem a parte em que
ele evita invenção.

COMO A AMOSTRA É MONTADA (75 perguntas, 3 lotes)

  30 irrespondíveis e 50 respondíveis (17 factual, 17 multi_hop, 16
  comparative). As irrespondíveis são SOBRE-AMOSTRADAS de propósito — a
  população é 1/4 de cada tipo, e aqui elas são mais de 1/3 —, pelo motivo em
  `repartir_irrespondiveis`: os pares que decidem o teste de abstenção são
  raros e têm de entrar inteiros. O custo do outro lado é pequeno: 57 → 50
  respondíveis alarga o IC 95% do kappa de `correctness` de 0,18 para 0,19.

  DENTRO DE CADA TIPO RESPONDÍVEL, metade acerta e metade erra no top-5 — o
  acerto da recuperação é o mecanismo pelo qual o RAG ajudaria, então é a
  estratificação que importa. Quando a célula não tem metade, entra inteira: só
  existem 5 factuais que o sistema erra em todo o acervo (Strict Recall@5 da
  factual = 70/75), então a factual sai 14 acerta + 5 erra, e não há cota que
  mude isso.

  AS RESPONDÍVEIS SAEM DE DENTRO DAS 78 da auditoria de recuperação, não de todo
  o acervo. É de propósito: o `qid` fica compartilhado com a outra planilha, e
  dá para cruzar as duas e responder o que nenhuma responde sozinha — quando o
  especialista disse que os 5 trechos NÃO bastavam, a resposta saiu errada? As
  78 já são uma amostra estratificada pelas mesmas células, então subamostrar
  delas preserva a estratificação.

  AS IRRESPONDÍVEIS LEVAM TODOS OS PARES DISCORDANTES. Num teste pareado só o
  par discordante carrega informação: dos 75 irrespondíveis do qwen3_27b, 57
  são pares em que os dois braços fizeram a mesma coisa e não entram no
  p-valor. Quem decide são os 18 em que um se absteve e o outro afirmou. Eles
  entram inteiros, nas duas direções, mais os 2 em que os DOIS afirmaram, que é
  o caso mais grave; o resto completa com "os dois se abstiveram". Assim o
  teste do especialista tem a mesma força que o do painel sobre as 75, numa
  planilha de 25.

  Isso corrige um erro da versão anterior, que tomava as 5 contra o RAG e só 6
  das 13 a favor: o teste do sinal caía de p=0,096 (todas) para p=0,754 (a
  amostra). O viés não era sobre-amostrar o difícil, era sobre-amostrar UMA
  DIREÇÃO.

  A AMOSTRA NÃO É PROPORCIONAL, e a análise tem de pesar por célula para
  qualquer TAXA (quantas vezes o sistema se abstém, quantas vezes acerta). O
  que sai direto daqui são as COMPARAÇÕES pareadas. É a mesma troca da
  auditoria de recuperação: metade acerta e metade erra, quando a população é
  54/46. Auditar só o que o sistema acerta mediria satisfação, não qualidade.

QUATRO DECISÕES DE DESENHO:

  O BRAÇO É CEGO. As duas respostas aparecem como "A" e "B" em ordem sorteada
  por pergunta, BALANCEADA por lote: metade das perguntas de cada lote tem o
  `rag` em A. Com sorteio independente por par o desequilíbrio é grande — na
  primeira versão o `rag` caiu em A em 49 das 78 e em B em 29 —, e um
  especialista que percebe "a A costuma ser a melhor" fica ancorado.

  O CEGAMENTO É PARCIAL, E ISSO FICA MEDIDO. A própria redação delata o braço em
  boa parte dos casos: nas 78 do qwen3_27b, 45 respostas do `rag` dizem
  "contexto fornecido", "no contexto" ou "trecho", contra 6 do `norag`. Não
  normalizo essas frases — "com base no contexto fornecido" em vez de nomear a
  norma é um defeito que o especialista deve poder penalizar, e reescrever a
  saída do sistema esconderia o que a auditoria procura. O que compensa é a
  planilha pedir nota ABSOLUTA de cada resposta, e não só a preferência: a nota
  absoluta sobrevive ao palpite sobre o braço, a preferência não.

  O TIPO DA PERGUNTA NÃO APARECE, e isso agora é essencial, não cosmético. Se a
  planilha dissesse quais são as irrespondíveis, estaria entregando a resposta
  da própria pergunta que se quer medir — se o sistema deveria ter se abstido. É
  por isso que a escala de cada resposta inclui o valor ABSTEVE, que é
  descritivo e vale para os dois casos, em vez de uma nota que só faz sentido
  sabendo se a pergunta tinha resposta. Quem decide isso é o especialista, na
  coluna "existe norma que responda isso?", com NÃO SEI permitido.

  NADA DO MODELO NA PLANILHA DO JUIZ. Sem nome do gerador, sem nota dos juízes
  LLM, sem resposta-referência. Ver a referência faria o especialista procurar
  confirmação, e a auditoria passaria a medir concordância com o gabarito —
  exatamente o que ela existe para checar de fora. Tudo isso vai na CHAVE.

UM GERADOR POR VEZ. O padrão é o `qwen3_27b`, que é o gerador de produção
(`agent/agent.py`, `qwen/qwen3.8-27b`). `--gerador ambos` monta as 75 × 2, mas
aí a mesma pergunta aparece duas vezes e a segunda leitura sai contaminada pela
primeira; se for usar, mande os lotes de cada gerador em momentos separados.

Uso:
  python eval/experimento_embedding/montar_auditoria_geracao.py
  python eval/experimento_embedding/montar_auditoria_geracao.py --total 100
  python eval/experimento_embedding/montar_auditoria_geracao.py --gerador ambos
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parents[1]

PERGUNTAS_DA_RECUPERACAO = AQUI / "auditoria/auditoria_perguntas.csv"
GABARITO = RAIZ / "eval/data/golden_dezembro.jsonl"
RANKINGS = RAIZ / "eval/results/retrieval/rankings"
SCORES = RAIZ / "eval/results/indexes/rerank_scores_bge_reranker_v2_m3.json"
GERACAO = RAIZ / "eval/results/generation"
SAIDA = AQUI / "auditoria_geracao"

K = 5
SEMENTE = 2026
GERADORES = ("qwen3_27b", "gptoss_120b")
JUIZES = ("gemini", "glm", "llama", "mistral")
# `refusal` entra porque nas irrespondiveis ele e o critério operante: ali
# correctness e completeness sao "na" legitimo (nao ha resposta-referencia) e o
# que o painel mede e se o sistema se absteve. Sem ele a chave nao serviria para
# a metade irrespondivel da planilha.
CRIT_CHAVE = ("correctness", "completeness", "refusal", "hallucination",
              "citation", "answer_relevance")
RECUSA = "Não tenho informações suficientes"
TIPOS_RESPONDIVEIS = ("factual", "multi_hop", "comparative")

# Ordem em que a cota das irrespondiveis e completada: as celulas informativas
# primeiro. "os dois se abstiveram" e 55 das 75 e nao tem nada para julgar.
ORDEM_CELULAS_IRR = (
    ("respondeu", "respondeu"),
    ("respondeu", "absteve"),
    ("absteve", "respondeu"),
    ("absteve", "absteve"),
)
SIGNIFICADO_IRR = {
    ("absteve", "absteve"): "os dois se abstiveram (comportamento correto)",
    ("absteve", "respondeu"): "rag se absteve, norag afirmou",
    ("respondeu", "absteve"): "rag afirmou, norag se absteve",
    ("respondeu", "respondeu"): "os DOIS afirmaram - o caso mais grave",
}

for _f in (sys.stdout, sys.stderr):
    try:
        _f.reconfigure(encoding="utf-8")
    except Exception:
        pass


# ── geometria da célula da resposta ─────────────────────────────────────────
# O Excel não cresce célula mesclada com wrap_text sozinho, e o formato não
# aceita linha acima de 409 pontos. A resposta que não couber entra com um aviso
# NO COMEÇO, para o especialista não julgar um texto cortado sem saber.
LARGURA_RESPOSTA = 118
CHARS_POR_LINHA = 112
PONTOS_POR_LINHA = 11.5
ALTURA_MAXIMA = 409
FONTE_RESPOSTA = 9
AVISO_CORTE = ("[RESPOSTA LONGA — o Excel corta o fim desta célula. O texto "
               "completo está em auditoria_geracao_respostas.csv, mesma "
               "pergunta e mesma letra.]")

CAB_XLSX = [
    ("lote", 6), ("qid", 8), ("pergunta", 40),
    ("resposta", 9), ("texto da resposta", LARGURA_RESPOSTA),
    ("JUIZ: o que ela afirma está certo?", 15),
    ("JUIZ: as fontes conferem?", 14),
    ("JUIZ: inventou? S/N", 11),
    ("JUIZ: observação da resposta", 26),
    ("JUIZ: existe norma que responda isso?", 15),
    ("JUIZ: qual responde melhor?", 14),
    ("JUIZ: por quê", 28),
]
COL_RESPOSTA = 5
COLS_DO_JUIZ_POR_RESPOSTA = (6, 7, 8, 9)
COLS_DO_BLOCO = (1, 2, 3, 10, 11, 12)
COLS_DO_JUIZ_NO_BLOCO = (10, 11, 12)


def limpar_enunciado(t: str) -> str:
    """Colapsa espaço em branco. Só para a PERGUNTA e a referência."""
    return re.sub(r"\s+", " ", (t or "").strip())


def limpar_resposta(t: str) -> str:
    """Normaliza a quebra de linha e NADA MAIS.

    A resposta do sistema é markdown com listas e cabeçalhos; colapsar a quebra
    de linha, como se faz com o texto do trecho, destruiria a estrutura que o
    especialista precisa ler. Só uniformiza CRLF e corta sequência de três ou
    mais linhas vazias, que no Excel vira célula alta e vazia.
    """
    t = (t or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    return re.sub(r"\n{3,}", "\n\n", t)


def escrever_csv(caminho: Path, cabecalho: list[str], linhas: list[list]) -> None:
    # utf-8-sig e ';': é o que o Excel em português abre com as colunas já
    # separadas, sem passar pelo assistente de importação.
    with caminho.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh, delimiter=";", quoting=csv.QUOTE_ALL)
        w.writerow(cabecalho)
        w.writerows(linhas)
    print(f"  ok {caminho.relative_to(RAIZ)}  ({len(linhas)} linhas)")


def _linhas_de_tela(texto: str) -> int:
    """Quantas linhas de tela o texto ocupa na célula da resposta."""
    return sum(max(1, len(l) // CHARS_POR_LINHA + 1)
               for l in texto.split("\n")) + 1


def ajustar_resposta(texto: str) -> tuple[str, float, bool]:
    """Devolve (texto para a célula, altura da linha, cortou?).

    Se o texto não couber nos 409 pontos que o formato permite, entra com
    AVISO_CORTE no COMEÇO — e não no fim, que é exatamente a parte que o Excel
    esconde. Julgar um texto cortado sem saber que ele está cortado produziria
    nota baixa por omissão que não é do sistema.
    """
    n = _linhas_de_tela(texto)
    if n * PONTOS_POR_LINHA <= ALTURA_MAXIMA:
        return texto, max(30.0, n * PONTOS_POR_LINHA), False
    return f"{AVISO_CORTE}\n\n{texto}", float(ALTURA_MAXIMA), True


def escrever_xlsx(caminho: Path, blocos: list[dict]) -> None:
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    from openpyxl.utils import get_column_letter
    from openpyxl.worksheet.datavalidation import DataValidation

    wb = Workbook()
    ws = wb.active
    ws.title = "auditoria"

    cinza = PatternFill("solid", fgColor="EFEFEF")
    amarelo = PatternFill("solid", fgColor="FFF4CC")
    topo = Border(top=Side(style="medium", color="999999"))
    alto = Alignment(vertical="top", wrap_text=True)
    centro = Alignment(vertical="center", horizontal="center", wrap_text=True)

    for i, (nome, largura) in enumerate(CAB_XLSX, start=1):
        c = ws.cell(row=1, column=i, value=nome)
        c.font = Font(bold=True, color="FFFFFF")
        c.fill = PatternFill("solid", fgColor="444444")
        c.alignment = centro
        ws.column_dimensions[get_column_letter(i)].width = largura
    ws.freeze_panes = "D2"
    ws.row_dimensions[1].height = 40

    linha = 2
    cortadas = 0
    for n, b in enumerate(blocos):
        ini = linha
        for r in b["respostas"]:
            texto, altura, cortou = ajustar_resposta(r["texto"])
            cortadas += cortou
            vals = [b["lote"], b["qid"], b["pergunta"] if linha == ini else None,
                    r["letra"], texto, None, None, None, None, None, None, None]
            for col, v in enumerate(vals, start=1):
                c = ws.cell(row=linha, column=col, value=v)
                c.alignment = alto
                if col == COL_RESPOSTA:
                    c.font = Font(size=FONTE_RESPOSTA)
                if col in COLS_DO_JUIZ_POR_RESPOSTA:
                    c.fill = amarelo
                elif n % 2 == 0:
                    c.fill = cinza
                if linha == ini:
                    c.border = topo
            ws.row_dimensions[linha].height = altura
            linha += 1
        fim = linha - 1
        # pergunta e vereditos do par valem para o bloco inteiro
        for col in COLS_DO_BLOCO:
            ws.merge_cells(start_row=ini, start_column=col,
                           end_row=fim, end_column=col)
            c = ws.cell(row=ini, column=col)
            c.alignment = centro if col in (1, 2, 10, 11) else alto
            if col in COLS_DO_JUIZ_NO_BLOCO:
                c.fill = amarelo

    validacoes = [
        ('"3,2,1,ABSTEVE"', "F",
         "Use 3, 2, 1 ou ABSTEVE, ou deixe vazio se não puder julgar."),
        ('"S,PARCIAL,N,SEM FONTE"', "G", "Use S, PARCIAL, N ou SEM FONTE."),
        ('"S,N"', "H", "Use S ou N."),
        ('"SIM,NÃO,NÃO SEI"', "J", "Use SIM, NÃO ou NÃO SEI."),
        ('"A,B,EMPATE,NENHUMA"', "K", "Use A, B, EMPATE ou NENHUMA."),
    ]
    for formula, col, erro in validacoes:
        dv = DataValidation(type="list", formula1=formula, allow_blank=True,
                            showErrorMessage=True, error=erro,
                            errorTitle="Valor inválido")
        ws.add_data_validation(dv)
        dv.add(f"{col}2:{col}{linha - 1}")

    guia = wb.create_sheet("como preencher")
    guia.column_dimensions["A"].width = 112
    for i, (texto, negrito) in enumerate(GUIA_XLSX, start=1):
        c = guia.cell(row=i, column=1, value=texto)
        c.alignment = Alignment(vertical="top", wrap_text=True)
        c.font = Font(bold=negrito, size=13 if negrito else 11)

    wb.save(caminho)
    print(f"  ok {caminho.relative_to(RAIZ)}  ({len(blocos)} perguntas, "
          f"{linha - 2} linhas de resposta)")
    if cortadas:
        print(f"     {cortadas} respostas nao cabem na celula e entraram com "
              f"aviso de corte; o texto inteiro esta no .csv")


GUIA_XLSX = [
    ("Auditoria da geração — como preencher", True),
    ("", False),
    ("Uma pergunta entra; o sistema responde em no máximo uma página, com uma "
     "seção '## Fontes utilizadas' no fim. Esta planilha traz, para cada "
     "pergunta, DUAS respostas do sistema em configurações diferentes, "
     "chamadas A e B. Qual configuração é qual você não sabe, e a ordem é "
     "sorteada a cada pergunta.", False),
    ("", False),
    ("Na aba 'auditoria', cada pergunta ocupa um bloco de 2 linhas. As colunas "
     "amarelas são suas.", False),
    ("", False),
    ("ALGUMAS PERGUNTAS DESTA PLANILHA NÃO TÊM RESPOSTA EM NORMA NENHUMA. Elas "
     "foram escritas assim de propósito, e não estão marcadas. Para essas, o "
     "comportamento certo do sistema é dizer que não encontrou base — e "
     "responder como se soubesse é o pior defeito possível. É por isso que "
     "duas das colunas existem: 'ABSTEVE' na nota, e 'existe norma que "
     "responda isso?' no bloco.", True),
    ("", False),
    ("1) Para cada resposta: 'o que ela afirma está certo?'", True),
    ("3 = tudo o que ela afirma está certo, e ela entrega o que foi pedido", False),
    ("2 = mistura acerto e erro, ou entrega só parte", False),
    ("1 = afirma coisa errada", False),
    ("ABSTEVE = ela não afirma nada; diz que não tem informação suficiente", False),
    ("", False),
    ("ABSTEVE é descritivo, não é nota: não decida aqui se abster-se foi certo "
     "ou errado. Quem decide isso é a coluna 'existe norma que responda isso?'.", True),
    ("Se você não tiver como julgar, deixe em branco e escreva o motivo em "
     "'observação'. Chute não ajuda.", False),
    ("", False),
    ("2) 'as fontes conferem?' — olhe a seção '## Fontes utilizadas'", True),
    ("S = as normas citadas existem e sustentam o que a resposta afirma", False),
    ("PARCIAL = sustentam parte, ou falta a norma principal", False),
    ("N = citam norma que não sustenta aquilo, ou que não existe", False),
    ("SEM FONTE = a resposta não cita fonte nenhuma", False),
    ("", False),
    ("3) 'inventou? S/N' — marque S se a resposta afirma lei, artigo, número "
     "ou prazo que você sabe que não é aquilo. É a pergunta mais importante da "
     "planilha: resposta errada com aparência de citação é o pior defeito "
     "possível num sistema jurídico.", True),
    ("", False),
    ("4) Por pergunta: 'existe norma que responda isso?'", True),
    ("SIM = existe norma (brasileira ou internacional) que responde", False),
    ("NÃO = a pergunta pede algo que norma nenhuma estabelece", False),
    ("NÃO SEI = você não tem como afirmar nem negar", False),
    ("", False),
    ("NÃO SEI é resposta legítima e é melhor que chute. Esta coluna é o que "
     "permite interpretar o ABSTEVE: abster-se numa pergunta sem resposta é o "
     "certo; abster-se numa que tem resposta é falhar.", False),
    ("", False),
    ("5) Por pergunta: 'qual responde melhor?'", True),
    ("A ou B = aquela responde melhor a pergunta", False),
    ("EMPATE = as duas servem igualmente, ou falham igualmente", False),
    ("NENHUMA = nenhuma das duas serve", False),
    ("", False),
    ("Se você marcou que NÃO existe norma que responda, 'melhor' é a que disse "
     "que não encontrou base — não a que respondeu bonito.", True),
    ("Diga em 'por quê' o que decidiu: a norma certa, o número certo, a "
     "omissão, a invenção.", False),
    ("", False),
    ("Quatro coisas que evitam mal-entendido", True),
    ("Julgue cada resposta CONTRA A PERGUNTA e contra o que a norma diz, não "
     "uma contra a outra. As duas podem ser 3, e as duas podem ser 1.", False),
    ("Pelo jeito de escrever, você provavelmente vai conseguir adivinhar qual "
     "configuração é qual em várias perguntas. Não tem problema — mas julgue o "
     "CONTEÚDO da resposta, não o palpite. A nota de cada resposta é o que mais "
     "importa aqui; a preferência entre as duas é secundária.", True),
    ("As duas respostas da mesma pergunta podem ser muito diferentes em "
     "tamanho, e tamanho não é qualidade: a mais longa pode ser a que inventa.", False),
    ("Os enunciados estão em português. As respostas citam normas brasileiras "
     "e internacionais; o acervo é bilíngue, e nome de norma em inglês é "
     "normal, não é defeito.", False),
    ("", False),
    ("Algumas respostas são longas demais para a célula e começam com um aviso "
     "em maiúsculas dizendo isso. Nesses casos leia o texto completo no arquivo "
     ".csv que acompanha a planilha, na mesma pergunta e na mesma letra — não "
     "julgue pela parte que aparece.", False),
    ("", False),
    ("Faça o LOTE 1 inteiro primeiro — ele já é uma amostra válida.", True),
]


INSTRUCOES = """# Auditoria da geração — instruções

Obrigado por auditar. É uma planilha e uma passada por pergunta. Não abra
`auditoria_geracao_CHAVE_nao_abrir_antes.csv`: ela diz qual resposta é de qual
configuração, quais perguntas não têm resposta e qual era a resposta esperada.
Consultá-la antes invalida o julgamento.

## O que o sistema faz

Uma pergunta entra; o sistema responde em no máximo uma página, terminando com
uma seção `## Fontes utilizadas`. O acervo são 54 normas brasileiras e
internacionais, em português e inglês.

Para cada pergunta, a planilha traz **duas respostas**, em configurações
diferentes do sistema, chamadas **A** e **B**. Qual é qual você não sabe, e a
ordem é sorteada a cada pergunta.

## Uma coisa importante antes de começar

**Algumas perguntas desta planilha não têm resposta em norma nenhuma.** Foram
escritas assim de propósito, e não estão marcadas. Para essas, o comportamento
certo do sistema é dizer que não encontrou base — e responder como se soubesse é
o pior defeito possível.

Por isso a planilha separa duas coisas que costumam ser confundidas: **o que a
resposta faz** (afirma? se absteve?) e **o que ela deveria fazer** (a pergunta
tinha resposta?). A primeira você preenche por resposta; a segunda, por
pergunta.

## O que preencher, por resposta

| coluna | o que é |
|---|---|
| `o que ela afirma está certo?` | **3** tudo certo e entrega o que foi pedido · **2** mistura acerto e erro, ou entrega parte · **1** afirma coisa errada · **ABSTEVE** não afirma nada, diz que não tem informação suficiente |
| `as fontes conferem?` | **S** as normas citadas existem e sustentam o que a resposta afirma · **PARCIAL** sustentam parte, ou falta a principal · **N** citam norma que não sustenta aquilo, ou que não existe · **SEM FONTE** |
| `inventou? S/N` | **S** se a resposta afirma lei, artigo, número ou prazo que você sabe que não é aquilo |
| `observação da resposta` | o que a nota não conta |

`ABSTEVE` é **descritivo, não é nota**: não decida aí se abster-se foi certo ou
errado. Se não tiver como julgar, **deixe em branco** e diga o motivo na
observação.

## O que preencher, por pergunta

| coluna | o que é |
|---|---|
| `existe norma que responda isso?` | **SIM** existe norma, brasileira ou internacional, que responde · **NÃO** a pergunta pede algo que norma nenhuma estabelece · **NÃO SEI** você não tem como afirmar nem negar |
| `qual responde melhor?` | **A**, **B**, **EMPATE** (servem ou falham igualmente), **NENHUMA** |
| `por quê` | a norma certa, o número certo, a omissão, a invenção |

`NÃO SEI` é resposta legítima e é melhor que chute. Essa coluna é o que permite
interpretar o `ABSTEVE`: abster-se numa pergunta sem resposta é o certo;
abster-se numa que tem resposta é falhar.

E se você marcou que **não** existe norma que responda, "melhor" é a que disse
que não encontrou base — não a que respondeu bonito.

## Três coisas que mudam o resultado

- Julgue cada resposta **contra a pergunta** e contra o que a norma diz, não
  uma contra a outra. As duas podem ser 3, e as duas podem ser 1.
- **Você provavelmente vai adivinhar qual configuração é qual** em várias
  perguntas, pelo jeito de escrever. Não tem problema — mas julgue o conteúdo,
  não o palpite. A nota de cada resposta é o que mais importa; a preferência
  entre as duas é secundária, justamente porque é a mais sensível ao palpite.
- **Tamanho não é qualidade.** As duas respostas podem diferir muito em
  extensão, e a mais longa costuma ser a que inventa.

Algumas respostas não cabem na célula do Excel e começam com um aviso em
maiúsculas dizendo isso. Nesses casos leia o texto completo em
`auditoria_geracao_respostas.csv`, na mesma pergunta e na mesma letra — não
julgue pela parte que aparece.

## Ordem sugerida

Faça o **lote 1** inteiro primeiro. Cada lote é uma amostra estratificada
válida por si, com os quatro tipos de pergunta na mesma proporção.

## O que vai ser feito com isso

Duas coisas. (1) Medir a concordância entre o seu julgamento e o de um painel de
quatro modelos de linguagem que já julgou estas mesmas respostas — o painel é
barato e escala, mas precisa de uma âncora humana, e já deu problema: dois dos
quatro se recusaram a pontuar centenas de respostas alegando motivos que o
próprio enunciado contradizia. (2) Decidir, de fora, se a busca em normas
melhora a resposta ou se o modelo já respondia bem sem ela — e se ela evita que
o sistema invente quando não há o que responder.

Boa parte das perguntas respondíveis são as **mesmas** da auditoria da
recuperação, de propósito: dá para cruzar as duas e ver se, quando os 5 trechos
não bastavam, a resposta saiu errada.
"""


def carregar_judged(gerador: str, cenario: str) -> dict:
    caminho = GERACAO / f"judged_{gerador}_{cenario}_icl0.jsonl"
    if not caminho.exists():
        sys.exit(
            f"ERRO: nao achei {caminho.relative_to(RAIZ)}.\n"
            f"Os julgamentos da rubrica mista-v2 estao na branch "
            f"helen/nova-rubrica-e-rotacao-corrigida; na main ainda nao. "
            f"Marque essa branch, ou rode depois do merge.")
    with caminho.open(encoding="utf-8") as fh:
        return {r["qid"]: r for r in (json.loads(l) for l in fh if l.strip())}


def carregar_acerto() -> dict:
    """Para cada qid respondivel, se o top-5 do BGE+reranker cobre o gabarito.

    MESMA definicao de `montar_auditoria.py`: Strict Recall@5, conjuntivo --
    acerta so quando TODO grupo do qrels tem pelo menos um trecho no top-5.
    Repito o calculo em vez de importar porque o outro script faz o trabalho
    dele no import (le o corpus de 11 MB), e aqui so o rotulo interessa.
    """
    faltam = [p for p in (GABARITO, RANKINGS / "B_dense_bge_m3.json",
                          RANKINGS / "B_bge_m3_rerank.json", SCORES)
              if not p.exists()]
    if faltam:
        sys.exit("ERRO: faltam arquivos para calcular o acerto do top-5:\n  "
                 + "\n  ".join(str(p.relative_to(RAIZ)) for p in faltam))
    gold = {g["qid"]: g for g in
            (json.loads(l) for l in GABARITO.open(encoding="utf-8") if l.strip())}
    bge = json.loads((RANKINGS / "B_dense_bge_m3.json").read_text(encoding="utf-8"))
    rr = json.loads((RANKINGS / "B_bge_m3_rerank.json").read_text(encoding="utf-8"))
    notas = json.loads(SCORES.read_text(encoding="utf-8"))["scores"]

    def top5(q: str) -> list[str]:
        if q in rr:
            return rr[q][:K]
        pool = bge.get(q, [])[:100]
        nq = notas.get(q, {})
        return sorted(pool, key=lambda c: -nq.get(c, -1e9))[:K]

    acerto = {}
    for q, g in gold.items():
        gr = g["qrels_grupos"]
        if not gr or q not in bge or not notas.get(q):
            continue
        topo = set(top5(q))
        acerto[q] = all(any(c in topo for c in g_) for g_ in gr)
    return gold, acerto


def repartir_irrespondiveis(total: int, celulas: dict) -> tuple[dict, int]:
    """Quantas de cada célula de comportamento entram, e quantas são obrigatórias.

    NUM TESTE PAREADO SÓ O PAR DISCORDANTE CARREGA INFORMAÇÃO. Dos 75
    irrespondíveis do qwen3_27b, 57 são pares em que os dois braços fizeram a
    mesma coisa; eles não entram no p-valor do teste do sinal. Quem decide são
    os 18 discordantes — 13 em que o `rag` se absteve e o `norag` afirmou, e 5
    no sentido contrário.

    Por isso a regra aqui não é cota igual. É:

      OBRIGATÓRIAS — entram TODAS, sempre:
        · os dois pares discordantes, nas DUAS direções. Levar todos faz o teste
          do especialista ter a mesma força que o do painel sobre as 75
          inteiras, numa planilha de 25. Levar parte de uma direção e tudo da
          outra foi o erro da versão anterior: ela tomou as 5 contra o RAG e só
          6 das 13 a favor, e o teste caiu de p=0,096 para p=0,754.
        · "os dois afirmaram", que é o caso mais grave do sistema inteiro e é
          raríssimo (2 no qwen3_27b).

      PREENCHIMENTO — "os dois se abstiveram" completa até o total pedido. Essas
      não entram no p-valor, mas não são decoração: sem elas o bloco
      irrespondível seria 100% caso de falha, e um especialista que lê vinte
      respostas ruins seguidas recalibra sem perceber.

    A amostra fica DELIBERADAMENTE desproporcional, então TAXA (quantas vezes o
    sistema se abstém) tem de ser reponderada por célula, ou vir do painel sobre
    as 75. O que sai direto daqui é a COMPARAÇÃO pareada.
    """
    obrig = {k: len(celulas.get(k, ())) for k in ORDEM_CELULAS_IRR
             if k != ("absteve", "absteve")}
    n_obrig = sum(obrig.values())
    aloc = dict(obrig)
    aloc[("absteve", "absteve")] = max(
        0, min(len(celulas.get(("absteve", "absteve"), ())), total - n_obrig))
    return aloc, n_obrig


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gerador", default="qwen3_27b",
                    choices=[*GERADORES, "ambos"],
                    help="padrao qwen3_27b, o gerador de producao")
    ap.add_argument("--total", type=int, default=80,
                    help="perguntas na planilha; padrao 80")
    ap.add_argument("--irrespondiveis", type=int, default=30,
                    help="quantas das --total sao irrespondiveis; padrao 25. "
                         "O piso e o numero de pares discordantes mais os 'os "
                         "dois afirmaram', que entram sempre; o resto completa "
                         "com 'os dois se abstiveram'. As respondiveis ficam "
                         "com a diferenca, repartida entre os tres tipos.")
    ap.add_argument("--lotes", type=int, default=3)
    ap.add_argument("--controles-no-ultimo-lote", type=int, default=2,
                    help="quantas perguntas de controle (aquelas em que os "
                         "dois bracos se abstiveram) ficam no ultimo lote; "
                         "padrao 2. O resto vai para os lotes prioritarios, "
                         "onde o trabalho de fato vai acontecer.")
    ap.add_argument("--todo-o-acervo", action="store_true",
                    help="sorteia as respondiveis de TODO o golden_dezembro "
                         "em vez de so das 78 da auditoria de recuperacao. "
                         "Perde o cruzamento por qid entre as duas auditorias.")
    args = ap.parse_args()
    geradores = list(GERADORES) if args.gerador == "ambos" else [args.gerador]

    gold, acerto = carregar_acerto()

    # ── de onde saem as respondiveis ────────────────────────────────────────
    if args.todo_o_acervo:
        pool_resp = [q for q, g in gold.items()
                     if g["question_type"] in TIPOS_RESPONDIVEIS and q in acerto]
        print(f"pool de respondiveis: todo o acervo, {len(pool_resp)} perguntas")
    else:
        if not PERGUNTAS_DA_RECUPERACAO.exists():
            sys.exit(f"ERRO: nao achei "
                     f"{PERGUNTAS_DA_RECUPERACAO.relative_to(RAIZ)}; rode "
                     f"montar_auditoria.py primeiro, ou use --todo-o-acervo.")
        with PERGUNTAS_DA_RECUPERACAO.open(encoding="utf-8-sig") as fh:
            pool_resp = [r["qid"] for r in csv.DictReader(fh, delimiter=";")]
        print(f"pool de respondiveis: as {len(pool_resp)} da auditoria de "
              f"recuperacao (o qid fica compartilhado entre as duas)")

    dados = {(g, c): carregar_judged(g, c)
             for g in geradores for c in ("rag", "norag")}
    primeiro = geradores[0]

    # ── cotas ───────────────────────────────────────────────────────────────
    # As irrespondiveis NAO entram na proporcao da populacao (que seria 1/4).
    # Elas sao sobre-amostradas de proposito: os pares que decidem o teste de
    # abstencao sao raros (18 das 75 no qwen3_27b) e tem de entrar inteiros.
    # As respondiveis ficam com a diferenca, repartida entre os tres tipos --
    # e o custo disso e pequeno: 57 -> 50 respondiveis alarga o IC do kappa de
    # correctness de 0,18 para 0,19.
    cota_tipo = {"unanswerable": args.irrespondiveis}
    sobra = args.total - args.irrespondiveis
    por_tipo = sobra // len(TIPOS_RESPONDIVEIS)
    resto = sobra - por_tipo * len(TIPOS_RESPONDIVEIS)
    for i, t in enumerate(TIPOS_RESPONDIVEIS):
        cota_tipo[t] = por_tipo + (1 if i < resto else 0)
    if sobra <= 0:
        sys.exit(f"ERRO: --irrespondiveis {args.irrespondiveis} nao deixa "
                 f"espaco para respondiveis em --total {args.total}.")

    rnd = random.Random(SEMENTE)

    # ── respondiveis: tipo x acerto, metade e metade quando da ─────────────
    cel_resp: dict = defaultdict(list)
    for q in pool_resp:
        t = gold[q]["question_type"]
        if t in TIPOS_RESPONDIVEIS and q in acerto:
            cel_resp[(t, "acerta" if acerto[q] else "erra")].append(q)
    for v in cel_resp.values():
        v.sort()
        rnd.shuffle(v)

    escolhidas: list[tuple[str, tuple]] = []
    print("\ncelulas das respondiveis (tipo x acerto no top-5):")
    for t in TIPOS_RESPONDIVEIS:
        a, e = cel_resp[(t, "acerta")], cel_resp[(t, "erra")]
        n_e = min(len(e), cota_tipo[t] // 2)
        n_a = min(len(a), cota_tipo[t] - n_e)
        n_e = min(len(e), cota_tipo[t] - n_a)    # se 'acerta' nao tinha o bastante
        print(f"  {t:<14} acerta {len(a):>3} disponiveis -> {n_a:<3}   "
              f"erra {len(e):>3} disponiveis -> {n_e:<3}"
              + ("   (celula menor que a metade: entra inteira)"
                 if n_e < cota_tipo[t] // 2 or n_a < cota_tipo[t] - cota_tipo[t] // 2
                 else ""))
        escolhidas += [(q, (t, "acerta")) for q in a[:n_a]]
        escolhidas += [(q, (t, "erra")) for q in e[:n_e]]

    # ── irrespondiveis: comportamento do par (rag x norag) ─────────────────
    # O rotulo vem do PRIMEIRO gerador. Com --gerador ambos as celulas do
    # segundo sao outras, e sortear por gerador daria duas listas de perguntas
    # diferentes -- o que impediria comparar os dois geradores nas mesmas
    # perguntas. A chave registra a celula de cada gerador separadamente.
    rag0, no0 = dados[(primeiro, "rag")], dados[(primeiro, "norag")]
    cel_irr: dict = defaultdict(list)
    for q, g in gold.items():
        if g["question_type"] != "unanswerable":
            continue
        if q not in rag0 or q not in no0:
            continue
        k = ("absteve" if RECUSA in rag0[q]["answer"] else "respondeu",
             "absteve" if RECUSA in no0[q]["answer"] else "respondeu")
        cel_irr[k].append(q)
    for v in cel_irr.values():
        v.sort()
        rnd.shuffle(v)

    aloc_irr, n_obrig = repartir_irrespondiveis(cota_tipo["unanswerable"],
                                                cel_irr)
    print(f"\ncelulas das irrespondiveis (comportamento rag x norag, "
          f"rotulo do {primeiro}):")
    for k in ORDEM_CELULAS_IRR:
        marca = "TODAS" if k != ("absteve", "absteve") else "preenche"
        print(f"  {k[0]:<10}{k[1]:<10} {len(cel_irr.get(k, ())):>3} disponiveis "
              f"-> {aloc_irr[k]:<3} [{marca:<8}] {SIGNIFICADO_IRR[k]}")
    disc = {k: len(cel_irr.get(k, ())) for k in
            (("absteve", "respondeu"), ("respondeu", "absteve"))}
    n_disc = sum(disc.values())
    print(f"  pares DISCORDANTES (os unicos que entram no teste pareado): "
          f"{n_disc} = {disc[('absteve', 'respondeu')]} pro-rag + "
          f"{disc[('respondeu', 'absteve')]} pro-norag, todos na planilha")
    if n_obrig > cota_tipo["unanswerable"]:
        print(f"  ! as obrigatorias ja sao {n_obrig}, acima das "
              f"{cota_tipo['unanswerable']} pedidas; usando {n_obrig} e a "
              f"planilha fica com {args.total - cota_tipo['unanswerable'] + n_obrig} "
              f"perguntas")
    for k in ORDEM_CELULAS_IRR:
        escolhidas += [(q, ("unanswerable", f"{k[0]}/{k[1]}"))
                       for q in cel_irr.get(k, ())[:aloc_irr[k]]]

    if len(escolhidas) < args.total:
        print(f"\n! pedi {args.total} e sairam {len(escolhidas)}: alguma celula "
              f"nao tinha o bastante. A tabela acima diz qual.")

    # ── lotes ───────────────────────────────────────────────────────────────
    # Cada celula e distribuida em rodada pelos lotes, e o PONTO DE PARTIDA
    # avanca de celula em celula. Sem isso o resto de cada divisao cai sempre no
    # lote 1: com 10 celulas de tamanho impar os lotes sairam 28/26/21.
    #
    # EXCECAO: as celulas que decidem o teste de abstencao -- os pares
    # discordantes e "os dois afirmaram" -- sao distribuidas so pelos PRIMEIROS
    # lotes, nao por todos. O motivo e prazo: espalhados 6/6/6, parar no lote 1
    # deixaria 6 pares e p=0,688; terminando os dois primeiros lotes o teste ja
    # esta completo (18 pares, todos os que existem). O preco e que o ultimo
    # lote deixa de dizer qualquer coisa sobre abstencao e fica so com as
    # respondiveis e as irrespondiveis faceis -- e o lote "se der tempo".
    #
    # As CONTROLES ("os dois se abstiveram") ficam quase todas nos lotes
    # prioritarios, pelo mesmo motivo de prazo: controle num lote que talvez
    # nunca seja preenchido e controle desperdicado, e e nos primeiros lotes que
    # se checa a calibracao do especialista antes de confiar no resto. Mas
    # `--controles-no-ultimo-lote` deixa algumas no ultimo: sem nenhuma, o
    # ultimo lote fica 100% respondivel, estruturalmente diferente dos outros, e
    # perde a propria aferição.
    LOTES_PRIORITARIOS = max(1, args.lotes - 1)
    CONTROLE = f"{'absteve'}/{'absteve'}"
    celulas_do_teste = {f"{a}/{b}" for a, b in ORDEM_CELULAS_IRR
                        if (a, b) != ("absteve", "absteve")}
    por_celula: dict = defaultdict(list)
    for q, cel in escolhidas:
        por_celula[cel].append(q)
    lote_de: dict = {}
    for n_cel, cel in enumerate(sorted(por_celula)):
        qs = por_celula[cel]
        if cel[1] == CONTROLE and args.lotes > 1:
            # as ultimas N vao para o ultimo lote; o resto se reparte nos
            # prioritarios
            n_fim = min(args.controles_no_ultimo_lote, len(qs))
            for q in qs[len(qs) - n_fim:]:
                lote_de[q] = str(args.lotes)
            qs = qs[:len(qs) - n_fim]
            for i, q in enumerate(qs):
                lote_de[q] = str((i + n_cel) % LOTES_PRIORITARIOS + 1)
            continue
        n_lotes = (LOTES_PRIORITARIOS if cel[1] in celulas_do_teste
                   else args.lotes)
        for i, q in enumerate(qs):
            lote_de[q] = str((i + n_cel) % n_lotes + 1)

    # ── sorteio A/B balanceado DENTRO DE CADA LOTE ──────────────────────────
    # Com shuffle independente em cada par o desequilibrio e grande: na versao
    # anterior o `rag` caiu na letra A em 49 das 78 e na B em 29. Um
    # especialista que percebe "a A costuma ser a melhor" fica ancorado.
    em_a: dict = {}
    for gerador in geradores:
        for lote in (str(i + 1) for i in range(args.lotes)):
            qs = sorted(q for q, _ in escolhidas if lote_de[q] == lote)
            marcas = [True] * ((len(qs) + 1) // 2) + [False] * (len(qs) // 2)
            rnd.shuffle(marcas)
            em_a.update({(gerador, q): m for q, m in zip(qs, marcas)})

    # Lote 1 primeiro, mas EMBARALHADO dentro do lote. Ordenar por célula
    # agruparia as irrespondíveis no fim de cada lote, e o especialista
    # perceberia o bloco: nove perguntas seguidas em que nenhuma resposta
    # convence entregam que aquelas ali são de outro tipo, que é exatamente o
    # que a planilha não pode dizer. Embaralhar com a semente mantém a ordem
    # reprodutível.
    ordem = sorted(escolhidas, key=lambda x: (lote_de[x[0]], x[1], x[0]))
    por_lote: dict = defaultdict(list)
    for item in ordem:
        por_lote[lote_de[item[0]]].append(item)
    ordem = []
    for lote in sorted(por_lote):
        bloco_do_lote = por_lote[lote]
        rnd.shuffle(bloco_do_lote)
        ordem += bloco_do_lote
    blocos, linhas_r, linhas_p, linhas_c = [], [], [], []
    pistas = {"rag": 0, "norag": 0}
    PISTA = re.compile(r"contexto fornecid|\bno contexto\b|\btrecho", re.I)

    for gerador in geradores:
        for qid, celula in ordem:
            rec = {c: dados[(gerador, c)].get(qid) for c in ("rag", "norag")}
            faltam = [c for c, r in rec.items() if r is None]
            if faltam:
                sys.exit(f"ERRO: {qid} nao esta em {gerador} "
                         f"({', '.join(faltam)}).")
            lote = lote_de[qid]
            par = ([("rag", rec["rag"]), ("norag", rec["norag"])]
                   if em_a[(gerador, qid)] else
                   [("norag", rec["norag"]), ("rag", rec["rag"])])
            enunciado = limpar_enunciado(gold[qid]["question"])
            bloco = {"lote": lote, "qid": qid, "pergunta": enunciado,
                     "respostas": []}

            for letra, (cen, r) in zip("AB", par):
                texto = limpar_resposta(r.get("answer"))
                if PISTA.search(texto):
                    pistas[cen] += 1
                bloco["respostas"].append({"letra": letra, "texto": texto})
                linhas_r.append([lote, qid, enunciado, letra, texto,
                                 "", "", "", ""])
                nt = notas_do_painel(r)
                cel_do_gerador = celula[1]
                if celula[0] == "unanswerable":
                    a = "absteve" if RECUSA in limpar_resposta(
                        dados[(gerador, "rag")][qid]["answer"]) else "respondeu"
                    b = "absteve" if RECUSA in limpar_resposta(
                        dados[(gerador, "norag")][qid]["answer"]) else "respondeu"
                    cel_do_gerador = f"{a}/{b}"
                linhas_c.append([
                    lote, qid, gerador, letra, cen,
                    celula[0], cel_do_gerador,
                    "SIM" if gold[qid]["qrels_grupos"] else "nao",
                    len(r.get("context_chunk_ids") or []),
                    "SIM" if RECUSA in texto else "nao",
                    len(texto),
                    *[nt[f"mediana_{c}"] for c in CRIT_CHAVE],
                    *[nt[f"{j}_{c}"] for j in JUIZES for c in CRIT_CHAVE],
                    limpar_enunciado(gold[qid].get("reference_answer") or ""),
                ])

            linhas_p.append([lote, qid, enunciado, "", "", ""])
            blocos.append(bloco)

    # ── relatorio do que saiu ───────────────────────────────────────────────
    print(f"\namostra: {len(blocos)} blocos "
          f"({len(escolhidas)} perguntas x {len(geradores)} gerador(es))")
    print(f"  {'lote':<8}" + "".join(f"{t[:11]:>13}" for t in
                                     (*TIPOS_RESPONDIVEIS, "unanswerable"))
          + f"{'total':>8}")
    for lote in (str(i + 1) for i in range(args.lotes)):
        conta = {t: sum(1 for q, c in escolhidas
                        if lote_de[q] == lote and c[0] == t)
                 for t in (*TIPOS_RESPONDIVEIS, "unanswerable")}
        print(f"  {lote:<8}" + "".join(f"{conta[t]:>13}" for t in conta)
              + f"{sum(conta.values()):>8}")
    print(f"\ncegamento (so para dimensionar, nao e defeito a corrigir):")
    n = len(escolhidas) * len(geradores)
    print(f"  respostas que falam de 'contexto'/'trecho': "
          f"rag {pistas['rag']}/{n}, norag {pistas['norag']}/{n}")
    if args.gerador == "ambos":
        print("\n! A mesma pergunta aparece DUAS vezes na planilha, uma por")
        print("  gerador. Mande os lotes de cada gerador em momentos separados,")
        print("  ou a segunda leitura sai contaminada pela primeira.")

    SAIDA.mkdir(exist_ok=True)
    print()
    escrever_xlsx(SAIDA / "AUDITORIA_GERACAO.xlsx", blocos)
    escrever_csv(SAIDA / "auditoria_geracao_respostas.csv", [
        "lote", "qid", "pergunta", "resposta_letra", "texto_da_resposta",
        "JUIZ_afirma_certo_3_2_1_ABSTEVE",
        "JUIZ_fontes_conferem_S_PARCIAL_N_SEMFONTE",
        "JUIZ_inventou_S_N", "JUIZ_observacao",
    ], linhas_r)
    escrever_csv(SAIDA / "auditoria_geracao_perguntas.csv", [
        "lote", "qid", "pergunta",
        "JUIZ_existe_norma_SIM_NAO_NAOSEI",
        "JUIZ_qual_melhor_A_B_EMPATE_NENHUMA", "JUIZ_por_que",
    ], linhas_p)
    escrever_csv(SAIDA / "auditoria_geracao_CHAVE_nao_abrir_antes.csv", [
        "lote", "qid", "gerador", "resposta_letra", "cenario",
        "tipo_de_pergunta", "celula_da_amostra", "tem_resposta_no_acervo",
        "n_trechos_no_contexto", "e_recusa", "caracteres_da_resposta",
        *[f"mediana_{c}" for c in CRIT_CHAVE],
        *[f"{j}_{c}" for j in JUIZES for c in CRIT_CHAVE],
        "resposta_referencia",
    ], linhas_c)
    (SAIDA / "auditoria_geracao_INSTRUCOES.md").write_text(
        INSTRUCOES, encoding="utf-8")
    print(f"  ok {(SAIDA / 'auditoria_geracao_INSTRUCOES.md').relative_to(RAIZ)}")

    print("\nMANDE SO o AUDITORIA_GERACAO.xlsx (e o .csv das respostas, para o")
    print("caso de alguma resposta nao caber na celula). A CHAVE NAO vai: ela")
    print("tem o braco de cada letra, quais perguntas sao irrespondiveis, a")
    print("resposta-referencia e as notas do painel.")


def notas_do_painel(rec: dict) -> dict:
    """As notas dos quatro juizes LLM, para a CHAVE.

    Nao entra na planilha do especialista: ver a nota da maquina antes de julgar
    ancora o julgamento, que e justamente o que esta auditoria checa de fora.
    `"na"` e mantido como texto para a diferenca entre "nao aplicavel" e "nao
    pontuou" nao virar celula vazia na analise.
    """
    out = {}
    scores = rec.get("scores") or {}
    for j in JUIZES:
        s = scores.get(j)
        for c in CRIT_CHAVE:
            v = (s or {}).get(c) if isinstance(s, dict) else None
            out[f"{j}_{c}"] = v if isinstance(v, int) else ("na" if v == "na" else "")
    for c in CRIT_CHAVE:
        vals = sorted(v for v in (out[f"{j}_{c}"] for j in JUIZES)
                      if isinstance(v, int))
        if not vals:
            out[f"mediana_{c}"] = ""
        else:
            m = len(vals) // 2
            med = vals[m] if len(vals) % 2 else (vals[m - 1] + vals[m]) / 2
            out[f"mediana_{c}"] = f"{med:g}"
    return out


if __name__ == "__main__":
    main()
