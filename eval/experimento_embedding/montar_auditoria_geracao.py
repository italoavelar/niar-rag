#!/usr/bin/env python3
"""Planilha de auditoria da GERAÇÃO: a mesma pergunta, a resposta com e sem RAG.

PARA QUE SERVE. A comparação `rag` × `norag` hoje é julgada por um painel de
quatro LLMs (rubrica `mista-v2`). Esse painel precisa de uma âncora humana: em
05/11 dois dos quatro juízes se recusaram a pontuar 809 notas de
`correctness`/`completeness` alegando motivos que o próprio prompt contradizia,
e isso só apareceu porque alguém leu as justificativas. Um especialista julgando
as mesmas respostas dá (1) concordância especialista × painel, por critério, e
(2) um veredito independente sobre a pergunta do artigo: a recuperação melhora a
resposta, ou o modelo já sabia?

AS MESMAS 78 PERGUNTAS DA AUDITORIA DE RECUPERAÇÃO, nos mesmos lotes. Não é
conveniência: é o que permite cruzar as duas auditorias por `qid` e responder a
pergunta que nenhuma delas responde sozinha — quando o especialista disse que os
5 trechos NÃO bastavam, a resposta gerada saiu errada? O conjunto é lido de
`auditoria/auditoria_perguntas.csv`, não re-sorteado, para os dois serem
idênticos por construção.

QUATRO DECISÕES DE DESENHO, pelos mesmos motivos da auditoria de recuperação:

  O BRAÇO É CEGO. As duas respostas aparecem como "A" e "B" em ordem sorteada
  por pergunta. Se a planilha dissesse qual é "com RAG", o rótulo viraria
  âncora. O mapa A/B -> rag/norag fica só na chave.

  O CEGAMENTO É PARCIAL, E ISSO FICA MEDIDO. A própria redação delata o braço
  em boa parte dos casos. Medido em 05/11 nas 78 do qwen3_27b: 45 respostas do
  `rag` dizem "contexto fornecido", "no contexto" ou "trecho", contra 6 do
  `norag`; e "Nenhuma fonte encontrada" aparece em 36 do `norag` contra 4 do
  `rag` (o `norag` recusa 36 vezes, o `rag` 4). Então o sorteio tira a âncora
  do RÓTULO e da POSIÇÃO, não a informação que está no texto.

  Não normalizo essas frases, e a escolha é deliberada: "com base no contexto
  fornecido" em vez de nomear a norma é um defeito de qualidade que o
  especialista deve poder penalizar. Reescrever a saída do sistema para cegar
  melhor esconderia justamente o que a auditoria procura. O que compensa é a
  planilha pedir nota ABSOLUTA de cada resposta, e não só a preferência: a nota
  absoluta sobrevive ao palpite sobre o braço, a preferência não.

  NADA DO MODELO NA PLANILHA DO JUIZ. Sem nome do gerador, sem nota dos juízes
  LLM, sem resposta-referência. Ver a referência faria o especialista procurar
  confirmação, e a auditoria passaria a medir concordância com o gabarito —
  exatamente o que ela existe para checar de fora.

  A ESCALA É A MESMA DO PAINEL (1-3), para a concordância ser direta, sem
  mapeamento. Inclusive no caso em que o painel errou: uma RECUSA vale 1, não
  "não se aplica" — todas as 78 perguntas têm resposta no acervo, então
  abster-se aqui é falhar.

UM GERADOR POR VEZ. O padrão é o `qwen3_27b`, que é o gerador de produção
(`agent/agent.py`, `qwen/qwen3.8-27b`). `--gerador ambos` monta as 78 × 2, mas
aí a mesma pergunta aparece duas vezes na planilha e a segunda leitura fica
contaminada pela primeira; se for usar, mande os lotes de cada gerador em
momentos separados.

Uso:
  python eval/experimento_embedding/montar_auditoria_geracao.py
  python eval/experimento_embedding/montar_auditoria_geracao.py --gerador ambos
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parents[1]

PERGUNTAS_DA_RECUPERACAO = AQUI / "auditoria/auditoria_perguntas.csv"
GERACAO = RAIZ / "eval/results/generation"
SAIDA = AQUI / "auditoria_geracao"

SEMENTE = 2026
GERADORES = ("qwen3_27b", "gptoss_120b")
JUIZES = ("gemini", "glm", "llama", "mistral")
CRIT_CHAVE = ("correctness", "completeness", "hallucination", "citation")
RECUSA = "Não tenho informações suficientes"

for _f in (sys.stdout, sys.stderr):
    try:
        _f.reconfigure(encoding="utf-8")
    except Exception:
        pass


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


# Geometria da célula da resposta, usada para estimar a altura da linha. O
# Excel não cresce célula mesclada com wrap_text sozinho, e o formato não aceita
# linha acima de 409 pontos — a resposta que não couber entra com um aviso no
# começo, para o especialista não julgar um texto cortado sem saber.
LARGURA_RESPOSTA = 118
CHARS_POR_LINHA = 112
PONTOS_POR_LINHA = 11.5
ALTURA_MAXIMA = 409
FONTE_RESPOSTA = 9
AVISO_CORTE = ("[RESPOSTA LONGA — o Excel corta o fim desta célula. O texto "
               "completo está em auditoria_geracao_respostas.csv, mesma "
               "pergunta e mesma letra.]")

CAB_XLSX = [
    ("lote", 6), ("qid", 8), ("pergunta", 42),
    ("resposta", 9), ("texto da resposta", LARGURA_RESPOSTA),
    ("JUIZ: correção 1-3", 13),
    ("JUIZ: as fontes conferem?", 15),
    ("JUIZ: inventou? S/N", 12),
    ("JUIZ: observação da resposta", 28),
    ("JUIZ: qual responde melhor?", 15),
    ("JUIZ: por quê", 30),
]


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
    cabe = _linhas_de_tela(texto) * PONTOS_POR_LINHA <= ALTURA_MAXIMA
    if cabe:
        return texto, max(30.0, _linhas_de_tela(texto) * PONTOS_POR_LINHA), False
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
    ws.row_dimensions[1].height = 34

    linha = 2
    cortadas = 0
    for n, b in enumerate(blocos):
        ini = linha
        for r in b["respostas"]:
            texto, altura, cortou = ajustar_resposta(r["texto"])
            cortadas += cortou
            vals = [b["lote"], b["qid"], b["pergunta"] if linha == ini else None,
                    r["letra"], texto, None, None, None, None, None, None]
            for col, v in enumerate(vals, start=1):
                c = ws.cell(row=linha, column=col, value=v)
                c.alignment = alto
                if col == 5:
                    c.font = Font(size=FONTE_RESPOSTA)
                if col in (6, 7, 8, 9):
                    c.fill = amarelo
                elif n % 2 == 0:
                    c.fill = cinza
                if linha == ini:
                    c.border = topo
            ws.row_dimensions[linha].height = altura
            linha += 1
        fim = linha - 1
        # pergunta e veredito do par valem para o bloco inteiro
        for col in (1, 2, 3, 10, 11):
            ws.merge_cells(start_row=ini, start_column=col,
                           end_row=fim, end_column=col)
            c = ws.cell(row=ini, column=col)
            c.alignment = centro if col in (1, 2, 10) else alto
            if col in (10, 11):
                c.fill = amarelo

    validacoes = [
        ('"1,2,3"', "F", "Use 1, 2 ou 3, ou deixe vazio se não puder julgar."),
        ('"S,PARCIAL,N,SEM FONTE"', "G", "Use S, PARCIAL, N ou SEM FONTE."),
        ('"S,N"', "H", "Use S ou N."),
        ('"A,B,EMPATE,NENHUMA"', "J", "Use A, B, EMPATE ou NENHUMA."),
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
    ("1) Para cada resposta, preencha 'correção 1-3'", True),
    ("3 = correta e completa: responde o que foi perguntado, sem erro", False),
    ("2 = parcialmente correta: mistura acerto e erro, ou omite o essencial", False),
    ("1 = factualmente errada, OU não entrega a resposta", False),
    ("", False),
    ("Uma RECUSA ('Não tenho informações suficientes...') vale 1. Todas as "
     "perguntas desta planilha TÊM resposta nas normas do acervo, então "
     "abster-se aqui é falhar, não é prudência.", True),
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
    ("4) Depois das duas, preencha 'qual responde melhor?' na primeira linha "
     "do bloco", True),
    ("A ou B = aquela responde melhor a pergunta", False),
    ("EMPATE = as duas servem igualmente, ou falham igualmente", False),
    ("NENHUMA = nenhuma das duas serve", False),
    ("", False),
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
configuração e traz a resposta-referência, e consultá-la antes invalida o
julgamento.

## O que o sistema faz

Uma pergunta entra; o sistema responde em no máximo uma página, terminando com
uma seção `## Fontes utilizadas`. O acervo são 54 normas brasileiras e
internacionais, em português e inglês.

Para cada pergunta, a planilha traz **duas respostas**, em configurações
diferentes do sistema, chamadas **A** e **B**. Qual é qual você não sabe, e a
ordem é sorteada a cada pergunta.

## O que preencher, por resposta

| coluna | o que é |
|---|---|
| `correção 1-3` | **3** correta e completa · **2** parcialmente correta, mistura acerto e erro ou omite o essencial · **1** factualmente errada, ou não entrega a resposta |
| `as fontes conferem?` | **S** as normas citadas existem e sustentam o que a resposta afirma · **PARCIAL** sustentam parte, ou falta a principal · **N** citam norma que não sustenta aquilo, ou que não existe · **SEM FONTE** |
| `inventou? S/N` | **S** se a resposta afirma lei, artigo, número ou prazo que você sabe que não é aquilo |
| `observação da resposta` | o que a nota não conta |

Uma **recusa** (`Não tenho informações suficientes para responder com
segurança`) vale **1**. Todas as perguntas desta planilha **têm** resposta nas
normas do acervo, então abster-se aqui é falhar, não é prudência.

Se não tiver como julgar, **deixe em branco** e diga o motivo na observação.

## O que preencher, por pergunta

Depois de julgar as duas, na primeira linha do bloco: **qual responde melhor?**
`A`, `B`, `EMPATE` (as duas servem igualmente, ou falham igualmente) ou
`NENHUMA`. E o motivo em `por quê`.

## Três coisas que mudam o resultado

- Julgue cada resposta **contra a pergunta** e contra o que a norma diz, não
  uma contra a outra. As duas podem ser 3, e as duas podem ser 1.
- **Você provavelmente vai adivinhar qual configuração é qual** em várias
  perguntas, pelo jeito de escrever. Não tem problema — mas julgue o conteúdo,
  não o palpite. A nota de cada resposta é o que mais importa; a preferência
  entre as duas é secundária, justamente porque ela é a mais sensível ao
  palpite.
- **Tamanho não é qualidade.** As duas respostas podem diferir muito em
  extensão, e a mais longa costuma ser a que inventa.

Algumas respostas não cabem na célula do Excel e começam com um aviso em
maiúsculas dizendo isso. Nesses casos leia o texto completo em
`auditoria_geracao_respostas.csv`, na mesma pergunta e na mesma letra — não
julgue pela parte que aparece.

## Ordem sugerida

Faça o **lote 1** inteiro primeiro. Ele já é uma amostra estratificada válida.

## O que vai ser feito com isso

Duas coisas. (1) Medir a concordância entre o seu julgamento e o de um painel de
quatro modelos de linguagem que já julgou estas mesmas respostas — o painel é
barato e escala, mas precisa de uma âncora humana, e já deu problema: dois dos
quatro se recusaram a pontuar centenas de respostas alegando motivos que o
próprio enunciado contradizia. (2) Decidir, de fora, se a busca em normas
melhora a resposta ou se o modelo já respondia bem sem ela — que é a pergunta
do artigo.

São as **mesmas perguntas** da auditoria da recuperação, nos mesmos lotes, de
propósito: dá para cruzar as duas e ver se, quando os 5 trechos não bastavam, a
resposta saiu errada.
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gerador", default="qwen3_27b",
                    choices=[*GERADORES, "ambos"],
                    help="padrao qwen3_27b, o gerador de producao")
    args = ap.parse_args()
    geradores = list(GERADORES) if args.gerador == "ambos" else [args.gerador]

    if not PERGUNTAS_DA_RECUPERACAO.exists():
        sys.exit(f"ERRO: nao achei {PERGUNTAS_DA_RECUPERACAO.relative_to(RAIZ)}; "
                 f"rode montar_auditoria.py primeiro.")
    with PERGUNTAS_DA_RECUPERACAO.open(encoding="utf-8-sig") as fh:
        base = [(r["lote"], r["qid"], r["pergunta"])
                for r in csv.DictReader(fh, delimiter=";")]
    print(f"perguntas herdadas da auditoria de recuperacao: {len(base)}")

    dados = {(g, c): carregar_judged(g, c)
             for g in geradores for c in ("rag", "norag")}

    rnd = random.Random(SEMENTE)
    blocos, linhas_r, linhas_p, linhas_c = [], [], [], []
    recusas = {g: {"rag": 0, "norag": 0} for g in geradores}

    # SORTEIO BALANCEADO, não moeda por pergunta. Com `shuffle` independente em
    # cada par o desequilíbrio é grande: medido nas 78, o `rag` caiu na letra A
    # em 49 e na B em 29. Um especialista que percebe "a A costuma ser a
    # melhor" fica ancorado, que é exatamente o que o cegamento deveria
    # impedir. Aqui metade de CADA LOTE tem o `rag` em A, sorteado com semente
    # fixa — o cegamento se mantém e a planilha é reprodutível.
    em_a: dict[tuple, bool] = {}
    for gerador in geradores:
        for lote in dict.fromkeys(l for l, _, _ in base):
            qs = [q for l, q, _ in base if l == lote]
            marcas = [True] * ((len(qs) + 1) // 2) + [False] * (len(qs) // 2)
            rnd.shuffle(marcas)
            em_a.update({(gerador, q): m for q, m in zip(qs, marcas)})

    for gerador in geradores:
        for lote, qid, pergunta in base:
            rec = {c: dados[(gerador, c)].get(qid) for c in ("rag", "norag")}
            faltam = [c for c, r in rec.items() if r is None]
            if faltam:
                sys.exit(f"ERRO: {qid} nao esta em {gerador} "
                         f"({', '.join(faltam)}).")
            if not rec["rag"].get("should_answer"):
                sys.exit(f"ERRO: {qid} esta marcada como irrespondivel, e a "
                         f"auditoria herdada deveria ter so respondiveis.")

            par = ([("rag", rec["rag"]), ("norag", rec["norag"])]
                   if em_a[(gerador, qid)] else
                   [("norag", rec["norag"]), ("rag", rec["rag"])])
            enunciado = limpar_enunciado(pergunta)
            bloco = {"lote": lote, "qid": qid, "pergunta": enunciado,
                     "respostas": []}

            for letra, (cen, r) in zip("AB", par):
                texto = limpar_resposta(r.get("answer"))
                e_recusa = RECUSA in texto
                if e_recusa:
                    recusas[gerador][cen] += 1
                bloco["respostas"].append({"letra": letra, "texto": texto})
                linhas_r.append([lote, qid, enunciado, letra, texto,
                                 "", "", "", ""])
                nt = notas_do_painel(r)
                linhas_c.append([
                    lote, qid, gerador, letra, cen,
                    rec["rag"].get("question_type", ""),
                    len(r.get("context_chunk_ids") or []),
                    "SIM" if e_recusa else "nao",
                    len(texto),
                    *[nt[f"mediana_{c}"] for c in CRIT_CHAVE],
                    *[nt[f"{j}_{c}"] for j in JUIZES for c in CRIT_CHAVE],
                    limpar_enunciado(rec["rag"].get("reference_answer") or ""),
                ])

            linhas_p.append([lote, qid, enunciado, "", ""])
            blocos.append(bloco)

    print("\nrecusas no conjunto auditado (so para dimensionar o cegamento):")
    for g in geradores:
        print(f"  {g:<13} rag {recusas[g]['rag']:>3}/{len(base)}   "
              f"norag {recusas[g]['norag']:>3}/{len(base)}")
    if args.gerador == "ambos":
        print("\n! A mesma pergunta aparece DUAS vezes na planilha, uma por")
        print("  gerador. Mande os lotes de cada gerador em momentos separados,")
        print("  ou a segunda leitura sai contaminada pela primeira.")

    SAIDA.mkdir(exist_ok=True)
    print()
    escrever_xlsx(SAIDA / "AUDITORIA_GERACAO.xlsx", blocos)
    escrever_csv(SAIDA / "auditoria_geracao_respostas.csv", [
        "lote", "qid", "pergunta", "resposta_letra", "texto_da_resposta",
        "JUIZ_correcao_1a3", "JUIZ_fontes_conferem_S_PARCIAL_N_SEMFONTE",
        "JUIZ_inventou_S_N", "JUIZ_observacao",
    ], linhas_r)
    escrever_csv(SAIDA / "auditoria_geracao_perguntas.csv", [
        "lote", "qid", "pergunta",
        "JUIZ_qual_melhor_A_B_EMPATE_NENHUMA", "JUIZ_por_que",
    ], linhas_p)
    escrever_csv(SAIDA / "auditoria_geracao_CHAVE_nao_abrir_antes.csv", [
        "lote", "qid", "gerador", "resposta_letra", "cenario",
        "tipo_de_pergunta", "n_trechos_no_contexto", "e_recusa",
        "caracteres_da_resposta",
        *[f"mediana_{c}" for c in CRIT_CHAVE],
        *[f"{j}_{c}" for j in JUIZES for c in CRIT_CHAVE],
        "resposta_referencia",
    ], linhas_c)
    (SAIDA / "auditoria_geracao_INSTRUCOES.md").write_text(
        INSTRUCOES, encoding="utf-8")
    print(f"  ok {(SAIDA / 'auditoria_geracao_INSTRUCOES.md').relative_to(RAIZ)}")

    print("\nMANDE SO o AUDITORIA_GERACAO.xlsx (e o .csv das respostas, para o")
    print("caso de alguma resposta nao caber na celula). A CHAVE NAO vai: ela")
    print("tem o braco de cada letra, a resposta-referencia e as notas do painel.")


if __name__ == "__main__":
    main()
