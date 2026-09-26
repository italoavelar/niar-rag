#!/usr/bin/env python3
"""Gera o gabarito em lotes, com o Claude em sessão: sorteia de um lado, integra do outro.

Para quem
─────────
Para quem está gerando as 400 perguntas novas da Fase 1. Os dois subcomandos são
as duas pontas do mesmo trabalho:

    lotes.py sortear   → escreve um .md pronto para colar na sessão do Claude
    (colar, o Claude responde JSON, salvar como .out.json)
    lotes.py integrar  → confere o JSON e escreve no golden_qa.jsonl

A regra que o `sortear` existe para proteger
────────────────────────────────────────────
**Quem escolhe o trecho é o sorteio, não o modelo — nem a pessoa.** Se alguém
abrir o corpus e escolher passagens "boas", o benchmark passa a medir o que essa
pessoa (ou o gerador) acha interessante, e não o que o acervo exige. Item
escolhido pelo modelo não serve para avaliar modelo; item escolhido a dedo não
serve para avaliar nada. Por isso o sorteio é aleatório, com semente registrada
no manifesto.

O .md sai **sem os chunk_id**. O modelo não precisa deles e, se os receber,
tende a devolvê-los transcritos errado. O casamento número → id fica no
manifesto `.ids.json` e é feito aqui na integração.

A trava que o `integrar` existe para aplicar
────────────────────────────────────────────
Cada citação de `evidencia` tem de aparecer **literalmente** no trecho de onde
saiu. É o que torna impossível repetir a q0075, onde a resposta afirmava "quatro
níveis" e o trecho anotado enumerava dois: a pergunta ficou impossível por
construção, nem a recuperação perfeita a satisfazia. Aqui a citação que não casa
derruba o registro antes de ele entrar no gabarito.

Por que digitar id à mão não é opção: dos 33 ids escritos manualmente em
`build_gold_manual.py`, **zero** existem no corpus de hoje — estão no formato
posicional antigo (`_p4_c2`), que morreu quando o id passou a vir do conteúdo.

O que este script NÃO faz
─────────────────────────
Não anota vizinhos com grau 1. O gerador vê 1 ou 2 trechos de ~4.900 e não tem
como saber que outro responde igual ou melhor — julgar relevância é a Fase 2,
com pool dos três braços. Aqui só entra grau 2: "este é o trecho de onde a
pergunta saiu".

Uso
───
  python eval/tools/lotes.py sortear --tipo factual --n 10 --semente 42
  python eval/tools/lotes.py integrar --lote lote_01_factual --conferir
  python eval/tools/lotes.py integrar --lote lote_01_factual --aplicar
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import random
import re
import shutil
import sys
import unicodedata
from collections import defaultdict
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
EVAL = PROJECT_ROOT / "eval"
sys.path.insert(0, str(EVAL))

for _fluxo in (sys.stdout, sys.stderr):
    try:
        _fluxo.reconfigure(encoding="utf-8")
    except Exception:
        pass

CORPUS = PROJECT_ROOT / "data/processed/documents.jsonl"
GOLDEN = EVAL / "data/golden_qa.jsonl"
QUERIES = EVAL / "data/queries.csv"
QRELS = EVAL / "data/qrels.csv"
AUDITORIA = PROJECT_ROOT / "docs/auditoria_acervo.csv"
LOTES = EVAL / "lotes"
GERADOR = EVAL / "01_build_golden.py"

TIPOS = ("factual", "multi_hop", "comparative", "unanswerable")

# Expressões que denunciam pergunta escrita olhando o trecho. Quem pergunta ao
# LEME nunca viu o documento; "segundo o texto" torna a pergunta dependente de um
# contexto que o usuário real não tem.
META = re.compile(
    r"(segundo o (texto|trecho|documento)|de acordo com o (texto|trecho|documento)|"
    r"n[oa] (texto|trecho|documento) (acima|citado|apresentado)|conforme o (texto|trecho))",
    re.IGNORECASE,
)

REF_UNANS = ("Não encontrei informações suficientes nas fontes recuperadas "
             "para responder com segurança.")


# ── utilidades ───────────────────────────────────────────────────────────────

def normalizar(t: str) -> str:
    """Colapsa espaço e tira acento/caixa. Para comparar citação com trecho sem
    tropeçar em quebra de linha reposicionada pelo recorte."""
    t = unicodedata.normalize("NFKD", (t or "").lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    return " ".join(t.split())


def norm_pergunta(q: str) -> str:
    return " ".join((q or "").lower().split())


def carregar_corpus() -> dict[str, dict]:
    if not CORPUS.exists():
        sys.exit(f"corpus não encontrado: {CORPUS}")
    corpus = {}
    for linha in CORPUS.open(encoding="utf-8"):
        if linha.strip():
            r = json.loads(linha)
            corpus[r["id"]] = r
    return corpus


def carregar_gabarito() -> list[dict]:
    if not GOLDEN.exists():
        return []
    return [json.loads(l) for l in GOLDEN.open(encoding="utf-8") if l.strip()]


def documentos_aprovados() -> set[str] | None:
    """Documentos que a Fase 0 aprovou. None = auditoria ainda não preenchida."""
    if not AUDITORIA.exists():
        return None
    aprovados, preenchidas = set(), 0
    for r in csv.DictReader(AUDITORIA.open(encoding="utf-8-sig")):
        classif = (r.get("classificacao") or "").strip().lower()
        if classif:
            preenchidas += 1
            if classif.startswith(("integro", "íntegro", "ok", "aprovad")):
                aprovados.add(r["document_id"])
    return aprovados if preenchidas else None


def sys_prompt() -> str:
    """Lê a constante SYS do gerador sem executar o módulo (que exige chave de API).

    Ler do código em vez de repetir aqui é de propósito: se o prompt da sessão
    divergir do prompt do gerador, as perguntas deixam de ser comparáveis entre si.
    """
    arvore = ast.parse(GERADOR.read_text(encoding="utf-8"))
    for no in arvore.body:
        if (isinstance(no, ast.Assign) and isinstance(no.targets[0], ast.Name)
                and no.targets[0].id == "SYS"):
            return ast.literal_eval(no.value)
    sys.exit(f"constante SYS não encontrada em {GERADOR}")


def unidades_neutras(corpus: dict[str, dict], alvo: int = 2600,
                     semente: int = 0) -> dict[str, dict]:
    """Monta as passagens mostradas ao gerador, SEM coincidir com nenhuma das
    estratégias de recorte que o projeto compara.

    POR QUE ISTO EXISTE. A pergunta nasce com o tamanho da passagem que a gerou.
    Se a passagem for um trecho de 1.200 caracteres, o gabarito favorece o recorte
    de 1.200. Se for a página, favorece o recorte por página. Em qualquer caso o
    benchmark deixa de servir para comparar estratégias de recorte — que é uma das
    perguntas do projeto.

    A REGRA: a unidade de geração não pode coincidir com NENHUMA estratégia sob
    teste.

        C1 Page              → não pode ser a página
        C4 Recursive 1000/200→ não pode ser janela de tamanho fixo alinhada a grade
        C5 Semantic          → não pode ser fronteira temática
        C6 Legal Structure   → não pode ser artigo/seção

    O que sobra e não é nenhuma delas: uma **janela de deslocamento aleatório
    dentro do documento, ajustada para começar e terminar em fronteira de frase**.
    O deslocamento aleatório garante que ela não se alinhe a começo de página, a
    grade fixa nem a início de artigo; a fronteira de frase existe só para o texto
    não começar no meio de uma palavra, e frase não é estratégia sob teste.

    A ÂNCORA NÃO É A PASSAGEM. O que fica anotado é a **citação literal** que o
    gerador devolve em `evidencia`, validada contra o texto na integração. Citação
    é fato, e fato sobrevive a qualquer recorte — a menos que o recorte a parta ao
    meio, que é exatamente o defeito a medir. É o desenho de *supporting facts* do
    HotpotQA.
    """
    rng = random.Random(semente)
    por_doc: dict[str, list[str]] = defaultdict(list)
    for cid, r in corpus.items():
        por_doc[r["metadata"].get("document_id")].append(cid)

    unidades = {}
    for doc, cids in sorted(por_doc.items()):
        # reconstrói o documento na ordem, com a sobreposição do recorte atual
        # removida na junção — a passagem não deve herdar artefato de recorte.
        cids = sorted(cids, key=lambda c: (corpus[c]["metadata"].get("page") or 0, c))
        inteiro = "\n".join(corpus[c]["text"] for c in cids)
        if len(inteiro) < alvo // 2:
            continue

        # fronteiras de frase; a janela encaixa entre duas delas
        cortes = [0] + [m.end() for m in re.finditer(r"[.!?]\s", inteiro)] + [len(inteiro)]
        if len(cortes) < 3:
            continue

        # deslocamento aleatório: nem grade fixa, nem início de página
        passo = max(1, alvo // 2)
        inicio = rng.randrange(0, max(1, min(passo, len(inteiro) // 2)))
        n = 0
        while inicio < len(inteiro) - alvo // 3:
            fim_alvo = inicio + alvo
            ini_frase = min(cortes, key=lambda x: abs(x - inicio))
            fim_frase = min(cortes, key=lambda x: abs(x - fim_alvo))
            if fim_frase <= ini_frase:
                break
            texto = inteiro[ini_frase:fim_frase].strip()
            if len(texto) >= alvo // 2:
                n += 1
                titulo = corpus[cids[0]]["metadata"].get("title") or doc
                unidades[f"{doc}::j{n}"] = {
                    "document_id": doc,
                    "rotulo": f"{titulo} — passagem {n}",
                    # os chunk_id são conveniência: dizem de onde a passagem veio
                    # no recorte ATUAL. Não são a definição da evidência.
                    "chunks": [c for c in cids
                               if corpus[c]["text"][:60] and
                               normalizar(corpus[c]["text"][:60]) in normalizar(texto)],
                    "texto": texto,
                }
            inicio = fim_frase if fim_frase > inicio else inicio + alvo
    return unidades


def rotulo(rec: dict) -> str:
    md = rec["metadata"]
    titulo = (md.get("title") or md.get("document_id") or "").strip()
    pagina = md.get("page")
    return f"{titulo}" + (f" — p. {pagina}" if pagina else "")


def proximo_numero(tipo: str) -> int:
    LOTES.mkdir(parents=True, exist_ok=True)
    usados = [int(m.group(1)) for p in LOTES.glob(f"lote_*_{tipo}.ids.json")
              if (m := re.match(r"lote_(\d+)_", p.name))]
    return max(usados, default=0) + 1


# ── sortear ──────────────────────────────────────────────────────────────────

def cmd_sortear(args) -> None:
    corpus = carregar_corpus()
    gabarito = carregar_gabarito()
    random.seed(args.semente)

    aprovados = documentos_aprovados()
    if aprovados is None:
        print("⚠ auditoria da Fase 0 ainda não preenchida — sorteando sobre os 63 documentos.")
        elegiveis = dict(corpus)
    else:
        elegiveis = {c: r for c, r in corpus.items()
                     if r["metadata"].get("document_id") in aprovados}
        print(f"✓ {len(aprovados)} documentos aprovados na Fase 0 → {len(elegiveis)} trechos elegíveis")

    # não repetir trecho que já é âncora de alguma pergunta
    ja_ancorados = {c for r in gabarito for c in (r.get("qrels") or {})}
    elegiveis = {c: r for c, r in elegiveis.items() if c not in ja_ancorados}
    elegiveis = {c: r for c, r in elegiveis.items() if len(r["text"]) >= args.min_chars}
    print(f"  {len(ja_ancorados)} trechos já ancorados foram excluídos; "
          f"sobram {len(elegiveis)} com ≥ {args.min_chars} caracteres")

    # Unidades independentes de recorte — ver `unidades_neutras`.
    unid = unidades_neutras(elegiveis, semente=args.semente)
    print(f"  {len(unid)} unidades neutras (página ou janela) montadas a partir dos trechos elegíveis")

    itens: list[dict] = []
    if args.tipo == "factual":
        if len(unid) < args.n:
            sys.exit(f"só há {len(unid)} unidades para {args.n} pedidos")
        for i, u in enumerate(random.sample(sorted(unid), args.n), 1):
            itens.append({"n": i, "unidades": [u], "chunks": unid[u]["chunks"]})

    elif args.tipo == "multi_hop":
        por_doc = defaultdict(list)
        for u, d in unid.items():
            por_doc[d["document_id"]].append(u)
        candidatos = [d for d, us in por_doc.items() if len(us) >= 2]
        if not candidatos:
            sys.exit("nenhum documento com duas unidades")
        for i in range(1, args.n + 1):
            doc = random.choice(candidatos)
            ua, ub = random.sample(sorted(por_doc[doc]), 2)
            itens.append({"n": i, "unidades": [ua, ub],
                          "chunks": unid[ua]["chunks"] + unid[ub]["chunks"]})

    elif args.tipo == "comparative":
        por_doc = defaultdict(list)
        for u, d in unid.items():
            por_doc[d["document_id"]].append(u)
        if len(por_doc) < 2:
            sys.exit("são necessários pelo menos dois documentos")
        for i in range(1, args.n + 1):
            da, db = random.sample(sorted(por_doc), 2)
            ua, ub = random.choice(sorted(por_doc[da])), random.choice(sorted(por_doc[db]))
            itens.append({"n": i, "unidades": [ua, ub],
                          "chunks": unid[ua]["chunks"] + unid[ub]["chunks"]})

    else:  # unanswerable — não sai de trecho nenhum
        itens = [{"n": i, "unidades": [], "chunks": []} for i in range(1, args.n + 1)]

    numero = proximo_numero(args.tipo)
    nome = f"lote_{numero:02d}_{args.tipo}"
    md = _montar_md(nome, args.tipo, itens, corpus, args.semente, unid)

    (LOTES / f"{nome}.md").write_text(md, encoding="utf-8")
    (LOTES / f"{nome}.ids.json").write_text(json.dumps({
        "lote": nome,
        "tipo": args.tipo,
        "semente": args.semente,
        "gerado_em": datetime.now().isoformat(timespec="seconds"),
        "itens": itens,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n✓ {LOTES / (nome + '.md')}        ← cole ESTE arquivo inteiro na sessão do Claude")
    print(f"✓ {LOTES / (nome + '.ids.json')}   ← manifesto, não mexer")
    print(f"\nDepois salve a resposta do Claude como: {LOTES / (nome + '.out.json')}")
    print(f"E rode: python eval/tools/lotes.py integrar --lote {nome} --conferir")


def _montar_md(nome, tipo, itens, corpus, semente, unid=None) -> str:
    p = [f"<!-- {nome} · semente {semente} · gerado por eval/tools/lotes.py -->",
         "", sys_prompt(), "", "---", ""]

    if tipo == "factual":
        p += [f"Abaixo estão {len(itens)} passagens. Para CADA UMA, gere **uma** pergunta cuja "
              "resposta esteja contida e completa naquela passagem.", ""]
    elif tipo == "multi_hop":
        p += [f"Abaixo estão {len(itens)} pares de passagens **do mesmo documento**. Para CADA PAR, "
              "gere **uma** pergunta cuja resposta exija combinar os dois — não respondível por "
              "apenas um. Em `evidencia`, cite literalmente de CADA trecho.", ""]
    elif tipo == "comparative":
        p += [f"Abaixo estão {len(itens)} pares de passagens de **documentos diferentes**. Para CADA "
              "PAR, gere **uma** pergunta que relacione ou compare o que os dois dizem sobre um "
              "ponto em comum. A pergunta deve nomear os dois documentos ou regimes. Em "
              "`evidencia`, cite literalmente de CADA documento.", ""]
    else:
        temas = sorted({r["metadata"].get("theme", "") for r in corpus.values()
                        if r["metadata"].get("theme")})
        p += [f"O acervo trata de: {', '.join(temas)}.", "",
              f"Gere {len(itens)} perguntas que pareçam plausíveis em saúde/direito/IA mas que "
              "**não possam** ser respondidas por esse acervo (legislação de outros países, "
              "estatísticas específicas, casos clínicos individuais, temas fora do escopo). "
              "Não use `evidencia` nem `answer` — a resposta é sempre a mesma recusa padrão.", ""]

    if tipo != "unanswerable":
        p += ["**Recusar é resultado esperado.** Se o(s) trecho(s) não sustentarem uma pergunta "
              "autossuficiente, devolva `{\"n\": N, \"question\": null}` para aquele item. "
              "Descartar é barato; pergunta irrespondível contamina o benchmark inteiro.", ""]

    p += ["## Formato da resposta", "",
          "Responda **apenas** com um array JSON, um objeto por item, usando o mesmo `n`:", "",
          "```json", "["]
    if tipo == "unanswerable":
        p += ['  {"n": 1, "question": "...", "persona": "saude|desenvolvedor|pesquisador"},',
              '  {"n": 2, "question": "..."}']
    else:
        p += ['  {"n": 1, "question": "...", "answer": "...", "evidencia": ["citação literal"],',
              '   "difficulty": "easy|medium|hard", "persona": "saude|desenvolvedor|pesquisador"},',
              '  {"n": 2, "question": null}']
    p += ["]", "```", "", "---", ""]

    if tipo == "unanswerable":
        return "\n".join(p) + "\n"

    for it in itens:
        us = [unid[u] for u in it.get("unidades", [])]
        if len(us) == 1:
            p += [f"## Passagem {it['n']} — {us[0]['rotulo']}", "", us[0]["texto"].strip(), ""]
        elif len(us) == 2:
            p += [f"## Par {it['n']}", "",
                  f"### A — {us[0]['rotulo']}", "", us[0]["texto"].strip(), "",
                  f"### B — {us[1]['rotulo']}", "", us[1]["texto"].strip(), ""]
    return "\n".join(p) + "\n"


# ── integrar ─────────────────────────────────────────────────────────────────

def cmd_integrar(args) -> None:
    base = LOTES / args.lote
    manifesto_p, saida_p = Path(f"{base}.ids.json"), Path(f"{base}.out.json")
    for p in (manifesto_p, saida_p):
        if not p.exists():
            sys.exit(f"não encontrado: {p}")

    manifesto = json.loads(manifesto_p.read_text(encoding="utf-8"))
    tipo = manifesto["tipo"]
    por_n = {it["n"]: it["chunks"] for it in manifesto["itens"]}

    bruto = saida_p.read_text(encoding="utf-8").strip()
    if bruto.startswith("```"):                       # cerca de código colada junto
        bruto = re.sub(r"^```[a-z]*\n|\n```$", "", bruto)
    try:
        respostas = json.loads(bruto)
    except json.JSONDecodeError as e:
        sys.exit(f"{saida_p.name} não é JSON válido: {e}")
    if isinstance(respostas, dict):
        respostas = respostas.get("pairs") or respostas.get("itens") or [respostas]

    corpus = carregar_corpus()
    gabarito = carregar_gabarito()
    vistas = {norm_pergunta(r["question"]) for r in gabarito}

    aceitos, recusas, rejeitados = [], [], []

    for resp in respostas:
        n = resp.get("n")
        if n not in por_n:
            rejeitados.append((n, "item 'n' não existe no manifesto"))
            continue
        pergunta = (resp.get("question") or "").strip()
        if not pergunta:
            recusas.append(n)
            continue

        chunks = por_n[n]
        faltando = [c for c in chunks if c not in corpus]
        if faltando:
            rejeitados.append((n, f"chunk_id fora do corpus: {faltando}"))
            continue
        if norm_pergunta(pergunta) in vistas:
            rejeitados.append((n, "pergunta duplicada"))
            continue
        if (m := META.search(pergunta)):
            rejeitados.append((n, f"expressão meta: {m.group(0)!r}"))
            continue

        if tipo == "unanswerable":
            resposta, evidencia = REF_UNANS, []
        else:
            resposta = (resp.get("answer") or "").strip()
            evidencia = [e for e in (resp.get("evidencia") or []) if isinstance(e, str) and e.strip()]
            if not resposta:
                rejeitados.append((n, "sem resposta-referência"))
                continue
            if not evidencia:
                rejeitados.append((n, "sem evidência — a regra de contenção exige citação literal"))
                continue

            # A TRAVA: cada citação tem de estar literalmente em algum trecho do item.
            textos = {c: normalizar(corpus[c]["text"]) for c in chunks}
            casou_em = {}
            problema = None
            for cit in evidencia:
                achou = [c for c, t in textos.items() if normalizar(cit) in t]
                if not achou:
                    problema = f"citação não aparece em nenhum trecho: {cit[:70]!r}"
                    break
                casou_em[cit] = achou[0]
            if problema:
                rejeitados.append((n, problema))
                continue
            if tipo in ("multi_hop", "comparative") and len(set(casou_em.values())) < 2:
                rejeitados.append((n, "citação de um trecho só — não combina os dois"))
                continue

        vistas.add(norm_pergunta(pergunta))
        aceitos.append({
            "question": pergunta,
            "reference_answer": resposta,
            "question_type": tipo,
            "difficulty": resp.get("difficulty") or ("hard" if tipo == "comparative" else "medium"),
            "persona": resp.get("persona") or "",
            "evidencia": evidencia,
            "chunks": chunks,
            "_n": n,
        })

    # ── relatório ────────────────────────────────────────────────────────────
    total = len(respostas)
    print(f"\nLOTE {args.lote}  ({tipo})")
    print(f"  itens no manifesto     : {len(por_n)}")
    print(f"  respostas recebidas    : {total}")
    print(f"  aceitos                : {len(aceitos)}")
    print(f"  recusados pelo modelo  : {len(recusas)}  {recusas if recusas else ''}")
    print(f"  REJEITADOS na validação: {len(rejeitados)}")
    for n, motivo in rejeitados:
        print(f"      item {n}: {motivo}")

    if tipo == "unanswerable" and aceitos:
        print("\n  ⚠ 'unanswerable' não passa pela trava de evidência. Antes de aceitar o lote,")
        print("    confirme à mão que o acervo realmente NÃO responde a cada uma delas —")
        print("    uma 'unanswerable' que o acervo responde é falso negativo plantado no gabarito.")

    if not args.aplicar:
        print("\n(conferência apenas — nada foi escrito. Use --aplicar para gravar.)")
        return
    if rejeitados and not args.forcar:
        sys.exit("\n✗ há itens rejeitados. Corrija e regere, ou use --forcar para gravar só os aceitos.")
    if not aceitos:
        sys.exit("\n✗ nada a gravar.")

    _gravar(aceitos, corpus, gabarito, manifesto)


def _gravar(aceitos, corpus, gabarito, manifesto) -> None:
    from lib.retrievers import detect_lang

    carimbo = datetime.now().strftime("%Y%m%d-%H%M%S")
    if GOLDEN.exists():
        shutil.copy2(GOLDEN, f"{GOLDEN}.bak-pre-lote-{carimbo}")

    proximo = max((int(r["qid"][1:]) for r in gabarito if r["qid"][1:].isdigit()), default=0)
    novos = []
    for item in aceitos:
        proximo += 1
        chunks = item["chunks"]
        # Grau 2 e só. Vizinho com grau 1 é heurística posicional, não julgamento:
        # quem julga relevância é a Fase 2, sobre o pool dos três braços.
        qrels = {c: 2 for c in chunks}
        langs = {detect_lang(corpus[c]["text"]) for c in chunks}
        novos.append({
            "qid": f"q{proximo:04d}",
            "question": item["question"],
            "reference_answer": item["reference_answer"],
            "question_type": item["question_type"],
            "source_lang": ("en" if langs == {"en"} else "pt" if langs == {"pt"}
                            else ("mixed" if langs else None)),
            "difficulty": item["difficulty"],
            "theme": ", ".join(sorted({corpus[c]["metadata"].get("theme", "") for c in chunks}))
                     if chunks else "fora-de-escopo",
            "source_docs": sorted({corpus[c]["metadata"].get("document_id", "") for c in chunks}),
            "qrels": qrels,
            "qrels_text": {c: corpus[c]["text"] for c in chunks},
            "evidencia": item["evidencia"],
            "persona": item["persona"],
            "status": "ok",
            "procedencia": f"{manifesto['lote']} (semente {manifesto['semente']})",
        })

    with GOLDEN.open("a", encoding="utf-8") as f:
        for r in novos:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    todos = gabarito + novos
    _derivados(todos)
    print(f"\n✓ {len(novos)} perguntas acrescentadas "
          f"({novos[0]['qid']} … {novos[-1]['qid']}) — gabarito agora tem {len(todos)}")
    print(f"✓ backup: {Path(GOLDEN).name}.bak-pre-lote-{carimbo}")


def _derivados(registros) -> None:
    respondiveis = [r for r in registros if r.get("qrels") and r.get("status", "ok") == "ok"]
    with QUERIES.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["QueryId", "Query"])
        for r in respondiveis:
            w.writerow([r["qid"], r["question"]])
    with QRELS.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["QueryId", "ChunkId", "Relevance"])
        for r in respondiveis:
            for cid, g in r["qrels"].items():
                w.writerow([r["qid"], cid, g])
    print(f"✓ derivados: {QUERIES.name} ({len(respondiveis)} q) + {QRELS.name}")


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("sortear", help="sorteia trechos e escreve o .md para colar no Claude")
    s.add_argument("--tipo", choices=TIPOS, required=True)
    s.add_argument("--n", type=int, default=10, help="quantos itens no lote (padrão 10)")
    s.add_argument("--semente", type=int, required=True, help="registrada no manifesto")
    s.add_argument("--min-chars", type=int, default=400,
                   help="ignora trechos curtos demais para sustentar pergunta (padrão 400)")
    s.set_defaults(func=cmd_sortear)

    i = sub.add_parser("integrar", help="confere o JSON do Claude e grava no gabarito")
    i.add_argument("--lote", required=True, help="nome base, ex.: lote_01_factual")
    i.add_argument("--conferir", action="store_true", help="só valida, não escreve (padrão)")
    i.add_argument("--aplicar", action="store_true", help="grava no golden_qa.jsonl")
    i.add_argument("--forcar", action="store_true", help="grava os aceitos mesmo havendo rejeitados")
    i.set_defaults(func=cmd_integrar)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
