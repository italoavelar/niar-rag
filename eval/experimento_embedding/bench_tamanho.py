#!/usr/bin/env python3
"""Etapa 1 do benchmark de recorte: QUAL TAMANHO responde melhor às perguntas.

A pergunta é uma só, e estreita de propósito: mantendo tudo igual — mesma
extração, mesma estratégia (janela recursiva), mesmo embedding, mesmo top-k,
mesmas perguntas — **qual tamanho de trecho coloca toda a evidência necessária
no top-k?**

Só depois de fixar o tamanho é que faz sentido comparar ESTRATÉGIAS (página,
artigo, semântico) — misturar as duas perguntas embaralha o resultado.

A AMOSTRA é estratificada, não sorteada. O que importa ao recorte é a forma do
documento, e o acervo tem quatro formas bem distintas:

    normativo articulado PT   onde existe "Art. 5º, § 2º" para cortar
    normativo articulado EN   articulado em outra língua (GDPR)
    guidance em prosa EN      NIST/WHO/UNESCO — sem artigo, só títulos de seção
    prosa PT                  controle

Sortear 15 de 54 poderia não trazer nenhum articulado EN. Estratificar garante
que cada forma esteja representada, que é o que a comparação precisa.

A SOBREPOSIÇÃO acompanha o tamanho (20%), senão duas variáveis mudam juntas e
não se sabe qual causou o quê.

Uso:
  python eval/tools/bench_tamanho.py amostra --n 15
  python eval/tools/bench_tamanho.py recortar
  python eval/tools/bench_tamanho.py recortar --tamanhos 400, 1200, 3500
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import random
import re
import statistics as st
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

for _f in (sys.stdout, sys.stderr):
    try:
        _f.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Corpus já limpo do ruído de extração (ver limpar.py). Partir do bruto traria
# andaime de tabela e rodapé institucional para dentro das quatro condições.
CORPUS = AQUI / "dados/corpus_limpo.jsonl"
PAGINAS = PROJECT_ROOT / "data/processed/paginas_limpas.jsonl"
AMOSTRA = AQUI / "dados/amostra.json"
DESTINO = AQUI / "dados"

TAMANHOS = (400, 1200, 3500)

# Parent-child (small-to-big): indexa o filho, entrega o pai.
#
# Por que filho 400 e pai 3500, e não um tamanho novo: os dois números já são
# condições isoladas da grade. Se o parent-child vencer com um pai de 4.570 —
# tamanho que nenhuma outra condição testa — não dá para saber se ganhou pelo
# MECANISMO (buscar pequeno, entregar grande) ou por 4.570 ser um bom tamanho.
# Reusando os números da grade, o mecanismo fica isolado.
PC_FILHO = int(os.getenv("PC_FILHO", "400"))
# O pai de 3500 mistura duas coisas: a ESTRATEGIA parent-child e a UNIDADE DE
# SAIDA de 3500 caracteres. Para separar as duas e medir a estrategia sozinha,
# o pai vira parametro: PC_PAI=1200 da o parent-child na largura que venceu.
PC_PAI = int(os.getenv("PC_PAI", "3500"))
PC_FILHOS_BUSCADOS, PC_PAIS_MAX = 12, 3
SEPARADORES = ["\n\n", "\n", ". ", "; ", ", ", " "]
ARTIGO = re.compile(r"(Art\.\s*\d+[º°]?|Artigo\s+\d+[º°]?|Article\s+\d+)")

# Marcadores grosseiros de idioma: basta separar PT de EN, não identificar língua.
PT = re.compile(r"\b(que|não|para|com|dos|das|pelo|pela|deve|será)\b", re.I)
EN = re.compile(r"\b(the|of|and|shall|that|with|for|this|are)\b", re.I)


def carregar_corpus() -> dict[str, list[dict]]:
    por_doc: dict[str, list[dict]] = collections.defaultdict(list)
    for l in CORPUS.open(encoding="utf-8"):
        if l.strip():
            r = json.loads(l)
            por_doc[r["metadata"]["document_id"]].append(r)
    return por_doc


ENCADEAMENTO_MINIMO = 8     # pares Art.N/Art.N+1 presentes no documento
TRECHOS_MINIMO = 30         # documento menor que isso nao tem o que recortar


def forma(trechos: list[dict]) -> str:
    """Classifica o documento pela forma que importa ao recorte.

    O critério é SEQUÊNCIA, não contagem nem densidade — as duas erram:

      contagem  confunde "ter artigos" com "citar artigos". O
                `regulatory_considerations_ai_health_WHO_2024` traz 7 marcadores,
                todos citando GDPR e AI Act; é guidance em prosa.

      densidade  penaliza norma longa. O `european_health_data_space_EU_2025` é
                 regulamento articulado com 121 marcadores, mas tem considerandos
                 extensos antes dos artigos — densidade 2,3, abaixo de qualquer
                 limiar razoável, e ainda assim é articulado.

    Documento articulado numera os próprios artigos em SEQUÊNCIA: Art. 1, Art. 2,
    Art. 3… Documento que cita artigos alheios traz números esparsos (5, 9, 24,
    35). O sinal é o **encadeamento**: quantos números têm o sucessor presente.

    Cobertura de faixa também não serve, e o GDPR mostra por quê: ele numera os
    Artigos 1 a 99 em sequência perfeita, mas cita os Artigos 263, 290 e 338 do
    TFUE. Esses três outliers esticam a faixa de 99 para 338 e derrubam a
    cobertura para 0,31 — reprovando o documento mais articulado do acervo.
    Contar pares consecutivos ignora outlier por construção.
    """
    texto = " ".join(t["text"] for t in trechos)
    numeros = {int(n) for n in re.findall(
        r"(?:Art\.|Artigo|Article)\s*(\d{1,3})", texto)}
    encadeados = sum(1 for n in numeros if n + 1 in numeros)
    ingles = len(EN.findall(texto)) > len(PT.findall(texto))
    tipo = "articulado" if encadeados >= ENCADEAMENTO_MINIMO else "prosa"
    return f"{tipo}_{'EN' if ingles else 'PT'}"


def emissor(trechos: list[dict]) -> str:
    """Quem publicou. O campo `issuer` está vazio nos 54 documentos; `author`
    está preenchido em todos, então é ele. Corta no primeiro ';' porque há
    coautoria ("FDA; Health Canada; MHRA") e o que importa é o emissor principal.
    """
    autor = (trechos[0]["metadata"].get("author") or "?").strip()
    return autor.split(";")[0].strip()[:40]


# ── amostra estratificada ────────────────────────────────────────────────────

COTAS = {"articulado_PT": 5, "prosa_EN": 5, "articulado_EN": 2, "prosa_PT": 3}


def cmd_amostra(args) -> None:
    por_doc = carregar_corpus()
    formas = {d: forma(ts) for d, ts in por_doc.items()}
    grupos: dict[str, list[str]] = collections.defaultdict(list)
    for d, f in formas.items():
        grupos[f].append(d)

    print("O ACERVO, pela forma que importa ao recorte:")
    for f in ("articulado_PT", "articulado_EN", "prosa_PT", "prosa_EN"):
        ds = grupos.get(f, [])
        n = sum(len(por_doc[d]) for d in ds)
        print(f"  {f:16s} {len(ds):3d} documentos, {n:5d} trechos")

    rng = random.Random(args.semente)
    escolhidos: list[str] = []
    for f, cota in COTAS.items():
        # Documento pequeno demais não tem o que recortar — fica fora.
        candidatos = sorted((d for d in grupos.get(f, [])
                             if len(por_doc[d]) >= TRECHOS_MINIMO),
                            key=lambda d: -len(por_doc[d]))
        rng.shuffle(candidatos)

        # DIVERSIDADE DE EMISSOR dentro do estrato. Sem isto, "articulado_PT"
        # tenderia a encher de resoluções do CFM (9 dos 54 documentos do acervo),
        # e o resultado passaria a descrever o estilo de redação de um emissor
        # em vez da forma do documento. Escolhe um por emissor primeiro; só
        # repete emissor se faltar documento.
        pegos, usados = [], set()
        for d in candidatos:
            if len(pegos) >= cota:
                break
            if emissor(por_doc[d]) not in usados:
                pegos.append(d)
                usados.add(emissor(por_doc[d]))
        for d in candidatos:
            if len(pegos) >= cota:
                break
            if d not in pegos:
                pegos.append(d)

        if len(pegos) < cota:
            print(f"  ⚠ {f}: só {len(pegos)} disponíveis para cota {cota}")
        escolhidos += pegos

    print(f"\nAMOSTRA ({len(escolhidos)} documentos):")
    for d in escolhidos:
        print(f"  {formas[d]:16s} {d[:40]:40s} {len(por_doc[d]):4d} tr  {emissor(por_doc[d])[:26]}")

    AMOSTRA.parent.mkdir(parents=True, exist_ok=True)
    AMOSTRA.write_text(json.dumps(
        {"semente": args.semente, "cotas": COTAS,
         "documentos": {d: formas[d] for d in escolhidos}},
        ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n✓ {AMOSTRA}")


# ── recorte por tamanho ──────────────────────────────────────────────────────

def cortar(texto: str, tamanho: int, sobreposicao: int) -> list[str]:
    if len(texto) <= tamanho:
        return [texto] if texto.strip() else []
    partes, inicio = [], 0
    while inicio < len(texto):
        fim = min(inicio + tamanho, len(texto))
        if fim < len(texto):
            for sep in SEPARADORES:
                achado = texto.rfind(sep, inicio + tamanho // 2, fim)
                if achado != -1:
                    fim = achado + len(sep)
                    break
        pedaco = texto[inicio:fim].strip()
        if pedaco:
            partes.append(pedaco)
        if fim >= len(texto):
            break
        inicio = max(fim - sobreposicao, inicio + 1)
    return partes


def reconstruir(trechos: list[dict], janela: int = 400) -> str:
    """Remonta o texto do documento a partir dos trechos, removendo a sobreposição.

    O `documents.jsonl` já vem recortado em 1.200 com sobreposição de 200, então
    concatenar os trechos duplicaria as bordas — e cada recorte novo herdaria a
    duplicação do antigo. Aqui a sobreposição é detectada e descartada: para cada
    trecho, procura-se o maior sufixo do acumulado que seja prefixo do próximo.

    Conferido contra o texto limpo de página (a fonte antes de qualquer recorte):
    similaridade 0,989 no Código de Ética, 0,999 no GDPR, 0,998 no FDA. O resíduo
    é sobreposição que a dedup não alcançou — irrelevante para medir tamanho.

    A vantagem de reconstruir em vez de reextrair: funciona para HTML também, que
    não tem página. Assim os 15 documentos da amostra entram, não só os PDFs.
    """
    trechos = sorted(trechos, key=lambda x: (x["metadata"].get("page") or 0, x["id"]))
    texto = trechos[0]["text"]
    for t in trechos[1:]:
        s = t["text"]
        corte = 0
        for n in range(min(janela, len(texto), len(s)), 20, -1):
            if texto[-n:] == s[:n]:
                corte = n
                break
        resto = s[corte:]
        if not resto:
            continue
        # Sem sobreposição detectada, colar direto gruda as pontas e inventa
        # palavra: "revoke such a decision." + "extending such a code" virava
        # "decision.extending". Separador explícito evita o artefato — que não
        # favorece condição nenhuma (todas partem deste mesmo texto), mas suja a
        # passagem de onde as perguntas serão escritas.
        if corte == 0 and not texto[-1:].isspace() and not resto[:1].isspace():
            texto += "\n"
        texto += resto
    return texto


def cmd_recortar(args) -> None:
    from chunk_id import chunk_id_por_conteudo

    if not AMOSTRA.exists():
        sys.exit("amostra não definida — rode `bench_tamanho.py amostra` antes.")
    amostra = json.loads(AMOSTRA.read_text(encoding="utf-8"))["documentos"]

    por_doc = carregar_corpus()
    paginas: dict[str, list[dict]] = {}
    for doc in amostra:
        if doc not in por_doc:
            print(f"⚠ {doc} não está no corpus — fora do recorte")
            continue
        # a "página" aqui é só o veículo do texto reconstruído; o recorte novo
        # não conhece página nenhuma.
        paginas[doc] = [{"page": 1, "text": reconstruir(por_doc[doc]),
                         "metadata": por_doc[doc][0]["metadata"]}]
    print(f"documentos reconstruídos: {len(paginas)} de {len(amostra)}\n")

    DESTINO.mkdir(parents=True, exist_ok=True)

    # ── parent-child ─────────────────────────────────────────────────────────
    # O arquivo guarda os FILHOS (é o que vai para o índice), cada um carregando
    # o texto do PAI em `metadata.pai_texto`. A avaliação busca filho, mapeia
    # para pai, deduplica e limita a PC_PAIS_MAX — é o pai que vai ao gerador,
    # e é nele que a evidência tem de estar.
    if args.parent_child:
        filhos = []
        for doc, pgs in sorted(paginas.items()):
            inteiro = "\n".join(p["text"] for p in sorted(pgs, key=lambda x: x["page"]))
            for pai in cortar(inteiro, PC_PAI, round(PC_PAI * 0.20)):
                pai_id = chunk_id_por_conteudo(doc, pai)
                for filho in cortar(pai, PC_FILHO, round(PC_FILHO * 0.20)):
                    filhos.append({
                        "id": chunk_id_por_conteudo(doc, filho),
                        "text": filho,
                        "metadata": {**pgs[0]["metadata"], "document_id": doc,
                                     "page": None, "forma": amostra[doc],
                                     "recorte": "PC", "pai_id": pai_id,
                                     "pai_texto": pai,
                                     "pais_max": PC_PAIS_MAX,
                                     "filhos_buscados": PC_FILHOS_BUSCADOS},
                    })
        # PC.jsonl continua sendo o pai de 3500, para nao invalidar o que ja foi
        # embutido. Variantes ganham sufixo proprio.
        caminho = DESTINO / ("PC.jsonl" if PC_PAI == 3500 else f"PC{PC_PAI}.jsonl")
        with caminho.open("w", encoding="utf-8") as f:
            for t in filhos:
                f.write(json.dumps(t, ensure_ascii=False) + "\n")
        pais = {t["metadata"]["pai_id"] for t in filhos}
        tf = [len(t["text"]) for t in filhos]
        print(f"  {'PC':6s} filho {PC_FILHO} / pai {PC_PAI} | {len(filhos):5d} filhos em "
              f"{len(pais)} pais | filho mediana {st.median(tf):.0f}")
        print(f"         busca top-{PC_FILHOS_BUSCADOS} filhos, entrega no máximo "
              f"{PC_PAIS_MAX} pais (~{PC_PAIS_MAX*PC_PAI} chars de contexto)")

    for tamanho in args.tamanhos:
        sobreposicao = round(tamanho * 0.20)
        trechos = []
        for doc, pgs in sorted(paginas.items()):
            inteiro = "\n".join(p["text"] for p in sorted(pgs, key=lambda x: x["page"]))
            for pedaco in cortar(inteiro, tamanho, sobreposicao):
                trechos.append({
                    "id": chunk_id_por_conteudo(doc, pedaco),
                    "text": pedaco,
                    "metadata": {**pgs[0]["metadata"], "document_id": doc,
                                 "page": None, "forma": amostra[doc],
                                 "recorte": f"T{tamanho}"},
                })
        nome = f"T{tamanho}"
        caminho = DESTINO / f"{nome}.jsonl"
        with caminho.open("w", encoding="utf-8") as f:
            for t in trechos:
                f.write(json.dumps(t, ensure_ascii=False) + "\n")
        tam = [len(t["text"]) for t in trechos]
        print(f"  {nome:6s} sobrep. {sobreposicao:4d} | {len(trechos):5d} trechos | "
              f"mediana {st.median(tam):6.0f} | min {min(tam):5d} | max {max(tam):5d}")

    print(f"\n✓ {DESTINO}")
    print("\nPróximo: embutir cada um (Colab, se houver GPU) e avaliar com")
    print("  python eval/tools/bench_chunking.py avaliar --nome T800 "
          "--corpus data/processed/chunking/T800.jsonl")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("amostra", help="escolhe a amostra estratificada")
    a.add_argument("--n", type=int, default=15, help="informativo; as cotas mandam")
    a.add_argument("--semente", type=int, default=42)
    a.set_defaults(func=cmd_amostra)

    b = sub.add_parser("recortar", help="gera um corpus por tamanho")
    b.add_argument("--tamanhos", type=int, nargs="+", default=list(TAMANHOS))
    b.add_argument("--sem-parent-child", dest="parent_child", action="store_false",
                   help="não gera a condição parent-child")
    b.set_defaults(func=cmd_recortar)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
