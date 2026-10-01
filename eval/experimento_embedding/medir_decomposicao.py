#!/usr/bin/env python3
"""Mede a decomposição CEGA nos três tipos, contra a consulta única.

O SISTEMA QUE ESTÁ SENDO MEDIDO é o que daria para implantar: para cada
pergunta, o `decompor_tudo.py` decidiu sozinho — vendo só o enunciado — se ela
pede uma informação ou duas. Se disse DUAS, busca-se cada sub-pergunta, reordena-
se cada pool com o cross-encoder e monta-se o top-5 ALTERNANDO entre os dois. Se
disse UMA, usa-se a consulta única reranqueada, que é o sistema de hoje.

Nenhuma etapa consulta o tipo da pergunta nem o gabarito. É essa a diferença em
relação à medição de 29/09, em que a decomposição só rodava nas comparativas
porque alguém sabia que elas eram comparativas.

ORÇAMENTO IGUAL, SENÃO NÃO VALE. Cada sub-pergunta traz 50 candidatos, então o
sistema decomposto vê no máximo 85 trechos distintos contra os 100 da consulta
única — MENOS material, não mais. Sem essa trava, qualquer ganho poderia ser
apenas efeito de ter olhado mais corpus.

TRÊS MONTAGENS, porque reprovar a pior não reprova a ideia:
    score global   junta tudo e ordena por nota. É o controle: se a fusão não
                   fizer nada, o resultado tende ao da consulta única.
    reserva        garante a vaga do melhor de cada lado, depois preenche.
    intercalado    alterna estritamente. Foi a que funcionou em 29/09, porque é
                   a única que ignora a nota ENTRE os lados — e o mecanismo da
                   falha é justamente o segundo lado perder para o quinto trecho
                   do primeiro.

Uso:
  python eval/experimento_embedding/medir_decomposicao.py --pools     # BGE local
  python eval/experimento_embedding/medir_decomposicao.py --rerank    # CPU, retomável
  python eval/experimento_embedding/medir_decomposicao.py             # tabelas
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from math import comb
from pathlib import Path

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parents[1]
EVAL = RAIZ / "eval"
sys.path.insert(0, str(EVAL))
sys.path.insert(0, str(EVAL / "retrieval_eval"))
sys.path.insert(0, str(RAIZ / "src"))

GABARITO = EVAL / "data/golden_dezembro.jsonl"
DECOMP = EVAL / "results/indexes/decomposicao_cega.json"
POOLS = EVAL / "results/indexes/pools_decomposicao_cega.json"
NOTAS = EVAL / "results/indexes/rerank_scores_decomposicao_cega.json"
RANK_UNICO = EVAL / "results/retrieval/rankings/B_bge_m3_rerank.json"

K = 5
PROFUNDIDADE = 50
MAX_LENGTH = 384
RERANKER = "BAAI/bge-reranker-v2-m3"

for _f in (sys.stdout, sys.stderr):
    try:
        _f.reconfigure(encoding="utf-8")
    except Exception:
        pass


def mcnemar(v: int, d: int) -> float:
    n = v + d
    if n == 0:
        return 1.0
    k = min(v, d)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / (2 ** n))


def ordenar(cands, notas):
    return sorted(cands, key=lambda c: -notas.get(c, -1e9))


def score_global(o1, o2, n1, n2, k):
    juntos = {}
    for c in o1:
        juntos[c] = max(juntos.get(c, -1e9), n1.get(c, -1e9))
    for c in o2:
        juntos[c] = max(juntos.get(c, -1e9), n2.get(c, -1e9))
    return sorted(juntos, key=lambda c: -juntos[c])[:k]


def reserva(o1, o2, n1, n2, k):
    saida = []
    for lista in (ordenar(o1, n1), ordenar(o2, n2)):
        for c in lista:
            if c not in saida:
                saida.append(c)
                break
    resto = [c for c in score_global(o1, o2, n1, n2, 10 ** 6) if c not in saida]
    return (saida + resto)[:k]


def intercalado(o1, o2, n1, n2, k):
    a, b = ordenar(o1, n1), ordenar(o2, n2)
    saida, i, j = [], 0, 0
    while len(saida) < k and (i < len(a) or j < len(b)):
        for lista, idx in ((a, "i"), (b, "j")):
            pos = i if idx == "i" else j
            while pos < len(lista) and lista[pos] in saida:
                pos += 1
            if pos < len(lista) and len(saida) < k:
                saida.append(lista[pos])
                pos += 1
            if idx == "i":
                i = pos
            else:
                j = pos
    return saida[:k]


ESTRATEGIAS = {"score global": score_global, "reserva": reserva,
               "intercalado": intercalado}


def carregar(p: Path, padrao):
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    return padrao


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pools", action="store_true",
                    help="busca BGE das sub-perguntas (local, sem API)")
    ap.add_argument("--rerank", action="store_true",
                    help="pontua os pares com o cross-encoder (CPU, retomável)")
    ap.add_argument("--profundidade", type=int, default=PROFUNDIDADE)
    args = ap.parse_args()

    gold = {g["qid"]: g for g in
            (json.loads(l) for l in GABARITO.open(encoding="utf-8") if l.strip())}
    decomp = carregar(DECOMP, {})
    if not decomp:
        print(f"✗ falta {DECOMP.name} — rode decompor_tudo.py --aplicar primeiro.")
        return
    unico = carregar(RANK_UNICO, {})

    alvo = [q for q, g in gold.items()
            if g.get("qrels_grupos") and g["question"] in decomp]
    duas = [q for q in alvo if decomp[gold[q]["question"]]["partes"] == 2]
    print(f"respondíveis com decomposição: {len(alvo)}")
    print(f"  que o modelo dividiu em duas: {len(duas)}")
    print(f"  que ele julgou atômicas     : {len(alvo) - len(duas)}\n")

    # ── fase 1: pools das sub-perguntas ─────────────────────────────────────
    pools = carregar(POOLS, {})
    if args.pools:
        pendentes = [q for q in duas if q not in pools]
        print(f"pools a buscar: {len(pendentes)}")
        if pendentes:
            from lib.common import load_config, load_corpus, resolve
            from _common import dense_ranking
            cfg = load_config(str(EVAL / "config.yaml"))
            corpus = load_corpus(resolve(cfg["paths"]["corpus"]))
            queries = []
            for q in pendentes:
                d = decomp[gold[q]["question"]]
                queries.append((f"{q}#sub1", d["sub1"]))
                queries.append((f"{q}#sub2", d["sub2"]))
            t0 = time.time()
            r = dense_ranking(cfg, "bge_m3", corpus, queries)
            print(f"  {len(r)} buscas em {time.time()-t0:.0f}s")
            for q in pendentes:
                a, b = r.get(f"{q}#sub1"), r.get(f"{q}#sub2")
                if a and b:
                    pools[q] = {"sub1": a, "sub2": b}
            POOLS.write_text(json.dumps(pools, ensure_ascii=False),
                             encoding="utf-8")
            print(f"  ✓ {POOLS.name} — {len(pools)} perguntas")
        return

    # ── fase 2: notas do cross-encoder ──────────────────────────────────────
    notas = carregar(NOTAS, {"modelo": RERANKER, "max_length": MAX_LENGTH,
                             "profundidade": args.profundidade, "scores": {}})
    if args.rerank:
        if not pools:
            print(f"✗ falta {POOLS.name} — rode --pools primeiro.")
            return
        from lib.common import load_config, load_corpus, resolve
        from embedding_text import build_embedding_text
        cfg = load_config(str(EVAL / "config.yaml"))
        corpus = load_corpus(resolve(cfg["paths"]["corpus"]))

        pares, indice = [], []
        for q in duas:
            if q not in pools:
                continue
            for lado in ("sub1", "sub2"):
                chave = f"{q}#{lado}"
                if chave in notas["scores"]:
                    continue
                d = decomp[gold[q]["question"]]
                for cid in pools[q][lado][:args.profundidade]:
                    pares.append((d[lado], build_embedding_text(corpus[cid])))
                    indice.append((chave, cid))
        print(f"pares a pontuar: {len(pares)}")
        if not pares:
            print("  nada pendente.")
            return
        from sentence_transformers import CrossEncoder
        print(f"carregando {RERANKER} em CPU...")
        ce = CrossEncoder(RERANKER, max_length=MAX_LENGTH, device="cpu",
                          trust_remote_code=True)
        t0 = time.time()
        # Em blocos, gravando a cada um: o lote inteiro leva horas em CPU e
        # perder tudo por uma interrupção seria besteira.
        BLOCO = 2000
        for i in range(0, len(pares), BLOCO):
            fatia = pares[i:i + BLOCO]
            idx = indice[i:i + BLOCO]
            v = ce.predict(fatia, batch_size=16, show_progress_bar=True)
            for (chave, cid), nota in zip(idx, v):
                notas["scores"].setdefault(chave, {})[cid] = float(nota)
            NOTAS.write_text(json.dumps(notas, ensure_ascii=False),
                             encoding="utf-8")
            feito = min(i + BLOCO, len(pares))
            dt = time.time() - t0
            print(f"  {feito}/{len(pares)} pares · {dt/60:.1f} min · "
                  f"{dt/feito:.3f}s por par · resta "
                  f"{(len(pares)-feito)*dt/feito/60:.0f} min")
        print(f"  ✓ {NOTAS.name}")
        return

    # ── fase 3: medir ───────────────────────────────────────────────────────
    if not notas["scores"]:
        print(f"✗ falta {NOTAS.name} — rode --pools e depois --rerank.")
        return

    def acerto(ordem, q) -> int:
        t = set(ordem[:K])
        return int(all(any(c in t for c in gr) for gr in gold[q]["qrels_grupos"]))

    prontas = [q for q in duas
               if q in pools and f"{q}#sub1" in notas["scores"]
               and f"{q}#sub2" in notas["scores"]]
    print(f"com notas prontas: {len(prontas)} das {len(duas)} divididas\n")

    def montar(q, fn):
        d = pools[q]
        o1 = d["sub1"][:args.profundidade]
        o2 = d["sub2"][:args.profundidade]
        n1 = notas["scores"][f"{q}#sub1"]
        n2 = notas["scores"][f"{q}#sub2"]
        return fn(o1, o2, n1, n2, K)

    # o sistema sempre-ligado: decompõe onde o modelo disse 2, senão único
    def sistema(q, fn):
        if q in prontas:
            return montar(q, fn)
        return unico.get(q, [])

    base_total = sum(acerto(unico.get(q, []), q) for q in alvo)
    print(f"LINHA DE BASE — consulta única reranqueada: "
          f"{base_total} de {len(alvo)}  ({base_total/len(alvo):.3f})\n")

    print(f"{'montagem':16s}{'acertos':>9s}{'SR@5':>8s}{'vit':>6s}{'der':>6s}"
          f"{'p':>9s}")
    print("─" * 54)
    for nome, fn in ESTRATEGIAS.items():
        tot = v = d = 0
        for q in alvo:
            a = acerto(sistema(q, fn), q)
            b = acerto(unico.get(q, []), q)
            tot += a
            v += int(b and not a)
            d += int(a and not b)
        print(f"{nome:16s}{tot:9d}{tot/len(alvo):8.3f}{v:6d}{d:6d}"
              f"{mcnemar(v, d):9.4f}")
    print("\n  vit = a única acerta e a decomposta não; der = o inverso.")
    print(f"  Orçamento: {args.profundidade}+{args.profundidade} candidatos "
          f"contra 100 da única.")

    # ── por tipo, só a intercalada ──────────────────────────────────────────
    print("\n\nINTERCALADA, POR TIPO\n")
    cab = (f"{'tipo':16s}{'n':>5s}{'dividida':>10s}{'única':>8s}"
           f"{'decomp':>9s}{'ganho':>8s}{'disc.':>9s}{'p':>9s}")
    print(cab)
    print("─" * len(cab))
    for t in ("factual", "multi_hop", "comparative"):
        qs = [q for q in alvo if gold[q]["question_type"] == t]
        if not qs:
            continue
        nd = sum(1 for q in qs if q in prontas)
        b = sum(acerto(unico.get(q, []), q) for q in qs)
        a = sum(acerto(sistema(q, intercalado), q) for q in qs)
        v = sum(1 for q in qs if acerto(unico.get(q, []), q)
                and not acerto(sistema(q, intercalado), q))
        d = sum(1 for q in qs if acerto(sistema(q, intercalado), q)
                and not acerto(unico.get(q, []), q))
        print(f"{t:16s}{len(qs):5d}{nd:10d}{b:8d}{a:9d}{a-b:+8d}"
              f"{f'{v} a {d}':>9s}{mcnemar(v, d):9.4f}")
    print("\n  'dividida' = quantas o modelo decidiu partir em duas.")

    # ── só nas que o modelo dividiu, para separar decisão de execução ───────
    print("\n\nSÓ NAS QUE O MODELO DIVIDIU — a execução funciona?\n")
    print(f"{'montagem':16s}{'n':>5s}{'única':>8s}{'decomp':>9s}{'ganho':>8s}"
          f"{'disc.':>9s}{'p':>9s}")
    print("─" * 55)
    for nome, fn in ESTRATEGIAS.items():
        b = sum(acerto(unico.get(q, []), q) for q in prontas)
        a = sum(acerto(montar(q, fn), q) for q in prontas)
        v = sum(1 for q in prontas if acerto(unico.get(q, []), q)
                and not acerto(montar(q, fn), q))
        d = sum(1 for q in prontas if acerto(montar(q, fn), q)
                and not acerto(unico.get(q, []), q))
        print(f"{nome:16s}{len(prontas):5d}{b:8d}{a:9d}{a-b:+8d}"
              f"{f'{v} a {d}':>9s}{mcnemar(v, d):9.4f}")
    print("\n  Esta tabela separa as duas perguntas: a de cima mistura o acerto")
    print("  da DECISÃO de dividir com o da EXECUÇÃO; esta isola a execução.")


if __name__ == "__main__":
    main()
