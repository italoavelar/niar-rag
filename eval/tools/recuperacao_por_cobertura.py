#!/usr/bin/env python3
"""Recuperação orientada à cobertura de evidências: união, sem fusão e sem reordenação.

DECOMPOSIÇÃO DE **NECESSIDADES DE EVIDÊNCIA**, não de pergunta.

A distinção não é vocabulário — ela determina o desenho, e é o que separa este
experimento do `decompose_eval.py`, que falhou:

    decomposição de PERGUNTA    trata as partes como sub-perguntas a responder.
                                Faz sentido, então, fundir as respostas e
                                repontuar tudo contra a pergunta INTEIRA — que é
                                exatamente o que o decompose_eval.py faz, e o que
                                desfaz a decomposição: a competição volta a ser
                                global e volta a ser ganha pelo lado dominante.

    decomposição de NECESSIDADE aqui não se tenta resolver o raciocínio da
    DE EVIDÊNCIA                pergunta. Só se afirma: "para responder isto o
                                recuperador precisa achar a evidência A E a
                                evidência B". Cada necessidade é buscada em seus
                                próprios termos, e o resultado é a UNIÃO. Nada é
                                repontuado contra a pergunta inteira, porque a
                                pergunta inteira não é o alvo da busca — as
                                evidências são.

A HIPÓTESE. Perguntas que exigem várias evidências (comparação, "X e Y",
relação entre normas) são mal servidas por uma consulta única: o vetor da
pergunta fica ENTRE os dois assuntos e perto de nenhum.

    Q: "Compare as regras de X e Y"
      → necessidade 1: evidências sobre X   → retrieval(n1)
      → necessidade 2: evidências sobre Y   → retrieval(n2)
      → união, deduplicada

O QUE ISTO TEM DE DIFERENTE do que já foi tentado e descartado:

    particao_por_lado.py      reparte um orçamento FIXO de 5 vagas (3+2). O total
                              não cresce; ganha-se cobertura de um lado tirando
                              vaga do outro. Piorou.
    decompose_eval.py         decompõe mas depois FUNDE por RRF e repontua tudo
                              contra a pergunta original — o que desfaz a
                              decomposição. Recall subiu, top-5 não mudou.
    ESTE                      a união NÃO tem teto. O contexto cresce, e é por
                              isso que o tamanho do contexto é medida de
                              primeira classe aqui, não detalhe.

O CONTROLE QUE DECIDE. Uma união de top-5 + top-5 pode chegar a 10 trechos.
Comparar isso com o top-5 da consulta única seria trapaça. Então o baseline é
reportado no MESMO tamanho de conjunto: se a união entrega 10 trechos, a
comparação honesta é contra o top-10 da consulta única.

    Se união(top5(Q1), top5(Q2)) ~= top10(Q), a decomposição não acrescenta
    nada — bastava pedir mais trechos à consulta original.

MÉTRICA PRINCIPAL: **Joint Recall@k** — proporção de perguntas para as quais
TODOS os trechos necessários foram recuperados. É o `complete_evidence_at_k` de
`eval/lib/metrics.py:142`, com o nome que a literatura usa. Reportado ao lado do
Recall (fração das âncoras), do MRR, do tamanho do contexto e da latência.

O DETECTOR. Esta etapa NÃO testa o detector de "pergunta multi-evidência": usa o
rótulo `question_type` do gabarito como detector-oráculo. Assim mede-se a
hipótese de recuperação isoladamente. Um detector real é questão à parte, e o
resultado aqui é o TETO do que ele poderia entregar.

SEM LLM NOVO. As necessidades de evidência são lidas do cache
`eval/results_decomp/subqueries_cache.json` (50 pares, gerados em 09/set para o
experimento de decomposição de pergunta). O cache é o mesmo; o que muda é o que
se faz com ele. Chunking, embedding e ausência de reranker são exatamente os do
baseline.

⚠️ Em produção, formular as necessidades exige um decompositor — hoje, um LLM.
Rodar ESTE script não gasta API porque o cache existe, mas o método não é
livre de LLM. Um decompositor sem LLM (por padrão sintático, por exemplo) é
questão separada e não testada.

Uso:
  python eval/tools/recuperacao_por_cobertura.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVAL = PROJECT_ROOT / "eval"
sys.path.insert(0, str(EVAL))

for _fluxo in (sys.stdout, sys.stderr):
    try:
        _fluxo.reconfigure(encoding="utf-8")
    except Exception:
        pass

CORPUS = PROJECT_ROOT / "data/processed/documents.jsonl"
GOLDEN = EVAL / "data/golden_qa.jsonl"
NECESSIDADES = EVAL / "results_decomp/subqueries_cache.json"
INDICES = EVAL / "results/indexes"

MULTI = ("multi_hop", "comparative")


def carregar():
    cands = sorted(INDICES.glob("dense_bge_m3_context-v1_*.npz"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    if not cands:
        sys.exit(f"nenhum índice BGE em {INDICES}")
    d = np.load(cands[0], allow_pickle=True)
    print(f"índice: {cands[0].name}  ({d['matrix'].shape[0]} trechos)")
    M, ids = d["matrix"].astype("float32"), list(d["ids"])

    corpus = {r["id"]: r for r in
              (json.loads(l) for l in CORPUS.open(encoding="utf-8") if l.strip())}
    gold = [json.loads(l) for l in GOLDEN.open(encoding="utf-8") if l.strip()]
    resp = [g for g in gold if g.get("qrels")]
    pos = {c: i for i, c in enumerate(ids)}
    fora = [c for g in resp for c in g["qrels"] if c not in pos]
    if fora:
        sys.exit(f"✗ {len(fora)} âncoras fora do índice — corpus e índice dessincronizados.")
    return M, ids, corpus, resp


def main() -> None:
    M, ids, corpus, resp = carregar()
    # chaves "sub1"/"sub2" são o formato do cache; aqui elas são NECESSIDADES.
    necessidades = json.loads(NECESSIDADES.read_text(encoding="utf-8"))

    alvo = [g for g in resp if g["question_type"] in MULTI]
    com_nec = [g for g in alvo if g["question"] in necessidades]
    print(f"perguntas multi-evidência no gabarito : {len(alvo)}")
    print(f"   com necessidades em cache           : {len(com_nec)}")
    if len(com_nec) < len(alvo):
        print(f"   ⚠ {len(alvo)-len(com_nec)} sem cache ficam de fora da comparação")

    ancoras = {g["qid"]: [c for c in g["qrels"]] for g in com_nec}
    tipos = {g["qid"]: g["question_type"] for g in com_nec}

    from sentence_transformers import SentenceTransformer
    print("\ncarregando BGE-m3 (CPU)...", flush=True)
    mod = SentenceTransformer("BAAI/bge-m3", device="cpu")

    def emb(textos):
        return mod.encode(textos, normalize_embeddings=True, batch_size=8,
                          show_progress_bar=False).astype("float32")

    # ── latência: medida na mesma máquina, mesmo lote ────────────────────────
    t0 = time.perf_counter()
    Q = emb([g["question"] for g in com_nec])
    SQ = Q @ M.T
    t_unica = time.perf_counter() - t0

    t0 = time.perf_counter()
    N1v = emb([necessidades[g["question"]]["sub1"] for g in com_nec])
    N2v = emb([necessidades[g["question"]]["sub2"] for g in com_nec])
    S1, S2 = N1v @ M.T, N2v @ M.T
    t_decomp = time.perf_counter() - t0

    n = len(com_nec)
    print(f"\nlatência de recuperação (n={n}, CPU, necessidades já em cache):")
    print(f"   consulta única      : {t_unica*1000/n:6.1f} ms/pergunta")
    print(f"   duas necessidades   : {t_decomp*1000/n:6.1f} ms/pergunta   "
          f"({t_decomp/t_unica:.1f}x)")

    ord_q = np.argsort(-SQ, axis=1)[:, :60]
    ord_1 = np.argsort(-S1, axis=1)[:, :60]
    ord_2 = np.argsort(-S2, axis=1)[:, :60]

    def avaliar(conjuntos, ordenados):
        """conjuntos: lista de set(chunk_id) por pergunta (ordem irrelevante).
        ordenados: lista de listas ordenadas, para MRR."""
        linhas = {}
        for t in MULTI + ("TOTAL",):
            idxs = [i for i, g in enumerate(com_nec)
                    if t == "TOTAL" or tipos[g["qid"]] == t]
            joint = rec = mrr = tam = chars = 0
            for i in idxs:
                a = ancoras[com_nec[i]["qid"]]
                s = conjuntos[i]
                joint += all(x in s for x in a)
                rec += sum(1 for x in a if x in s) / len(a)
                for r, c in enumerate(ordenados[i], 1):
                    if c in a:
                        mrr += 1 / r
                        break
                tam += len(s)
                chars += sum(len(corpus[c]["text"]) for c in s)
            k = len(idxs)
            linhas[t] = (joint, k, rec / k, mrr / k, tam / k, chars / k)
        return linhas

    def conj_unica(k):
        cs = [{ids[j] for j in ord_q[i][:k]} for i in range(n)]
        os_ = [[ids[j] for j in ord_q[i][:k]] for i in range(n)]
        return cs, os_

    def conj_uniao(m):
        cs, os_ = [], []
        for i in range(n):
            a = [ids[j] for j in ord_1[i][:m]]
            b = [ids[j] for j in ord_2[i][:m]]
            vistos, inter = set(), []
            for c in [x for par in zip(a, b) for x in par]:   # intercala as duas necessidades
                if c not in vistos:
                    vistos.add(c); inter.append(c)
            cs.append(vistos); os_.append(inter)
        return cs, os_

    cfgs = []
    for k in (5, 10, 16, 20):
        cfgs.append((f"consulta única  top-{k}", *conj_unica(k)))
    for m in (3, 5, 8, 10):
        cfgs.append((f"UNIÃO das 2 necessidades top-{m}", *conj_uniao(m)))

    print("\n" + "=" * 104)
    print("RECUPERAÇÃO ORIENTADA À COBERTURA — união sem teto, sem reordenação")
    print("=" * 104)
    cab = f"{'configuração':32s} | {'trechos':>8}{'chars':>8} | "
    cab += " | ".join(f"{t[:11]:^22s}" for t in ("multi_hop", "comparative", "TOTAL"))
    print(cab)
    print(f"{'':32s} | {'no ctx':>8}{'no ctx':>8} | " +
          " | ".join(f"{'JointR':>8s}{'Recall':>7s}{'MRR':>7s}" for _ in range(3)))
    print("-" * 104)
    for nome, cs, os_ in cfgs:
        L = avaliar(cs, os_)
        tot = L["TOTAL"]
        cels = []
        for t in ("multi_hop", "comparative", "TOTAL"):
            j, k, rec, mrr, _, _ = L[t]
            cels.append(f"{j:4d}/{k:<3d}{rec:7.1%}{mrr:7.3f}")
        print(f"{nome:32s} | {tot[4]:8.1f}{tot[5]:8.0f} | " + " | ".join(cels))
    print("-" * 104)
    print("JointR = Joint Recall@k: perguntas com TODOS os trechos necessários no conjunto final")
    print("Recall = fração média das âncoras recuperadas   |   chars = tamanho do contexto")
    print("\nA comparação honesta é entre linhas de MESMO tamanho de contexto.")


if __name__ == "__main__":
    main()
