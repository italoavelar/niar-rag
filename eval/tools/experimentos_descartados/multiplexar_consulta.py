#!/usr/bin/env python3
"""Multiplexar a consulta: combinar pergunta + tradução EN + as duas sub-perguntas ajuda?

Contexto. O `Memoria.md` §1.1 registrava uma tabela animadora: combinando as
quatro formulações, a posição mediana da âncora **dentro do próprio documento**
caía de 8 para 3 (n=49 documentos exigidos pelas comparativas). E o custo é zero
— os dois caches já existem:

    eval/results/indexes/query_translations.json    (100 traduções PT→EN)
    eval/results_decomp/subqueries_cache.json       (50 pares de sub-perguntas)

Parecia ganho de graça. Este script mediu no que decide.

A ARMADILHA DA RÉGUA. "Posição dentro do documento" é grandeza CONDICIONAL: ela
pressupõe que já se está no documento certo. A pergunta, porém, exige as âncoras
no top-5 GLOBAL, onde elas competem com 4.897 trechos. Melhorar a ordem interna
não resolve a competição externa — e foi exatamente isso que aconteceu.

Por isso aqui medimos as duas formas que importam, sempre lado a lado:

    >=1       perguntas que trazem PELO MENOS UM dos trechos necessários
    AS DUAS   perguntas que trazem TODOS os trechos necessários

A segunda é a que decide. Metade da evidência não responde a pergunta: se o
multi-hop precisa do Art. 5 e do Art. 12 e a busca traz só o Art. 5, a resposta
sai incompleta. Foi confundir as duas que produziu o "multi-hop resolvida em 84%"
(era o >=1; pela régua estrita eram 3 de 25) — ver `Memoria.md` §1.1.

RESULTADO (16/set/2026) — refutado. Top-5, BGE-m3 denso sem reordenação:

    consulta                  multi_hop (>=1 · as duas)   comparative (>=1 · as duas)
    base — só a pergunta            14/25 ·  2/25              13/25 ·  2/25
    media(q,en)                     16/25 ·  2/25              13/25 ·  2/25
    media(q,s1,s2)                  10/25 ·  2/25              12/25 ·  2/25
    media(q,en,s1,s2)               12/25 ·  3/25              13/25 ·  2/25
    maxsim(q,en)                    14/25 ·  1/25               9/25 ·  0/25
    maxsim(q,en,s1,s2)              16/25 ·  1/25              12/25 ·  1/25
    RRF(q,en,s1,s2)                  9/25 ·  0/25              12/25 ·  2/25

Nenhuma variante move a comparativa. No multi-hop há TROCA, não ganho: ou sobe o
>=1 e o "as duas" fica parado, ou sobe o "as duas" em uma pergunta e o >=1 cai de
14 para 12. Três das seis fusões PIORAM — o maxsim(q,en) derruba a comparativa de
2/25 para zero.

Procedência: a tabela do §1.1 não tinha script no repositório. Varredura de
16/set não achou nenhum arquivo de multiplexação em `eval/`; os caches são usados
só pelo braço `A_bm25_mt` (BM25 com consulta traduzida — CLIR legítimo, outra
coisa). Esta é a primeira avaliação nas réguas que decidem.

Uso:
  python eval/tools/experimentos_descartados/multiplexar_consulta.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]   # .../eval/tools/experimentos_descartados/x.py
EVAL = PROJECT_ROOT / "eval"

for _fluxo in (sys.stdout, sys.stderr):
    try:
        _fluxo.reconfigure(encoding="utf-8")
    except Exception:
        pass

CORPUS = PROJECT_ROOT / "data/processed/documents.jsonl"
GOLDEN = EVAL / "data/golden_qa.jsonl"
TRADUCOES = EVAL / "results/indexes/query_translations.json"
SUBPERGUNTAS = EVAL / "results_decomp/subqueries_cache.json"
INDICES = EVAL / "results/indexes"

TIPOS = ("factual", "multi_hop", "comparative")


def indice_bge() -> tuple[np.ndarray, list[str]]:
    """O .npz mais recente do BGE-m3 com perfil context-v1."""
    cands = sorted(INDICES.glob("dense_bge_m3_context-v1_*.npz"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    if not cands:
        sys.exit(f"nenhum índice BGE em {INDICES} — rode eval/tools/reembutir_incremental.py")
    d = np.load(cands[0], allow_pickle=True)
    print(f"índice: {cands[0].name}  ({d['matrix'].shape[0]} trechos)")
    return d["matrix"].astype("float32"), list(d["ids"])


def main() -> None:
    M, ids = indice_bge()
    pos = {c: i for i, c in enumerate(ids)}

    gold = [json.loads(l) for l in GOLDEN.open(encoding="utf-8") if l.strip()]
    resp = [g for g in gold if g.get("qrels")]
    tipos = [g["question_type"] for g in resp]
    ancoras = [[c for c in g["qrels"] if c in pos] for g in resp]

    faltando = [c for g in resp for c in g["qrels"] if c not in pos]
    if faltando:
        sys.exit(f"✗ {len(faltando)} âncoras fora do índice — corpus e índice estão dessincronizados.")

    trad = json.loads(TRADUCOES.read_text(encoding="utf-8"))
    subq = json.loads(SUBPERGUNTAS.read_text(encoding="utf-8"))
    print(f"perguntas respondíveis: {len(resp)}")
    for t in TIPOS:
        idx = [i for i, x in enumerate(tipos) if x == t]
        en = sum(1 for i in idx if resp[i]["question"] in trad)
        sb = sum(1 for i in idx if resp[i]["question"] in subq)
        print(f"   {t:12s} n={len(idx):3d}  com tradução EN={en:3d}  com sub-perguntas={sb:3d}")

    from sentence_transformers import SentenceTransformer
    print("\ncarregando BGE-m3 (CPU)...", flush=True)
    mod = SentenceTransformer("BAAI/bge-m3", device="cpu")

    def emb(textos):
        return mod.encode(textos, normalize_embeddings=True, batch_size=8,
                          show_progress_bar=False).astype("float32")

    # Quando falta tradução ou sub-pergunta, cai de volta na pergunta original:
    # o objetivo é medir o GANHO da combinação, não punir a ausência de cache.
    Q = emb([g["question"] for g in resp])
    EN = emb([trad.get(g["question"], g["question"]) for g in resp])
    S1 = emb([(subq.get(g["question"]) or {}).get("sub1") or g["question"] for g in resp])
    S2 = emb([(subq.get(g["question"]) or {}).get("sub2") or g["question"] for g in resp])
    print("consultas embutidas", flush=True)

    def metricas(ordem):
        out = {}
        for t in TIPOS:
            idx = [i for i, x in enumerate(tipos) if x == t]
            um = todas = 0
            for i in idx:
                top5 = {ids[j] for j in ordem[i][:5]}
                a = ancoras[i]
                um += any(x in top5 for x in a)
                todas += all(x in top5 for x in a)
            out[t] = (um, todas, len(idx))
        return out

    def media(vs):
        V = sum(vs) / len(vs)
        return np.argsort(-((V / np.linalg.norm(V, axis=1, keepdims=True)) @ M.T), axis=1)[:, :200]

    def maxsim(vs):
        return np.argsort(-np.maximum.reduce([v @ M.T for v in vs]), axis=1)[:, :200]

    def rrf(vs, k=60):
        listas = [np.argsort(-(v @ M.T), axis=1)[:, :200] for v in vs]
        saida = np.zeros((len(resp), 200), dtype=int)
        for i in range(len(resp)):
            sc = defaultdict(float)
            for L in listas:
                for r, j in enumerate(L[i], 1):
                    sc[j] += 1.0 / (k + r)
            saida[i] = np.array(sorted(sc, key=lambda j: -sc[j])[:200])
        return saida

    cfgs = {
        "BASE — só a pergunta": np.argsort(-(Q @ M.T), axis=1)[:, :200],
        "média(q,en)": media([Q, EN]),
        "média(q,s1,s2)": media([Q, S1, S2]),
        "média(q,en,s1,s2)": media([Q, EN, S1, S2]),
        "maxsim(q,en)": maxsim([Q, EN]),
        "maxsim(q,en,s1,s2)": maxsim([Q, EN, S1, S2]),
        "RRF(q,en,s1,s2)": rrf([Q, EN, S1, S2]),
    }

    print("\n" + "=" * 96)
    print("MULTIPLEXAR A CONSULTA — top-5, as duas formas de contar")
    print("=" * 96)
    print(f"{'configuração':26s} | " + " | ".join(f"{t[:11]:^20s}" for t in TIPOS))
    print(f"{'':26s} | " + " | ".join(f"{'>=1':>9s}{'AS DUAS':>11s}" for _ in TIPOS))
    print("-" * 96)
    for nome, ordem in cfgs.items():
        m = metricas(ordem)
        cels = [f"{m[t][0]:5d}/{m[t][2]:<3d}{m[t][1]:7d}/{m[t][2]:<3d}" for t in TIPOS]
        print(f"{nome:26s} | " + " | ".join(cels))
    print("-" * 96)
    print(">=1     = perguntas que trazem PELO MENOS UM trecho necessário no top-5")
    print("AS DUAS = perguntas que trazem TODOS os trechos necessários — a régua que decide")


if __name__ == "__main__":
    main()
