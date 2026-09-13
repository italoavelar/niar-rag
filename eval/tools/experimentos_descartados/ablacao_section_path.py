#!/usr/bin/env python3
"""Ablação: o `section_path` no prefixo de embedding ajuda a recuperar?

Contexto. `src/embedding_text.py:20-58` embute cada trecho prefixado por
`[título · emissor · seção]`. Título e emissor são IDÊNTICOS para todos os
trechos irmãos de um documento, então não discriminam nada dentro dele — a
seção é a única parte do prefixo que varia entre irmãos. Hoje ela está
preenchida em apenas 608 dos 5.160 trechos (12%), porque o extrator só a
atribui quando o trecho COMEÇA com o marcador ("Art. 3º"); a continuação do
mesmo artigo fica vazia.

Antes de investir em propagar a seção para os trechos seguintes — o que traz
risco real de marcar errado —, esta ablação mede se ela ajuda de fato.

Desenho: pareado, por trecho. Reembute os 608 trechos que têm seção, agora SEM
ela no prefixo, substitui na matriz e compara a posição da âncora do gabarito
antes e depois. Todos os 608 são reembutidos (não só as âncoras) porque eles
competem entre si no ranking.

Limitação declarada: só 12 âncoras do gabarito têm seção preenchida. Com n=12 o
teste enxerga efeito grande; ausência de sinal aqui significa "não há efeito
grande", nunca "não há efeito".

Uso:
  python eval/tools/ablacao_section_path.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]   # .../eval/tools/experimentos_descartados/x.py
EVAL = PROJECT_ROOT / "eval"
sys.path.insert(0, str(EVAL))
sys.path.insert(0, str(EVAL / "retrieval_eval"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

for _f in (sys.stdout, sys.stderr):
    try:
        _f.reconfigure(encoding="utf-8")
    except Exception:
        pass

from _common import load_config, load_corpus, resolve, setup_io   # noqa: E402
from embedding_text import (                                       # noqa: E402
    corpus_embedding_fingerprint, bge_cache_filename, build_embedding_text,
)
from lib.metrics import document_of                                # noqa: E402


def main() -> None:
    cfg = load_config()
    setup_io()
    t0 = time.time()

    corpus = load_corpus(resolve(cfg["paths"]["corpus"]))
    fp = corpus_embedding_fingerprint(corpus.values())
    npz = np.load(EVAL / "results/indexes" / bge_cache_filename("bge_m3", fp), allow_pickle=True)
    matriz = npz["matrix"].astype(np.float32)
    ids = [str(i) for i in npz["ids"]]
    pos = {c: i for i, c in enumerate(ids)}
    print(f"[{time.time()-t0:5.1f}s] corpus e matriz ({len(ids)} trechos)")

    com_secao = [c for c in ids
                 if (corpus[c]["metadata"].get("section_path") or "").strip()]
    print(f"trechos com section_path: {len(com_secao)}")

    gold = [json.loads(l) for l in open(EVAL / "data/golden_qa.jsonl", encoding="utf-8") if l.strip()]
    alvos = []   # (qid, pergunta, chunk_ancora)
    for r in gold:
        for cid, grau in (r.get("qrels") or {}).items():
            if grau == 2 and cid in corpus and (corpus[cid]["metadata"].get("section_path") or "").strip():
                alvos.append((r["qid"], r["question"], cid, r["question_type"]))
    print(f"âncoras do gabarito com seção preenchida: {len(alvos)}")
    if not alvos:
        print("nada a medir."); return

    from lib.embedders import build_embedder
    emb = build_embedder("bge_m3", cfg)
    print(f"[{time.time()-t0:5.1f}s] BGE-m3 carregado")

    # ── reembute os 608 SEM a seção ─────────────────────────────────────────
    textos_sem = []
    for cid in com_secao:
        reg = corpus[cid]
        copia = {"text": reg["text"],
                 "metadata": {k: v for k, v in reg["metadata"].items() if k != "section_path"}}
        textos_sem.append(build_embedding_text(copia))
    print(f"reembutindo {len(textos_sem)} trechos sem a seção...")
    vetores_sem = np.vstack([emb.embed_query(t).astype(np.float32) for t in textos_sem])
    print(f"[{time.time()-t0:5.1f}s] reembedding pronto")

    matriz_sem = matriz.copy()
    for cid, v in zip(com_secao, vetores_sem):
        matriz_sem[pos[cid]] = v

    # ── compara posição da âncora, com e sem ────────────────────────────────
    def rank_de(cid, qvec, M, restrito_ao_doc=False):
        sc = M @ qvec
        if restrito_ao_doc:
            doc = document_of(cid)
            idx = [i for i, c in enumerate(ids) if document_of(c) == doc]
            ordem = sorted(idx, key=lambda i: -sc[i])
            return ordem.index(pos[cid]) + 1
        return int((sc > sc[pos[cid]]).sum()) + 1

    print(f"\n{'qid':<8}{'tipo':<13}{'seção':<24}{'global c/':>10}{'global s/':>10}{'no doc c/':>11}{'no doc s/':>11}")
    linhas = []
    for qid, pergunta, cid, tipo in alvos:
        qv = emb.embed_query(pergunta).astype(np.float32)
        g_com = rank_de(cid, qv, matriz)
        g_sem = rank_de(cid, qv, matriz_sem)
        d_com = rank_de(cid, qv, matriz, restrito_ao_doc=True)
        d_sem = rank_de(cid, qv, matriz_sem, restrito_ao_doc=True)
        sec = (corpus[cid]["metadata"].get("section_path") or "")[:22]
        print(f"{qid:<8}{tipo:<13}{sec:<24}{g_com:>10}{g_sem:>10}{d_com:>11}{d_sem:>11}")
        linhas.append((qid, tipo, g_com, g_sem, d_com, d_sem))

    g_com = np.array([l[2] for l in linhas]); g_sem = np.array([l[3] for l in linhas])
    d_com = np.array([l[4] for l in linhas]); d_sem = np.array([l[5] for l in linhas])
    print(f"\n{'':<12}{'com seção':>12}{'sem seção':>12}{'melhora?':>12}")
    print(f"{'global med.':<12}{np.median(g_com):>12.0f}{np.median(g_sem):>12.0f}"
          f"{('sim' if np.median(g_com) < np.median(g_sem) else 'não'):>12}")
    print(f"{'no doc med.':<12}{np.median(d_com):>12.0f}{np.median(d_sem):>12.0f}"
          f"{('sim' if np.median(d_com) < np.median(d_sem) else 'não'):>12}")
    melhor = int((g_com < g_sem).sum()); pior = int((g_com > g_sem).sum())
    print(f"\nglobal: a seção melhora em {melhor}, piora em {pior}, empata em {len(linhas)-melhor-pior} (n={len(linhas)})")
    melhor_d = int((d_com < d_sem).sum()); pior_d = int((d_com > d_sem).sum())
    print(f"no doc: a seção melhora em {melhor_d}, piora em {pior_d}, empata em {len(linhas)-melhor_d-pior_d}")
    print("\n⚠ n pequeno: este teste só enxerga efeito grande. Ausência de sinal aqui")
    print("  significa 'não há efeito grande', não 'não há efeito'.")
    print(f"\ntempo total: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
