#!/usr/bin/env python3
"""Recalcula só os vetores dos trechos que mudaram, reaproveitando o resto.

Por que existe
──────────────
Reextrair o corpus muda o `corpus_embedding_fingerprint`, e o nome do cache
`.npz` inclui esse fingerprint — então todo o cache "vence" de uma vez. Mas
reembutir 4.897 trechos em CPU custa horas, e na prática quase nada mudou: o id
vem do conteúdo, então trecho cujo texto não mudou mantém id E vetor.

Este script faz a conta: reaproveita o que dá, calcula só o resto.

O critério de reaproveitamento é estrito — não basta o id existir no cache
antigo, o **texto embutido** (`build_embedding_text`, com o prefixo de contexto)
precisa ser idêntico. Se alguém preencher `section_path` depois, o texto embutido
muda sem o id mudar, e reaproveitar o vetor daria um índice silenciosamente
errado. Por isso a conferência é feita contra o corpus de onde o cache saiu.

Uso
───
  python eval/tools/reembutir_incremental.py --modelo bge_m3
  python eval/tools/reembutir_incremental.py --modelo bge_m3 --aplicar
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
EVAL = PROJECT_ROOT / "eval"
for _p in (EVAL, EVAL / "retrieval_eval", PROJECT_ROOT / "src"):
    sys.path.insert(0, str(_p))

for _fluxo in (sys.stdout, sys.stderr):
    try:
        _fluxo.reconfigure(encoding="utf-8")
    except Exception:
        pass

from embedding_text import (  # noqa: E402
    build_embedding_text, corpus_embedding_fingerprint, bge_cache_filename,
)

INDEXES = EVAL / "results/indexes"


def carregar_corpus(caminho: Path) -> list[dict]:
    with caminho.open(encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description="Reembute só o que mudou.")
    ap.add_argument("--modelo", default="bge_m3", help="chave do embedder (eval/config.yaml)")
    ap.add_argument("--aplicar", action="store_true", help="grava o cache novo")
    ap.add_argument("--corpus-antigo", type=Path, default=None,
                    help="corpus de onde o cache antigo saiu (padrão: o backup mais recente)")
    args = ap.parse_args()
    t0 = time.time()

    corpus = carregar_corpus(PROJECT_ROOT / "data/processed/documents.jsonl")
    fp_novo = corpus_embedding_fingerprint(corpus)
    destino = INDEXES / bge_cache_filename(args.modelo, fp_novo)
    print("── Reembutimento incremental ──")
    print(f"modelo: {args.modelo}   corpus: {len(corpus)} trechos")
    print(f"fingerprint novo: {fp_novo[:16]}…")
    if destino.exists():
        print(f"\n✓ {destino.name} já existe — nada a fazer.")
        return

    antigos = sorted(INDEXES.glob(f"dense_{args.modelo}_context-*.npz"))
    if not antigos:
        raise SystemExit(f"Nenhum cache anterior de {args.modelo}; rode o pipeline completo.")
    origem = antigos[-1]

    # De onde o cache antigo saiu, para conferir o texto embutido.
    if args.corpus_antigo:
        caminho_antigo = args.corpus_antigo
    else:
        backups = sorted(PROJECT_ROOT.glob("data/processed/documents.jsonl.bak-*"))
        if not backups:
            raise SystemExit("Sem backup do corpus antigo; passe --corpus-antigo.")
        caminho_antigo = backups[-1]
    texto_antigo = {r["id"]: build_embedding_text(r)
                    for r in carregar_corpus(caminho_antigo)}
    print(f"cache anterior : {origem.name}")
    print(f"corpus anterior: {caminho_antigo.name}")

    with np.load(origem, allow_pickle=True) as z:
        matriz_antiga = z["matrix"]
        ids_antigos = [str(i) for i in z["ids"]]
    pos_antiga = {c: i for i, c in enumerate(ids_antigos)}

    reaproveita, calcular = [], []
    for reg in corpus:
        cid = reg["id"]
        # Estrito: id conhecido E texto embutido idêntico ao que gerou o vetor.
        if cid in pos_antiga and texto_antigo.get(cid) == build_embedding_text(reg):
            reaproveita.append(cid)
        else:
            calcular.append(reg)
    print(f"\n  vetores reaproveitados : {len(reaproveita)}")
    print(f"  a calcular             : {len(calcular)}")
    if not args.aplicar:
        print(f"\n(simulação — rode com --aplicar para gravar {destino.name})")
        return

    matriz = np.zeros((len(corpus), matriz_antiga.shape[1]), dtype=matriz_antiga.dtype)
    indice = {r["id"]: i for i, r in enumerate(corpus)}
    for cid in reaproveita:
        matriz[indice[cid]] = matriz_antiga[pos_antiga[cid]]

    if calcular:
        from _common import load_config
        from lib.embedders import build_embedder
        emb = build_embedder(args.modelo, load_config())
        print(f"[{time.time()-t0:5.1f}s] {args.modelo} carregado; embutindo {len(calcular)}…")
        for k, reg in enumerate(calcular, 1):
            matriz[indice[reg["id"]]] = emb.embed_query(
                build_embedding_text(reg)).astype(matriz.dtype)
            if k % 50 == 0:
                print(f"     {k}/{len(calcular)}  [{time.time()-t0:5.1f}s]")

    np.savez_compressed(
        destino,
        matrix=matriz,
        ids=np.array([r["id"] for r in corpus], dtype=object),
        embedding_text_fingerprint=np.array(fp_novo),
    )
    print(f"\n✓ {destino.name}  ({matriz.shape[0]} vetores, dim {matriz.shape[1]})")

    # ── verificação: os reaproveitados são bit a bit os mesmos? ──────────────
    with np.load(destino, allow_pickle=True) as z:
        conf = z["matrix"]
        ids_conf = [str(i) for i in z["ids"]]
    pos_nova = {c: i for i, c in enumerate(ids_conf)}
    iguais = sum(1 for c in reaproveita
                 if np.array_equal(conf[pos_nova[c]], matriz_antiga[pos_antiga[c]]))
    normas = np.linalg.norm(conf, axis=1)
    print(f"verificação: {iguais}/{len(reaproveita)} reaproveitados idênticos ao cache antigo")
    print(f"             norma dos vetores: min {normas.min():.4f}  max {normas.max():.4f}")
    print(f"             vetores zerados (falha): {int((normas == 0).sum())}")
    print(f"\ntempo: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
