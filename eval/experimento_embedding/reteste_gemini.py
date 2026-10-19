#!/usr/bin/env python3
"""O braço Gemini, reconferido: ele repete? e a busca aproximada custa quanto?

DUAS PERGUNTAS, UMA RODADA

  1. CONSISTÊNCIA. O `B_dense_gemini.json` que está em disco foi gravado numa
     data; a coleção do Qdrant é produção e pode ter mudado. Reconsultar hoje,
     com as mesmas perguntas, diz se o número do relatório ainda é o número.

  2. APROXIMAÇÃO. O Gemini é o único braço que busca no Qdrant, que usa HNSW e
     devolve vizinho APROXIMADO; os abertos usam matriz local e busca exata.
     Isso está declarado como limitação no relatório e nunca foi medido. Os
     vetores de documento do Gemini estão todos em disco, então a busca exata
     pode ser feita aqui, sem reindexar nada.

A diferença entre as duas é o que se compara com o que:

    disco  × Qdrant hoje   → consistência (mudou a coleção? mudou a API?)
    Qdrant × exata local   → o preço do HNSW

NÃO ESCREVE EM PRODUÇÃO. A única chamada ao Qdrant é de busca.

O QUE CUSTA. Só o passo `--embutir` gasta API: uma chamada de embedding por
pergunta, RETRIEVAL_QUERY, igual à produção. São 225 consultas curtas, e o
resultado fica em .npz — rodar de novo não paga de novo.

Uso:
  python eval/experimento_embedding/reteste_gemini.py --embutir   # paga API
  python eval/experimento_embedding/reteste_gemini.py --buscar    # Qdrant + exata
  python eval/experimento_embedding/reteste_gemini.py             # tabelas
"""

from __future__ import annotations

import argparse
import json
import sys
from math import comb
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parents[1]
EVAL = RAIZ / "eval"
sys.path.insert(0, str(EVAL))
sys.path.insert(0, str(RAIZ / "src"))

GABARITO = EVAL / "data/golden_dezembro.jsonl"
CORPUS = RAIZ / "data/processed/documents.jsonl"
VETORES_DOC = RAIZ / "data/processed/embeddings_context_v1_backup.jsonl"
CACHE_Q = EVAL / "results/indexes/dense_gemini_queries.npz"
RANK_DISCO = EVAL / "results/retrieval/rankings/B_dense_gemini.json"
RANK_QDRANT = EVAL / "results/indexes/reteste_gemini_qdrant.json"
RANK_EXATO = EVAL / "results/indexes/reteste_gemini_exato.json"

TOP_K = 100
K = 5

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


def populacao() -> dict:
    """As respondíveis do subconjunto balanceado — a população do artigo."""
    gold = {g["qid"]: g for g in
            (json.loads(l) for l in GABARITO.open(encoding="utf-8") if l.strip())}
    return {q: g for q, g in gold.items() if g["qrels_grupos"]}


def embutir(gold: dict) -> None:
    from lib.common import load_config
    from lib.embedders import build_embedder
    cfg = load_config(str(EVAL / "config.yaml"))
    emb = build_embedder("gemini", cfg)

    qids = sorted(gold)
    textos = [gold[q]["question"] for q in qids]
    print(f"embutindo {len(textos)} consultas com RETRIEVAL_QUERY "
          f"(lotes de {emb.batch_size}, {emb.sleep}s entre lotes)")
    M = emb._embed(textos, emb.task_query)
    CACHE_Q.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE_Q, qids=np.array(qids), matriz=M)
    print(f"✓ {CACHE_Q.name} — {M.shape[0]} × {M.shape[1]}")


def carregar_consultas() -> tuple[list[str], np.ndarray]:
    if not CACHE_Q.exists():
        sys.exit(f"✗ falta {CACHE_Q.name} — rode --embutir primeiro (gasta API).")
    d = np.load(CACHE_Q, allow_pickle=False)
    return [str(q) for q in d["qids"]], d["matriz"].astype(np.float32)


def buscar_qdrant(qids, M) -> None:
    """Reconsulta a coleção de produção. Leitura apenas."""
    import requests
    from urllib.parse import urlsplit
    from lib.common import get_env, load_config
    cfg = load_config(str(EVAL / "config.yaml"))
    colecao = cfg["embedders"]["gemini"]["qdrant_collection"]
    sp = urlsplit(get_env("QDRANT_URL", required=True).rstrip("/"))
    url = f"{sp.scheme or 'https'}://{sp.hostname}:{sp.port or 6333}"
    cab = {"api-key": get_env("QDRANT_API_KEY", required=True),
           "Content-Type": "application/json"}

    info = requests.get(f"{url}/collections/{colecao}", headers=cab, timeout=60)
    info.raise_for_status()
    r = info.json()["result"]
    print(f"coleção {colecao}: {r.get('points_count')} pontos")

    out = {}
    base = f"{url}/collections/{colecao}/points"
    for i, (q, v) in enumerate(zip(qids, M), start=1):
        resp = requests.post(f"{base}/search", headers=cab, timeout=60, json={
            "vector": v.tolist(), "limit": TOP_K, "with_payload": True})
        resp.raise_for_status()
        pts = resp.json().get("result") or []
        out[q] = [str(p["payload"]["id_original"]) for p in pts
                  if (p.get("payload") or {}).get("id_original") is not None]
        if i % 50 == 0:
            print(f"  {i}/{len(qids)}")
    RANK_QDRANT.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(f"✓ {RANK_QDRANT.name} — {len(out)} perguntas")


def buscar_exato(qids, M) -> None:
    """Top-100 por produto interno exato, contra os vetores que estão em disco.

    A junção é pelo TEXTO DE EMBEDDING reconstruído, não pelo id e não pelo
    `embedding_text_fingerprint` gravado no backup. Medido em 05/10:

        por id do trecho                      2.105 de 6.134
        pelo fingerprint gravado no backup        0 de 6.134
        pelo texto de embedding reconstruído  6.134 de 6.134

    O id falha porque houve migração para id por conteúdo depois de 22/09. O
    fingerprint gravado falha porque a RECEITA de `build_embedding_text` mudou
    desde então **sem que o rótulo do perfil mudasse** — o backup diz
    `context-v1` e o código de hoje também, mas os hashes divergem para o mesmo
    registro. O texto reconstruído é a única chave que responde à pergunta certa:
    "o texto que gerou este vetor é o texto que o corpus usa hoje?"."""
    from embedding_text import build_embedding_text
    alvo: dict[str, str] = {}
    for linha in CORPUS.open(encoding="utf-8"):
        if linha.strip():
            r = json.loads(linha)
            alvo[build_embedding_text(r)] = r["id"]
    ids, vets, achados = [], [], set()
    with VETORES_DOC.open(encoding="utf-8") as f:
        for linha in f:
            if not linha.strip():
                continue
            d = json.loads(linha)
            cid = alvo.get(build_embedding_text(d["document"]))
            if cid is not None and cid not in achados:
                achados.add(cid)
                ids.append(cid)
                vets.append(d["vector"])
    vivos = set(alvo.values())
    D = np.asarray(vets, dtype=np.float32)
    D /= np.linalg.norm(D, axis=1, keepdims=True)
    print(f"vetores de documento: {D.shape[0]} de {len(vivos)} trechos do corpus")
    if D.shape[0] < len(vivos):
        print(f"  ATENÇÃO: {len(vivos) - D.shape[0]} trechos sem vetor — "
              f"a busca exata não os pode devolver")

    out = {}
    for q, v in zip(qids, M):
        v = v / np.linalg.norm(v)
        s = D @ v
        topo = np.argpartition(-s, TOP_K)[:TOP_K]
        out[q] = [ids[i] for i in topo[np.argsort(-s[topo])]]
    RANK_EXATO.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(f"✓ {RANK_EXATO.name} — {len(out)} perguntas")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--embutir", action="store_true",
                    help="embute as consultas pela API Gemini (GASTA API)")
    ap.add_argument("--buscar", action="store_true",
                    help="reconsulta o Qdrant e faz a busca exata local")
    args = ap.parse_args()

    gold = populacao()
    print(f"população: {len(gold)} respondíveis de {GABARITO.name}\n")

    if args.embutir:
        embutir(gold)
        return
    if args.buscar:
        qids, M = carregar_consultas()
        buscar_qdrant(qids, M)
        print()
        buscar_exato(qids, M)
        return

    faltam = [p.name for p in (RANK_QDRANT, RANK_EXATO) if not p.exists()]
    if faltam:
        sys.exit(f"✗ falta {', '.join(faltam)} — rode --buscar.")

    disco = json.loads(RANK_DISCO.read_text(encoding="utf-8"))
    hoje = json.loads(RANK_QDRANT.read_text(encoding="utf-8"))
    exato = json.loads(RANK_EXATO.read_text(encoding="utf-8"))
    pop = sorted(q for q in gold if q in disco and q in hoje and q in exato)
    print(f"comparando {len(pop)} perguntas\n")

    def acerto(r, q, k=K):
        topo = set(r[q][:k])
        return int(all(any(c in topo for c in gr) for gr in gold[q]["qrels_grupos"]))

    rotulos = {"em disco": disco, "Qdrant hoje": hoje, "exata local": exato}
    print("── acerto completo no top-5 ──\n")
    print(f"{'ranking':16s}{'acertos':>9s}{'CE@5':>8s}{'CE@100':>9s}")
    print("─" * 42)
    for nome, r in rotulos.items():
        a5 = sum(acerto(r, q) for q in pop)
        a100 = sum(acerto(r, q, TOP_K) for q in pop)
        print(f"{nome:16s}{a5:9d}{a5/len(pop):8.3f}{a100/len(pop):9.3f}")

    print("\n\n── quanto cada par difere ──\n")
    print(f"{'par':34s}{'top-5 igual':>13s}{'top-100 igual':>15s}"
          f"{'Jaccard@100':>13s}")
    print("─" * 75)
    pares = (("em disco", "Qdrant hoje"), ("Qdrant hoje", "exata local"),
             ("em disco", "exata local"))
    for a, b in pares:
        ra, rb = rotulos[a], rotulos[b]
        ig5 = sum(1 for q in pop if ra[q][:K] == rb[q][:K])
        ig100 = sum(1 for q in pop if set(ra[q][:TOP_K]) == set(rb[q][:TOP_K]))
        jac = np.mean([len(set(ra[q][:TOP_K]) & set(rb[q][:TOP_K])) /
                       len(set(ra[q][:TOP_K]) | set(rb[q][:TOP_K])) for q in pop])
        print(f"{f'{a} × {b}':34s}{ig5:9d}/{len(pop):<3d}"
              f"{ig100:11d}/{len(pop):<3d}{jac:13.3f}")

    print("\n\n── e muda o acerto? (McNemar exato) ──\n")
    print(f"{'par':34s}{'vit':>5s}{'der':>5s}{'p':>10s}")
    print("─" * 54)
    for a, b in pares:
        ra, rb = rotulos[a], rotulos[b]
        v = sum(1 for q in pop if acerto(ra, q) and not acerto(rb, q))
        d = sum(1 for q in pop if acerto(rb, q) and not acerto(ra, q))
        print(f"{f'{a} × {b}':34s}{v:5d}{d:5d}{mcnemar(v, d):10.4f}")
    print("\n  vit = o primeiro acerta e o segundo não.")


if __name__ == "__main__":
    main()
