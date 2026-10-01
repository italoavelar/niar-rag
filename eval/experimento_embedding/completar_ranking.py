#!/usr/bin/env python3
"""Completa os rankings para as perguntas que entraram depois da última rodada.

O PROBLEMA QUE ISTO RESOLVE. Acrescentar uma pergunta ao acervo invalida a
comparação inteira, porque `analisar_bracos.py` descarta braço que não ranqueou
todas as respondíveis — corretamente, já que média sobre populações diferentes
não compara nada. Mas refazer os três braços por causa de UMA pergunta custa,
no Gemini, 393 chamadas de API que não mudam nada.

Este script calcula só o que falta e costura no arquivo existente. Em 30/09/2026
foi escrito para a `m016`, a 16ª multi-hop dispersa, que fechou o alvo de 75
multi_hop.

O QUE CADA BRAÇO CUSTA, e por isso cada um tem sua opção:

    --bge      local, CPU, zero de API. Embute a pergunta com o BAAI/bge-m3 e
               busca na matriz do corpus que já está em cache.
    --rerank   local, CPU. Reordena o top-100 do BGE com o cross-encoder. São
               100 pares por pergunta — segundos, não as 28 horas da rodada
               inteira.
    --gemini   CHAMA A API PAGA: uma embutidura de query por pergunta faltante,
               e uma busca na coleção do Qdrant de produção. Só roda se pedido
               explicitamente, para não gastar por descuido.

A RECEITA DO RERANKER É COPIADA DE `eval/colab_f3_rerank.py`, e tem que ser:
`max_length=384`, par `(pergunta, build_embedding_text(trecho))` — o texto
CONTEXTUAL, com o prefixo `[título · emissor · seção]` —, base `B_dense_bge_m3`,
100 candidatos. Pontuar a pergunta nova com outra receita faria dela um ponto
fora da curva por artefato de medição.

Uso:
  python eval/experimento_embedding/completar_ranking.py                 # relatório
  python eval/experimento_embedding/completar_ranking.py --bge --rerank --aplicar
  python eval/experimento_embedding/completar_ranking.py --gemini --aplicar
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parents[1]
EVAL = RAIZ / "eval"
sys.path.insert(0, str(EVAL))
sys.path.insert(0, str(EVAL / "retrieval_eval"))
sys.path.insert(0, str(RAIZ / "src"))

for _f in (sys.stdout, sys.stderr):
    try:
        _f.reconfigure(encoding="utf-8")
    except Exception:
        pass

RANKINGS = EVAL / "results/retrieval/rankings"
SCORES = EVAL / "results/indexes/rerank_scores_bge_reranker_v2_m3.json"
MAX_LENGTH = 384
MAX_CANDIDATOS = 100
RERANKER = "BAAI/bge-reranker-v2-m3"


def carregar(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def gravar(p: Path, dados: dict) -> None:
    p.write_text(json.dumps(dados, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gabarito", default=str(EVAL / "data/golden_acervo.jsonl"),
                    help="jsonl que define quem PRECISA ter ranking")
    ap.add_argument("--bge", action="store_true")
    ap.add_argument("--gemini", action="store_true",
                    help="CUSTA API: uma embutidura de query por pergunta")
    ap.add_argument("--rerank", action="store_true")
    ap.add_argument("--aplicar", action="store_true")
    args = ap.parse_args()

    from lib.common import load_config, load_corpus, resolve
    from _common import dense_ranking
    from embedding_text import build_embedding_text

    cfg = load_config(str(EVAL / "config.yaml"))
    corpus = load_corpus(resolve(cfg["paths"]["corpus"]))

    gold = [json.loads(l) for l in Path(args.gabarito).open(encoding="utf-8")
            if l.strip()]
    por_qid = {g["qid"]: g for g in gold}
    respondiveis = [g["qid"] for g in gold if g.get("qrels_grupos")]
    print(f"gabarito : {Path(args.gabarito).name} — {len(gold)} perguntas, "
          f"{len(respondiveis)} respondíveis")
    print(f"corpus   : {len(corpus)} trechos\n")

    alvos = {
        "B_dense_bge_m3": ("bge_m3", args.bge),
        "B_dense_gemini": ("gemini", args.gemini),
    }
    faltas: dict[str, list[str]] = {}
    for arq in list(alvos) + ["B_bge_m3_rerank"]:
        d = carregar(RANKINGS / f"{arq}.json")
        faltas[arq] = [q for q in respondiveis if q not in d]
        print(f"  {arq:18s} {len(d):4d} consultas · falta {len(faltas[arq])}"
              f"  {faltas[arq][:8]}")
    if not any(faltas.values()):
        print("\nNada a completar.")
        return
    print()

    # ── braços densos ───────────────────────────────────────────────────────
    for arq, (nome, pedido) in alvos.items():
        if not faltas[arq]:
            continue
        if not pedido:
            print(f"  {arq}: {len(faltas[arq])} faltando — passe "
                  f"--{'gemini' if nome == 'gemini' else 'bge'} para calcular")
            continue
        queries = [(q, por_qid[q]["question"]) for q in faltas[arq]]
        t0 = time.time()
        novo = dense_ranking(cfg, nome, corpus, queries)
        print(f"  {arq}: {len(novo)} calculados em {time.time()-t0:.0f}s")
        if args.aplicar:
            d = carregar(RANKINGS / f"{arq}.json")
            d.update(novo)
            gravar(RANKINGS / f"{arq}.json", d)
            print(f"    ✓ gravado — agora {len(d)} consultas")

    # ── reranker ────────────────────────────────────────────────────────────
    if faltas["B_bge_m3_rerank"]:
        if not args.rerank:
            print(f"  B_bge_m3_rerank: {len(faltas['B_bge_m3_rerank'])} faltando "
                  f"— passe --rerank para calcular")
        else:
            base = carregar(RANKINGS / "B_dense_bge_m3.json")
            pendentes = [q for q in faltas["B_bge_m3_rerank"] if q in base]
            sem_base = [q for q in faltas["B_bge_m3_rerank"] if q not in base]
            if sem_base:
                print(f"  ✗ sem top-100 do BGE, rode --bge primeiro: {sem_base}")
            if pendentes:
                pares, indice = [], []
                for q in pendentes:
                    for cid in base[q][:MAX_CANDIDATOS]:
                        pares.append((por_qid[q]["question"],
                                      build_embedding_text(corpus[cid])))
                        indice.append((q, cid))
                print(f"  reranker: {len(pares)} pares "
                      f"({len(pendentes)} perguntas) — carregando {RERANKER}")
                from sentence_transformers import CrossEncoder
                t0 = time.time()
                ce = CrossEncoder(RERANKER, max_length=MAX_LENGTH, device="cpu",
                                  trust_remote_code=True)
                notas = ce.predict(pares, batch_size=16, show_progress_bar=True)
                dt = time.time() - t0
                print(f"    {len(pares)} pares em {dt:.0f}s "
                      f"({dt/max(len(pares),1):.3f}s por par)")

                por_query: dict[str, dict[str, float]] = {}
                for (q, cid), nota in zip(indice, notas):
                    por_query.setdefault(q, {})[cid] = float(nota)

                if args.aplicar:
                    d = carregar(RANKINGS / "B_bge_m3_rerank.json")
                    for q, mapa in por_query.items():
                        d[q] = sorted(mapa, key=lambda c: -mapa[c])
                    gravar(RANKINGS / "B_bge_m3_rerank.json", d)
                    print(f"    ✓ ranking gravado — agora {len(d)} consultas")

                    s = carregar(SCORES)
                    s["scores"].update(por_query)
                    s["n_pares"] = s.get("n_pares", 0) + len(pares)
                    gravar(SCORES, s)
                    print(f"    ✓ notas gravadas — agora "
                          f"{len(s['scores'])} perguntas pontuadas")

    if not args.aplicar:
        print("\n(relatório apenas — use --aplicar para gravar)")


if __name__ == "__main__":
    main()
