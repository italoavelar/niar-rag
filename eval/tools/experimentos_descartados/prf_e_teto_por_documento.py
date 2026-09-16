#!/usr/bin/env python3
"""Duas tentativas de quebrar o monopólio do top-5 — as duas falharam.

O PROBLEMA QUE AS DUAS ATACAVAM. Medido em 13/set: o documento mais frequente
ocupa **mediana 5 das 5 vagas** do top-5, nos três tipos de pergunta. Na
comparativa isso é fatal — as duas âncoras estão em documentos DIFERENTES, então
um top-5 monopolizado por um deles nunca responde a pergunta.

    (A) TETO POR DOCUMENTO   no máximo N trechos do mesmo documento no top-5.
                             Diferente da "partição 3+2 por lado", não precisa
                             saber QUAL é cada lado — foi isso que derrubou a
                             partição, que exigia um roteador.

    (B) PRF / ROCCHIO DENSO  q' = q + beta * média(top-k), renormalizado. A ideia
                             veio do diagnóstico da lacuna de vocabulário: se
                             aproximar a consulta do texto da resposta ajuda,
                             usar os próprios trechos bem colocados como
                             aproximação deveria ajudar — e sem LLM nenhum.

RESULTADO (13/set/2026) — as duas refutadas, por motivos opostos e instrutivos.

(A) O teto DESTRÓI o multi-hop:

    teto              factual R@5   multi_hop R@5   multi_hop AS DUAS   comparative R@5
    sem teto (hoje)      28,2%          31,2%            2/25               23,7%
    máx. 3 por doc       24,4%          23,4%            1/25               25,4%
    máx. 2 por doc       19,8%          12,5%            0/25               22,0%
    máx. 1 por doc       12,2%           7,8%            0/25               16,9%

    Por quê: âncoras em documentos DISTINTOS por pergunta — factual 1,
    multi_hop 1, comparative 2. O teto foi desenhado para a comparativa e
    aplicado a todos; no multi-hop ele expulsa justamente a segunda âncora, que
    está no MESMO documento da primeira. E na comparativa mal move.

(B) O PRF AMPLIFICA o monopólio em vez de corrigi-lo. Doze configurações
    (k em {3,5,10} x beta em {0,3; 0,5; 0,7; 1,0}), todas planas ou piores. A
    comparativa é a mais prejudicada: mediana da melhor âncora de 5 para 8-18,
    R@20 de 40,7% para 25-37%.

    Por quê: o centroide do top-k é dominado pelo lado que já ganhava. Realimentar
    com ele empurra a consulta para MAIS LONGE do lado que faltava.

CONCLUSÃO REGISTRADA (Memoria.md §9.4): não perseguir o monopólio. Ele AJUDA
factual e multi-hop, cujas âncoras estão no mesmo documento; só a comparativa
sofre, e as duas tentativas de quebrá-lo pioraram o conjunto. Medido também que
o prefixo contextual `[título · emissor · seção]` responde por cerca de um terço
da coesão interna (0,758 dentro vs 0,495 entre, com prefixo; 0,669 vs 0,491 sem)
— os outros dois terços são intrínsecos ao acervo.

O monopólio é sintoma. A doença é a consulta — ver Memoria.md §9.1.

Uso:
  python eval/tools/experimentos_descartados/prf_e_teto_por_documento.py
"""

from __future__ import annotations

import json
import statistics as st
import sys
from collections import Counter
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
INDICES = EVAL / "results/indexes"
TIPOS = ("factual", "multi_hop", "comparative")


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
    faltando = [c for g in resp for c in g["qrels"] if c not in pos]
    if faltando:
        sys.exit(f"✗ {len(faltando)} âncoras fora do índice — corpus e índice dessincronizados.")
    return M, ids, pos, corpus, resp


def main() -> None:
    M, ids, pos, corpus, resp = carregar()
    tipos = [g["question_type"] for g in resp]
    ancoras = [list(g["qrels"]) for g in resp]
    doc_de = {c: corpus[c]["metadata"].get("document_id") for c in ids}

    from sentence_transformers import SentenceTransformer
    print("carregando BGE-m3 (CPU)...", flush=True)
    mod = SentenceTransformer("BAAI/bge-m3", device="cpu")
    Q = mod.encode([g["question"] for g in resp], normalize_embeddings=True,
                   batch_size=8, show_progress_bar=False).astype("float32")
    S0 = Q @ M.T
    ORD = np.argsort(-S0, axis=1)[:, :200]

    print("\nâncoras em documentos DISTINTOS, por pergunta (mediana):")
    for t in TIPOS:
        idx = [i for i, x in enumerate(tipos) if x == t]
        print(f"   {t:12s} {st.median([len({doc_de[a] for a in ancoras[i]}) for i in idx]):.0f}")

    print("\nmonopólio do top-5 (quantas das 5 vagas o documento mais frequente ocupa):")
    for t in TIPOS:
        idx = [i for i, x in enumerate(tipos) if x == t]
        mono = [Counter(doc_de[ids[j]] for j in ORD[i][:5]).most_common(1)[0][1] for i in idx]
        print(f"   {t:12s} mediana {st.median(mono):.0f} de 5   (média {st.mean(mono):.1f})")

    # ── (A) teto por documento ───────────────────────────────────────────────
    def seleciona(i, teto, k=5):
        sel, conta = [], Counter()
        for j in ORD[i]:
            c = ids[j]
            if conta[doc_de[c]] >= teto:
                continue
            sel.append(c); conta[doc_de[c]] += 1
            if len(sel) == k:
                break
        return set(sel)

    print("\n" + "=" * 74)
    print("(A) TETO POR DOCUMENTO no top-5")
    print("=" * 74)
    print(f"{'teto':18s} | " + " | ".join(f"{t[:11]:>17s}" for t in TIPOS))
    print(f"{'':18s} | " + " | ".join(f"{'R@5':>8s}{'AS DUAS':>9s}" for _ in TIPOS))
    print("-" * 74)
    for teto in (99, 3, 2, 1):
        cels = []
        for t in TIPOS:
            idx = [i for i, x in enumerate(tipos) if x == t]
            acer = tot = todas = 0
            for i in idx:
                sel = seleciona(i, teto)
                acer += sum(1 for a in ancoras[i] if a in sel); tot += len(ancoras[i])
                todas += all(a in sel for a in ancoras[i])
            cels.append(f"{acer/tot:8.1%}{todas:5d}/{len(idx):<3d}")
        rot = "sem teto (hoje)" if teto == 99 else f"máx {teto} por doc"
        print(f"{rot:18s} | " + " | ".join(cels))

    # ── (B) PRF / Rocchio ────────────────────────────────────────────────────
    def avalia(S):
        ordem = np.argsort(-S, axis=1)[:, :500]
        out = {}
        for t in TIPOS:
            idx = [i for i, x in enumerate(tipos) if x == t]
            r5 = r20 = tot = 0
            melhores = []
            for i in idx:
                onde = {ids[j]: r for r, j in enumerate(ordem[i], 1)}
                p = sorted(onde.get(a, 9999) for a in ancoras[i])
                melhores.append(p[0]); tot += len(p)
                r5 += sum(1 for x in p if x <= 5)
                r20 += sum(1 for x in p if x <= 20)
            out[t] = (st.median(melhores), r5 / tot, r20 / tot)
        return out

    print("\n" + "=" * 86)
    print("(B) PRF DENSO (Rocchio) — varredura de k e beta")
    print("=" * 86)
    print(f"{'config':22s} | " + " | ".join(f"{t[:11]:>19s}" for t in TIPOS))
    print(f"{'':22s} | " + " | ".join(f"{'med':>5s}{'R@5':>7s}{'R@20':>7s}" for _ in TIPOS))
    print("-" * 86)

    def linha(nome, m):
        cels = [f"{m[t][0]:5.0f}{m[t][1]:7.1%}{m[t][2]:7.1%}" for t in TIPOS]
        print(f"{nome:22s} | " + " | ".join(cels))

    linha("BASE (pergunta)", avalia(S0))
    for k in (3, 5, 10):
        topk = np.argsort(-S0, axis=1)[:, :k]
        centro = np.stack([M[topk[i]].mean(0) for i in range(len(Q))])
        for beta in (0.3, 0.5, 0.7, 1.0):
            Qp = Q + beta * centro
            Qp /= np.linalg.norm(Qp, axis=1, keepdims=True)
            linha(f"PRF k={k} beta={beta}", avalia(Qp @ M.T))
    print("-" * 86)
    print("AS DUAS = perguntas com TODOS os trechos necessários no top-5 — a régua que decide")


if __name__ == "__main__":
    main()
