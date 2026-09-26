#!/usr/bin/env python3
"""Banco de provas de estratégias de recorte (chunking), sem reanotar nada.

O PROBLEMA QUE ESTE SCRIPT RESOLVE. Cada estratégia de recorte produz trechos
diferentes, logo ids diferentes. Comparar estratégias por id exigiria reancorar o
gabarito uma vez por estratégia — quatro reancoragens, quatro chances de erro, e
nenhuma garantia de que as quatro ficaram equivalentes.

A SAÍDA: avaliar por TEXTO, não por id. A evidência de uma pergunta é um fato, e
um fato vive numa frase. De cada âncora de grau 2 extrai-se um **núcleo**: os 160
caracteres centrais da sua maior frase. A estratégia acerta quando o núcleo
aparece, literalmente, em algum trecho recuperado.

    Medido no recorte atual: os 139 núcleos de grau 2 são encontrados —
    139/139, sendo 135 em um único trecho. Os 4 ambíguos são efeito da
    sobreposição de 200 caracteres, e não atrapalham: basta o núcleo estar
    no conjunto recuperado.

Por que 160 caracteres: curto o bastante para sobreviver a fronteiras de recorte
(a mediana da unidade jurídica é 304), longo o bastante para ser único no acervo.
Se um recorte parte a frase ao meio, ele PERDE o núcleo — e isso é exatamente o
defeito que se quer medir, não um artefato da medição.

MÉTRICA PRINCIPAL: **Joint Recall@k** — perguntas cujos núcleos estão TODOS no
conjunto recuperado. Ver `Histórico.md`, seção "Como tudo é medido".

O QUE SE MANTÉM CONSTANTE: embedding (BGE-m3), índice, recuperador, top-k,
perguntas e métricas. Varia só o recorte. Comparação de LLM ou de embedding é
outro experimento.

Uso:
  python eval/tools/bench_chunking.py nucleos                      # extrai, roda uma vez
  python eval/tools/bench_chunking.py avaliar --nome atual \\
      --corpus data/processed/documents.jsonl
  python eval/tools/bench_chunking.py avaliar --nome C4_rec1000 \\
      --corpus data/processed/chunking/C4.jsonl
  python eval/tools/bench_chunking.py tabela                       # consolida tudo
"""

from __future__ import annotations

import argparse
import json
import re
import statistics as st
import sys
import time
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVAL = PROJECT_ROOT / "eval"

for _f in (sys.stdout, sys.stderr):
    try:
        _f.reconfigure(encoding="utf-8")
    except Exception:
        pass

GOLDEN = EVAL / "data/golden_qa.jsonl"
AQUI = Path(__file__).resolve().parent
NUCLEOS = AQUI / "dados/nucleos_evidencia.json"
RESULTADOS = AQUI / "resultados"

NUCLEO_CHARS = 160
FRASE = re.compile(r"[^.!?;]{60,}")
KS = (5, 10, 20)


def normalizar(t: str) -> str:
    t = unicodedata.normalize("NFKD", (t or "").lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    return " ".join(t.split())


# ── núcleos ──────────────────────────────────────────────────────────────────

def cmd_nucleos(args) -> None:
    """Extrai um núcleo por âncora de grau 2. Independe de qualquer recorte."""
    gold = [json.loads(l) for l in GOLDEN.open(encoding="utf-8") if l.strip()]
    resp = [g for g in gold if g.get("qrels")]

    saida, sem = {}, []
    for g in resp:
        nucleos = []
        for cid, grau in g["qrels"].items():
            if grau < 2:
                continue
            texto = (g.get("qrels_text") or {}).get(cid, "")
            frases = FRASE.findall(texto)
            if not frases:
                sem.append((g["qid"], cid))
                continue
            n = normalizar(max(frases, key=len))
            meio = n[max(0, (len(n) - NUCLEO_CHARS) // 2):][:NUCLEO_CHARS]
            if len(meio) >= 40:
                nucleos.append(meio)
            else:
                sem.append((g["qid"], cid))
        if nucleos:
            saida[g["qid"]] = {"tipo": g["question_type"],
                               "pergunta": g["question"],
                               "nucleos": nucleos}

    NUCLEOS.parent.mkdir(parents=True, exist_ok=True)
    NUCLEOS.write_text(json.dumps(saida, ensure_ascii=False, indent=1), encoding="utf-8")
    n_tot = sum(len(v["nucleos"]) for v in saida.values())
    print(f"✓ {NUCLEOS}")
    print(f"  perguntas com núcleo : {len(saida)}")
    print(f"  núcleos (grau 2)     : {n_tot}")
    if sem:
        print(f"  ⚠ sem frase utilizável: {len(sem)} → {sem[:4]}")
    for t in ("factual", "multi_hop", "comparative"):
        sub = [v for v in saida.values() if v["tipo"] == t]
        if sub:
            print(f"    {t:13s} {len(sub):3d} perguntas, "
                  f"mediana {st.median([len(v['nucleos']) for v in sub]):.0f} núcleo(s)")


# ── avaliação de um recorte ──────────────────────────────────────────────────

def estatisticas(corpus: list[dict]) -> dict:
    tam = [len(c["text"]) for c in corpus]
    docs = {c["metadata"].get("document_id") for c in corpus}
    return {
        "trechos": len(corpus),
        "documentos": len(docs),
        "media": round(st.mean(tam), 1),
        "mediana": st.median(tam),
        "minimo": min(tam),
        "maximo": max(tam),
        "p10": int(np.percentile(tam, 10)),
        "p90": int(np.percentile(tam, 90)),
        "abaixo_300": sum(1 for t in tam if t < 300),
        "acima_2500": sum(1 for t in tam if t > 2500),
    }


def cmd_avaliar(args) -> None:
    if not NUCLEOS.exists():
        sys.exit("núcleos não extraídos — rode `bench_chunking.py nucleos` antes.")
    nuc = json.loads(NUCLEOS.read_text(encoding="utf-8"))

    caminho = Path(args.corpus)
    if not caminho.is_absolute():
        caminho = PROJECT_ROOT / caminho
    if not caminho.exists():
        sys.exit(f"corpus não encontrado: {caminho}")
    corpus = [json.loads(l) for l in caminho.open(encoding="utf-8") if l.strip()]
    for c in corpus:
        if "id" not in c or "text" not in c:
            sys.exit("cada linha do corpus precisa ter 'id' e 'text'.")

    est = estatisticas(corpus)
    print(f"\n══ {args.nome} ══")
    print(f"  {est['trechos']} trechos em {est['documentos']} documentos")
    print(f"  tamanho: mediana {est['mediana']}  média {est['media']}  "
          f"p10 {est['p10']}  p90 {est['p90']}  min {est['minimo']}  max {est['maximo']}")
    print(f"  abaixo de 300 chars: {est['abaixo_300']}   acima de 2500: {est['acima_2500']}")

    # ── sanidade: os núcleos sobrevivem a este recorte? ──────────────────────
    textos = [normalizar(c["text"]) for c in corpus]
    todos = [n for v in nuc.values() for n in v["nucleos"]]
    presentes = sum(1 for n in todos if any(n in t for t in textos))
    print(f"\n  núcleos que SOBREVIVEM ao recorte: {presentes}/{len(todos)} "
          f"({presentes/len(todos):.0%})")
    if presentes < len(todos):
        print("    (os que somem foram partidos por uma fronteira de trecho — "
              "é defeito do recorte, e conta como perda)")

    # ── embutir e buscar ─────────────────────────────────────────────────────
    from sentence_transformers import SentenceTransformer
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from embedding_text import build_embedding_text, embedding_text_hash

    # Reaproveita vetor já calculado, de qualquer cache .npz em disco. Embutir
    # 4.897 trechos em CPU leva horas; para o recorte atual isso já foi feito, e
    # entre duas estratégias boa parte dos trechos é idêntica. A chave é o hash
    # DO TEXTO (`embedding_text_hash`): o id não serve, porque cada estratégia
    # gera ids próprios.
    cache = {}
    por_id = {r["id"]: r for r in corpus}
    for npz in sorted((EVAL / "results/indexes").glob("dense_bge_m3_context-v1_*.npz")):
        d = np.load(npz, allow_pickle=True)
        mat = d["matrix"]
        for j, cid in enumerate(list(d["ids"])):
            r = por_id.get(cid)
            if r is not None:
                cache.setdefault(embedding_text_hash(r), mat[j])

    chaves = [embedding_text_hash(c) for c in corpus]
    faltam = [i for i, h in enumerate(chaves) if h not in cache]
    print(f"\n  vetores reaproveitados do cache: {len(corpus)-len(faltam)}/{len(corpus)}"
          f"   a calcular: {len(faltam)}")

    mod = None
    t_emb = 0.0
    if faltam:
        print("  carregando BGE-m3 (CPU)...", flush=True)
        mod = SentenceTransformer("BAAI/bge-m3", device="cpu")
        t0 = time.perf_counter()
        novos = mod.encode([build_embedding_text(corpus[i]) for i in faltam],
                           normalize_embeddings=True, batch_size=8,
                           show_progress_bar=True).astype("float32")
        t_emb = time.perf_counter() - t0
        for i, v in zip(faltam, novos):
            cache[chaves[i]] = v
        print(f"  embutidos {len(faltam)} em {t_emb/60:.1f} min")
    Mx = np.stack([cache[h] for h in chaves]).astype("float32")

    qids = list(nuc)
    if mod is None:
        print("  carregando BGE-m3 só para as consultas...", flush=True)
        mod = SentenceTransformer("BAAI/bge-m3", device="cpu")
    Qv = mod.encode([nuc[q]["pergunta"] for q in qids], normalize_embeddings=True,
                    batch_size=8, show_progress_bar=False).astype("float32")
    ordem = np.argsort(-(Qv @ Mx.T), axis=1)[:, :max(KS)]

    # ── métricas ─────────────────────────────────────────────────────────────
    linhas = {}
    for k in KS:
        por_tipo = {}
        for t in ("factual", "multi_hop", "comparative"):
            idx = [i for i, q in enumerate(qids) if nuc[q]["tipo"] == t]
            joint = rec = 0
            for i in idx:
                juntos = " ".join(textos[j] for j in ordem[i][:k])
                achou = [n for n in nuc[qids[i]]["nucleos"] if n in juntos]
                joint += len(achou) == len(nuc[qids[i]]["nucleos"])
                rec += len(achou) / len(nuc[qids[i]]["nucleos"])
            por_tipo[t] = {"n": len(idx), "joint": joint, "recall": round(rec / len(idx), 4)}
        linhas[k] = por_tipo

    print(f"\n  {'k':>4}  " + "  ".join(f"{t[:11]:^22s}" for t in
                                        ("factual", "multi_hop", "comparative")))
    print(f"  {'':>4}  " + "  ".join(f"{'JointRecall':>13s}{'Rec':>9s}" for _ in range(3)))
    for k in KS:
        cels = []
        for t in ("factual", "multi_hop", "comparative"):
            d = linhas[k][t]
            cels.append(f"{d['joint']:9d}/{d['n']:<3d}{d['recall']:9.1%}")
        print(f"  {k:>4}  " + "  ".join(cels))

    RESULTADOS.mkdir(parents=True, exist_ok=True)
    destino = RESULTADOS / f"{args.nome}.json"
    destino.write_text(json.dumps({
        "nome": args.nome,
        "corpus": str(caminho.relative_to(PROJECT_ROOT)),
        "estatisticas": est,
        "nucleos_sobreviventes": f"{presentes}/{len(todos)}",
        "segundos_para_embutir": round(t_emb, 1),
        "metricas": {str(k): linhas[k] for k in KS},
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n✓ {destino}")


# ── consolidação ─────────────────────────────────────────────────────────────

def cmd_tabela(args) -> None:
    arqs = sorted(RESULTADOS.glob("*.json"))
    if not arqs:
        sys.exit(f"nada em {RESULTADOS}")
    dados = [json.loads(p.read_text(encoding="utf-8")) for p in arqs]
    k = str(args.k)

    print(f"\nCOMPARAÇÃO DE ESTRATÉGIAS DE RECORTE — Joint Recall@{k}\n")
    print(f"{'estratégia':18s}{'trechos':>9}{'mediana':>9}{'núcleos':>10}  " +
          "  ".join(f"{t[:11]:>13s}" for t in ("factual", "multi_hop", "comparative")))
    print("-" * 96)
    for d in dados:
        m = d["metricas"].get(k)
        if not m:
            continue
        cels = [f"{m[t]['joint']:8d}/{m[t]['n']:<3d}" for t in
                ("factual", "multi_hop", "comparative")]
        print(f"{d['nome']:18s}{d['estatisticas']['trechos']:>9}"
              f"{d['estatisticas']['mediana']:>9}{d['nucleos_sobreviventes']:>10}  "
              + "  ".join(cels))
    print("-" * 96)
    print("núcleos = quantos dos 139 sobrevivem ao recorte (não foram partidos por fronteira)")
    print("Joint Recall = perguntas com TODOS os núcleos no top-k")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("nucleos", help="extrai os núcleos de evidência do gabarito")
    a.set_defaults(func=cmd_nucleos)

    b = sub.add_parser("avaliar", help="avalia um corpus já recortado")
    b.add_argument("--nome", required=True, help="rótulo da estratégia, ex.: C4_rec1000")
    b.add_argument("--corpus", required=True, help="caminho do .jsonl com id/text/metadata")
    b.set_defaults(func=cmd_avaliar)

    c = sub.add_parser("tabela", help="consolida os resultados já gravados")
    c.add_argument("--k", type=int, default=5)
    c.set_defaults(func=cmd_tabela)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
