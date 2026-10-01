#!/usr/bin/env python3
"""Exporta o acervo no formato que o pipeline de avaliação já lê.

POR QUE UMA PONTE, E NÃO EDITAR OS CENÁRIOS. O `eval/` inteiro — os quatro
cenários de recuperação, o `julgar_relevancia.py`, o `rerank_eval.py` e as etapas
de geração — lê um só arquivo, `paths.golden_qa`, no formato de `golden_qa.jsonl`:
um objeto por linha com `qid`, `question`, `qrels` e metadados. Reescrever cada
consumidor para ler `acervo.json` seria mudar seis scripts e arriscar seis
divergências. Traduzir uma vez e apontar o `config.yaml` muda UM lugar.

O QUE MUDA DE NATUREZA NA TRADUÇÃO. No formato antigo `qrels` era dado anotado à
mão, gravado dentro da pergunta. Aqui ele é **derivado**: vem do `qrels.csv` que
o `gerar_qrels.py` produz das citações literais contra o corpus em uso. Então
este arquivo é uma VISTA, não uma fonte — regerar depois de mexer no corpus é
obrigatório, e editá-lo à mão é perder a edição na próxima exportação. A fonte é
`acervo.json` mais o corpus.

AS IRRESPONDÍVEIS ENTRAM, sem `qrels`. Elas não têm trecho relevante por
definição, mas o pipeline precisa recuperar para elas: é o contexto plausível que
a etapa de geração usa para testar se o modelo RECUSA. `load_gold` só monta
qrels para quem tem, e `save_scenario` deixa as colunas de métrica vazias nelas
— nDCG sem documento relevante é indefinido, não zero.

`source_lang` das irrespondíveis sai como "n/a", não "pt": não existe documento
de origem cuja língua declarar, e enfiá-las no balde "pt" contaminaria a
comparação PT×EN que é uma das leituras do artigo.

Uso:
  python eval/experimento_embedding/exportar_gabarito.py --aplicar
  python eval/experimento_embedding/exportar_gabarito.py --acervo dados/acervo.json --saida ../data/golden_acervo.jsonl --aplicar
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

AQUI = Path(__file__).resolve().parent
PROJECT_ROOT = AQUI.parents[1]

for _f in (sys.stdout, sys.stderr):
    try:
        _f.reconfigure(encoding="utf-8")
    except Exception:
        pass


def _normalizar(t: str) -> str:
    t = (t or "").replace("­", "")
    t = unicodedata.normalize("NFKD", t.lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    return " ".join(t.split())


def agrupar_por_citacao(evidencia, corpus_norm):
    """Para cada citação, TODOS os trechos que a contêm.

    POR QUE O AGRUPAMENTO É NECESSÁRIO. A sobreposição de 240 caracteres faz a
    mesma citação aparecer em dois trechos vizinhos, e os dois entram no
    `qrels.csv` como grau 2 — corretamente, porque qualquer um deles serve. Mas
    `complete_evidence_at_k` exige que TODOS os trechos de grau 2 estejam no
    top-k, e passa a cobrar os dois. Isso SUBESTIMA o Junta@k: medido em
    27/09/2026, 35 das 225 perguntas respondíveis (15,6%) têm ao menos uma
    citação em mais de um trecho, somando 22 trechos exigidos a mais.

    Com o agrupamento, a regra volta a ser a certa: a citação está satisfeita se
    QUALQUER trecho do seu grupo chegou ao top-k, e a pergunta acerta se todas as
    suas citações estão satisfeitas.
    """
    grupos = []
    for cit in evidencia:
        alvo = _normalizar(cit)
        grupos.append([cid for cid, t in corpus_norm if alvo in t])
    return grupos


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--acervo", default="dados/subconjunto_dezembro.json")
    ap.add_argument("--qrels", default=str(PROJECT_ROOT / "eval/data/qrels.csv"))
    ap.add_argument("--saida", default=str(PROJECT_ROOT / "eval/data/golden_dezembro.jsonl"))
    ap.add_argument("--queries", default=str(PROJECT_ROOT / "eval/data/queries_dezembro.csv"))
    ap.add_argument("--aplicar", action="store_true")
    args = ap.parse_args()

    caminho = AQUI / args.acervo if not Path(args.acervo).is_absolute() else Path(args.acervo)
    acervo = json.loads(caminho.read_text(encoding="utf-8"))

    corpus_norm = []
    with (PROJECT_ROOT / "data/processed/documents.jsonl").open(encoding="utf-8") as fh:
        for linha in fh:
            if linha.strip():
                r = json.loads(linha)
                corpus_norm.append((r["id"], _normalizar(r["text"])))

    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    with open(args.qrels, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            qrels[r["QueryId"]][r["ChunkId"]] = int(r["Relevance"])

    saida, sem_qrels = [], []
    for q in acervo:
        qid = str(q["n"])
        docs = ([q["documento"]] if q.get("documento") else list(q.get("documentos") or []))
        respondivel = q["question_type"] != "unanswerable"
        grades = qrels.get(qid, {})
        if respondivel and not grades:
            sem_qrels.append(qid)
        saida.append({
            "qid": qid,
            "question": q["question"],
            "reference_answer": q.get("answer") or "",
            "qrels": grades,
            "qrels_text": list(q.get("evidencia") or []),
            # Um grupo por citação; a citação vale se QUALQUER trecho do grupo
            # voltar. Ver agrupar_por_citacao().
            "qrels_grupos": agrupar_por_citacao(q.get("evidencia") or [], corpus_norm),
            "source_docs": docs,
            "question_type": q["question_type"],
            "difficulty": q.get("difficulty") or "",
            "theme": q.get("theme") or "",
            "source_lang": q.get("source_lang") or "n/a",
            "status": "ok",
            "origem": q.get("origem", ""),
        })

    print(f"acervo : {caminho.name} — {len(acervo)} perguntas")
    print(f"qrels  : {Path(args.qrels).name} — {sum(len(v) for v in qrels.values())} linhas, "
          f"{len(qrels)} perguntas")
    print(f"\n{'tipo':16s} {'perguntas':>10s} {'com qrels':>10s} {'trechos':>8s}")
    print("─" * 48)
    for tipo in ("factual", "multi_hop", "comparative", "unanswerable"):
        rs = [r for r in saida if r["question_type"] == tipo]
        cq = [r for r in rs if r["qrels"]]
        print(f"{tipo:16s} {len(rs):10d} {len(cq):10d} {sum(len(r['qrels']) for r in rs):8d}")
    print("─" * 48)
    print(f"{'TOTAL':16s} {len(saida):10d} {sum(1 for r in saida if r['qrels']):10d} "
          f"{sum(len(r['qrels']) for r in saida):8d}")
    print(f"\nlíngua: {dict(Counter(r['source_lang'] for r in saida))}")

    if sem_qrels:
        print(f"\n✗ {len(sem_qrels)} perguntas RESPONDÍVEIS sem qrels: {sem_qrels[:10]}")
        print("  Rode `gerar_qrels.py --aplicar` contra o corpus atual antes de exportar.")
        if args.aplicar:
            print("\n✗ não gravei.")
            return

    if not args.aplicar:
        print("\n(relatório apenas — use --aplicar para gravar)")
        return

    with open(args.saida, "w", encoding="utf-8", newline="\n") as fh:
        for r in saida:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(args.queries, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["QueryId", "Query"])
        for r in saida:
            w.writerow([r["qid"], r["question"]])
    print(f"\n✓ {args.saida}")
    print(f"✓ {args.queries}")
    print("  Aponte eval/config.yaml → paths.golden_qa e paths.queries_csv para estes.")


if __name__ == "__main__":
    main()
