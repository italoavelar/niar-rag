#!/usr/bin/env python3
"""O top-5 do braço recomendado, empacotado para a etapa de geração.

O QUE A GERAÇÃO RECEBE. Exatamente o que o sistema de produção entregaria a um
gerador: as cinco passagens que o `BGE-m3 + bge-reranker-v2-m3` põe no topo, em
ordem de ranking, com o texto íntegro e a procedência de cada uma. O gerador do
LEME responde SÓ com esses cinco trechos, então este arquivo é o contrato de
entrada dele.

POR QUE VAI O GABARITO JUNTO. Cada registro carrega `citacoes_exigidas` e
`evidencia_completa_no_top5`. Sem isso não se separa falha de geração de falha de
recuperação: numa pergunta cuja evidência não chegou ao top-5, o gerador CERTO é
o que recusa, e puni-lo por não responder mediria a recuperação de novo. São 104
das 225 nessa situação — quase metade, então a distinção não é detalhe.

AS IRRESPONDÍVEIS VÃO EM ARQUIVO SEPARADO. Elas não têm resposta no acervo e o
acerto é recusar. Ficam fora do arquivo principal para não se misturarem com as
respondíveis, e existem porque metade do problema de um RAG jurídico é não
inventar norma.

Uso:
  python eval/experimento_embedding/exportar_top5.py
  python eval/experimento_embedding/exportar_top5.py --k 10 --sem-xlsx
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parents[1]
EVAL = RAIZ / "eval"

GABARITO = EVAL / "data/golden_dezembro.jsonl"
RANKING = EVAL / "results/retrieval/rankings/B_bge_m3_rerank.json"
# O ranking reranqueado acima só cobre as RESPONDÍVEIS (268 do acervo), porque a
# métrica do artigo não usa as outras. Para as irrespondíveis o top-5 é montado
# aqui do mesmo jeito que aquele arquivo foi montado: pool do BGE reordenado
# pelas notas do cross-encoder, que existem para as 393.
POOL_BASE = EVAL / "results/retrieval/rankings/B_dense_bge_m3.json"
NOTAS = EVAL / "results/indexes/rerank_scores_bge_reranker_v2_m3.json"
CORPUS = RAIZ / "data/processed/documents.jsonl"
SAIDA = AQUI / "geracao"

BRACO = "BGE-m3 + bge-reranker-v2-m3 (pool 100)"

for _f in (sys.stdout, sys.stderr):
    try:
        _f.reconfigure(encoding="utf-8")
    except Exception:
        pass


def montar(gold: dict, rank: dict, corpus: dict, k: int) -> tuple[list, list]:
    respondiveis, irrespondiveis = [], []
    for qid in sorted(gold, key=lambda q: (gold[q]["question_type"], q)):
        g = gold[qid]
        if qid not in rank:
            continue
        grupos = g.get("qrels_grupos") or []
        topo = rank[qid][:k]
        if len(topo) < k:
            continue
        ids_topo = set(topo)
        reg = {
            "qid": qid,
            "pergunta": g["question"],
            "tipo": g["question_type"],
            "idioma": g["source_lang"],
            "origem": g["origem"],
            "respondivel": bool(grupos),
            "braco": BRACO,
            "trechos": [],
        }
        for pos, cid in enumerate(topo, start=1):
            d = corpus.get(cid)
            if d is None:
                continue
            m = d.get("metadata") or {}
            reg["trechos"].append({
                "posicao": pos,
                "chunk_id": cid,
                "documento": m.get("document_id"),
                "titulo": m.get("title"),
                "pagina": m.get("page"),
                "secao": m.get("section_path"),
                "texto": d.get("text", ""),
            })
        if grupos:
            reg["resposta_referencia"] = g["reference_answer"]
            reg["citacoes_exigidas"] = g.get("qrels_text") or []
            reg["evidencia_completa_no_top5"] = all(
                any(c in ids_topo for c in gr) for gr in grupos)
            reg["citacoes_que_chegaram"] = sum(
                1 for gr in grupos if any(c in ids_topo for c in gr))
            reg["citacoes_exigidas_n"] = len(grupos)
            respondiveis.append(reg)
        else:
            reg["motivo_da_ausencia"] = g.get("theme") or g.get("difficulty")
            irrespondiveis.append(reg)
    return respondiveis, irrespondiveis


def escrever_xlsx(respondiveis: list, caminho: Path) -> None:
    try:
        import openpyxl
        from openpyxl.styles import Alignment, Font, PatternFill
    except ImportError:
        print("  (openpyxl ausente — xlsx não gerado)")
        return
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "top-5 para geracao"
    cab = ["qid", "tipo", "idioma", "evidência completa no top-5",
           "pergunta", "pos.", "documento", "título", "pág.", "texto do trecho",
           "resposta de referência", "citações exigidas"]
    ws.append(cab)
    for c in range(1, len(cab) + 1):
        cel = ws.cell(row=1, column=c)
        cel.font = Font(bold=True, color="FFFFFF")
        cel.fill = PatternFill("solid", fgColor="10204E")
        cel.alignment = Alignment(wrap_text=True, vertical="center")
    larguras = {"A": 9, "B": 13, "C": 8, "D": 13, "E": 54, "F": 6, "G": 30,
                "H": 34, "I": 7, "J": 78, "K": 54, "L": 54}
    for col, w in larguras.items():
        ws.column_dimensions[col].width = w
    linha = 2
    for reg in respondiveis:
        ini = linha
        for t in reg["trechos"]:
            ws.cell(row=linha, column=6, value=t["posicao"])
            ws.cell(row=linha, column=7, value=t["documento"])
            ws.cell(row=linha, column=8, value=t["titulo"])
            ws.cell(row=linha, column=9, value=t["pagina"])
            cel = ws.cell(row=linha, column=10, value=t["texto"])
            cel.alignment = Alignment(wrap_text=True, vertical="top")
            linha += 1
        if linha == ini:
            continue
        for col, val in ((1, reg["qid"]), (2, reg["tipo"]), (3, reg["idioma"]),
                         (4, "sim" if reg["evidencia_completa_no_top5"] else "NÃO"),
                         (5, reg["pergunta"]),
                         (11, reg.get("resposta_referencia", "")),
                         (12, "\n\n".join(reg.get("citacoes_exigidas", [])))):
            ws.merge_cells(start_row=ini, start_column=col,
                           end_row=linha - 1, end_column=col)
            cel = ws.cell(row=ini, column=col, value=val)
            cel.alignment = Alignment(wrap_text=True, vertical="top")
    ws.freeze_panes = "A2"
    wb.save(caminho)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--sem-xlsx", action="store_true")
    args = ap.parse_args()

    gold = {g["qid"]: g for g in
            (json.loads(l) for l in GABARITO.open(encoding="utf-8") if l.strip())}
    rank = json.loads(RANKING.read_text(encoding="utf-8"))

    # completa as que faltam (as irrespondíveis) reordenando o pool do BGE pelas
    # notas do cross-encoder — a mesma receita do arquivo reranqueado
    faltando = [q for q in gold if q not in rank]
    if faltando:
        pool = json.loads(POOL_BASE.read_text(encoding="utf-8"))
        notas = json.loads(NOTAS.read_text(encoding="utf-8"))["scores"]
        feitas = 0
        for q in faltando:
            cands, n = pool.get(q), notas.get(q)
            if not cands or not n:
                continue
            rank[q] = sorted(cands, key=lambda c: -n.get(c, -1e9))
            feitas += 1
        print(f"  reordenadas aqui: {feitas} de {len(faltando)} ausentes do "
              f"{RANKING.name}")
    corpus = {}
    for linha in CORPUS.open(encoding="utf-8"):
        if linha.strip():
            d = json.loads(linha)
            corpus[d["id"]] = d
    print(f"gabarito: {len(gold)} perguntas · corpus: {len(corpus)} trechos")
    print(f"braço   : {BRACO}\n")

    resp, irresp = montar(gold, rank, corpus, args.k)
    SAIDA.mkdir(parents=True, exist_ok=True)

    p = SAIDA / f"top{args.k}_respondiveis.jsonl"
    p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in resp),
                 encoding="utf-8")
    print(f"✓ {p.name}  {len(resp)} perguntas")

    p2 = SAIDA / f"top{args.k}_irrespondiveis.jsonl"
    p2.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in irresp),
                  encoding="utf-8")
    print(f"✓ {p2.name}  {len(irresp)} perguntas (o acerto é recusar)")

    if not args.sem_xlsx:
        p3 = SAIDA / f"TOP{args.k}_PARA_GERACAO.xlsx"
        escrever_xlsx(resp, p3)
        if p3.exists():
            print(f"✓ {p3.name}  {len(resp)} perguntas, {args.k} linhas cada")

    completas = sum(1 for r in resp if r["evidencia_completa_no_top5"])
    print(f"\nevidência completa no top-{args.k}: {completas} de {len(resp)}"
          f"  ({completas/len(resp):.3f})")
    print(f"  nas outras {len(resp)-completas} o gerador NÃO tem como responder")
    print("  inteiro — o comportamento certo nelas é responder o que dá e dizer")
    print("  o que falta, não completar de memória.")


if __name__ == "__main__":
    main()
