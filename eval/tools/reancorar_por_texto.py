#!/usr/bin/env python3
"""Reancora o gabarito depois de uma reextração do corpus, casando por TEXTO.

Quando rodar
────────────
Depois de mexer no acervo — remover documento, recoletar, mudar `MIN_CHUNK_SIZE`
ou `TABLE_CHUNK_SIZE` — e reexecutar `src/extract_to_jsonl.py` /
`src/extract_html_to_jsonl.py`. É o passo que mantém as anotações de pé sem
ninguém reanotar nada.

Por que funciona
────────────────
O id de um trecho é `<document_id>_<sha256 do texto>` (`src/chunk_id.py`), então:

  • **remover um documento não toca os ids dos outros.** O hash depende só do
    texto daquele trecho. Só somem os ids do documento removido;
  • mudar o recorte muda o id **apenas dos trechos cujo texto mudou**;
  • e `golden_qa.jsonl` guarda `qrels_text`, o texto de cada âncora. Quem tem o
    texto reancora por conteúdo, sem depender do id.

O que acontece com cada âncora
──────────────────────────────
  id ainda existe                  → nada a fazer
  texto contido em 1 trecho novo   → remapeia (o trecho cresceu por fusão)
  texto contém 1 trecho novo       → remapeia (o trecho encolheu por corte)
  texto casa com vários            → AMBÍGUA, fica para revisão humana
  documento inteiro sumiu          → pergunta marcada como inválida
  nada casa                        → PERDIDA, fica para revisão humana

Perguntas inválidas **não são apagadas**. Elas continuam no `golden_qa.jsonl`
com `status` e `status_motivo`, e ficam de fora do `qrels.csv` — que é o que a
engine de recuperação lê. Nada de histórico se perde, e o número de perguntas
efetivas do benchmark passa a ser auditável.

Uso
───
  python eval/tools/reancorar_por_texto.py              # simulação (padrão)
  python eval/tools/reancorar_por_texto.py --aplicar    # escreve, com backup
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
EVAL = PROJECT_ROOT / "eval"
sys.path.insert(0, str(PROJECT_ROOT / "src"))

for _fluxo in (sys.stdout, sys.stderr):
    try:
        _fluxo.reconfigure(encoding="utf-8")
    except Exception:
        pass

from chunk_id import normalizar_texto  # noqa: E402

# Marcação estrutural que o empacotador põe e tira: o invólucro da tabela e a
# linha de colunas, repetida em cada pedaço quando a tabela é partida. Quando um
# recorte junta duas metades de tabela, o texto antigo deixa de ser substring do
# novo — não porque o conteúdo mudou, mas porque o `[/TABELA]` do meio sumiu e o
# cabeçalho deixou de se repetir. Comparar sem o invólucro compara conteúdo.
_MARCACAO = re.compile(r"\[/?TABELA\]|Colunas:[^\n]*\n?|Nota:[^\n]*")


def so_conteudo(texto: str) -> str:
    return normalizar_texto(_MARCACAO.sub(" ", texto or ""))


# A âncora guardada em `qrels_text` é uma FOTO do texto no dia em que foi
# julgada — com o ruído de extração que o corpus tinha naquele dia. Quando a
# limpeza remove esse ruído, o texto antigo deixa de ser substring do novo e a
# âncora se perde, embora o conteúdo seja o mesmo. Medido em 22/set: 48 linhas
# de qrels e 10 perguntas sumiriam por isso.
#
# Limpar a âncora com as MESMAS regras antes de comparar resolve. As regras
# moram em eval/experimento_embedding/limpar.py; o lugar natural delas é src/,
# e é para lá que devem ir quando alguém mexer nisto de novo.
sys.path.insert(0, str(EVAL / "experimento_embedding"))
try:
    from limpar import documento_com_numeracao_de_linha as _doc_numerado
    from limpar import limpar as _limpar_corpus
except Exception:  # pragma: no cover - a reancoragem funciona sem esta passada
    _limpar_corpus = _doc_numerado = None


def como_o_corpus_limpo(texto: str, numerado: bool = False) -> str:
    """A âncora antiga passada pelas mesmas regras que limparam o corpus.

    `numerado` precisa vir de fora: remover numeração de linha é decisão do
    DOCUMENTO, e limpar a âncora sem ela não reproduz o que aconteceu com o
    corpus. Foi o que derrubou 22 âncoras e 7 perguntas na primeira tentativa —
    as do `ai_device_software_guidance_FDA_2025`, cujo texto guardado ainda
    tinha os números de linha que o corpus já não tem.
    """
    if not texto or _limpar_corpus is None:
        return ""
    return normalizar_texto(_limpar_corpus(texto, numerado=numerado)[0])

CORPUS = PROJECT_ROOT / "data/processed/documents.jsonl"
QRELS = EVAL / "data/qrels.csv"
GOLDEN = EVAL / "data/golden_qa.jsonl"

OK = "ok"
DOC_REMOVIDO = "invalida_documento_removido"
REVISAR = "revisar"


def carregar_corpus(caminho: Path) -> list[dict]:
    with caminho.open(encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def backup(caminho: Path, carimbo: str) -> None:
    shutil.copy2(caminho, caminho.with_suffix(caminho.suffix + f".bak-pre-reancoragem-{carimbo}"))


def main() -> None:
    ap = argparse.ArgumentParser(description="Reancora o gabarito por casamento de texto.")
    ap.add_argument("--aplicar", action="store_true", help="escreve (sem isto, só simula)")
    ap.add_argument("--corpus", type=Path, default=CORPUS,
                    help="corpus candidato, para conferir o custo ANTES de trocar o de produção")
    args = ap.parse_args()
    carimbo = datetime.now().strftime("%Y%m%d-%H%M%S")

    if args.aplicar and args.corpus != CORPUS:
        raise SystemExit("--aplicar só com o corpus de produção; --corpus é para simular.")

    corpus = carregar_corpus(args.corpus)
    ids_novos = {r["id"] for r in corpus}
    docs_novos = {r["metadata"]["document_id"] for r in corpus}
    por_doc: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for r in corpus:
        por_doc[r["metadata"]["document_id"]].append(
            (r["id"], normalizar_texto(r["text"]), so_conteudo(r["text"])))

    gold = [json.loads(l) for l in GOLDEN.open(encoding="utf-8") if l.strip()]

    print("── Reancoragem do gabarito por casamento de texto ──")
    print(f"modo: {'APLICAR' if args.aplicar else 'SIMULAÇÃO'}")
    print(f"corpus novo: {len(corpus)} trechos, {len(docs_novos)} documentos")
    print(f"gabarito: {len(gold)} perguntas\n")

    contagem = {"intacta": 0, "remapeada": 0, "ambigua": 0, "perdida": 0, "doc_removido": 0}
    pendencias: list[tuple[str, str, str]] = []

    for reg in gold:
        qrels = reg.get("qrels") or {}
        textos = reg.get("qrels_text") or {}
        if not qrels:                                   # unanswerable: nada a reancorar
            reg["status"] = OK
            continue

        novos_qrels: dict[str, int] = {}
        novos_textos: dict[str, str] = {}
        status, motivos = OK, []

        for cid, grau in qrels.items():
            # O document_id é o prefixo do id até o hash final. Preferir o texto
            # guardado; o id é só o ponto de partida da busca.
            doc = cid.rsplit("_", 1)[0]
            alvo = normalizar_texto(textos.get(cid, ""))

            if cid in ids_novos:
                contagem["intacta"] += 1
                novos_qrels[cid] = grau
                novos_textos[cid] = textos.get(cid, "")
                continue

            if doc not in docs_novos:
                contagem["doc_removido"] += 1
                status = DOC_REMOVIDO
                motivos.append(f"documento {doc} não está mais no acervo")
                continue

            if not alvo:
                contagem["perdida"] += 1
                status = REVISAR
                motivos.append(f"âncora {cid} sem qrels_text — impossível reancorar")
                pendencias.append((reg["qid"], cid, "sem qrels_text"))
                continue

            contidos = [c for c, t, _ in por_doc[doc] if alvo in t]
            contem = [c for c, t, _ in por_doc[doc] if t and t in alvo]
            candidatos = contidos or contem
            if not candidatos:
                # 2a passada: ignora a marcação estrutural de tabela, que o
                # empacotador reescreve quando duas metades voltam a ser uma.
                nu = so_conteudo(textos.get(cid, ""))
                candidatos = [c for c, _, sc in por_doc[doc] if nu and nu in sc]

            if not candidatos:
                # 3a passada: a âncora ainda traz o ruído que a limpeza tirou do
                # corpus. Passa a âncora pelas mesmas regras e compara de novo.
                # Tenta as DUAS formas de limpar a ancora: sem e com remocao de
                # numeracao de linha. Detectar o documento aqui nao funciona --
                # o corpus ja foi limpo, entao a numeracao ja sumiu dele e a
                # densidade nao acusa mais nada. Tentar as duas nao arrisca:
                # remover numeracao exige uma corrida de 4+ inteiros seguidos,
                # que prosa normativa nao tem, e a primeira tentativa que casar
                # vence.
                for numerado in (False, True):
                    lc = como_o_corpus_limpo(textos.get(cid, ""), numerado)
                    candidatos = [c for c, t, _ in por_doc[doc] if lc and lc in t]
                    if not candidatos:
                        candidatos = [c for c, t, _ in por_doc[doc] if lc and t and t in lc]
                    if candidatos:
                        break

            if len(candidatos) == 1:
                contagem["remapeada"] += 1
                novo = candidatos[0]
                novos_qrels[novo] = max(grau, novos_qrels.get(novo, 0))
                novos_textos[novo] = next(t for c, t, _ in por_doc[doc] if c == novo)
            elif len(candidatos) > 1:
                contagem["ambigua"] += 1
                status = REVISAR
                motivos.append(f"âncora {cid} casa com {len(candidatos)} trechos novos")
                pendencias.append((reg["qid"], cid, f"ambígua ({len(candidatos)} candidatos)"))
            else:
                contagem["perdida"] += 1
                status = REVISAR
                motivos.append(f"âncora {cid} não foi encontrada no corpus novo")
                pendencias.append((reg["qid"], cid, "não encontrada"))

        if status == OK and len(novos_qrels) < len({*qrels}):
            # Fusão colapsou duas âncoras numa só. Não é erro, mas muda a
            # contagem de evidência exigida — precisa ser visto.
            motivos.append(
                f"{len(qrels)} âncoras viraram {len(novos_qrels)} (fusão de trechos)"
            )
        reg["qrels"] = novos_qrels
        reg["qrels_text"] = novos_textos
        reg["status"] = status
        if motivos:
            reg["status_motivo"] = "; ".join(motivos)
        elif "status_motivo" in reg:
            del reg["status_motivo"]

    print("── Âncoras ──")
    for rotulo, chave in (
        ("id inalterado", "intacta"),
        ("remapeada por texto", "remapeada"),
        ("ambígua (revisão humana)", "ambigua"),
        ("perdida (revisão humana)", "perdida"),
        ("documento removido", "doc_removido"),
    ):
        print(f"  {rotulo:<28}{contagem[chave]:>6}")

    por_status: dict[str, list[str]] = defaultdict(list)
    for reg in gold:
        por_status[reg.get("status", OK)].append(reg["qid"])
    print("\n── Perguntas ──")
    for st in (OK, DOC_REMOVIDO, REVISAR):
        qs = por_status.get(st, [])
        if not qs:
            continue
        print(f"  {st:<32}{len(qs):>4}")
        if st != OK:
            print(f"      {', '.join(sorted(qs)[:14])}{' …' if len(qs) > 14 else ''}")
    mudou = [r for r in gold if r.get("status_motivo") and r.get("status") == OK]
    if mudou:
        print(f"\n  {len(mudou)} pergunta(s) válidas com observação — conferir:")
        for r in mudou[:10]:
            print(f"      {r['qid']}: {r['status_motivo']}")

    if pendencias:
        print(f"\n⚠ {len(pendencias)} âncora(s) exigem decisão humana:")
        for qid, cid, motivo in pendencias[:12]:
            print(f"      {qid}  {cid}  → {motivo}")

    if not args.aplicar:
        print("\n(simulação — rode com --aplicar para escrever)")
        return

    backup(GOLDEN, carimbo)
    with GOLDEN.open("w", encoding="utf-8") as fh:
        for reg in gold:
            fh.write(json.dumps(reg, ensure_ascii=False) + "\n")
    print(f"\n✓ {GOLDEN.name}")

    # qrels.csv só com o que a engine deve avaliar: perguntas válidas e com âncora.
    backup(QRELS, carimbo)
    linhas = [
        {"QueryId": r["qid"], "ChunkId": cid, "Relevance": grau}
        for r in gold
        if r.get("status") == OK
        for cid, grau in (r.get("qrels") or {}).items()
    ]
    with QRELS.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["QueryId", "ChunkId", "Relevance"])
        w.writeheader()
        w.writerows(linhas)
    avaliaveis = len({l["QueryId"] for l in linhas})
    print(f"✓ {QRELS.name}  ({len(linhas)} linhas, {avaliaveis} perguntas avaliáveis)")

    orfas = sum(1 for l in linhas if l["ChunkId"] not in ids_novos)
    print(f"\nverificação: {orfas} linha(s) de qrels apontando para fora do corpus")
    print("Próximos passos: reembutir os trechos novos e atualizar o payload do Qdrant")
    print("(eval/tools/migrar_ids_qdrant.py).")


if __name__ == "__main__":
    main()
