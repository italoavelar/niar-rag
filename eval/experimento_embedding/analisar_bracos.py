#!/usr/bin/env python3
"""Compara os braços de recuperação sobre as 300 perguntas de dezembro.

POR QUE NÃO USAR SÓ O `metrics.csv` DO PIPELINE. O pipeline calcula
`evidencia_completa@k` com `complete_evidence_at_k`, que exige TODOS os trechos de
grau 2 no top-k. Os nossos qrels são derivados de citação literal, e a
sobreposição de 240 caracteres faz a mesma citação aparecer em dois trechos
vizinhos — os dois entram como grau 2, corretamente, porque qualquer um serve.
Exigir os dois SUBESTIMA o Junta@k em 35 das 225 perguntas respondíveis (15,6%).

Aqui o Junta@k é calculado sobre `qrels_grupos`: um grupo por citação, a citação
vale se QUALQUER trecho do grupo chegou ao top-k, e a pergunta acerta se todas as
citações estão satisfeitas. É a definição que o gerador de qrels já declara
("basta um deles vir no top-k") e que a métrica do pipeline não implementa.

O QUE ESTE SCRIPT NÃO FAZ. nDCG sai daqui pela fórmula padrão sobre o qrels
plano, e sofre a incompletude do gabarito: trecho não julgado conta zero. Para
`Junta@k` e `Recall@k` isso é irrelevante — se a pergunta precisa do trecho X e o
braço trouxe X, é acerto, seja o que for julgado. Por isso a leitura principal é
Junta@5, e o nDCG entra como secundário. Ver DESENHO_DEZEMBRO.md.

Uso:
  python eval/experimento_embedding/analisar_bracos.py
  python eval/experimento_embedding/analisar_bracos.py --k 10
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

AQUI = Path(__file__).resolve().parent
PROJECT_ROOT = AQUI.parents[1]
RANKINGS = PROJECT_ROOT / "eval/results/retrieval/rankings"
# DUAS POPULAÇÕES, de propósito, e esta é a distinção que decide o número.
#
#   golden_dezembro.jsonl   a vista BALANCEADA 75/75/75 (225 respondíveis). É a
#                           população DO ARTIGO, decidida em 30/09/2026, e o
#                           padrão deste script.
#   golden_acervo.jsonl     o acervo inteiro, 393 perguntas / 268 respondíveis.
#                           É o que `paths.golden_qa` aponta, porque o PIPELINE
#                           (02_retrieval_eval, finalize) mede sobre ele.
#
# Não é detalhe de apresentação: o mesmo par de braços dá p = 0,0989 na
# balanceada e p = 0,0055 no acervo inteiro, porque a vantagem do empilhamento
# aberto está concentrada no estrato factual e o balanceamento corta a factual
# de 118 para 75. Rodar sem saber qual população está em uso é como escolher a
# conclusão no escuro — por isso o nome do arquivo e a contagem são impressos
# em toda execução.
#
# O padrão NÃO vem mais do config: o config serve o pipeline, que mede o acervo
# inteiro, e herdar dele fazia este script reportar a população errada para o
# artigo. Use --gabarito para trocar.
GABARITO_ARTIGO = PROJECT_ROOT / "eval/data/golden_dezembro.jsonl"


def _golden() -> Path:
    """O gabarito que o PIPELINE usa — não o do artigo. Só com --pipeline."""
    sys.path.insert(0, str(PROJECT_ROOT / "eval"))
    from lib.common import load_config, resolve
    return resolve(load_config()["paths"]["golden_qa"])
SAIDA = AQUI / "resultados/bracos_dezembro.json"

for _f in (sys.stdout, sys.stderr):
    try:
        _f.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ordem de apresentação; só entram os que têm ranking em disco.
BRACOS = ["A_bm25", "A_bm25_mt", "B_dense_bge_m3", "B_dense_gemini",
          "B_bge_m3_rerank", "B_gemini_rerank", "C_fusion"]

ROTULO = {
    "A_bm25": "BM25",
    "A_bm25_mt": "BM25+trad",
    "B_dense_bge_m3": "BGE-m3",
    "B_dense_gemini": "Gemini",
    "B_bge_m3_rerank": "BGE+rerank",
    "B_gemini_rerank": "Gemini+rerank",
    "C_fusion": "Fusão RRF",
}


# ── métricas ────────────────────────────────────────────────────────────────

def junta(ranked, grupos, k) -> float:
    """1.0 se TODA citação tem ao menos um dos seus trechos no top-k."""
    top = set(ranked[:k])
    return 1.0 if all(any(c in top for c in gr) for gr in grupos) else 0.0


def completude(ranked, grupos, k) -> float:
    """Fração das citações satisfeitas — versão graduada do Junta@k."""
    if not grupos:
        return 0.0
    top = set(ranked[:k])
    return sum(1 for gr in grupos if any(c in top for c in gr)) / len(grupos)


def recall_alguma(ranked, grupos, k) -> float:
    """1.0 se ao menos UMA citação chegou — o critério frouxo, para contraste."""
    top = set(ranked[:k])
    return 1.0 if any(any(c in top for c in gr) for gr in grupos) else 0.0


def ndcg(ranked, rel: dict, k: int) -> float:
    ganhos = [rel.get(c, 0) for c in ranked[:k]]
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(ganhos))
    ideal = sorted(rel.values(), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def mrr(ranked, grupos, k) -> float:
    relevantes = {c for gr in grupos for c in gr}
    for i, c in enumerate(ranked[:k], start=1):
        if c in relevantes:
            return 1.0 / i
    return 0.0


def cobertura_documento(ranked, grupos, k, doc_de) -> float:
    """1.0 se todo documento exigido teve uma citação SUA no top-k.

    Para comparativa é o que decide se os dois lados vieram. Trecho qualquer do
    documento certo não conta — não é ele que sustenta a resposta.
    """
    if not grupos:
        return 0.0
    top = set(ranked[:k])
    exigidos = {doc_de(gr[0]) for gr in grupos if gr}
    cobertos = {doc_de(gr[0]) for gr in grupos if any(c in top for c in gr)}
    return 1.0 if exigidos <= cobertos else 0.0


# ── McNemar exato ───────────────────────────────────────────────────────────

def mcnemar(a: list[float], b: list[float]) -> tuple[int, int, float]:
    """Teste exato bilateral para pares binários. Devolve (vitórias de a,
    vitórias de b, p). Só os pares DISCORDANTES informam: onde os dois acertam
    ou os dois erram não há evidência sobre qual é melhor."""
    vb = sum(1 for x, y in zip(a, b) if x > y)
    vc = sum(1 for x, y in zip(a, b) if y > x)
    n = vb + vc
    if n == 0:
        return vb, vc, 1.0
    menor = min(vb, vc)
    cauda = sum(math.comb(n, i) for i in range(menor + 1)) / (2 ** n)
    return vb, vc, min(1.0, 2 * cauda)


def benjamini_hochberg(ps: list[float]) -> list[float]:
    n = len(ps)
    ordem = sorted(range(n), key=lambda i: ps[i])
    ajust = [0.0] * n
    anterior = 1.0
    for posicao, i in enumerate(reversed(ordem), start=1):
        rank = n - posicao + 1
        valor = min(anterior, ps[i] * n / rank)
        ajust[i] = valor
        anterior = valor
    return ajust


# ── principal ───────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--bracos", default=None,
                    help="lista separada por vírgula; padrão = todos com ranking. "
                         "Restringir importa para a CORREÇÃO: com dois braços há um "
                         "teste só, e corrigir para múltiplos testes que não foram "
                         "feitos é punir a comparação por companhia que ela não tem.")
    ap.add_argument("--gabarito", default=None,
                    help="jsonl a analisar; padrão = a vista balanceada do "
                         "artigo (golden_dezembro.jsonl)")
    ap.add_argument("--pipeline", action="store_true",
                    help="usa paths.golden_qa do config — o acervo inteiro, que "
                         "é o que o pipeline mede. NÃO é a população do artigo.")
    ap.add_argument("--aplicar", action="store_true")
    args = ap.parse_args()
    k = args.k

    if args.gabarito:
        caminho = Path(args.gabarito)
    elif args.pipeline:
        caminho = _golden()
    else:
        caminho = GABARITO_ARTIGO
    gold = {g["qid"]: g for g in
            (json.loads(l) for l in caminho.open(encoding="utf-8") if l.strip())}
    doc_por_chunk, ordem = {}, {}
    with (PROJECT_ROOT / "data/processed/documents.jsonl").open(encoding="utf-8") as fh:
        for i, linha in enumerate(fh):
            if linha.strip():
                r = json.loads(linha)
                doc_por_chunk[r["id"]] = r["metadata"]["document_id"]
                ordem[r["id"]] = i

    # DISPERSÃO calculada aqui, uniformemente, para TODAS as famílias. O campo
    # `dispersao_trechos` do acervo só existe nas famílias geradas depois de
    # 27/09; cortar por um campo que metade das perguntas não tem produziria um
    # estrato "None" que é procedência disfarçada de medida.
    def faixa_dispersao(grupos) -> str:
        if len(grupos) < 2:
            return "1 citação"
        pos = [min(ordem[c] for c in gr) for gr in grupos if gr]
        d = max(pos) - min(pos)
        if d == 0:
            return "d=0 (mesmo trecho)"
        if d == 1:
            return "d=1 (vizinho)"
        if d <= 20:
            return "d=2..20"
        return "d>20 (disperso)"
    doc_de = lambda cid: doc_por_chunk.get(cid, cid)  # noqa: E731

    pedidos = [b.strip() for b in args.bracos.split(",")] if args.bracos else None
    disponiveis = [b for b in BRACOS if (RANKINGS / f"{b}.json").exists()
                   and (pedidos is None or b in pedidos)]
    rankings = {}
    for b in disponiveis:
        rankings[b] = json.loads((RANKINGS / f"{b}.json").read_text(encoding="utf-8"))

    respondiveis = [q for q, g in gold.items() if g["qrels_grupos"]]
    faltando = {b: [q for q in respondiveis if q not in rankings[b]] for b in disponiveis}

    # VALIDAÇÃO DOS IDS DE TRECHO, e não só da cobertura de perguntas.
    #
    # Em 31/08 os sete rankings em disco ficaram mortos de uma vez: os ids dos
    # trechos deixaram de existir depois da migração para id endereçado por
    # conteúdo (12/09) e da limpeza do corpus (22/09). O ranking continua um
    # JSON válido, com todas as perguntas no lugar — e marca 0,000 em tudo,
    # porque nenhum id casa. Zero com cara de resultado é o pior defeito
    # possível numa tabela comparativa.
    #
    # A checagem de cobertura de perguntas, logo acima, só pegava esse caso por
    # acidente: o A_bm25_mt sobrevivia porque tinha os qids do piloto antigo, e
    # era descartado por faltarem perguntas, não por ter ids mortos. Se um dia
    # um ranking morto tivesse os qids certos, ele entraria na comparação
    # pesando 0,000 contra os braços vivos.
    LIMITE_IDS_VIVOS = 0.95
    vivos = {}
    for b in disponiveis:
        ids = {c for ordem in rankings[b].values() for c in ordem[:20]}
        vivos[b] = (sum(1 for c in ids if c in doc_por_chunk) / len(ids)) if ids else 0.0

    print(f"gabarito : {caminho.name} — {len(gold)} perguntas, {len(respondiveis)} respondíveis")
    print(f"braços   : {len(disponiveis)} com ranking em disco\n")
    for b in disponiveis:
        n = len(faltando[b])
        avisos = []
        if n:
            avisos.append(f"✗ {n} perguntas sem ranking")
        if vivos[b] < LIMITE_IDS_VIVOS:
            avisos.append(f"✗ só {vivos[b]:.0%} dos ids existem no corpus — MORTO")
        sufixo = ("  " + " · ".join(avisos)) if avisos else ""
        print(f"  {ROTULO[b]:14s} {len(rankings[b]):4d} consultas{sufixo}")

    validos = [b for b in disponiveis
               if not faltando[b] and vivos[b] >= LIMITE_IDS_VIVOS]
    if len(validos) < len(disponiveis):
        print("\n  Braços fora da comparação: ranking incompleto ou com ids que não")
        print("  existem mais no corpus. Rode-os de novo contra o corpus atual —")
        print("  NÃO compare um braço morto, ele marca 0,000 e parece resultado.")
    if not validos:
        print("\n✗ nenhum braço válido. Nada a comparar.")
        return
    print()

    # ── por pergunta ────────────────────────────────────────────────────────
    por_braco: dict[str, dict[str, list]] = {}
    for b in validos:
        m = defaultdict(list)
        for q in respondiveis:
            g, r = gold[q], rankings[b][q]
            gr = g["qrels_grupos"]
            m["junta"].append(junta(r, gr, k))
            m["compl"].append(completude(r, gr, k))
            m["recall"].append(recall_alguma(r, gr, k))
            m["ndcg"].append(ndcg(r, g["qrels"], k))
            m["mrr"].append(mrr(r, gr, 10))
            m["cobdoc"].append(cobertura_documento(r, gr, k, doc_de))
        por_braco[b] = m

    med = lambda v: sum(v) / len(v) if v else 0.0  # noqa: E731

    print(f"══ Geral, {len(respondiveis)} perguntas respondíveis, k={k} ══\n")
    print(f"{'braço':16s} {'Junta@k':>8s} {'Compl@k':>8s} {'Recall@k':>9s} "
          f"{'nDCG@k':>7s} {'MRR@10':>7s} {'CobDoc':>7s}")
    print("─" * 68)
    for b in sorted(validos, key=lambda x: -med(por_braco[x]["junta"])):
        m = por_braco[b]
        print(f"{ROTULO[b]:16s} {med(m['junta']):8.3f} {med(m['compl']):8.3f} "
              f"{med(m['recall']):9.3f} {med(m['ndcg']):7.3f} {med(m['mrr']):7.3f} "
              f"{med(m['cobdoc']):7.3f}")

    # ── cortes ──────────────────────────────────────────────────────────────
    for q in respondiveis:
        gold[q]["faixa_dispersao"] = faixa_dispersao(gold[q]["qrels_grupos"])

    for campo, titulo in (("question_type", "tipo"), ("source_lang", "idioma"),
                          ("faixa_dispersao", "dispersão da evidência"),
                          ("origem", "procedência")):
        grupos = sorted({gold[q][campo] for q in respondiveis})
        print(f"\n══ Junta@{k} por {titulo} ══\n")
        print(f"{'braço':16s} " + " ".join(f"{g[:13]:>14s}" for g in grupos))
        print("─" * (17 + 15 * len(grupos)))
        for b in sorted(validos, key=lambda x: -med(por_braco[x]["junta"])):
            linha = f"{ROTULO[b]:16s} "
            for g in grupos:
                idx = [i for i, q in enumerate(respondiveis) if gold[q][campo] == g]
                v = [por_braco[b]["junta"][i] for i in idx]
                linha += f"{med(v):9.3f} ({len(v):3d}) "
            print(linha)

    # ── significância ───────────────────────────────────────────────────────
    print(f"\n══ McNemar exato em Junta@{k} ══\n")
    pares, ps = [], []
    for i, a in enumerate(validos):
        for b in validos[i + 1:]:
            vb, vc, p = mcnemar(por_braco[a]["junta"], por_braco[b]["junta"])
            pares.append((a, b, vb, vc, p))
            ps.append(p)
    ajust = benjamini_hochberg(ps) if ps else []
    bonf = [min(1.0, p * len(ps)) for p in ps]
    print(f"{'comparação':32s} {'vit':>5s} {'der':>5s} {'p':>8s} {'BH':>8s} {'Bonf':>8s}")
    print("─" * 72)
    for (a, b, vb, vc, p), pbh, pbo in sorted(zip(pares, ajust, bonf),
                                              key=lambda x: x[0][4]):
        marca = "*" if pbh < 0.05 else " "
        print(f"{ROTULO[a]+' vs '+ROTULO[b]:32s} {vb:5d} {vc:5d} "
              f"{p:8.4f} {pbh:8.4f}{marca} {pbo:8.4f}")
    print("\n  vit/der = pares discordantes. * = sobrevive a Benjamini-Hochberg.")

    if args.aplicar:
        SAIDA.parent.mkdir(parents=True, exist_ok=True)
        SAIDA.write_text(json.dumps({
            "k": k,
            "n_respondiveis": len(respondiveis),
            "bracos": {b: {met: med(v) for met, v in m.items()}
                       for b, m in por_braco.items()},
            "por_pergunta": {b: {q: por_braco[b]["junta"][i]
                                 for i, q in enumerate(respondiveis)}
                             for b in validos},
            "mcnemar": [{"a": a, "b": b, "vitorias_a": vb, "vitorias_b": vc,
                         "p": p, "p_bh": pbh}
                        for (a, b, vb, vc, p), pbh in zip(pares, ajust)],
        }, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\n✓ {SAIDA}")


if __name__ == "__main__":
    main()
