#!/usr/bin/env python3
"""As tabelas do RELATORIO_RECUPERACAO.md, recalculadas da fonte.

Existe porque o relatório cita número. Cada
tabela do relatório sai daqui, com a população impressa ao lado, para que se
possa conferir linha por linha.

Os três braços que ficaram: BGE-m3, Gemini e BGE-m3+reranker. BM25 e fusão RRF
saíram da comparação por decisão de 30/09/2026 — são resultado do artigo do
WebMedia e não são candidatos a produção aqui.

A POPULAÇÃO É ÚNICA: as 267 perguntas respondíveis que os TRÊS braços
ranquearam. Métrica comparada entre braços em populações diferentes é a forma
mais silenciosa de errar uma comparação.

Uso:
  python eval/experimento_embedding/tabelas_relatorio.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from math import comb
from pathlib import Path

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parents[1]

# Padrão = a vista BALANCEADA de 75/75/75, que é a população do artigo desde a
# decisão de 30/09/2026. Para o acervo inteiro (118 factuais), passe
# --gabarito eval/data/golden_acervo.jsonl.
GABARITO = RAIZ / "eval/data/golden_dezembro.jsonl"
R = RAIZ / "eval/results/retrieval/rankings"
CORPUS = RAIZ / "data/processed/documents.jsonl"

BRACOS = {
    "BGE-m3": "B_dense_bge_m3",
    "Gemini": "B_dense_gemini",
    "BGE+reranker": "B_bge_m3_rerank",
}

for _f in (sys.stdout, sys.stderr):
    try:
        _f.reconfigure(encoding="utf-8")
    except Exception:
        pass


def mcnemar_exato(vitorias: int, derrotas: int) -> float:
    """Binomial exato bicaudal sobre os pares DISCORDANTES.

    Os pares concordantes não entram: uma pergunta que os dois braços acertam
    não informa nada sobre qual é melhor. É por isso que n aqui é sempre menor
    que o número de perguntas.
    """
    n = vitorias + derrotas
    if n == 0:
        return 1.0
    k = min(vitorias, derrotas)
    cauda = sum(comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * cauda)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gabarito", default=str(GABARITO))
    args = ap.parse_args()

    caminho = Path(args.gabarito)
    gold = {g["qid"]: g for g in
            (json.loads(l) for l in caminho.open(encoding="utf-8") if l.strip())}
    rk = {nome: json.loads((R / f"{arq}.json").read_text(encoding="utf-8"))
          for nome, arq in BRACOS.items()}
    print(f"GABARITO: {caminho.name}")

    # população comum: respondível E ranqueada pelos três
    pop = sorted(q for q, g in gold.items()
                 if g["qrels_grupos"] and all(q in d for d in rk.values()))
    print(f"POPULAÇÃO: {len(pop)} perguntas respondíveis ranqueadas pelos "
          f"{len(rk)} braços")
    print(f"  (o acervo tem {len(gold)} perguntas; "
          f"{sum(1 for g in gold.values() if not g['qrels_grupos'])} são "
          f"irrespondíveis e não entram aqui)\n")

    def acerto(braco: str, q: str, k: int) -> int:
        """Strict Recall@k: TODAS as citações exigidas no top-k.
        Uma citação conta se QUALQUER trecho que a contenha aparecer."""
        topo = set(rk[braco][q][:k])
        return int(all(any(c in topo for c in gr)
                       for gr in gold[q]["qrels_grupos"]))

    def completude(braco: str, q: str, k: int) -> float:
        """FRAÇÃO das citações exigidas que apareceram — versão graduada do
        Strict Recall. Não confundir com o recall frouxo abaixo: esta dá crédito
        parcial, aquele dá crédito total por uma citação só."""
        topo = set(rk[braco][q][:k])
        gr = gold[q]["qrels_grupos"]
        return sum(any(c in topo for c in g) for g in gr) / len(gr)

    def recall_frouxo(braco: str, q: str, k: int) -> int:
        """1 se AO MENOS UMA citação chegou — o critério que superestima, aqui
        só para contraste. É ele que dava 84% em multi-hop onde o estrito dava
        12%."""
        topo = set(rk[braco][q][:k])
        return int(any(any(c in topo for c in g)
                       for g in gold[q]["qrels_grupos"]))

    # ── tabela 1: os três braços ────────────────────────────────────────────
    print("── TABELA 1 · os três braços ──\n")
    cab = (f"{'braço':16s}{'SR@5':>9s}{'Compl@5':>9s}{'frouxo@5':>10s}"
           f"{'SR@10':>9s}{'SR@100':>9s}")
    print(cab)
    print("─" * len(cab))
    for nome in BRACOS:
        sr5 = sum(acerto(nome, q, 5) for q in pop) / len(pop)
        cp5 = sum(completude(nome, q, 5) for q in pop) / len(pop)
        fr5 = sum(recall_frouxo(nome, q, 5) for q in pop) / len(pop)
        sr10 = sum(acerto(nome, q, 10) for q in pop) / len(pop)
        sr100 = sum(acerto(nome, q, 100) for q in pop) / len(pop)
        print(f"{nome:16s}{sr5:9.3f}{cp5:9.3f}{fr5:10.3f}"
              f"{sr10:9.3f}{sr100:9.3f}")
    print("\n  SR@k = Strict Recall@k (todas as citações). Compl@5 = fração das")
    print("  citações. frouxo@5 = ao menos uma — o número que superestima.")
    print("  SR@100 é o TETO: nenhuma reordenação do")
    print("  pool passa disso. O do BGE+reranker é igual ao do BGE-m3 porque o")
    print("  reranker reordena exatamente o top-100 do BGE — reordenar não")
    print("  acrescenta documento.")

    # ── tabela 2: por idioma, com McNemar do par que decide ─────────────────
    for campo, rotulo, ordem in (
        ("source_lang", "idioma", ("pt", "en", "mixed")),
        ("question_type", "tipo", ("factual", "multi_hop", "comparative")),
    ):
        print(f"\n\n── TABELA · por {rotulo} ──\n")
        grupos = defaultdict(list)
        for q in pop:
            grupos[gold[q][campo]].append(q)
        cab = (f"{'estrato':14s}{'n':>5s}" +
               "".join(f"{n:>16s}" for n in BRACOS) +
               f"{'Gemini×BGE+rr':>18s}{'p':>9s}")
        print(cab)
        print("─" * len(cab))
        for est in ordem:
            qs = grupos.get(est, [])
            if not qs:
                continue
            linha = f"{est:14s}{len(qs):5d}"
            for nome in BRACOS:
                a = sum(acerto(nome, q, 5) for q in qs)
                linha += f"{a:8d} ({a/len(qs):.3f})"
            v = sum(1 for q in qs
                    if acerto("Gemini", q, 5) and not acerto("BGE+reranker", q, 5))
            d = sum(1 for q in qs
                    if acerto("BGE+reranker", q, 5) and not acerto("Gemini", q, 5))
            linha += f"{f'{v} a {d}':>18s}{mcnemar_exato(v, d):9.4f}"
            print(linha)
        print("\n  Cada célula: ACERTOS e (fração). n = perguntas no estrato.")
        print("  'Gemini×BGE+rr' = pares discordantes, na ordem "
              "Gemini-acerta-e-o-outro-não  a  o-inverso.")

    # ── tabela 3: McNemar global ────────────────────────────────────────────
    print("\n\n── TABELA · McNemar global, Strict Recall@5 ──\n")
    print(f"{'comparação':34s}{'vit':>5s}{'der':>5s}{'n disc':>8s}{'p':>10s}")
    print("─" * 62)
    nomes = list(BRACOS)
    for i in range(len(nomes)):
        for j in range(i + 1, len(nomes)):
            a, b = nomes[i], nomes[j]
            v = sum(1 for q in pop if acerto(a, q, 5) and not acerto(b, q, 5))
            d = sum(1 for q in pop if acerto(b, q, 5) and not acerto(a, q, 5))
            p = mcnemar_exato(v, d)
            print(f"{f'{a} vs {b}':34s}{v:5d}{d:5d}{v+d:8d}{p:10.4f}")
    print("\n  vit = a primeira acerta e a segunda não.")

    # ── teto de reordenação ─────────────────────────────────────────────────
    print("\n\n── onde ainda cabe ganho (BGE+reranker) ──\n")
    n_ok = sum(acerto("BGE+reranker", q, 5) for q in pop)
    teto = sum(acerto("BGE+reranker", q, 100) for q in pop)
    no_pool_fora_do_topo = teto - n_ok
    print(f"  acerta no top-5                      : {n_ok:3d} de {len(pop)}"
          f"  ({n_ok/len(pop):.3f})")
    print(f"  teto: tudo que está no pool de 100   : {teto:3d} de {len(pop)}"
          f"  ({teto/len(pop):.3f})")
    print(f"  tem tudo no pool mas não no top-5    : {no_pool_fora_do_topo:3d}"
          f"   ← orçamento da reordenação")
    print(f"  falta material no pool               : {len(pop)-teto:3d}"
          f"   ← falha de recuperação, não de ordem")


if __name__ == "__main__":
    main()
