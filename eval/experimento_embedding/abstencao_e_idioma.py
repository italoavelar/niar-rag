#!/usr/bin/env python3
"""Duas medições de graça, sobre dados que já estão em disco.

(1) A NOTA DO RERANKER SERVE DE SINAL DE ABSTENÇÃO?

As 125 perguntas irrespondíveis nunca entraram na avaliação de recuperação — por
definição não têm trecho relevante, então nDCG e recall não se aplicam. Mas o
reranker pontuou TODAS as 392, inclusive elas. Se a melhor nota de uma
irrespondível for sistematicamente mais baixa, existe um limiar de recusa: "se
nem o melhor candidato passa de X, não há resposta no acervo".

Isso é diferente do `score_threshold` que removemos de produção em 26/09. Aquele
era similaridade de cosseno entre pergunta e trecho, sem escala absoluta e nunca
medido. A nota do cross-encoder é de um modelo treinado para julgar relevância do
par, tem escala própria, e aqui ela É medida — contra 125 negativos escritos de
propósito. Se funcionar, é limiar com curva, não palpite.

Reporta AUC (probabilidade de uma respondível sorteada ter nota maior que uma
irrespondível sorteada) e, para alguns limiares, quanto se recusa de cada lado.

(2) DE QUEM É O GANHO DO RERANKER, POR IDIOMA?

O Gemini ganha do BGE+reranker em inglês (0,588 contra 0,544) e perde em
português (0,670 contra 0,766). Se o ganho do reranker estiver concentrado no
português, a leitura é que o empilhamento aberto alcança o proprietário NA
LÍNGUA em que o reranker é forte — e isso muda o que a conclusão pode afirmar
para quem opera em outra língua.

Uso:
  python eval/experimento_embedding/abstencao_e_idioma.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parents[1]

# ACERVO INTEIRO DE PROPÓSITO, não a vista balanceada do artigo. A abstenção se
# estima contra os NEGATIVOS, e o acervo tem 125 irrespondíveis contra as 75 da
# vista balanceada — com metades de 62/63 em vez de 37/38 para escolher e validar
# o limiar. Medido em 30/09: o AUC é o mesmo nas duas (0,846 e 0,845), ou seja o
# sinal não depende da composição; só a escolha do corte fica mais ruidosa com
# menos dados. Por isso o limiar recomendado sai daqui, e não da balanceada.
GABARITO = RAIZ / "eval/data/golden_acervo.jsonl"
SCORES = RAIZ / "eval/results/indexes/rerank_scores_bge_reranker_v2_m3.json"
R = RAIZ / "eval/results/retrieval/rankings"
K = 5

for _f in (sys.stdout, sys.stderr):
    try:
        _f.reconfigure(encoding="utf-8")
    except Exception:
        pass


def auc(positivos: list[float], negativos: list[float]) -> float:
    """Probabilidade de um positivo sorteado exceder um negativo sorteado.
    Calculada por contagem de pares — n é pequeno, não vale aproximar."""
    if not positivos or not negativos:
        return float("nan")
    maior = empate = 0
    for p in positivos:
        for n in negativos:
            if p > n:
                maior += 1
            elif p == n:
                empate += 1
    return (maior + 0.5 * empate) / (len(positivos) * len(negativos))


def main() -> None:
    gold = {g["qid"]: g for g in
            (json.loads(l) for l in GABARITO.open(encoding="utf-8") if l.strip())}
    scores = json.loads(SCORES.read_text(encoding="utf-8"))["scores"]

    # ── (1) abstenção ───────────────────────────────────────────────────────
    resp, irresp = [], []
    for qid, notas in scores.items():
        g = gold.get(qid)
        if not g or not notas:
            continue
        melhor = max(notas.values())
        (resp if g["qrels_grupos"] else irresp).append(melhor)

    print("(1) A NOTA DO RERANKER COMO SINAL DE ABSTENÇÃO\n")
    print(f"{'grupo':16s}{'n':>5s}{'mediana':>10s}{'média':>9s}{'mín':>8s}{'máx':>8s}")
    print("─" * 56)
    for nome, v in (("respondível", resp), ("irrespondível", irresp)):
        s = sorted(v)
        print(f"{nome:16s}{len(v):5d}{s[len(s)//2]:10.4f}"
              f"{sum(v)/len(v):9.4f}{s[0]:8.4f}{s[-1]:8.4f}")
    a = auc(resp, irresp)
    print(f"\n  AUC = {a:.3f}   (0,5 = a nota não distingue nada; 1,0 = separa perfeito)")

    print(f"\n{'limiar':>8s}{'recusa irrespondível':>24s}{'recusa respondível':>22s}")
    print("─" * 56)
    for lim in (0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99):
        ri = sum(1 for x in irresp if x < lim)
        rr = sum(1 for x in resp if x < lim)
        print(f"{lim:8.2f}{ri:14d} de {len(irresp):<6d}{rr:12d} de {len(resp):<6d}")
    print("\n  'recusa irrespondível' é acerto; 'recusa respondível' é o preço.")

    # ── (2) idioma ──────────────────────────────────────────────────────────
    rk = {}
    for nome, arq in (("BGE-m3", "B_dense_bge_m3"), ("Gemini", "B_dense_gemini"),
                      ("BGE+reranker", "B_bge_m3_rerank")):
        p = R / f"{arq}.json"
        if p.exists():
            rk[nome] = json.loads(p.read_text(encoding="utf-8"))

    def junta(ordem, grupos) -> int:
        t = set(ordem[:K])
        return int(all(any(c in t for c in gr) for gr in grupos))

    print("\n\n(2) O GANHO DO RERANKER, POR IDIOMA\n")
    idiomas = ("pt", "en", "mixed")
    alvos = {i: [q for q, g in gold.items()
                 if g["qrels_grupos"] and g["source_lang"] == i
                 and all(q in d for d in rk.values())]
             for i in idiomas}
    cab = f"{'braço':16s}" + "".join(f"{i:>16s}" for i in idiomas)
    print(cab)
    print("─" * len(cab))
    tab = defaultdict(dict)
    for nome, d in rk.items():
        linha = f"{nome:16s}"
        for i in idiomas:
            qs = alvos[i]
            v = sum(junta(d[q], gold[q]["qrels_grupos"]) for q in qs) / max(len(qs), 1)
            tab[nome][i] = v
            linha += f"{v:10.3f} ({len(qs):3d})"
        print(linha)

    if "BGE-m3" in tab and "BGE+reranker" in tab:
        print(f"\n{'ganho do reranker':16s}" +
              "".join(f"{tab['BGE+reranker'][i] - tab['BGE-m3'][i]:>+16.3f}"
                      for i in idiomas))
    if "Gemini" in tab and "BGE+reranker" in tab:
        print(f"{'aberto − Gemini':16s}" +
              "".join(f"{tab['BGE+reranker'][i] - tab['Gemini'][i]:>+16.3f}"
                      for i in idiomas))


if __name__ == "__main__":
    main()
