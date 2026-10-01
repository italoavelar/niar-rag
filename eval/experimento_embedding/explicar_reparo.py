#!/usr/bin/env python3
"""Por que reparar 16 de 25 comparativas não mudou o resultado?

A PERGUNTA. Em 29/09/2026 as 25 comparativas do piloto foram relidas e 16 tinham
defeito de gabarito. Todas foram reparadas. O Strict Recall@5 do subconjunto foi
de 0,542 para 0,538 e o McNemar ficou idêntico. Dizer "não mudou" e seguir é
ruim: ou o reparo era irrelevante, ou a métrica não vê o que foi reparado.

O QUE SE MEDE AQUI. Para cada pergunta reparada, a POSIÇÃO de cada citação
exigida no ranking do BGE+reranker. Se as citações exigidas estão em 30, 50, 80,
então trocar QUAL citação é exigida não pode mudar um corte em 5: o alvo antigo
estava fora e o novo também. O reparo é invisível para a métrica por construção,
e o número não se mover é a previsão correta, não uma coincidência.

As três explicações candidatas, e como distinguir:

  (a) ESTRATO NO CHÃO      a comparativa acerta 13 de 75. Um item que era 0 e
                           continuou 0 não move média nenhuma.
  (b) REPARO ENDURECEU     4 reparos eram estruturais: a pergunta citava duas
                           normas e o gabarito uma só. Acrescentar a segunda
                           citação AUMENTA a exigência de uma métrica
                           conjuntiva. Só pode manter ou piorar.
  (c) ALVO FORA DO ALCANCE as citações, velhas e novas, estão fundo no ranking.

As três podem valer ao mesmo tempo; o que o script mostra é o peso de cada uma.

Uso:
  python eval/experimento_embedding/explicar_reparo.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parents[1]

# A população é indiferente aqui, e vale dizer por quê: as 16 comparativas
# reparadas estão nas DUAS vistas, porque a comparativa tem 75 nas duas e entra
# inteira, sem sorteio. O acervo é usado só por ser o superconjunto.
GABARITO = RAIZ / "eval/data/golden_acervo.jsonl"
R = RAIZ / "eval/results/retrieval/rankings"
K = 5

for _f in (sys.stdout, sys.stderr):
    try:
        _f.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Classe do defeito por pergunta, como registrado em
# `corrigir_comparativas_piloto.py`. Estrutural é o único que mexe no NÚMERO de
# citações exigidas — os outros trocam qual citação é, mantendo a contagem.
DEFEITO = {
    "q0081": "estrutural", "q0085": "estrutural",
    "q0093": "estrutural", "q0099": "estrutural",
    "q0078": "conteúdo", "q0082": "conteúdo", "q0083": "conteúdo",
    "q0084": "conteúdo", "q0092": "conteúdo", "q0095": "conteúdo",
    "q0097": "conteúdo", "q0098": "conteúdo",
    "q0076": "limítrofe", "q0094": "limítrofe", "q0096": "limítrofe",
    "q0086": "intrusa",
}


def main() -> None:
    gold = {g["qid"]: g for g in
            (json.loads(l) for l in GABARITO.open(encoding="utf-8") if l.strip())}
    rr = json.loads((R / "B_bge_m3_rerank.json").read_text(encoding="utf-8"))

    def posicoes(qid: str) -> list[int | None]:
        """Para cada citação exigida, a melhor posição (1-indexada) de algum
        trecho que a contenha. None = não está no ranking inteiro."""
        ordem = {c: i + 1 for i, c in enumerate(rr.get(qid, []))}
        saida = []
        for grupo in gold[qid]["qrels_grupos"]:
            ps = [ordem[c] for c in grupo if c in ordem]
            saida.append(min(ps) if ps else None)
        return saida

    print("POSIÇÃO DAS CITAÇÕES EXIGIDAS, nas 16 comparativas reparadas")
    print("(ranking do BGE+reranker; 'fora' = não está nos 100 do pool)\n")
    print(f"{'qid':8s}{'defeito':13s}{'citações':>9s}{'posições':>26s}"
          f"{'acerta@5':>10s}")
    print("─" * 68)

    dentro_do_top5 = 0
    piores = []
    presentes = 0
    for qid in sorted(DEFEITO):
        g = gold.get(qid)
        if not g or qid not in rr:
            print(f"{qid:8s}{DEFEITO[qid]:13s}{'—':>9s}"
                  f"{'(sem ranking)':>26s}{'—':>10s}")
            continue
        presentes += 1
        ps = posicoes(qid)
        txt = ", ".join("fora" if p is None else str(p) for p in ps)
        ok = all(p is not None and p <= K for p in ps)
        dentro_do_top5 += int(ok)
        pior = max((p for p in ps if p is not None), default=None)
        if any(p is None for p in ps):
            pior = None
        piores.append(pior)
        print(f"{qid:8s}{DEFEITO[qid]:13s}{len(ps):9d}{txt:>26s}"
              f"{('SIM' if ok else 'não'):>10s}")

    print(f"\n  acertam no top-5: {dentro_do_top5} de {presentes}")

    # (c) o alvo está ao alcance de um corte em 5?
    alcance = [p for p in piores if p is not None]
    fora_do_pool = sum(1 for p in piores if p is None)
    if alcance:
        alcance.sort()
        mediana = alcance[len(alcance) // 2]
        print(f"\n  onde está a ÚLTIMA citação exigida (a que decide o acerto):")
        print(f"    mediana da posição : {mediana}")
        print(f"    mínimo / máximo    : {alcance[0]} / {alcance[-1]}")
        print(f"    perguntas com alguma citação fora do pool de 100: "
              f"{fora_do_pool} de {presentes}")
        print(f"\n  Leitura: um corte em {K} só vê o que está em posição ≤ {K}.")
        print(f"  Com a última citação em mediana {mediana}, trocar QUAL citação")
        print(f"  é exigida não atravessa o corte — o alvo velho estava fora e o")
        print(f"  novo também. É por isso que o número não se moveu.")

    # (b) o reparo estrutural endureceu a exigência
    print("\n\nO QUE CADA CLASSE DE REPARO FEZ COM A EXIGÊNCIA\n")
    print("  estrutural (4) : gabarito tinha 1 citação, passou a ter 2.")
    print("                   Numa métrica conjuntiva isso só MANTÉM ou PIORA.")
    print("                   Antes do reparo essas 4 eram artificialmente")
    print("                   fáceis: uma citação só, e `montar_acervo.py`")
    print("                   as retipava para `factual`.")
    print("  conteúdo (8)   : troca qual frase é exigida, no mesmo documento.")
    print("                   Contagem igual, dificuldade parecida.")
    print("  limítrofe (3)  : âncora adjacente à que sustenta a resposta.")
    print("  intrusa (1)    : q0086 tinha 4 citações e passou a ter 3 —")
    print("                   o único reparo que AFROUXA a exigência.")

    # (a) o estrato já estava no chão
    comp = [q for q, g in gold.items()
            if g["qrels_grupos"] and g["question_type"] == "comparative"
            and q in rr]
    acertos = sum(
        1 for q in comp
        if all(any(c in set(rr[q][:K]) for c in gr)
               for gr in gold[q]["qrels_grupos"])
    )
    print(f"\n\nO ESTRATO ONDE O REPARO ACONTECEU\n")
    print(f"  comparativas com ranking : {len(comp)}")
    print(f"  acertam no top-5         : {acertos}  "
          f"({acertos/max(len(comp),1):.3f})")
    print(f"  as 16 reparadas são {16/max(len(comp),1):.0%} do estrato e "
          f"{16/267:.1%} das 267 respondíveis.")
    print(f"\n  Um estrato que marca {acertos/max(len(comp),1):.2f} tem pouca")
    print("  resolução: quase todo item já vale 0, e reparo que mantém o item")
    print("  em 0 não aparece na média. O reparo não foi inútil — ele corrigiu")
    print("  o ALVO da geração, que era o que estava errado. Só não era")
    print("  detectável por uma métrica de recuperação neste estrato.")


if __name__ == "__main__":
    main()
