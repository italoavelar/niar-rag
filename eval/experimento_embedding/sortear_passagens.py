#!/usr/bin/env python3
"""Sorteia as passagens a partir das quais as perguntas do experimento serão escritas.

O PROBLEMA QUE ISTO RESOLVE. A pergunta nasce com o tamanho da passagem que a
gerou. Se todas saírem de passagens de 1.200 caracteres, o gabarito favorece o
recorte de 1.200 — que é exatamente uma das condições sob teste. O gabarito atual
tem esse defeito e é por isso que ele não serve para decidir tamanho.

A SAÍDA NÃO É ACHAR A JANELA NEUTRA — ela não existe, qualquer janela tem um
tamanho. A saída é **variar a janela e registrar qual gerou cada pergunta**:

    600, 1400 e 3000 caracteres, em proporção igual

Nenhum tamanho testado (400, 1200, 3500) coincide com uma janela de geração, e o
conjunto vem de uma mistura. Melhor ainda: como cada pergunta carrega a janela
que a gerou, o viés residual vira MEDIDA — dá para testar se a vantagem de um
recorte se concentra nas perguntas de janela parecida. Se concentrar, está
declarado; se não, o resultado é robusto.

Isso transforma uma limitação escondida em controle reportado.

A ÂNCORA NÃO É A PASSAGEM. O que fica anotado é a citação literal que o gerador
devolve, validada contra o texto. Citação é fato e sobrevive a qualquer recorte —
a menos que o recorte a parta ao meio, que é o defeito a medir.

Uso:
  python eval/experimento_embedding/sortear_passagens.py --n 100
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
PROJECT_ROOT = AQUI.parents[1]
sys.path.insert(0, str(AQUI))

for _f in (sys.stdout, sys.stderr):
    try:
        _f.reconfigure(encoding="utf-8")
    except Exception:
        pass

AMOSTRA = AQUI / "dados/amostra.json"
SAIDA = AQUI / "dados/passagens.json"
JANELAS = (600, 1400, 3000)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=100, help="passagens a sortear")
    ap.add_argument("--semente", type=int, default=2026)
    args = ap.parse_args()

    import bench_tamanho as B

    amostra = json.loads(AMOSTRA.read_text(encoding="utf-8"))["documentos"]
    por_doc = B.carregar_corpus()
    inteiros = {d: B.reconstruir(por_doc[d]) for d in amostra if d in por_doc}
    print(f"documentos: {len(inteiros)}")

    rng = random.Random(args.semente)
    # Reparte as vagas entre documentos e janelas, para que nenhum documento
    # domine e as três janelas fiquem equilibradas.
    docs = sorted(inteiros)
    plano = [(docs[i % len(docs)], JANELAS[i % len(JANELAS)]) for i in range(args.n)]
    rng.shuffle(plano)

    passagens, usados = [], set()
    for i, (doc, janela) in enumerate(plano, 1):
        texto = inteiros[doc]
        if len(texto) < janela + 200:
            continue
        # Fronteiras de frase: a passagem não deve começar no meio de uma palavra.
        # Frase não é estratégia sob teste, então não favorece nenhum recorte.
        cortes = [0] + [m.end() for m in re.finditer(r"[.!?]\s", texto)] + [len(texto)]
        for _ in range(30):
            inicio = rng.randrange(0, max(1, len(texto) - janela))
            ini = min(cortes, key=lambda x: abs(x - inicio))
            fim = min(cortes, key=lambda x: abs(x - (ini + janela)))
            if fim - ini < janela * 0.5:
                continue
            chave = (doc, ini // 500)
            if chave in usados:
                continue
            usados.add(chave)
            passagens.append({
                "n": len(passagens) + 1,
                "documento": doc,
                "forma": amostra[doc],
                "janela": janela,
                "inicio": ini,
                "texto": texto[ini:fim].strip(),
            })
            break

    SAIDA.write_text(json.dumps({
        "semente": args.semente, "janelas": list(JANELAS),
        "passagens": passagens}, ensure_ascii=False, indent=1), encoding="utf-8")

    import collections
    print(f"\n✓ {SAIDA}")
    print(f"  passagens: {len(passagens)}")
    print(f"  por janela : {dict(sorted(collections.Counter(p['janela'] for p in passagens).items()))}")
    print(f"  por forma  : {dict(collections.Counter(p['forma'] for p in passagens))}")
    print(f"  por documento: mín {min(collections.Counter(p['documento'] for p in passagens).values())}, "
          f"máx {max(collections.Counter(p['documento'] for p in passagens).values())}")


if __name__ == "__main__":
    main()
