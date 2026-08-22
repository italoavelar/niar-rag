#!/usr/bin/env python3
"""
eval/tools/check_answers.py
═══════════════════════════
Sanidade das respostas geradas: detecta o modo de falha do gerador de raciocínio
(<think> aberto consumindo o max_tokens) e respostas truncadas.

Uma resposta conta como COMPLETA se termina em pontuação final OU no link da
fonte — este último é o fecho normal da seção "## Fontes utilizadas", e tratá-lo
como truncamento produz falso positivo.

Uso:  python eval/tools/check_answers.py [judged_*.jsonl ...]
      (sem argumentos, verifica todos os judged_*.jsonl de results/generation)
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

EVAL = Path(__file__).resolve().parent.parent


def completa(a: str) -> bool:
    t = a.rstrip()
    if re.search(r"(#+|\*\*|\|)\s*$", t):      # construto markdown cortado no meio
        return False
    if re.search(r"https?://\S+$", t):          # fecha no link da fonte: normal
        return True
    if re.search(r'https?://[^"]+"\}\s*$', t):  # link embrulhado em JSON
        return True
    return t.endswith((".", "!", "?", ")"))


def verifica(path: Path) -> bool:
    recs = [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]
    think = [r for r in recs if "<think>" in r["answer"].lower()]
    resto = [r for r in recs if r not in think]
    trunc = [r for r in resto if not completa(r["answer"])]
    boas = len(resto) - len(trunc)

    print(f"\n── {path.name}  ({len(recs)} respostas)")
    print(f"   completas          : {boas}")
    print(f"   com <think> aberto : {len(think)}")
    print(f"   truncadas          : {len(trunc)}")
    for r in think[:3]:
        print(f"     ! {r['qid']} começa com: {r['answer'][:60]!r}")
    for r in trunc[:5]:
        print(f"     ! {r['qid']} termina em: ...{r['answer'].rstrip()[-55:]!r}")
    if think:
        print("   → o reasoning_effort não surtiu efeito; confira se o modelo aceita o parâmetro")
    return not think and not trunc


def main():
    alvos = [Path(a) for a in sys.argv[1:]]
    if not alvos:
        alvos = sorted((EVAL / "results" / "generation").glob("judged_*.jsonl"))
    if not alvos:
        sys.exit("! nenhum judged_*.jsonl encontrado — rode a geração antes.")
    tudo_ok = all(verifica(p) for p in alvos)
    print("\n" + ("✓ tudo limpo" if tudo_ok else "! há respostas comprometidas — ver acima"))


if __name__ == "__main__":
    main()
