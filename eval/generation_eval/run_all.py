#!/usr/bin/env python3
"""
run_all.py — roda TODOS os cenários de geração (A, B, C, no_rag) × ICL{sem, com},
cada um julgado pelos 2 juízes. Carrega gerador/juízes/rankings uma única vez.
Substitui o antigo 03_generation_eval.py.

Uso:
  python eval/generation_eval/run_all.py
  python eval/generation_eval/run_all.py --limit 30 --no-judge
  python eval/generation_eval/run_all.py --scenarios C no_rag
Depois: python eval/04_agreement.py  e  python eval/tools/export_judged_xlsx.py
"""
from __future__ import annotations
import argparse
from _common import Setup, load_config
import scenario_A_bm25
import scenario_B_dense
import scenario_C_fusion
import scenario_no_rag

RUNNERS = {"A": scenario_A_bm25, "B": scenario_B_dense,
           "C": scenario_C_fusion, "no_rag": scenario_no_rag}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--scenarios", nargs="+", default=["A", "B", "C", "no_rag"],
                    choices=list(RUNNERS))
    args = ap.parse_args()

    S = Setup(cfg=load_config(args.config), limit=args.limit, no_judge=args.no_judge)
    for sc in args.scenarios:
        RUNNERS[sc].run(S=S)

    print("\n✓ geração concluída. Próximos: python eval/04_agreement.py  e  "
          "python eval/tools/export_judged_xlsx.py")


if __name__ == "__main__":
    main()
