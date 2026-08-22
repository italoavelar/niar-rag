#!/usr/bin/env python3
"""
Cenário B (geração) — Qwen responde a partir do contexto do DENSO implantado
(retrieval.fusion.dense_for_fusion, ex.: gemini), COM e SEM ICL. Requer a
recuperação (ranking B_dense_<emb>) já rodada.

Uso:  python eval/generation_eval/scenario_B_dense.py [--limit N] [--no-judge]
"""
from __future__ import annotations
import argparse, sys
from _common import Setup, run_and_save


def run(S=None, cfg=None, icl_only=None):
    S = S or Setup(cfg)
    system = f"B_dense_{S.dense_for_gen}"
    rk = S.rankings(system)
    if rk is None:
        sys.exit(f"! ranking {system} ausente — rode a recuperação (retrieval_eval) antes.")
    # icl_only permite dividir o trabalho entre pessoas/contas: cada uma roda uma
    # variante. Sem o parâmetro, roda as duas (comportamento do run_all.py).
    variantes = S.icl_variants if icl_only is None else [icl_only]
    for icl in variantes:
        run_and_save(S, f"{system}_icl{int(icl)}", rk, icl)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int); ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--icl", type=int, choices=[0, 1],
                    help="roda SÓ uma variante: 0 = sem ICL, 1 = com ICL")
    a = ap.parse_args()
    run(S=Setup(limit=a.limit, no_judge=a.no_judge),
        icl_only=None if a.icl is None else bool(a.icl))
