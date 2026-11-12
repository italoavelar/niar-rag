#!/usr/bin/env python3
"""
eval/generation_eval/run_matriz_geracao.py
══════════════════════════════════════════
Runner da MATRIZ de geração (eixos 2 e 5), lendo os TOP-5 já montados pela
curadoria (fonte canônica, balanceada 75/75/75):

    eval/experimento_embedding/geracao/top5_respondiveis.jsonl    (225)
    eval/experimento_embedding/geracao/top5_irrespondiveis.jsonl  (75)   → testa RECUSA

Total: 300 perguntas. Para CADA gerador em `generation.generators`, roda DOIS
cenários:
  • RAG     — responde a partir dos 5 trechos da própria linha (prompt RAG_EVAL_SYSTEM)
  • no_rag  — responde do próprio conhecimento, SEM contexto (prompt NO_RAG_EVAL_SYSTEM)
e julga com os juízes de `judges.models`. Grava:

    results/generation/judged_<gerador>_<cenario>_icl{0,1}.jsonl
       cenario ∈ {rag, norag}

PROMPTS: usa eval/lib/eval_prompts.py (específicos da avaliação, SEM a ferramenta
de busca). O prompt de PRODUÇÃO (agent/utils/prompt.py) NÃO é usado aqui.

Uso:
  python eval/generation_eval/run_matriz_geracao.py                 # 300, todos geradores, rag+norag, sem e com ICL
  python eval/generation_eval/run_matriz_geracao.py --limit 4 --no-judge
  python eval/generation_eval/run_matriz_geracao.py --generators qwen3_27b --cenarios rag --icl 0
  python eval/generation_eval/run_matriz_geracao.py --só-respondiveis
Depois:  python eval/04_agreement.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from lib.common import (load_config, resolve, load_jsonl, save_jsonl,  # noqa: E402
                        ensure_dir, setup_io, set_seed)
from lib.rag_runner import RagRunner, _load_icl_block, strip_reasoning  # noqa: E402
from lib.judges import build_judges, judge_answer                       # noqa: E402
from lib.eval_prompts import RAG_EVAL_SYSTEM, NO_RAG_EVAL_SYSTEM         # noqa: E402

GERACAO_DIR = "experimento_embedding/geracao"
RESP = "top5_respondiveis.jsonl"
IRR  = "top5_irrespondiveis.jsonl"


def _contexto_dos_trechos(trechos) -> str:
    blocos = []
    for i, t in enumerate(sorted(trechos, key=lambda x: x.get("posicao", 0)), 1):
        blocos.append(
            f"📄 DOCUMENTO {i}:\n{t.get('texto','[indisponível]')}\n"
            f"🔗 FONTE: {t.get('titulo','?')} — {t.get('documento','')}\n"
            + "-" * 60
        )
    return "\n".join(blocos) if blocos else "[nenhum documento recuperado]"


def _carrega_perguntas(cfg, so_respondiveis: bool):
    cand = [resolve(f"{GERACAO_DIR}/{RESP}"),
            resolve(cfg["paths"]["results_dir"]).parent / GERACAO_DIR / RESP]
    resp_path = next((p for p in cand if Path(p).exists()), None)
    if resp_path is None:
        sys.exit(f"! não achei {RESP}. Procurei em: {[str(p) for p in cand]}")
    irr_path = Path(str(resp_path).replace(RESP, IRR))
    perguntas = list(load_jsonl(resp_path))
    if not so_respondiveis and irr_path.exists():
        perguntas += load_jsonl(irr_path)
    print(f"perguntas: {len(perguntas)} "
          f"({sum(1 for r in perguntas if r.get('respondivel'))} respondíveis, "
          f"{sum(1 for r in perguntas if not r.get('respondivel'))} irrespondíveis)")
    return perguntas


def _gera(runner, cenario, pergunta, trechos, icl, n_shots):
    """Gera uma resposta no cenário dado. Retorna (answer, context_text, chunk_ids)."""
    icl_block = _load_icl_block(n_shots) if icl else ""
    if cenario == "rag":
        context = _contexto_dos_trechos(trechos)
        system = RAG_EVAL_SYSTEM + icl_block
        user = f"Contexto recuperado:\n{context}\n\nPergunta do usuário: {pergunta}"
        chunk_ids = [t.get("chunk_id") for t in trechos]
    else:  # norag
        context = ""
        system = NO_RAG_EVAL_SYSTEM + icl_block
        user = f"Pergunta do usuário: {pergunta}"
        chunk_ids = []
    raw = runner.llm.chat(user, system=system)
    return strip_reasoning(raw), context, chunk_ids


def _processa(gerador_nome, cenario, runner, perguntas, icl, judges, criteria,
              od, n_shots):
    from tqdm import tqdm
    tag = f"{gerador_nome}_{cenario}_icl{int(icl)}"
    final = od / f"judged_{tag}.jsonl"
    part = od / f"judged_{tag}.partial.jsonl"

    done = {}
    if part.exists():
        for line in part.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try: r = json.loads(line); done[r["qid"]] = r
                except Exception: pass
    if final.exists():
        feitas = load_jsonl(final)
        if {r["qid"] for r in perguntas} <= {r["qid"] for r in feitas}:
            print(f"══ {tag}: já concluído ({len(feitas)}) — pulando."); return feitas

    print(f"══ {tag} ══" + (f"  (retomando {len(done)})" if done else ""))
    answers = []
    with part.open("a", encoding="utf-8") as fh:
        for r in tqdm(perguntas, desc=f"[{tag}]", leave=False):
            if r["qid"] in done:
                answers.append(done[r["qid"]]); continue
            ans, context, chunk_ids = _gera(runner, cenario, r["pergunta"],
                                            r.get("trechos", []), icl, n_shots)
            rec = {
                "qid": r["qid"], "question": r["pergunta"],
                "question_type": r.get("tipo", ""),
                "should_answer": bool(r.get("respondivel")),
                "reference_answer": r.get("resposta_referencia", ""),
                # Sem estes tres campos nao se separa falha de GERACAO de
                # falha de RECUPERACAO. Em 104 das 225 respondiveis a
                # evidencia completa NAO chegou ao top-5: ali o gerador
                # certo e o que responde o que da e diz o que falta, e
                # puni-lo em completeness/correctness seria medir a
                # recuperacao de novo. Vao para o registro, NAO para o
                # prompt do juiz — contar ao juiz enviesaria a nota.
                "evidencia_completa_no_top5": r.get("evidencia_completa_no_top5"),
                "citacoes_que_chegaram": r.get("citacoes_que_chegaram"),
                "citacoes_exigidas_n": r.get("citacoes_exigidas_n"),
                "cenario": cenario, "answer": ans, "icl": icl,
                "context_chunk_ids": chunk_ids, "context_text": context,
            }
            ok = True
            if judges:
                rec["scores"] = {name: judge_answer(cli, rec, criteria)
                                 for name, cli in judges}
                ok = not any("_error" in s for s in rec["scores"].values())
            rec.pop("context_text", None)
            answers.append(rec)
            if ok:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n"); fh.flush()

    save_jsonl(answers, final)
    ruins = sum(1 for a in answers
                if any("_error" in s for s in (a.get("scores") or {}).values()))
    if ruins == 0 and part.exists():
        part.unlink()
    print(f"  ✓ {len(answers)} respostas → {final.name}"
          + (f"  ({ruins} com juiz em erro — re-rode)" if ruins else ""))
    return answers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--generators", nargs="+")
    ap.add_argument("--cenarios", nargs="+", choices=["rag", "norag"],
                    default=["rag", "norag"], help="padrão: ambos")
    ap.add_argument("--icl", type=int, choices=[0, 1])
    ap.add_argument("--só-respondiveis", "--so-respondiveis", dest="so_resp",
                    action="store_true")
    args = ap.parse_args()

    setup_io()
    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))

    gen_matrix = cfg["generation"].get("generators")
    if not gen_matrix:
        sys.exit("! cfg.generation.generators ausente (ver config_bloco_eixos_2_5.yaml).")

    perguntas = _carrega_perguntas(cfg, args.so_resp)
    if args.limit:
        perguntas = perguntas[:args.limit]

    criteria = cfg["judges"]["criteria"]
    judges = None if args.no_judge else build_judges(cfg)
    n_shots = cfg["generation"]["icl"]["n_shots"]
    icl_vars = ([bool(args.icl)] if args.icl is not None
                else [bool(x) for x in cfg["generation"]["icl"]["variants"]])
    od = ensure_dir(resolve(cfg["paths"]["results_dir"]) / "generation")

    quais = args.generators
    geradores = [g for g in gen_matrix if (quais is None or g["name"] in quais)]
    if quais and not geradores:
        sys.exit(f"! nenhum gerador casou com {quais}. "
                 f"Disponíveis: {[g['name'] for g in gen_matrix]}")
    print(f"Geradores: {', '.join(g['name'] for g in geradores)}")
    print(f"Cenários: {', '.join(args.cenarios)}")
    if judges:
        print(f"Juízes: {', '.join(m['name'] for m in cfg['judges']['models'])}")

    from lib.rag_runner import RagRunner
    for g in geradores:
        spec = {k: g[k] for k in
                ("provider", "model", "temperature", "max_tokens",
                 "reasoning_effort", "backends") if k in g}
        runner = RagRunner(spec, corpus={},
                           context_top_k=cfg["generation"]["context_top_k"])
        for cenario in args.cenarios:
            for icl in icl_vars:
                _processa(g["name"], cenario, runner, perguntas, icl, judges,
                          criteria, od, n_shots)

    print("\n✓ matriz concluída. Próximo: python eval/04_agreement.py")


if __name__ == "__main__":
    main()
