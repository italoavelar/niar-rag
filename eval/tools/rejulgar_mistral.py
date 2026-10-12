#!/usr/bin/env python3
"""
eval/tools/rejulgar_mistral.py
══════════════════════════════
Re-julga as respostas JÁ GERADAS com UM novo juiz (Mistral), reaproveitando as
notas de Llama e GLM que já estão nos judged_*.jsonl. Substitui o Gemma.

Por que isso: a geração não muda; Llama e GLM já custaram e são bons. Só o juiz
novo roda → 1/3 do custo de re-julgar os três. O contexto (necessário ao juiz)
é reconstruído a partir dos trechos de origem, idêntico ao que os outros juízes
viram.

Uso:
  python eval/tools/rejulgar_mistral.py --juiz-id mistralai/mistral-small-3.2-24b-instruct
  python eval/tools/rejulgar_mistral.py --juiz-id <id> --limit 50   # teste
  python eval/tools/rejulgar_mistral.py --juiz-id <id> --nome mistral --remover gemma

Escreve por cima dos judged_*.jsonl (faz backup .bak antes). Checkpoint: pula
registros que já têm o novo juiz.
"""
from __future__ import annotations
import argparse, json, shutil, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EVAL = HERE.parent
sys.path.insert(0, str(EVAL))
from lib.common import load_config, resolve, setup_io          # noqa: E402
from lib.llm_clients import LLMClient                          # noqa: E402
from lib.judges import judge_answer                            # noqa: E402

GERACAO_DIR = "experimento_embedding/geracao"
RESP = "top5_respondiveis.jsonl"
IRR  = "top5_irrespondiveis.jsonl"


def _contexto_dos_trechos(trechos) -> str:
    blocos = []
    for i, t in enumerate(sorted(trechos, key=lambda x: int(x.get("posicao", 0))), 1):
        blocos.append(
            f"📄 DOCUMENTO {i}:\n{t.get('texto','[indisponível]')}\n"
            f"🔗 FONTE: {t.get('titulo','?')} — {t.get('documento','')}\n"
            + "-" * 60)
    return "\n".join(blocos) if blocos else "[nenhum documento recuperado]"


def _carrega_trechos(cfg):
    """{qid: trechos} a partir dos jsonl de origem (resp + irresp)."""
    base = resolve(f"{GERACAO_DIR}")
    out = {}
    for nome in (RESP, IRR):
        p = base / nome
        if not p.exists():
            p = resolve(cfg["paths"]["results_dir"]).parent / GERACAO_DIR / nome
        if p.exists():
            for l in open(p, encoding="utf-8"):
                if l.strip():
                    r = json.loads(l); out[r["qid"]] = r.get("trechos", [])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--juiz-id", required=True, help="id do modelo no OpenRouter")
    ap.add_argument("--nome", default="mistral", help="nome do juiz no registro")
    ap.add_argument("--provider", default="openrouter")
    ap.add_argument("--remover", default="gemma", help="juiz a descartar (ou 'nenhum')")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--max-tokens", type=int, default=3072)
    args = ap.parse_args()

    setup_io()
    cfg = load_config(args.config)
    criteria = cfg["judges"]["criteria"]
    scale = cfg["judges"]["scale"]
    gen_dir = resolve(cfg["paths"]["results_dir"]) / "generation"

    cli = LLMClient(provider=args.provider, model=args.juiz_id,
                    temperature=cfg["judges"].get("temperature", 0),
                    max_tokens=args.max_tokens)
    trechos_por_qid = _carrega_trechos(cfg)

    arquivos = sorted(gen_dir.glob("judged_*.jsonl"))
    arquivos = [a for a in arquivos if not a.name.endswith("_TESTE.jsonl")]
    print(f"Juiz novo: {args.nome} ({args.juiz_id})")
    print(f"Arquivos: {len(arquivos)} | remove: {args.remover}"
          + ("  [MODO TESTE: grava em *_TESTE.jsonl, não toca nos originais]"
             if args.limit else ""))

    from tqdm import tqdm
    total_mudou = 0
    for fp in arquivos:
        regs = [json.loads(l) for l in open(fp, encoding="utf-8") if l.strip()]
        if args.limit:
            regs = regs[:args.limit]
            destino = fp.with_name(fp.stem + "_TESTE.jsonl")   # NÃO sobrescreve
        else:
            destino = fp
            bak = fp.with_suffix(".jsonl.bak")                 # backup uma vez
            if not bak.exists():
                shutil.copy(fp, bak)

        mudou = 0
        for r in tqdm(regs, desc=f"[{fp.name}]", leave=False):
            if args.nome in r.get("scores", {}):
                continue                      # checkpoint: já tem o juiz novo
            # reconstrói o contexto que o juiz precisa
            tre = trechos_por_qid.get(r["qid"], [])
            r["context_text"] = (_contexto_dos_trechos(tre)
                                 if r.get("cenario") == "rag" else "")
            nota = judge_answer(cli, r, criteria, scale)
            r.setdefault("scores", {})[args.nome] = nota
            r.pop("context_text", None)
            if not args.limit and args.remover and args.remover != "nenhum":
                r["scores"].pop(args.remover, None)   # no teste, mantém tudo
            mudou += 1

        with open(destino, "w", encoding="utf-8") as f:
            for r in regs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  ✓ {destino.name}: {mudou} re-julgados")
        total_mudou += mudou

    if args.limit:
        print(f"\n✓ TESTE concluído ({total_mudou} registros). Inspecione os "
              f"*_TESTE.jsonl e veja se o {args.nome} discrimina (não satura em 3).")
    else:
        print("\n✓ re-julgamento concluído. Rode: python eval/04_agreement.py")


if __name__ == "__main__":
    main()
