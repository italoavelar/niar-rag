#!/usr/bin/env python3
"""
eval/tools/rejulgar_juiz.py
═══════════════════════════
Acrescenta (ou substitui) UM juiz nas respostas já geradas, reaproveitando as
notas dos juízes que já estão nos judged_*.jsonl. Generalização do
rejulgar_mistral.py — serve para qualquer juiz (aberto ou proprietário).

Exemplos:
  # Gemini como 4º juiz de REFERÊNCIA (acrescenta, não remove ninguém):
  python eval/tools/rejulgar_juiz.py --juiz-id google/gemini-2.5-flash \
      --nome gemini --provider openrouter --remover nenhum

  # teste reaproveitável de 50/arquivo:
  python eval/tools/rejulgar_juiz.py --juiz-id google/gemini-2.5-flash \
      --nome gemini --remover nenhum --limit-real 50

Escreve por cima dos judged_*.jsonl (backup .bak). Checkpoint: pula registros
que já têm o juiz novo. O contexto é reconstruído a partir dos trechos de origem,
idêntico ao que os outros juízes viram.
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
    # --nome e --remover eram "mistral" e "gemma" por padrao: fossilizavam um
    # painel que ja mudou duas vezes. Errar o --nome rotula o juiz errado no
    # dado, e o --remover apagava um juiz sem ninguem pedir. Agora o nome e
    # obrigatorio e remover exige dizer quem.
    ap.add_argument("--nome", required=True,
                    help="nome do juiz no registro (ex.: glm, mistral)")
    ap.add_argument("--provider", default="openrouter")
    ap.add_argument("--remover", default="nenhum",
                    help="juiz a DESCARTAR dos registros; o padrao e 'nenhum', "
                         "isto e, so acrescenta")
    ap.add_argument("--limit", type=int,
                    help="TESTE descartável: grava em *_TESTE.jsonl e NÃO aplica o --remover")
    ap.add_argument("--limit-real", type=int,
                    help="TESTE reaproveitável: julga os N primeiros NO arquivo real "
                         "(com .bak, aplica o --remover). O comando completo "
                         "depois continua de onde parou via checkpoint.")
    ap.add_argument("--max-tokens", type=int, default=3072)
    args = ap.parse_args()
    if args.limit and args.limit_real:
        ap.error("use --limit OU --limit-real, não os dois")
    n_real = args.limit_real          # recorta no arquivo real, reaproveitável

    setup_io()
    cfg = load_config(args.config)
    criteria = cfg["judges"]["criteria"]
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
        regs_all = [json.loads(l) for l in open(fp, encoding="utf-8") if l.strip()]
        if args.limit:
            # teste descartável: só os N primeiros, grava em cópia, sem remover
            regs = regs_all[:args.limit]
            destino = fp.with_name(fp.stem + "_TESTE.jsonl")   # NÃO sobrescreve
        else:
            # arquivo real (com backup). Em --limit-real, julga só os N primeiros
            # mas regrava o arquivo INTEIRO (os demais ficam sem o mistral e o
            # comando completo os pega depois via checkpoint).
            regs = regs_all
            destino = fp
            bak = fp.with_suffix(".jsonl.bak")                 # backup uma vez
            if not bak.exists():
                shutil.copy(fp, bak)

        mudou = 0
        julgados = 0                           # quantos de fato passaram pelo juiz
        for r in tqdm(regs, desc=f"[{fp.name}]", leave=False):
            # --limit-real: julga só os N primeiros deste arquivo; o resto é
            # regravado intacto e o comando completo o pega depois (checkpoint).
            if n_real is not None and julgados >= n_real:
                break
            if args.nome in r.get("scores", {}):
                continue                      # checkpoint: já tem o juiz novo
            julgados += 1
            # reconstrói o contexto que o juiz precisa
            tre = trechos_por_qid.get(r["qid"], [])
            r["context_text"] = (_contexto_dos_trechos(tre)
                                 if r.get("cenario") == "rag" else "")
            nota = judge_answer(cli, r, criteria)
            r.setdefault("scores", {})[args.nome] = nota
            r.pop("context_text", None)
            if not args.limit and args.remover and args.remover != "nenhum":
                r["scores"].pop(args.remover, None)   # no teste descartável, mantém tudo
            mudou += 1

        with open(destino, "w", encoding="utf-8") as f:
            for r in regs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  ✓ {destino.name}: {mudou} re-julgados")
        total_mudou += mudou

    if args.limit:
        print(f"\n✓ TESTE descartável concluído ({total_mudou} registros em "
              f"*_TESTE.jsonl). Veja se o {args.nome} discrimina (não satura em 3). "
              f"Os originais NÃO foram tocados.")
    elif n_real is not None:
        print(f"\n✓ TESTE reaproveitável concluído ({total_mudou} registros, "
              f"gravados NOS arquivos reais com .bak). Gostou? Rode o comando "
              f"completo (sem --limit-real): o checkpoint pula estes {total_mudou} "
              f"e julga o resto. NÃO rode 04_agreement.py ainda — espere o completo.")
    else:
        print("\n✓ re-julgamento completo concluído. Rode: python eval/04_agreement.py")


if __name__ == "__main__":
    main()
