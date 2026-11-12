#!/usr/bin/env python3
"""
eval/tools/completar_criterios.py
═════════════════════════════════
Re-julga CIRURGICAMENTE apenas os critérios faltantes de UM juiz, mesclando só
esses critérios de volta no bloco e preservando o resto. Não acrescenta nem
remove juiz — para isso use rejulgar_juiz.py.

Motivação (auditoria nov/2026): o prompt antigo, no cenário sem-RAG para
perguntas respondíveis, afirmava "esta pergunta tem resposta no contexto" com o
bloco de contexto vazio. A contradição levava alguns juízes (mistral, llama) a
abandonar correctness/completeness com "na" INDEVIDO. Além disso answer_relevance
ficava indefinido numa recusa. O judges.py foi corrigido; este script completa as
notas afetadas SEM re-julgar o que já está certo.

Regra de ouro — "na" legítimo NUNCA é tocado. Para cada critério pedido, a
decisão de "precisa completar?" é tomada pelo REGISTRO, reusando a lógica do
judges.py novo:
  • correctness/completeness: precisa nota só se HÁ resposta-referência;
    sem referência o "na" é correto e fica.
  • answer_relevance: precisa nota só se a resposta NÃO é recusa; numa recusa o
    "na" é correto e fica.
  • faithfulness/hallucination: precisa nota só se HÁ contexto; sem contexto
    (sem-RAG) o "na" é correto e fica.
Um registro só é re-julgado se, após essa decisão, algum critério pedido está
sem nota válida (None, ausente, ou "na" onde deveria haver número).

Exemplos:
  # completar os dois juízes afetados, 3 critérios:
  python eval/tools/completar_criterios.py --nome mistral \\
      --juiz-id mistralai/mistral-small-3.2-24b-instruct \\
      --completar correctness completeness answer_relevance
  python eval/tools/completar_criterios.py --nome llama \\
      --juiz-id <id-llama-no-openrouter> \\
      --completar correctness completeness answer_relevance

  # TESTE antes (grava em *_TESTE.jsonl, não toca nos originais):
  python eval/tools/completar_criterios.py --nome mistral \\
      --juiz-id mistralai/mistral-small-3.2-24b-instruct \\
      --completar correctness completeness answer_relevance --limit 50

Escreve por cima dos judged_*.jsonl (backup .bak uma vez). O contexto é
reconstruído a partir dos trechos de origem, idêntico ao que os juízes viram.
"""
from __future__ import annotations
import argparse, json, shutil, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EVAL = HERE.parent
sys.path.insert(0, str(EVAL))
from lib.common import load_config, resolve, setup_io          # noqa: E402
from lib.llm_clients import LLMClient                          # noqa: E402
from lib.judges import (                                       # noqa: E402
    judge_answer, _tem_contexto, _eh_recusa,
    CONDICIONAIS, SEM_SENTIDO_SE_RECUSA,
)

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


def _deveria_ter_nota(crit: str, record: dict) -> bool:
    """O critério DEVE ter nota numérica neste registro? (reusa a lógica de 'na'
    do judges.py, pela via oposta: se a condição de 'na' vale, NÃO deve ter nota)."""
    tem_ref = bool((record.get("reference_answer") or "").strip())
    if crit in CONDICIONAIS and not tem_ref:
        return False                       # "na" legítimo: sem referência
    if crit in ("faithfulness", "hallucination") and not _tem_contexto(record):
        return False                       # "na" legítimo: sem contexto
    if crit in SEM_SENTIDO_SE_RECUSA and _eh_recusa(record.get("answer", "")):
        return False                       # "na" legítimo: resposta é recusa
    return True


def _precisa_completar(nota_atual, crit: str, record: dict) -> bool:
    """True se este critério deveria ter nota mas a atual é inválida/ausente."""
    if not _deveria_ter_nota(crit, record):
        return False                       # "na" legítimo — não tocar
    # deveria ter número; precisa completar se está ausente, None, ou "na"
    if nota_atual is None:
        return True
    if isinstance(nota_atual, str) and nota_atual.strip().lower() in (
            "na", "n/a", "não se aplica", "nao se aplica"):
        return True
    return False                           # já tem número válido — não tocar


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--nome", required=True, help="juiz a completar (ex.: mistral)")
    ap.add_argument("--juiz-id", required=True, help="id do modelo no OpenRouter")
    ap.add_argument("--provider", default="openrouter")
    ap.add_argument("--completar", nargs="+", required=True,
                    help="critérios a completar (ex.: correctness completeness answer_relevance)")
    ap.add_argument("--limit", type=int,
                    help="TESTE: grava em *_TESTE.jsonl, não toca nos originais")
    ap.add_argument("--apenas", nargs="+",
                    help="processa SÓ os arquivos cujo nome contém um destes trechos "
                         "(ex.: qwen3 gptoss_120b). Útil quando um mesmo juiz aparece "
                         "em arquivos de painéis diferentes (geração vs recuperação).")
    ap.add_argument("--max-tokens", type=int, default=3072)
    args = ap.parse_args()

    setup_io()
    cfg = load_config(args.config)
    criteria_all = cfg["judges"]["criteria"]           # o prompt pede todos os 7
    alvo = set(args.completar)
    desconhecidos = alvo - set(criteria_all)
    if desconhecidos:
        ap.error(f"critérios fora da rubrica: {sorted(desconhecidos)}")
    gen_dir = resolve(cfg["paths"]["results_dir"]) / "generation"

    cli = LLMClient(provider=args.provider, model=args.juiz_id,
                    temperature=cfg["judges"].get("temperature", 0),
                    max_tokens=args.max_tokens)
    trechos_por_qid = _carrega_trechos(cfg)

    arquivos = sorted(gen_dir.glob("judged_*.jsonl"))
    arquivos = [a for a in arquivos if not a.name.endswith("_TESTE.jsonl")]
    if args.apenas:
        arquivos = [a for a in arquivos
                    if any(tok in a.name for tok in args.apenas)]
    print(f"Completar juiz: {args.nome} ({args.juiz_id})")
    if args.apenas:
        print(f"Filtro --apenas: {args.apenas}")
    print(f"Critérios: {sorted(alvo)} | arquivos: {len(arquivos)}"
          + ("  [MODO TESTE: grava em *_TESTE.jsonl]" if args.limit else ""))
    for a in arquivos:
        print(f"    • {a.name}")

    from tqdm import tqdm
    total_regs_tocados = 0
    total_notas_preenchidas = 0
    for fp in arquivos:
        regs = [json.loads(l) for l in open(fp, encoding="utf-8") if l.strip()]
        if args.limit:
            destino = fp.with_name(fp.stem + "_TESTE.jsonl")
            regs_proc = regs[:args.limit]
        else:
            destino = fp
            regs_proc = regs
            bak = fp.with_suffix(".jsonl.bak")
            if not bak.exists():
                shutil.copy(fp, bak)

        tocados = 0
        preenchidas = 0
        for r in tqdm(regs_proc, desc=f"[{fp.name}]", leave=False):
            bloco = r.get("scores", {}).get(args.nome)
            if not isinstance(bloco, dict):
                continue                   # este juiz não está no registro; pula
            # decide, critério a critério, o que precisa completar
            faltantes = [c for c in alvo
                         if _precisa_completar(bloco.get(c), c, r)]
            if not faltantes:
                continue                   # nada a fazer neste registro
            # reconstrói o contexto exatamente como no julgamento original
            tre = trechos_por_qid.get(r["qid"], [])
            r["context_text"] = (_contexto_dos_trechos(tre)
                                 if r.get("cenario") == "rag" else "")
            # re-julga mandando o prompt COMPLETO (todos os critérios), para a
            # condição ser a mesma da coleta original; mescla só os faltantes.
            nova = judge_answer(cli, r, criteria_all)
            r.pop("context_text", None)
            for c in faltantes:
                if c in nova:
                    bloco[c] = nova[c]
                    preenchidas += 1
            # preserva/atualiza metadados sem apagar as justificativas existentes
            just_old = bloco.get("_justifications", {})
            just_new = nova.get("_justifications", {})
            for c in faltantes:
                if c in just_new:
                    just_old[c] = just_new[c]
            bloco["_justifications"] = just_old
            bloco["_rubrica"] = nova.get("_rubrica", bloco.get("_rubrica"))
            # _fora_escala: limpa entradas dos critérios que acabamos de resolver
            fe = bloco.get("_fora_escala", {})
            for c in faltantes:
                fe.pop(c, None)
            if fe:
                bloco["_fora_escala"] = fe
            elif "_fora_escala" in bloco:
                bloco.pop("_fora_escala")
            tocados += 1

        with open(destino, "w", encoding="utf-8") as f:
            for r in regs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  ✓ {destino.name}: {tocados} registros, {preenchidas} notas preenchidas")
        total_regs_tocados += tocados
        total_notas_preenchidas += preenchidas

    print(f"\n✓ Concluído: {total_regs_tocados} registros tocados, "
          f"{total_notas_preenchidas} notas preenchidas.")
    if args.limit:
        print("  [TESTE] originais intactos. Confira os *_TESTE.jsonl.")
    else:
        print("  Rode os dois juízes (mistral e llama) e só então: "
              "python eval/04_agreement.py")


if __name__ == "__main__":
    main()
