#!/usr/bin/env python3
"""Julga relevância de TRECHO × PERGUNTA — o pooling da Fase 2.

O que isto é, e o que não é
───────────────────────────
O `eval/lib/judges.py` julga a **resposta do sistema** contra a resposta-referência.
Isto aqui é outra coisa: julga se um **trecho recuperado** responde à pergunta,
dupla a dupla. É o que transforma "não julgado" em "julgado irrelevante" — hoje,
nas comparativas, 13 de 294 duplas do pool têm julgamento, e as outras 281 contam
como erro só porque ninguém olhou.

Três subcomandos, três donos
────────────────────────────
    pool      João e Ítalo — junta o top-N dos três braços numa lista de duplas
    julgar    João e Ítalo — roda o juiz LLM sobre essas duplas
    calibrar  HELEN — diz se o juiz merece confiança

O que o juiz NÃO vê
───────────────────
A dupla chega ao juiz como (pergunta, texto do trecho) e nada mais. Ele não sabe
qual braço trouxe o trecho, nem se aquele é o trecho de onde a pergunta nasceu.
As colunas `origem` e `e_ancora` existem no CSV — são a base da calibração — mas
nunca entram no prompt. Se entrassem, o julgamento ficaria contaminado e a taxa
de controle não mediria nada.

O controle de âncora
────────────────────
O trecho de origem entra no pool como qualquer outro. Ele é relevante **por
construção**: a pergunta foi escrita a partir dele. Se o juiz der 0 nele, o erro
é do juiz. Essa taxa é a medida mais barata de saber se o julgamento serve, e é
o portão que a Helen opera antes de liberar o lote inteiro.

Escala
──────
    2  responde diretamente a pergunta
    1  contribui, mas não basta sozinho
    0  não contribui

Uso
───
  python eval/tools/julgar_relevancia.py pool --profundidade 5
  python eval/tools/julgar_relevancia.py julgar --juiz llama --piloto 100
  python eval/tools/julgar_relevancia.py calibrar --julgamentos eval/results/julgamentos_llama.csv
  python eval/tools/julgar_relevancia.py calibrar --julgamentos ...llama.csv --segundo ...gptoss.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
EVAL = PROJECT_ROOT / "eval"
sys.path.insert(0, str(EVAL))

for _fluxo in (sys.stdout, sys.stderr):
    try:
        _fluxo.reconfigure(encoding="utf-8")
    except Exception:
        pass

CORPUS = PROJECT_ROOT / "data/processed/documents.jsonl"
GOLDEN = EVAL / "data/golden_qa.jsonl"
RANKINGS = EVAL / "results/retrieval/rankings"
POOL = EVAL / "results/pool_fase2.csv"
SAIDA_DIR = EVAL / "results"

TEXTO_MAX = 4000

JUIZ_SYS = (
    "Você avalia a relevância de trechos de normas jurídicas e de saúde para "
    "perguntas de usuários. Seja rigoroso e imparcial. Responda SEMPRE em JSON "
    "válido, sem texto extra."
)

ESCALA = (
    "Escala (inteiro):\n"
    "  2 = o trecho RESPONDE DIRETAMENTE a pergunta; quem o lesse saberia a resposta.\n"
    "  1 = o trecho CONTRIBUI (traz parte da resposta, contexto necessário, uma das\n"
    "      condições), mas sozinho não responde.\n"
    "  0 = o trecho NÃO CONTRIBUI para responder.\n\n"
    "Julgue apenas a relação entre ESTE trecho e ESTA pergunta. Não suponha o que\n"
    "outros trechos diriam. Não penalize o trecho por estar em inglês — o acervo é\n"
    "bilíngue e a pergunta é sempre em português."
)


def prompt_dupla(pergunta: str, texto: str) -> str:
    return (f"{ESCALA}\n\n"
            f"PERGUNTA:\n{pergunta}\n\n"
            f"TRECHO:\n\"\"\"\n{texto[:TEXTO_MAX]}\n\"\"\"\n\n"
            f'Responda somente: {{"nota": <0|1|2>, "justificativa": "<uma frase>"}}')


def carregar_corpus() -> dict[str, dict]:
    return {r["id"]: r for r in
            (json.loads(l) for l in CORPUS.open(encoding="utf-8") if l.strip())}


def carregar_gabarito() -> list[dict]:
    return [json.loads(l) for l in GOLDEN.open(encoding="utf-8") if l.strip()]


# ── pool ─────────────────────────────────────────────────────────────────────

def cmd_pool(args) -> None:
    corpus = carregar_corpus()
    gold = {g["qid"]: g for g in carregar_gabarito()
            if g.get("qrels") and g.get("status", "ok") == "ok"}
    print(f"perguntas respondíveis: {len(gold)}")

    arquivos = sorted(RANKINGS.glob("*.json"))
    if not arquivos:
        sys.exit(f"nenhum ranking em {RANKINGS} — rode os três braços antes.")

    braços, fora = {}, Counter()
    for p in arquivos:
        if args.bracos and p.stem not in args.bracos:
            continue
        dados = json.loads(p.read_text(encoding="utf-8"))
        invalidos = sum(1 for lst in dados.values() for c in lst[:args.profundidade]
                        if c not in corpus)
        fora[p.stem] = invalidos
        braços[p.stem] = dados
    print(f"braços: {', '.join(braços)}")

    if any(fora.values()):
        print("\n✗ há rankings apontando para trechos que não existem no corpus atual:")
        for nome, n in fora.items():
            if n:
                print(f"    {nome}: {n} ids fora do corpus")
        sys.exit("  → os rankings são anteriores à reextração. Rode os braços de novo.")

    # (qid, chunk) -> braços que trouxeram
    duplas: dict[tuple[str, str], set[str]] = defaultdict(set)
    for nome, dados in braços.items():
        for qid, lista in dados.items():
            if qid not in gold:
                continue
            for c in lista[:args.profundidade]:
                duplas[(qid, c)].add(nome)

    # as âncoras entram no pool mesmo que nenhum braço as tenha trazido:
    # sem elas não há controle de calibração.
    ancoras = set()
    for qid, g in gold.items():
        for c in g["qrels"]:
            if c in corpus:
                duplas[(qid, c)].add("ancora")
                ancoras.add((qid, c))

    POOL.parent.mkdir(parents=True, exist_ok=True)
    with POOL.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["qid", "chunk_id", "origem", "e_ancora"])
        for (qid, c), orig in sorted(duplas.items()):
            w.writerow([qid, c, "|".join(sorted(orig)), int((qid, c) in ancoras)])

    por_tipo = Counter(gold[q]["question_type"] for q, _ in duplas)
    print(f"\n✓ {POOL}")
    print(f"  duplas únicas : {len(duplas)}")
    print(f"  âncoras       : {len(ancoras)}  (controle de calibração)")
    print(f"  por tipo      : {dict(por_tipo)}")
    print(f"  por pergunta  : {len(duplas)/max(len(gold),1):.1f} em média")


# ── julgar ───────────────────────────────────────────────────────────────────

def cmd_julgar(args) -> None:
    from lib.common import load_config
    from lib.llm_clients import LLMClient

    if not POOL.exists():
        sys.exit(f"pool não encontrado: {POOL} — rode o subcomando `pool` antes.")
    cfg = load_config(args.config)

    modelo = next((m for m in cfg["judges"]["models"] if m["name"] == args.juiz), None)
    if modelo is None:
        nomes = [m["name"] for m in cfg["judges"]["models"]]
        sys.exit(f"juiz '{args.juiz}' não está em eval/config.yaml (há: {nomes})")

    corpus = carregar_corpus()
    gold = {g["qid"]: g for g in carregar_gabarito()}
    duplas = list(csv.DictReader(POOL.open(encoding="utf-8")))

    saida = Path(args.saida) if args.saida else SAIDA_DIR / f"julgamentos_{args.juiz}.csv"
    feitas = set()
    if saida.exists() and not args.recomecar:
        for r in csv.DictReader(saida.open(encoding="utf-8")):
            feitas.add((r["qid"], r["chunk_id"]))
        print(f"retomando: {len(feitas)} duplas já julgadas em {saida.name}")

    pendentes = [d for d in duplas if (d["qid"], d["chunk_id"]) not in feitas]

    if args.piloto:
        # O piloto PRECISA conter âncoras — são elas o controle. Metade âncora,
        # metade sorteada entre o resto, para a taxa de controle ter denominador.
        random.seed(args.semente)
        anc = [d for d in pendentes if d["e_ancora"] == "1"]
        out = [d for d in pendentes if d["e_ancora"] != "1"]
        n_anc = min(len(anc), args.piloto // 2)
        pendentes = random.sample(anc, n_anc) + random.sample(out, min(len(out), args.piloto - n_anc))
        random.shuffle(pendentes)
        print(f"PILOTO: {len(pendentes)} duplas ({n_anc} âncoras)")

    if not pendentes:
        print("nada a julgar.")
        return

    cliente = LLMClient(provider=modelo["provider"], model=modelo["model"],
                        temperature=cfg["judges"].get("temperature", 0), max_tokens=300)

    novo = not saida.exists() or args.recomecar
    saida.parent.mkdir(parents=True, exist_ok=True)
    with saida.open("w" if novo else "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if novo:
            w.writerow(["qid", "chunk_id", "nota", "justificativa", "juiz", "modelo", "quando"])
        erros = 0
        for i, d in enumerate(pendentes, 1):
            texto = corpus.get(d["chunk_id"], {}).get("text", "")
            pergunta = gold[d["qid"]]["question"]
            try:
                # O prompt recebe pergunta e trecho. Nada de origem nem de âncora.
                r = cliente.chat_json(prompt_dupla(pergunta, texto), system=JUIZ_SYS)
                nota = int(r.get("nota", -1))
                if nota not in (0, 1, 2):
                    raise ValueError(f"nota fora da escala: {r.get('nota')!r}")
                just = str(r.get("justificativa", ""))[:300]
            except Exception as e:
                erros += 1
                print(f"  ! {d['qid']}/{d['chunk_id'][:24]}: {type(e).__name__} {e}")
                continue
            w.writerow([d["qid"], d["chunk_id"], nota, just, args.juiz,
                        modelo["model"], datetime.now().isoformat(timespec="seconds")])
            f.flush()
            if i % 25 == 0:
                print(f"  {i}/{len(pendentes)}", flush=True)

    print(f"\n✓ {saida}   ({len(pendentes)-erros} julgadas, {erros} erros)")
    print(f"→ entregue para a Helen: python eval/tools/julgar_relevancia.py calibrar "
          f"--julgamentos {saida.relative_to(PROJECT_ROOT).as_posix()}")


# ── calibrar (Helen) ─────────────────────────────────────────────────────────

def kappa_cohen(a: list[int], b: list[int]) -> float:
    """κ de Cohen para rótulos ordinais tratados como nominais."""
    n = len(a)
    if n == 0:
        return float("nan")
    obs = sum(1 for x, y in zip(a, b) if x == y) / n
    ca, cb = Counter(a), Counter(b)
    esp = sum((ca[k] / n) * (cb[k] / n) for k in set(ca) | set(cb))
    return (obs - esp) / (1 - esp) if esp < 1 else float("nan")


def cmd_calibrar(args) -> None:
    gold = {g["qid"]: g for g in carregar_gabarito()}
    ancoras = {(q, c) for q, g in gold.items() for c in (g.get("qrels") or {})}

    jl = {(r["qid"], r["chunk_id"]): r
          for r in csv.DictReader(Path(args.julgamentos).open(encoding="utf-8"))}
    if not jl:
        sys.exit("arquivo de julgamentos vazio.")
    modelo = next(iter(jl.values()))["modelo"]

    print("=" * 70)
    print("RELATÓRIO DE CALIBRAÇÃO — Fase 2")
    print("=" * 70)
    print(f"arquivo   : {Path(args.julgamentos).name}")
    print(f"juiz      : {next(iter(jl.values()))['juiz']}  ({modelo})")
    print(f"duplas    : {len(jl)}")

    # 1) controle de âncora
    nas_anc = [(k, v) for k, v in jl.items() if k in ancoras]
    print(f"\n1) CONTROLE DE ÂNCORA  ({len(nas_anc)} âncoras julgadas)")
    if not nas_anc:
        print("   ✗ nenhuma âncora no arquivo — sem controle, o julgamento não é auditável.")
    else:
        dist = Counter(int(v["nota"]) for _, v in nas_anc)
        zeros = dist[0]
        taxa = zeros / len(nas_anc)
        for nota in (2, 1, 0):
            print(f"   nota {nota}: {dist[nota]:4d}  ({dist[nota]/len(nas_anc):5.1%})")
        print(f"\n   taxa de erro (nota 0 em âncora): {taxa:.1%}   limiar: {args.limiar:.0%}")
        print(f"   >>> {'APROVADO' if taxa <= args.limiar else 'REPROVADO'} <<<")
        if taxa > args.limiar:
            print("   O juiz deu 0 em trechos que são relevantes por construção.")
            print("   Troque o modelo ou o prompt e repita o piloto ANTES do lote grande.")

    # 2) distribuição geral
    print(f"\n2) DISTRIBUIÇÃO DAS NOTAS (todas as duplas)")
    dist = Counter(int(v["nota"]) for v in jl.values())
    for nota in (2, 1, 0):
        print(f"   nota {nota}: {dist[nota]:5d}  ({dist[nota]/len(jl):5.1%})")
    if dist[0] / len(jl) > 0.95:
        print("   ⚠ juiz quase sempre dá 0 — desconfie do prompt.")

    # 3) κ com o segundo juiz
    if args.segundo:
        j2 = {(r["qid"], r["chunk_id"]): r
              for r in csv.DictReader(Path(args.segundo).open(encoding="utf-8"))}
        comuns = sorted(set(jl) & set(j2))
        print(f"\n3) CONCORDÂNCIA ENTRE JUÍZES  ({len(comuns)} duplas em comum)")
        if len(comuns) < 30:
            print("   ✗ amostra pequena demais para κ ser informativo (mínimo ~30).")
        else:
            a = [int(jl[k]["nota"]) for k in comuns]
            b = [int(j2[k]["nota"]) for k in comuns]
            k = kappa_cohen(a, b)
            conc = sum(1 for x, y in zip(a, b) if x == y) / len(comuns)
            print(f"   segundo juiz : {next(iter(j2.values()))['modelo']}")
            print(f"   concordância bruta : {conc:.1%}")
            print(f"   κ de Cohen         : {k:.3f}   ({_leitura_kappa(k)})")
            div = [(k_, a_, b_) for k_, a_, b_ in zip(comuns, a, b) if a_ != b_]
            print(f"   divergentes        : {len(div)}")
            if args.divergentes and div:
                p = Path(args.divergentes)
                with p.open("w", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    w.writerow(["qid", "chunk_id", "nota_juiz1", "nota_juiz2", "nota_humana"])
                    for (q, c), x, y in div:
                        w.writerow([q, c, x, y, ""])
                print(f"   ✓ {p}  ← amostra humana para o Matheus (coluna nota_humana em branco)")
    else:
        print("\n3) CONCORDÂNCIA ENTRE JUÍZES — não calculada (falta --segundo)")

    print("\n" + "=" * 70)
    print("Para o artigo: taxa de erro no controle de âncora, κ entre os dois juízes,")
    print("e a ficha do juiz (modelo, versão, prompt, temperatura).")


def _leitura_kappa(k: float) -> str:
    if k != k:
        return "indefinido"
    for lim, rot in ((0.0, "sem concordância"), (0.20, "muito fraca"), (0.40, "fraca"),
                     (0.60, "moderada"), (0.80, "forte")):
        if k < lim:
            return rot
    return "quase perfeita"


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("pool", help="junta o top-N dos braços numa lista de duplas")
    p.add_argument("--profundidade", type=int, default=5, help="top-N por braço (padrão 5)")
    p.add_argument("--bracos", nargs="*", help="nomes dos rankings a usar (padrão: todos)")
    p.set_defaults(func=cmd_pool)

    j = sub.add_parser("julgar", help="roda o juiz LLM sobre as duplas do pool")
    j.add_argument("--juiz", required=True, help="nome em eval/config.yaml (ex.: llama, gptoss)")
    j.add_argument("--config")
    j.add_argument("--saida")
    j.add_argument("--piloto", type=int, help="julga só N duplas, metade âncoras (portão da Helen)")
    j.add_argument("--semente", type=int, default=42)
    j.add_argument("--recomecar", action="store_true", help="ignora o que já foi julgado")
    j.set_defaults(func=cmd_julgar)

    c = sub.add_parser("calibrar", help="HELEN: o juiz merece confiança?")
    c.add_argument("--julgamentos", required=True)
    c.add_argument("--segundo", help="julgamentos do segundo juiz, para κ")
    c.add_argument("--limiar", type=float, default=0.10,
                   help="taxa máxima de nota 0 em âncora (padrão 0,10)")
    c.add_argument("--divergentes", help="grava as duplas divergentes para revisão humana")
    c.set_defaults(func=cmd_calibrar)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
