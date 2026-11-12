#!/usr/bin/env python3
"""
eval/tools/matriz_qualidade.py
══════════════════════════════
Agrega a QUALIDADE dos geradores a partir dos judged_*.jsonl de geração, pelo
VOTO DE MAIORIA do painel aberto (llama + glm + mistral; gemini é referência e
fica fora do voto). Não chama API, não depende de config de geradores: lê os
dados julgados e calcula as médias.

Saídas (em results/generation/):
  matriz_qualidade.csv   — gerador × cenário × critério (média / taxa)
  matriz_por_tipo.csv    — correctness por tipo de pergunta (cenário RAG)
  matriz_recusa.csv      — comportamento de recusa nas duas pontas

Convenções:
  • TERNÁRIOS (1-3): faithfulness, answer_relevance, completeness, correctness
    → média das notas inteiras (ignora "na").
  • BINÁRIOS (0/1): citation, refusal, hallucination → taxa de 1 (% ok).
  • faithfulness/hallucination só no cenário RAG (sem contexto = indefinidos).
  • answer_relevance/correctness/completeness ignoram "na" (recusa, sem-ref).
  • O gerador é identificado pelo nome do arquivo (qwen3_27b / gptoss_120b).

Uso:  python eval/tools/matriz_qualidade.py
"""
from __future__ import annotations
import csv
import glob
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

PAINEL_ABERTO = ["llama", "glm", "mistral"]        # gemini é referência → fora
TERNARIOS = ["faithfulness", "answer_relevance", "completeness", "correctness"]
BINARIOS = ["citation", "refusal", "hallucination"]
SO_RAG = {"faithfulness", "hallucination"}         # indefinidos sem contexto

GEN_DIR = Path("eval/results/generation")


def gerador_do_arquivo(nome: str) -> str | None:
    if "qwen3_27b" in nome:
        return "qwen3_27b"
    if "gptoss_120b" in nome:
        return "gptoss_120b"
    return None                                     # retrieval/outros → ignora


def voto_maioria(valores):
    """Voto de maioria de uma lista de notas (ints). Ignora não-ints ('na').
    Devolve o valor mais frequente; empate → o menor (conservador). None se vazio."""
    nums = [v for v in valores if isinstance(v, int)]
    if not nums:
        return None
    cont = {}
    for v in nums:
        cont[v] = cont.get(v, 0) + 1
    maxc = max(cont.values())
    candidatos = sorted(v for v, c in cont.items() if c == maxc)
    return candidatos[0]


def carrega():
    """Lê os judged_*.jsonl de geração; devolve lista de (gerador, registro)."""
    out = []
    for fp in sorted(glob.glob(str(GEN_DIR / "judged_*.jsonl"))):
        if fp.endswith("_TESTE.jsonl"):
            continue
        g = gerador_do_arquivo(Path(fp).name)
        if g is None:
            continue
        for l in open(fp, encoding="utf-8"):
            if l.strip():
                out.append((g, json.loads(l)))
    return out


def nota_painel(r, crit):
    """Voto de maioria do painel aberto para um critério num registro."""
    s = r.get("scores", {})
    vals = [(s.get(j) or {}).get(crit) for j in PAINEL_ABERTO]
    return voto_maioria(vals)


def main():
    dados = carrega()
    if not dados:
        sys.exit("! nenhum judged de geração (qwen3_27b / gptoss_120b).")
    print(f"Registros de geração: {len(dados)}  | painel: {'+'.join(PAINEL_ABERTO)}")

    # ── 1) matriz gerador × cenário × critério ──────────────────────────────
    # acumula notas por (gerador, cenario, criterio)
    tern = defaultdict(list)          # média
    bina = defaultdict(list)          # taxa de 1
    for g, r in dados:
        cen = r.get("cenario", "rag" if r.get("context_chunk_ids") else "norag")
        for c in TERNARIOS:
            if c in SO_RAG and cen != "rag":
                continue
            v = nota_painel(r, c)
            if isinstance(v, int):
                tern[(g, cen, c)].append(v)
        for c in BINARIOS:
            if c in SO_RAG and cen != "rag":
                continue
            v = nota_painel(r, c)
            if isinstance(v, int):
                bina[(g, cen, c)].append(v)

    linhas = []
    for g in ("qwen3_27b", "gptoss_120b"):
        for cen in ("rag", "norag"):
            for c in TERNARIOS:
                vals = tern.get((g, cen, c))
                if vals:
                    linhas.append((g, cen, c, "media_1a3",
                                   round(statistics.mean(vals), 3), len(vals)))
            for c in BINARIOS:
                vals = bina.get((g, cen, c))
                if vals:
                    linhas.append((g, cen, c, "taxa_ok",
                                   round(100*statistics.mean(vals), 1), len(vals)))
    with open(GEN_DIR / "matriz_qualidade.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["gerador", "cenario", "criterio", "metrica", "valor", "n"])
        w.writerows(linhas)

    # ── 2) correctness por tipo de pergunta (cenário RAG) ───────────────────
    por_tipo = defaultdict(list)
    for g, r in dados:
        if r.get("cenario") != "rag":
            continue
        tp = r.get("question_type", "?")
        v = nota_painel(r, "correctness")
        if isinstance(v, int):
            por_tipo[(g, tp)].append(v)
    with open(GEN_DIR / "matriz_por_tipo.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["gerador", "question_type", "correctness_media", "n"])
        for (g, tp), vals in sorted(por_tipo.items()):
            w.writerow([g, tp, round(statistics.mean(vals), 3), len(vals)])

    # ── 3) comportamento de recusa (duas pontas) ────────────────────────────
    # respondíveis: % refusal=1 (respondeu certo quando devia responder)
    # irrespondíveis: % refusal=1 (recusou certo quando devia recusar)
    recusa = defaultdict(lambda: [0, 0])      # [acertos, total]
    for g, r in dados:
        cen = r.get("cenario")
        v = nota_painel(r, "refusal")
        if not isinstance(v, int):
            continue
        tipo_resp = "respondivel" if r.get("should_answer") else "irrespondivel"
        key = (g, cen, tipo_resp)
        recusa[key][1] += 1
        recusa[key][0] += v                    # refusal=1 é comportamento correto
    with open(GEN_DIR / "matriz_recusa.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["gerador", "cenario", "tipo_pergunta", "pct_comportamento_correto", "n"])
        for (g, cen, tp), (ac, tot) in sorted(recusa.items()):
            if tot:
                w.writerow([g, cen, tp, round(100*ac/tot, 1), tot])

    # ── console ─────────────────────────────────────────────────────────────
    print("\n=== QUALIDADE por critério (voto de maioria do painel aberto) ===")
    print(f"{'gerador':<13}{'cenario':<7}{'criterio':<18}{'valor':>8}{'n':>6}")
    print("-"*54)
    for g, cen, c, met, val, n in linhas:
        suf = "%" if met == "taxa_ok" else ""
        print(f"{g:<13}{cen:<7}{c:<18}{val:>7}{suf}{n:>6}")

    print("\n=== correctness por tipo (RAG) ===")
    for (g, tp), vals in sorted(por_tipo.items()):
        print(f"  {g:<13}{tp:<14}{round(statistics.mean(vals),3):>6}  (n={len(vals)})")

    print("\n=== recusa (duas pontas, % comportamento correto) ===")
    for (g, cen, tp), (ac, tot) in sorted(recusa.items()):
        if tot:
            print(f"  {g:<13}{cen:<7}{tp:<14}{round(100*ac/tot,1):>6}%  (n={tot})")

    print(f"\n✓ matriz_qualidade.csv · matriz_por_tipo.csv · matriz_recusa.csv "
          f"(em {GEN_DIR})")


if __name__ == "__main__":
    main()
