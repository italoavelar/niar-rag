#!/usr/bin/env python3
"""Análise da auditoria humana da recuperação, conforme o desenho pré-registrado.

O desenho está em `auditoria/DESENHO_ANALISE_AUDITORIA.md`, escrito em 26/11/2026
antes de qualquer nota humana ser cruzada com o gabarito. Este script só executa
o que está lá; desvio entra na seção "Desvios" do desenho, não aqui.

Entradas
  auditoria/AUDITORIA_RECUPERACAO.xlsx                    especialista 1 (78 perguntas)
  auditoria_anotadores/AUDITORIA_RECUPERACAO_anotador_2.xlsx  anotador 2 (lote 1, 30)
  auditoria/auditoria_trechos.csv e
  auditoria_anotadores/auditoria_trechos_anotador_2.csv   ligam linha -> trecho_id
  eval/data/golden_dezembro.jsonl                         qrels_grupos e tipo

Saídas (em auditoria/)
  resultados_auditoria.json      todos os números
  casos_para_leitura.csv         casos listados pelas análises 3 e 4

Uso:
  python eval/experimento_embedding/analisar_auditoria.py
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parents[1]
AUD = AQUI / "auditoria"
ANOT = AQUI / "auditoria_anotadores"
GABARITO = RAIZ / "eval/data/golden_dezembro.jsonl"

COL_REL = "JUIZ: relevância 0-3"
COL_OBS = "JUIZ: observação do trecho"
COL_VER = "JUIZ: os 5 bastam? S/PARCIAL/N"
COL_FALTA = "JUIZ: o que falta"
VER_ORD = {"N": 0, "PARCIAL": 1, "S": 2}
SEMENTE = 2026
N_BOOT = 10_000

# População (225 respondíveis, BGE-m3 + reranker): acertos / erros por tipo.
# Conferido com analisar_bracos.py (Complete Evidence@5 por tipo: 70, 38, 13 de 75).
POP = {("factual", True): 70, ("factual", False): 5,
       ("multi_hop", True): 38, ("multi_hop", False): 37,
       ("comparative", True): 13, ("comparative", False): 62}

for _f in (sys.stdout, sys.stderr):
    try:
        _f.reconfigure(encoding="utf-8")
    except Exception:
        pass


# ── leitura ────────────────────────────────────────────────────────────────
def ler(planilha: Path, trechos_csv: Path, quem: str) -> pd.DataFrame:
    d = pd.read_excel(planilha, sheet_name="auditoria")
    t = pd.read_csv(trechos_csv, sep=";")
    d["qid"] = d["qid"].ffill().astype(str)
    if len(d) != len(t) or (d["qid"].values != t["qid"].astype(str).values).any():
        sys.exit(f"! {quem}: planilha e {trechos_csv.name} desalinhadas")
    d["trecho_id"] = t["trecho_id"].values
    d["rel"] = pd.to_numeric(d[COL_REL], errors="coerce")
    d["anotador"] = quem
    return d


def vereditos(d: pd.DataFrame) -> pd.DataFrame:
    v = d.dropna(subset=[COL_VER])[["qid", COL_VER, COL_FALTA]].copy()
    v["ver"] = v[COL_VER].astype(str).str.strip().str.upper()
    if not v["ver"].isin(VER_ORD).all() or v["qid"].duplicated().any():
        sys.exit("! veredito fora de S/PARCIAL/N ou duplicado")
    return v.set_index("qid")


# ── estatística ────────────────────────────────────────────────────────────
def kappa(a, b, k: int, pesos: str | None) -> float:
    """κ de Cohen; pesos None (nominal) ou 'quad'. a, b: inteiros 0..k-1."""
    a, b = np.asarray(a, int), np.asarray(b, int)
    O = np.zeros((k, k))
    for x, y in zip(a, b):
        O[x, y] += 1
    O /= O.sum()
    E = np.outer(O.sum(1), O.sum(0))
    i, j = np.indices((k, k))
    W = ((i - j) ** 2 / (k - 1) ** 2) if pesos == "quad" else (i != j).astype(float)
    den = (W * E).sum()
    return float("nan") if den == 0 else float(1 - (W * O).sum() / den)


def kappa_boot(pares: pd.DataFrame, k: int, pesos: str | None) -> dict:
    """κ com IC 95% por bootstrap de pergunta (cluster)."""
    est = kappa(pares["a"], pares["b"], k, pesos)
    grupos = [g for _, g in pares.groupby("qid")]
    rng = random.Random(SEMENTE)
    vals = []
    for _ in range(N_BOOT):
        amostra = pd.concat([grupos[rng.randrange(len(grupos))] for _ in grupos])
        v = kappa(amostra["a"], amostra["b"], k, pesos)
        if not np.isnan(v):
            vals.append(v)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return {"kappa": round(est, 3), "ic95": [round(float(lo), 3), round(float(hi), 3)],
            "n_itens": int(len(pares)), "n_perguntas": len(grupos),
            "concordancia_bruta": round(float((pares["a"] == pares["b"]).mean()), 3)}


def confusao(a, b, rotulos) -> list:
    return pd.crosstab(pd.Categorical(a, rotulos), pd.Categorical(b, rotulos),
                       dropna=False).values.tolist()


def ponderada(df: pd.DataFrame, col: str) -> dict:
    """Taxa de `col` (0/1) ponderada pelas células tipo × acerto da população."""
    linhas, total = [], 0.0
    for (tipo, ac), N in POP.items():
        c = df[(df["tipo"] == tipo) & (df["auto"] == ac)]
        if len(c) == 0:
            continue
        p = c[col].mean()
        linhas.append((N, p))
    W = sum(N for N, _ in linhas)
    total = sum(N * p for N, p in linhas) / W
    return {"ponderada": round(float(total), 3), "sem_peso": round(float(df[col].mean()), 3),
            "n": int(len(df)), "populacao_coberta": int(W)}


# ── análise ────────────────────────────────────────────────────────────────
def main() -> None:
    gold = {str(r["qid"]): r for r in map(json.loads, GABARITO.open(encoding="utf-8"))}
    e1 = ler(AUD / "AUDITORIA_RECUPERACAO.xlsx", AUD / "auditoria_trechos.csv", "especialista_1")
    a2 = ler(ANOT / "AUDITORIA_RECUPERACAO_anotador_2.xlsx",
             ANOT / "auditoria_trechos_anotador_2.csv", "anotador_2")
    a2 = a2[a2["rel"].notna() | a2["qid"].isin(a2.dropna(subset=["rel"])["qid"])]
    a2 = a2[a2["qid"].isin(a2.dropna(subset=["rel"])["qid"])]

    # gabarito por trecho e acerto automático por pergunta (sobre os 5 exibidos)
    for d in (e1, a2):
        d["no_gab"] = [any(t in g for g in gold[q]["qrels_grupos"])
                       for q, t in zip(d["qid"], d["trecho_id"])]
    perg = (e1.groupby("qid")["trecho_id"].apply(set).rename("top5").to_frame())
    perg["tipo"] = [gold[q]["question_type"] for q in perg.index]
    perg["origem"] = [gold[q].get("origem") or gold[q].get("status") for q in perg.index]
    perg["auto"] = [all(any(c in s for c in g) for g in gold[q]["qrels_grupos"])
                    for q, s in zip(perg.index, perg["top5"])]
    lote1 = sorted(a2["qid"].unique())

    R: dict = {"desenho": "auditoria/DESENHO_ANALISE_AUDITORIA.md",
               "amostra": {
                   "especialista_1": {"perguntas": int(e1["qid"].nunique()),
                                      "trechos_com_nota": int(e1["rel"].notna().sum()),
                                      "vereditos": int(len(vereditos(e1)))},
                   "anotador_2": {"perguntas": len(lote1),
                                  "trechos_com_nota": int(a2["rel"].notna().sum()),
                                  "vereditos": int(len(vereditos(a2)))},
                   "celulas": {f"{t}|{'acerta' if a else 'erra'}": int(n) for (t, a), n in
                               perg.groupby(["tipo", "auto"]).size().items()},
               }}

    # ── Análise 1: concordância entre anotadores (lote 1) ──
    p = e1[e1["qid"].isin(lote1)].merge(a2[["qid", "trecho_id", "rel"]],
                                         on=["qid", "trecho_id"], suffixes=("_e1", "_a2"))
    p = p.dropna(subset=["rel_e1", "rel_a2"])
    pares_rel = pd.DataFrame({"qid": p["qid"], "a": p["rel_e1"].astype(int),
                              "b": p["rel_a2"].astype(int)})
    v1, v2 = vereditos(e1), vereditos(a2)
    comuns = sorted(set(v1.index) & set(v2.index))
    pares_ver = pd.DataFrame({"qid": comuns,
                              "a": [VER_ORD[v1.loc[q, "ver"]] for q in comuns],
                              "b": [VER_ORD[v2.loc[q, "ver"]] for q in comuns]})
    R["analise1_concordancia"] = {
        "nota_trecho": {**kappa_boot(pares_rel, 4, "quad"),
                        "confusao_linhas_e1_colunas_a2": confusao(pares_rel["a"], pares_rel["b"], [0, 1, 2, 3]),
                        "marginal_e1": pares_rel["a"].value_counts().sort_index().to_dict(),
                        "marginal_a2": pares_rel["b"].value_counts().sort_index().to_dict()},
        "veredito": {**kappa_boot(pares_ver, 3, "quad"),
                     "confusao_linhas_e1_colunas_a2_N_P_S": confusao(pares_ver["a"], pares_ver["b"], [0, 1, 2])},
    }

    # ── Análise 2: o especialista 1 é mais favorável? ──
    media = p.groupby("qid")[["rel_e1", "rel_a2"]].mean()
    dif = media["rel_e1"] - media["rel_a2"]
    w = wilcoxon(dif[dif != 0]) if (dif != 0).sum() else None
    mais = int((pares_ver["a"] > pares_ver["b"]).sum())
    menos = int((pares_ver["a"] < pares_ver["b"]).sum())
    sinal = binomtest(mais, mais + menos, 0.5).pvalue if mais + menos else 1.0
    p_w = float(w.pvalue) if w is not None else 1.0
    vies = (p_w < 0.05 and dif.mean() > 0) or (sinal < 0.05 and mais > menos)
    R["analise2_vies_especialista_1"] = {
        "nota_media_por_pergunta": {"e1": round(float(media["rel_e1"].mean()), 3),
                                    "a2": round(float(media["rel_a2"].mean()), 3),
                                    "diferenca_media_e1_menos_a2": round(float(dif.mean()), 3),
                                    "perguntas_e1_maior": int((dif > 0).sum()),
                                    "perguntas_e1_menor": int((dif < 0).sum()),
                                    "empates": int((dif == 0).sum()),
                                    "wilcoxon_p": round(p_w, 4)},
        "veredito": {"e1_mais_favoravel": mais, "e1_menos_favoravel": menos,
                     "teste_sinal_p": round(float(sinal), 4)},
        "regra": "vies se algum p < 0,05 na direcao e1 mais favoravel",
        "vies_detectado": bool(vies),
        "base_principal": "anotador_2 (30)" if vies else "especialista_1 (78)",
    }

    # ── Análises 3 e 4, por anotador ──
    casos = []
    for nome, d, v in (("especialista_1", e1, v1), ("anotador_2", a2, v2)):
        q = perg.loc[[x for x in v.index if x in perg.index]].copy()
        q["ver"] = v.loc[q.index, "ver"]
        q["humano_S"] = (q["ver"] == "S").astype(int)
        q["humano_SP"] = q["ver"].isin(["S", "PARCIAL"]).astype(int)
        q["auto_i"] = q["auto"].astype(int)
        res3 = {}
        for col, rot in (("humano_S", "principal_so_S"), ("humano_SP", "sensibilidade_S_ou_PARCIAL")):
            tab = pd.crosstab(q[col], q["auto_i"]).reindex(index=[0, 1], columns=[0, 1], fill_value=0)
            res3[rot] = {
                "tabela_linhas_humano_0_1_colunas_auto_0_1": tab.values.tolist(),
                "concordancia_bruta": round(float((q[col] == q["auto_i"]).mean()), 3),
                "kappa": round(kappa(q[col], q["auto_i"], 2, None), 3),
                "taxa_humano_acerto": ponderada(q, col),
                "por_tipo": {t: {"n": int(len(g)), "humano": int(g[col].sum()), "auto": int(g["auto_i"].sum()),
                                 "concordam": int((g[col] == g["auto_i"]).sum())}
                             for t, g in q.groupby("tipo")},
            }
        res3["tabela_3x2_linhas_N_P_S_colunas_auto_erra_acerta"] = (
            pd.crosstab(q["ver"], q["auto_i"]).reindex(index=["N", "PARCIAL", "S"], columns=[0, 1],
                                                       fill_value=0).values.tolist())
        res3["taxa_auto_acerto_na_amostra"] = ponderada(q, "auto_i")
        for qid, r in q.iterrows():
            if r["auto_i"] == 0 and r["ver"] == "S":
                casos.append({"anotador": nome, "analise": "3: auto erra, humano S", "qid": qid,
                              "tipo": r["tipo"], "o_que_falta": v.loc[qid, COL_FALTA]})
            if r["auto_i"] == 1 and r["ver"] == "N":
                casos.append({"anotador": nome, "analise": "3: auto acerta, humano N", "qid": qid,
                              "tipo": r["tipo"], "o_que_falta": v.loc[qid, COL_FALTA]})

        t = d.dropna(subset=["rel"]).copy()
        t["relev2"] = (t["rel"] >= 2).astype(int)
        t["relev3"] = (t["rel"] >= 3).astype(int)
        t["gab"] = t["no_gab"].astype(int)
        res4 = {}
        for col, rot in (("relev2", "principal_nota_ge_2"), ("relev3", "sensibilidade_nota_3")):
            tab = pd.crosstab(t[col], t["gab"]).reindex(index=[0, 1], columns=[0, 1], fill_value=0)
            res4[rot] = {"tabela_linhas_humano_0_1_colunas_gabarito_0_1": tab.values.tolist(),
                         "kappa": round(kappa(t[col], t["gab"], 2, None), 3)}
        res4["nota_por_status_no_gabarito"] = {
            "no_gabarito": t[t["gab"] == 1]["rel"].value_counts().sort_index().to_dict(),
            "fora_do_gabarito": t[t["gab"] == 0]["rel"].value_counts().sort_index().to_dict()}
        for _, r in t.iterrows():
            if r["rel"] == 3 and r["gab"] == 0:
                casos.append({"anotador": nome, "analise": "4: nota 3 fora do gabarito", "qid": r["qid"],
                              "tipo": perg.loc[r["qid"], "tipo"], "trecho_id": r["trecho_id"],
                              "observacao": r[COL_OBS]})
            if r["rel"] <= 1 and r["gab"] == 1:
                casos.append({"anotador": nome, "analise": "4: no gabarito, nota 0-1", "qid": r["qid"],
                              "tipo": perg.loc[r["qid"], "tipo"], "trecho_id": r["trecho_id"],
                              "nota": int(r["rel"]), "observacao": r[COL_OBS]})
        R[f"analise3_humano_x_auto__{nome}"] = res3
        R[f"analise4_trechos__{nome}"] = res4

    # repetição do especialista 1 só nas 30 do lote 1, para comparar com o anotador 2
    q30 = perg.loc[[x for x in v1.index if x in lote1]].copy()
    q30["humano_S"] = (v1.loc[q30.index, "ver"] == "S").astype(int)
    R["analise3_especialista_1_so_lote1"] = {
        "n": int(len(q30)),
        "concordancia_bruta_so_S": round(float((q30["humano_S"] == q30["auto"].astype(int)).mean()), 3),
        "kappa_so_S": round(kappa(q30["humano_S"], q30["auto"].astype(int), 2, None), 3)}

    (AUD / "resultados_auditoria.json").write_text(
        json.dumps(R, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    pd.DataFrame(casos).to_csv(AUD / "casos_para_leitura.csv", sep=";", index=False, encoding="utf-8-sig")
    print(json.dumps(R, ensure_ascii=False, indent=1, default=str))
    print(f"\n{len(casos)} casos em auditoria/casos_para_leitura.csv")


if __name__ == "__main__":
    main()
