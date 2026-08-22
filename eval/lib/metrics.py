"""
eval/lib/metrics.py
═══════════════════
Métricas de recuperação para o pipeline de avaliação RAG.

Inclui:
  • nDCG@k (graded, 2^rel - 1)         — métrica obrigatória
  • Recall@k, MRR@k, MAP               — complementares
    risco-sensível, compara cada sistema contra o POOL de sistemas.

Convenções de tipos:
    qrels    : {qid: {chunk_id: grade}}   grade inteiro >= 1 (0 = não-relevante)
    rankings : {qid: [chunk_id, ...]}     ordem decrescente de relevância prevista

Dependências: numpy (obrigatório), scipy (opcional; há fallback via math.erf).
"""
from __future__ import annotations

import math
from typing import Dict, List, Sequence

import numpy as np

try:                                    # CDF normal padrão
    from scipy.stats import norm
    def _phi(x: float) -> float:
        return float(norm.cdf(x))
except Exception:                       # fallback sem scipy
    def _phi(x: float) -> float:
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


# ── Métricas por query ──────────────────────────────────────────────────────

def dcg_at_k(ranked: Sequence[str], rel: Dict[str, int], k: int) -> float:
    return sum((2 ** rel.get(e, 0) - 1) / math.log2(i + 2)
               for i, e in enumerate(ranked[:k]))


def ndcg_at_k(ranked: Sequence[str], rel: Dict[str, int], k: int) -> float:
    """nDCG@k com ganho exponencial (idêntico ao usado no ir_pipeline)."""
    dcg = dcg_at_k(ranked, rel, k)
    ideal = sorted(rel.values(), reverse=True)[:k]
    idcg = sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def recall_at_k(ranked: Sequence[str], rel: Dict[str, int], k: int) -> float:
    relevant = {e for e, g in rel.items() if g > 0}
    if not relevant:
        return 0.0
    hit = sum(1 for e in ranked[:k] if e in relevant)
    return hit / len(relevant)


def precision_at_k(ranked: Sequence[str], rel: Dict[str, int], k: int) -> float:
    if k <= 0:
        return 0.0
    relevant = {e for e, g in rel.items() if g > 0}
    hit = sum(1 for e in ranked[:k] if e in relevant)
    return hit / k


def f1_at_k(ranked: Sequence[str], rel: Dict[str, int], k: int) -> float:
    """F1@k = média harmônica de Precision@k e Recall@k."""
    p = precision_at_k(ranked, rel, k)
    r = recall_at_k(ranked, rel, k)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def mrr_at_k(ranked: Sequence[str], rel: Dict[str, int], k: int) -> float:
    relevant = {e for e, g in rel.items() if g > 0}
    for i, e in enumerate(ranked[:k]):
        if e in relevant:
            return 1.0 / (i + 1)
    return 0.0


def average_precision(ranked: Sequence[str], rel: Dict[str, int], k: int = None) -> float:
    relevant = {e for e, g in rel.items() if g > 0}
    if not relevant:
        return 0.0
    if k is None:
        k = len(ranked)
    hits, score = 0, 0.0
    for i, e in enumerate(ranked[:k]):
        if e in relevant:
            hits += 1
            score += hits / (i + 1)
    return score / len(relevant)


# ── Agregação de um run completo ────────────────────────────────────────────

def per_query_ndcg(rankings: Dict[str, List[str]],
                   qrels: Dict[str, Dict[str, int]],
                   k: int) -> Dict[str, float]:
    """nDCG@k por query — insumo do GeoRisk."""
    return {qid: ndcg_at_k(rankings.get(qid, []), rel, k)
            for qid, rel in qrels.items()}


def evaluate_run(rankings: Dict[str, List[str]],
                 qrels: Dict[str, Dict[str, int]],
                 ndcg_k: Sequence[int] = (5, 10),
                 recall_k: Sequence[int] = (5, 10),
                 precision_k: Sequence[int] = (5,),
                 f1_k: Sequence[int] = (5,),
                 mrr_k: int = 10) -> dict:
    """Retorna métricas médias + dicionários por query (para significância/GeoRisk)."""
    qids = list(qrels.keys())
    out: dict = {"n_queries": len(qids), "per_query": {}, "mean": {}}

    def _agg(name, fn):
        pq = {qid: fn(rankings.get(qid, []), qrels[qid]) for qid in qids}
        out["per_query"][name] = pq
        out["mean"][name] = float(np.mean(list(pq.values()))) if pq else 0.0

    for k in ndcg_k:
        _agg(f"ndcg@{k}", lambda r, rel, k=k: ndcg_at_k(r, rel, k))
    for k in precision_k:
        _agg(f"p@{k}", lambda r, rel, k=k: precision_at_k(r, rel, k))
    for k in recall_k:
        _agg(f"recall@{k}", lambda r, rel, k=k: recall_at_k(r, rel, k))
    for k in f1_k:
        _agg(f"f1@{k}", lambda r, rel, k=k: f1_at_k(r, rel, k))
    _agg(f"mrr@{mrr_k}", lambda r, rel: mrr_at_k(r, rel, mrr_k))
    _agg("map", lambda r, rel: average_precision(r, rel))

    return out


# ── GeoRisk ─────────────────────────────────────────────────────────────────

def geo_risk_matrix(S: np.ndarray, alpha: float = 2.0) -> np.ndarray:
    """
    GeoRisk para uma matriz S de efetividade [n_sistemas × n_queries].

    Para cada sistema i e query j:
        E_ij   = S_i · T_j / N         (valor esperado tipo tabela de contingência)
        t_ij   = (S_ij − E_ij) / √E_ij
                 (multiplicado por (1+α) quando S_ij < E_ij → penaliza risco)
        zRisk_i = Σ_j t_ij
        GeoRisk_i = √( (S_i / n) · Φ(zRisk_i / n) )

    onde S_i = soma da linha, T_j = soma da coluna, N = soma total, n = nº de queries,
    Φ = CDF normal padrão.

    Ref.: Dinçer, Macdonald, Ounis. "Hypothesis testing for the risk-sensitive
    evaluation of retrieval systems." SIGIR 2014.
    """
    S = np.asarray(S, dtype=float)
    if S.ndim != 2:
        raise ValueError("S deve ser matriz [n_sistemas × n_queries].")
    m, n = S.shape
    Si = S.sum(axis=1)            # totais por sistema (linhas)
    Tj = S.sum(axis=0)            # totais por query  (colunas)
    N = S.sum()                   # total geral

    geo = np.zeros(m)
    if N <= 0:
        return geo

    for i in range(m):
        zrisk = 0.0
        for j in range(n):
            eij = Si[i] * Tj[j] / N
            if eij <= 0:
                continue
            t = (S[i, j] - eij) / math.sqrt(eij)
            if S[i, j] < eij:     # sistema ficou abaixo do esperado → penaliza
                t *= (1.0 + alpha)
            zrisk += t
        geo[i] = math.sqrt((Si[i] / n) * _phi(zrisk / n)) if Si[i] > 0 else 0.0
    return geo


def geo_risk(per_query_scores: Dict[str, Dict[str, float]],
             alpha: float = 2.0) -> Dict[str, float]:
    """
    Wrapper amigável do GeoRisk.

    per_query_scores : {nome_sistema: {qid: score}}   (ex.: nDCG@10 por query)
    Alinha as queries comuns a todos os sistemas e devolve {nome_sistema: georisk}.
    """
    systems = list(per_query_scores.keys())
    if not systems:
        return {}
    # interseção de qids presentes em todos os sistemas (comparação justa)
    common = set.intersection(*[set(per_query_scores[s].keys()) for s in systems])
    qids = sorted(common)
    if not qids:
        return {s: 0.0 for s in systems}

    S = np.array([[per_query_scores[s][q] for q in qids] for s in systems], dtype=float)
    geo = geo_risk_matrix(S, alpha=alpha)
    return {s: float(g) for s, g in zip(systems, geo)}


# ── Self-test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("── Self-test métricas ──")

    # nDCG: ranking perfeito = 1.0
    rel = {"a": 2, "b": 1, "c": 1}
    assert abs(ndcg_at_k(["a", "b", "c"], rel, 10) - 1.0) < 1e-9
    assert ndcg_at_k(["c", "b", "a"], rel, 10) < 1.0
    assert abs(recall_at_k(["a", "b", "x"], rel, 10) - 2 / 3) < 1e-9
    assert abs(mrr_at_k(["x", "a"], rel, 10) - 0.5) < 1e-9
    print("  nDCG / Recall / MRR  ✓")

    # GeoRisk — propriedade: sistemas idênticos → zRisk=0 → georisk=√(Si/n · 0.5)
    S_equal = np.array([[0.4, 0.8, 0.6], [0.4, 0.8, 0.6], [0.4, 0.8, 0.6]])
    g_eq = geo_risk_matrix(S_equal, alpha=2.0)
    Si = S_equal.sum(axis=1)[0]
    expected = math.sqrt((Si / 3) * 0.5)
    assert all(abs(g - expected) < 1e-9 for g in g_eq), (g_eq, expected)
    print(f"  GeoRisk sistemas idênticos = {g_eq[0]:.4f} (= √(Si/n·0.5))  ✓")

    # GeoRisk — sistema melhor na média pontua mais alto que um fraco.
    scores = {
        "forte": {"q1": 0.70, "q2": 0.72, "q3": 0.68, "q4": 0.71},
        "fraco": {"q1": 0.30, "q2": 0.28, "q3": 0.31, "q4": 0.29},
    }
    g = geo_risk(scores, alpha=2.0)
    print("  GeoRisk forte vs fraco:", {k: round(v, 4) for k, v in g.items()})
    assert g["forte"] > g["fraco"], g

    # Isola o EFEITO DE RISCO: 'steady' e 'swingy' têm a MESMA média (0.6),
    # mas o instável é penalizado por α nas queries em que cai abaixo do esperado.
    risk = {
        "steady": {"q1": 0.60, "q2": 0.60, "q3": 0.60, "q4": 0.60},
        "swingy": {"q1": 0.95, "q2": 0.25, "q3": 0.95, "q4": 0.25},
    }
    gr = geo_risk(risk, alpha=2.0)
    print("  GeoRisk steady vs swingy (mesma média):",
          {k: round(v, 4) for k, v in gr.items()})
    assert gr["steady"] > gr["swingy"], f"risco deveria penalizar swingy: {gr}"
    print("  risco penalizado (steady > swingy, mesma média)  ✓")

    print("Todos os testes passaram.")
