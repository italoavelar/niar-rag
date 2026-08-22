"""
eval/lib/stats.py
═════════════════
Ferramentas estatísticas do protocolo de avaliação (Etapas 04 e 05):

  • paired_randomization   — teste de randomização (permutação) pareado, bicaudal
  • wilcoxon_p             — Wilcoxon signed-rank (robustez não-paramétrica)
  • bootstrap_mean_diff_ci — IC 95% da diferença média pareada (tamanho de efeito)
  • cliffs_delta           — tamanho de efeito ordinal
  • holm_bonferroni / benjamini_hochberg — correção para múltiplas comparações
  • bootstrap_ci_pairs     — IC 95% genérico p/ métricas de concordância (κ, α, ρ)

Tudo pareado por consulta (mesmo conjunto de queries em todas as condições).
"""
from __future__ import annotations

from typing import Callable, List, Sequence, Tuple

import numpy as np

try:
    from scipy.stats import wilcoxon as _wilcoxon
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


def paired_randomization(x: Sequence[float], y: Sequence[float],
                         n: int = 10000, seed: int = 0) -> Tuple[float, float]:
    """Retorna (Δ médio = média(x−y), p-valor bicaudal)."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    d = x - y
    if len(d) == 0:
        return 0.0, 1.0
    obs = d.mean()
    if np.allclose(d, 0):
        return 0.0, 1.0
    rng = np.random.default_rng(seed)
    signs = rng.choice([1.0, -1.0], size=(n, len(d)))
    perm = (signs * d).mean(axis=1)
    p = (np.sum(np.abs(perm) >= abs(obs)) + 1) / (n + 1)
    return float(obs), float(p)


def wilcoxon_p(x: Sequence[float], y: Sequence[float]) -> float:
    """p-valor do Wilcoxon signed-rank pareado (bicaudal). NaN se indisponível."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    d = x - y
    if len(d) < 1 or np.allclose(d, 0):
        return 1.0
    if not _HAS_SCIPY:
        return float("nan")
    try:
        return float(_wilcoxon(x, y, zero_method="wilcox", alternative="two-sided").pvalue)
    except Exception:
        return float("nan")


def bootstrap_mean_diff_ci(x: Sequence[float], y: Sequence[float],
                           n_boot: int = 10000, alpha: float = 0.05,
                           seed: int = 0) -> Tuple[float, float, float]:
    """IC (1−alpha) da diferença média pareada via bootstrap percentil.
    Retorna (Δ, lo, hi)."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    d = x - y
    if len(d) == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    means = d[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(d.mean()), float(lo), float(hi)


def cliffs_delta(x: Sequence[float], y: Sequence[float]) -> float:
    """Cliff's δ ∈ [−1, 1]: P(x>y) − P(x<y) sobre todos os pares.
    >0 indica x tende a ser maior que y."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    nx, ny = len(x), len(y)
    if nx == 0 or ny == 0:
        return 0.0
    if nx * ny > 5_000_000:                      # evita matriz gigante
        gt = sum(int(np.count_nonzero(xi > y)) for xi in x)
        lt = sum(int(np.count_nonzero(xi < y)) for xi in x)
        return (gt - lt) / (nx * ny)
    diff = np.subtract.outer(x, y)
    return float(np.sign(diff).sum() / (nx * ny))


def holm_bonferroni(pvals: Sequence[float]) -> List[float]:
    """p-valores ajustados por Holm–Bonferroni (controle do erro família)."""
    p = np.asarray(pvals, float)
    m = len(p)
    if m == 0:
        return []
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adj[idx] = min(running, 1.0)
    return adj.tolist()


def benjamini_hochberg(pvals: Sequence[float]) -> List[float]:
    """p-valores ajustados por Benjamini–Hochberg (controle de FDR)."""
    p = np.asarray(pvals, float)
    m = len(p)
    if m == 0:
        return []
    order = np.argsort(p)
    adj = np.empty(m)
    prev = 1.0
    for rank in range(m - 1, -1, -1):
        idx = order[rank]
        prev = min(prev, p[idx] * m / (rank + 1))
        adj[idx] = min(prev, 1.0)
    return adj.tolist()


def correct(pvals: Sequence[float], method: str = "holm") -> List[float]:
    if method == "holm":
        return holm_bonferroni(pvals)
    if method == "bh":
        return benjamini_hochberg(pvals)
    return list(pvals)


def bootstrap_ci_pairs(fn: Callable[[np.ndarray, np.ndarray], float],
                       a: Sequence, b: Sequence, n_boot: int = 1000,
                       alpha: float = 0.05, seed: int = 0
                       ) -> Tuple[float, float] | Tuple[None, None]:
    """IC (1−alpha) de uma métrica de concordância fn(a, b) via bootstrap pareado."""
    a = np.asarray(a); b = np.asarray(b)
    if len(a) < 3:
        return None, None
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(a), size=len(a))
        v = fn(a[idx], b[idx])
        if v is not None and v == v:           # ignora None/NaN
            vals.append(v)
    if len(vals) < 10:
        return None, None
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


# ── Self-test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    x = rng.normal(0.7, 0.1, 60)
    y = rng.normal(0.6, 0.1, 60)

    delta, p = paired_randomization(x, y, n=5000)
    assert delta > 0 and p < 0.05, (delta, p)
    pw = wilcoxon_p(x, y)
    assert pw < 0.05 or pw != pw, pw
    d, lo, hi = bootstrap_mean_diff_ci(x, y, n_boot=2000)
    assert lo <= d <= hi and lo > 0, (d, lo, hi)
    print(f"  randomização Δ={delta:.3f} p={p:.4f} | Wilcoxon p={pw:.4f} | "
          f"IC95%=[{lo:.3f},{hi:.3f}]  ✓")

    assert abs(cliffs_delta([3, 3, 3], [1, 1, 1]) - 1.0) < 1e-9
    assert abs(cliffs_delta([1, 1, 1], [3, 3, 3]) + 1.0) < 1e-9
    print("  Cliff's δ extremos = ±1  ✓")

    raw = [0.01, 0.04, 0.03, 0.20]
    h = holm_bonferroni(raw); bh = benjamini_hochberg(raw)
    assert all(a >= b - 1e-12 for a, b in zip(h, raw)), h     # ajustado >= bruto
    assert all(0 <= v <= 1 for v in h + bh)
    print(f"  Holm={[round(v,3) for v in h]}  BH={[round(v,3) for v in bh]}  ✓")

    ci = bootstrap_ci_pairs(lambda a, b: float(np.mean(a == b)),
                            [1, 2, 3, 4, 5, 1, 2, 3], [1, 2, 3, 4, 4, 1, 2, 2], n_boot=500)
    assert ci[0] is not None and ci[0] <= ci[1]
    print(f"  bootstrap_ci_pairs = [{ci[0]:.3f},{ci[1]:.3f}]  ✓")
    print("stats OK")
