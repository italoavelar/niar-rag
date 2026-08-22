"""
eval/lib/refusal.py
═══════════════════
Métrica OBJETIVA de recusa (SEM depender do juiz): detecta se a resposta recusou —
pela frase padrão que o gerador é instruído a usar — e compara com should_answer.

  • recusa correta  — nas perguntas SEM resposta no acervo (should_answer=False):
                      recusou como devia.  → taxa_recusa_correta
  • falsa recusa    — nas perguntas COM resposta (should_answer=True): recusou por
                      engano (over-refusal).  → taxa_falsa_recusa

É reproduzível (casamento de string, sem custo de API) e "acurácia" passa a ser o
termo certo. O critério `refusal` do juiz continua existindo como sinal secundário.
"""
from __future__ import annotations
import re
import unicodedata
from typing import List

# Padrões de recusa, aplicados ao texto NORMALIZADO (minúsculas, sem acento).
# Tolerantes a reformulações do gerador — NÃO exigem a frase exata — mas exigem a
# NEGAÇÃO, para não confundir com afirmações do tipo "há informações suficientes...".
_REFUSAL_PATTERNS = [
    re.compile(r"nao(?:\s+\w+){1,3}\s+(?:informac\w+|dados)\s+suficient"),      # não encontrei/há/localizei... informações/dados suficientes
    re.compile(r"nao(?:\s+\w+){0,3}\s+poss\w+\s+responder\s+com\s+seguranca"),  # não é possível responder com segurança
    re.compile(r"nao\s+consig\w+\s+responder"),                                 # não consigo responder
]


def _norm(t: str) -> str:
    t = unicodedata.normalize("NFKD", (t or "").lower())
    return "".join(c for c in t if not unicodedata.combining(c))


def is_refusal(answer: str) -> bool:
    """True se a resposta recusou. Casa a frase padrão E variações comuns (não há /
    não foram encontradas / não localizei ... informações suficientes; não é possível
    responder com segurança), sempre exigindo a negação (evita falso positivo)."""
    t = _norm(answer)
    return any(p.search(t) for p in _REFUSAL_PATTERNS)


def refusal_rates(records: List[dict]) -> dict:
    """records = dicts com 'answer' e 'should_answer'. Retorna contagens e taxas:
    recusa correta (nas fora-de-escopo) e falsa recusa (nas respondíveis)."""
    unans = [r for r in records if not r.get("should_answer")]
    ans = [r for r in records if r.get("should_answer")]
    correct = sum(1 for r in unans if is_refusal(r.get("answer", "")))
    false = sum(1 for r in ans if is_refusal(r.get("answer", "")))
    return {
        "n_fora_escopo": len(unans),
        "recusas_corretas": correct,
        "taxa_recusa_correta": round(correct / len(unans), 4) if unans else None,
        "n_respondiveis": len(ans),
        "falsas_recusas": false,
        "taxa_falsa_recusa": round(false / len(ans), 4) if ans else None,
    }
