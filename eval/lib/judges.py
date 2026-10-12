"""
eval/lib/judges.py
══════════════════
LLM-as-a-judge com N juízes independentes (≠ família dos geradores avaliados),
rubrica de ESCALA MISTA e raciocínio-antes-da-nota (CoT).

── Decisões de desenho (fundamentadas na literatura) ───────────────────────
ESCALA MISTA (não 1–5 uniforme). A literatura mostra que LLMs não são bem
calibrados para pontuação de alta precisão e que escalas mais grossas (binária
ou poucos níveis) dão maior confiabilidade/concordância; critérios subjetivos
usam 3 níveis com âncoras explícitas por nível:
  • BINÁRIOS (0/1): citation, refusal, hallucination  — objetivos
  • TERNÁRIOS (1–3): faithfulness, answer_relevance, completeness, correctness
Refs: Yan (eugeneyan.com/writing/llm-evaluators) recomenda saída binária p/
objetivos; Evidently AI (LLM-as-a-judge guide) prefere baixa granularidade e
exige definir cada nível; Shepherd (arXiv:2308.04592) e UMbrela (arXiv:2406.06519)
mostram deriva e baixa acurácia nos níveis intermediários de Likert.

RACIOCÍNIO ANTES DA NOTA (CoT). O juiz primeiro escreve uma análise curta e só
então pontua. Melhora qualidade/consistência e NIVELA juízes de raciocínio e
não-raciocínio. Refs: Chiang et al. (arXiv:2310.05657); Constitutional AI
(arXiv:2212.08073). Mitigação de "pensar demais"/injetar conhecimento externo
(PoLL, arXiv:2404.18796): instrução explícita de se ater ao contexto/referência.

CRITÉRIOS CONDICIONAIS. correctness e completeness exigem resposta-referência;
nas perguntas irrespondíveis (should_answer=False) não há referência, então são
marcados "na" (não-aplicável) e ficam fora das médias e da concordância.

ANÁLISE. κ de Cohen super-penaliza dados ordinais; para os critérios ternários a
Etapa 04 reporta κ ponderado JUNTO COM Spearman ρ/Kendall. Ref: discussão em
Yan (idem) e UMbrela (idem). O 04_agreement já calcula κ, α e Spearman.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

from tqdm import tqdm

from .llm_clients import LLMClient

CONTEXT_CHAR_LIMIT = 12000

# ── Tipo de escala por critério ─────────────────────────────────────────────
# "bin"  → 0 ou 1
# "tern" → 1, 2 ou 3
# criterios condicionais (só com referência): correctness, completeness
BINARY = {"citation", "refusal", "hallucination"}
TERNARY = {"faithfulness", "answer_relevance", "completeness", "correctness"}
CONDICIONAIS = {"correctness", "completeness"}   # "na" quando não há referência


JUDGE_SYS = (
    "Você é um avaliador rigoroso e imparcial de respostas de um sistema RAG "
    "jurídico-médico. Avalie SOMENTE com base no material fornecido (contexto e "
    "resposta-referência); NÃO use conhecimento externo nem extrapole além do "
    "que está escrito. Primeiro raciocine brevemente, depois pontue. Responda "
    "SEMPRE em JSON válido, sem texto fora do JSON."
)

# Âncoras com TODOS os níveis definidos (não só os extremos).
_ANCHORS = (
    "ESCALAS (use exatamente os valores indicados):\n"
    "\n"
    "Binários (0 ou 1):\n"
    "- hallucination: 1 = NÃO inventou (toda lei/artigo/número citado existe no "
    "contexto); 0 = inventou algum fato, lei, artigo ou número ausente do contexto.\n"
    "- citation: 1 = a seção '## Fontes utilizadas' lista as fontes corretas que "
    "sustentam a resposta; 0 = ausente, vazia ou cita fontes erradas/não usadas.\n"
    "- refusal: avalie conforme o tipo da pergunta (indicado abaixo). "
    "Se a pergunta NÃO tem resposta no contexto: 1 = recusou corretamente; 0 = "
    "respondeu como se tivesse resposta. Se a pergunta TEM resposta no contexto: "
    "1 = NÃO recusou (respondeu); 0 = recusou indevidamente.\n"
    "\n"
    "Ternários (1, 2 ou 3):\n"
    "- faithfulness (sustentação no contexto): 3 = toda afirmação tem suporte "
    "explícito no contexto; 2 = a maior parte tem suporte, mas há afirmações sem "
    "respaldo claro; 1 = pouco ou nada sustentado pelo contexto.\n"
    "- answer_relevance (responde ao perguntado): 3 = responde exatamente o que "
    "foi perguntado; 2 = responde em parte ou tangencia o tema; 1 = foge do tema.\n"
    "- completeness (cobertura do essencial): 3 = cobre todos os pontos essenciais "
    "da resposta-referência; 2 = cobre parte; 1 = omite o essencial. "
    "[Se não houver resposta-referência, use \"na\".]\n"
    "- correctness (acerto factual vs. referência): 3 = concorda com a "
    "resposta-referência; 2 = parcialmente correta (mistura acerto e erro); 1 = "
    "factualmente errada. [Se não houver resposta-referência, use \"na\".]\n"
)

JUDGE_SYS = JUDGE_SYS  # (mantém nome usado abaixo)


def build_judges(cfg: dict) -> List[Tuple[str, LLMClient]]:
    jcfg = cfg["judges"]
    # Juízes de raciocínio (GLM) precisam de teto alto p/ raciocinar E escrever o
    # JSON; padrão 3072, sobreponível por juiz.
    default_mt = int(jcfg.get("max_tokens", 3072))
    judges = []
    for m in jcfg["models"]:
        judges.append((m["name"], LLMClient(
            provider=m["provider"], model=m["model"],
            temperature=jcfg.get("temperature", 0),
            max_tokens=int(m.get("max_tokens", default_mt)))))
    return judges


def _escala_txt(c: str) -> str:
    if c in BINARY:
        return "0 ou 1"
    if c in TERNARY:
        return '1, 2, 3 ou "na"' if c in CONDICIONAIS else "1, 2 ou 3"
    return "?"


def _prompt(record: dict, criteria: List[str], scale: int = None) -> str:
    tem_ref = bool((record.get("reference_answer") or "").strip())
    if record.get("should_answer"):
        answerability = ("Esta pergunta TEM resposta no contexto (o sistema "
                         "DEVERIA responder).")
    else:
        answerability = ("Esta pergunta NÃO tem resposta no contexto (o sistema "
                         "DEVERIA recusar). Não há resposta-referência: marque "
                         "correctness e completeness como \"na\".")
    # formato JSON pedido: cada critério com reasoning + score; os condicionais
    # aceitam "na" quando não há referência.
    campos = []
    for c in criteria:
        campos.append(
            f'"{c}": {{"reasoning": "<análise breve>", "score": <{_escala_txt(c)}>}}')
    fields = ", ".join(campos)
    ref = record.get("reference_answer", "") or "(não se aplica — pergunta sem resposta)"
    return (
        f"{_ANCHORS}\n"
        f"REGRA: raciocine em 'reasoning' ANTES de escolher 'score'. Atenha-se ao "
        f"contexto e à referência; não use conhecimento externo.\n\n"
        f"{answerability}\n\n"
        f"PERGUNTA:\n{record['question']}\n\n"
        f"RESPOSTA-REFERÊNCIA (gold):\n{ref}\n\n"
        f"CONTEXTO RECUPERADO (única base permitida):\n"
        f"{record.get('context_text','')[:CONTEXT_CHAR_LIMIT]}\n\n"
        f"RESPOSTA DO SISTEMA (avaliar):\n{record.get('answer','')}\n\n"
        f"Responda SOMENTE JSON no formato:\n{{{fields}}}"
    )


def _parse_score(c: str, v, record: dict):
    """Converte o score bruto para o tipo certo da escala do critério.
    Retorna int válido, a string 'na' (condicional sem referência) ou None (erro)."""
    tem_ref = bool((record.get("reference_answer") or "").strip())
    # condicional sem referência → não-aplicável
    if c in CONDICIONAIS and not tem_ref:
        return "na"
    if isinstance(v, str) and v.strip().lower() in ("na", "n/a", "não se aplica", "nao se aplica"):
        return "na"
    try:
        x = int(round(float(v)))
    except (TypeError, ValueError):
        return None
    if c in BINARY:
        return 1 if x >= 1 else 0            # tolera 0/1 (e satura acima)
    if c in TERNARY:
        return max(1, min(3, x))
    return None


def judge_answer(client: LLMClient, record: dict, criteria: List[str],
                 scale: int = None) -> dict:
    try:
        data = client.chat_json(_prompt(record, criteria), system=JUDGE_SYS)
    except Exception as e:
        return {"_error": str(e), **{c: None for c in criteria}}
    out, just = {}, {}
    for c in criteria:
        v = data.get(c) if isinstance(data, dict) else None
        if isinstance(v, dict):
            out[c] = _parse_score(c, v.get("score"), record)
            just[c] = v.get("reasoning", v.get("justification", ""))
        else:
            out[c] = _parse_score(c, v, record)
    out["_justifications"] = just
    return out


def run_judging(answers: List[dict], judges: List[Tuple[str, LLMClient]],
                criteria: List[str], scale: int = None) -> List[dict]:
    """Anota cada resposta com os scores de cada juiz: record['scores'][judge][crit]."""
    out = []
    for rec in tqdm(answers, desc="[juízes]", leave=False):
        scores = {}
        for nome, cli in judges:
            scores[nome] = judge_answer(cli, rec, criteria)
        rec["scores"] = scores
        out.append(rec)
    return out
