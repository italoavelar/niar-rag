"""
eval/lib/judges.py
══════════════════
LLM-as-a-judge com DOIS modelos independentes (≠ gerador Qwen para evitar viés
de auto-preferência): Llama-3.3-70B e gpt-oss-120b.

Cada juiz pontua 1..5 por critério, com justificativa, em JSON estruturado, temp 0:
  faithfulness     — resposta sustentada APENAS pelo contexto recuperado (anti-alucinação)
  answer_relevance — responde de fato à pergunta
  correctness      — concorda com a resposta-referência do gold set
  completeness     — cobre os pontos essenciais
  citation         — seção "Fontes utilizadas" presente e correta
  refusal          — (in)existência de recusa foi APROPRIADA à respondibilidade

A concordância entre os dois juízes é medida na Etapa 04.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

from tqdm import tqdm

from .llm_clients import LLMClient

# Teto do contexto no prompt do juiz. Precisa caber o contexto INTEIRO: o
# _format_context põe a linha "FONTE:" DEPOIS do texto de cada documento, então
# truncar corta justamente as fontes dos últimos docs e inviabiliza o critério
# `citation` (e enviesa faithfulness/hallucination, já que o juiz é instruído a
# tratar o contexto como única base permitida). Com context_top_k=5 o contexto
# real vai a ~7,1k chars; 12k dá folga. Custo irrisório: ~3-4k tokens em modelos
# de 128k de janela.
CONTEXT_CHAR_LIMIT = 12000

JUDGE_SYS = (
    "Você é um avaliador rigoroso e imparcial de respostas de um sistema RAG "
    "jurídico-médico. Avalie de forma objetiva, penalizando alucinação (afirmações "
    "não sustentadas pelo contexto). Responda SEMPRE em JSON válido, sem texto extra."
)

_ANCHORS = (
    "Escala por critério (inteiro 1..5): 1=péssimo, 2=ruim, 3=regular, 4=bom, 5=excelente.\n"
    "- faithfulness: 5 = toda afirmação tem suporte no contexto; 1 = sem suporte/contradiz.\n"
    "- hallucination: severidade de fatos, leis, artigos ou números INVENTADOS (ausentes do "
    "contexto). 5 = nenhuma invenção; 1 = inventa informação factual relevante.\n"
    "- answer_relevance: 5 = responde exatamente o perguntado; 1 = foge do tema.\n"
    "- correctness: 5 = bate com a resposta-referência; 1 = factualmente errada.\n"
    "- completeness: 5 = cobre tudo que importa; 1 = omite o essencial.\n"
    "- citation: 5 = cita fontes corretas em '## Fontes utilizadas'; 1 = ausente/errada.\n"
    "- refusal: para perguntas SEM resposta no corpus, 5 = recusou corretamente; "
    "para perguntas COM resposta, 5 = NÃO recusou indevidamente; 1 = comportamento oposto."
)


def build_judges(cfg: dict) -> List[Tuple[str, LLMClient]]:
    jcfg = cfg["judges"]
    judges = []
    for m in jcfg["models"]:
        judges.append((m["name"], LLMClient(
            provider=m["provider"], model=m["model"],
            temperature=jcfg.get("temperature", 0), max_tokens=1024)))
    return judges


def _prompt(record: dict, criteria: List[str], scale: int) -> str:
    answerability = ("Esta pergunta TEM resposta no corpus (deveria ser respondida)."
                     if record.get("should_answer")
                     else "Esta pergunta NÃO tem resposta no corpus (deveria ser recusada).")
    crits = ", ".join(criteria)
    fields = ", ".join(f'"{c}": {{"score": <1..{scale}>, "justification": "<curta>"}}'
                       for c in criteria)
    return (
        f"{_ANCHORS}\n\n{answerability}\n\n"
        f"PERGUNTA:\n{record['question']}\n\n"
        f"RESPOSTA-REFERÊNCIA (gold):\n{record.get('reference_answer','(n/a)')}\n\n"
        f"CONTEXTO RECUPERADO (única base permitida):\n{record.get('context_text','')[:CONTEXT_CHAR_LIMIT]}\n\n"
        f"RESPOSTA DO SISTEMA (avaliar):\n{record.get('answer','')}\n\n"
        f"Avalie os critérios [{crits}]. Responda SOMENTE JSON no formato:\n"
        f"{{{fields}}}"
    )


def judge_answer(client: LLMClient, record: dict, criteria: List[str],
                 scale: int = 5) -> dict:
    try:
        data = client.chat_json(_prompt(record, criteria, scale), system=JUDGE_SYS)
    except Exception as e:
        return {"_error": str(e), **{c: None for c in criteria}}
    out, just = {}, {}
    for c in criteria:
        v = data.get(c) if isinstance(data, dict) else None
        if isinstance(v, dict):
            out[c] = _clip(v.get("score"), scale)
            just[c] = v.get("justification", "")
        else:
            out[c] = _clip(v, scale)
    out["_justifications"] = just
    return out


def _clip(x, scale):
    try:
        return max(1, min(scale, int(round(float(x)))))
    except (TypeError, ValueError):
        return None


def run_judging(answers: List[dict], judges: List[Tuple[str, LLMClient]],
                criteria: List[str], scale: int = 5) -> List[dict]:
    """Anota cada resposta com os scores de cada juiz: record['scores'][judge][crit]."""
    out = []
    for rec in tqdm(answers, desc="[juízes]", leave=False):
        scores = {}
        for name, client in judges:
            scores[name] = judge_answer(client, rec, criteria, scale)
        out.append({**rec, "scores": scores})
    return out
