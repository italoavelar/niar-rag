"""
eval/lib/rag_runner.py
══════════════════════
Executa o caminho de GERAÇÃO do RAG de forma controlada para avaliação:
    pergunta → (retriever) contexto top-k → Qwen-32b → resposta

Diferenças propositais vs. produção (para isolar variáveis):
  • injeta o contexto recuperado diretamente (sem tool-calling) → fixa o gerador
    e permite atribuir diferenças a RECUPERAÇÃO + ICL;
  • sem o corte score_threshold=0.60 (já tratado no retriever);
  • toggle de ICL (few-shot) controlado por parâmetro.

Mantém o MESMO system prompt de produção (agent/utils/prompt.py) para fidelidade.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from .common import PROJECT_ROOT, resolve
from .llm_clients import LLMClient, strip_reasoning

# System prompt de produção (importa o real; fallback p/ cópia fiel)
try:
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))
    from agent.utils.prompt import CHAT_SYSTEM_PROMPT
except Exception:
    CHAT_SYSTEM_PROMPT = (
        "Você é um assistente especializado em recuperação e síntese de informações "
        "médicas e jurídicas (RAG). Responda exclusivamente com base no contexto "
        "recuperado; não invente. Se insuficiente, diga: 'Não encontrei informações "
        "suficientes nas fontes recuperadas para responder com segurança.' Sempre "
        "inclua a seção '## Fontes utilizadas'."
    )


def _format_context(chunk_ids: Sequence[str], corpus: Dict[str, dict]) -> str:
    blocks = []
    for i, cid in enumerate(chunk_ids, 1):
        d = corpus.get(cid, {})
        meta = d.get("metadata", {})
        blocks.append(
            f"📄 DOCUMENTO {i}:\n{d.get('text','[indisponível]')}\n"
            f"🔗 FONTE: {meta.get('title','?')} — {meta.get('source_url','')}\n"
            + "-" * 60
        )
    return "\n".join(blocks) if blocks else "[nenhum documento recuperado]"


def _load_icl_block(n_shots: int) -> str:
    path = resolve("data/icl_examples.json")
    if not Path(path).exists():
        return ""
    examples = json.loads(Path(path).read_text(encoding="utf-8"))[:n_shots]
    if not examples:
        return ""
    parts = ["\n\n=== EXEMPLOS DE RESPOSTAS BEM FUNDAMENTADAS (siga o estilo) ==="]
    for ex in examples:
        parts.append(
            f"\nPergunta: {ex['question']}\n"
            f"Contexto: {ex['context'][:600]}...\n"
            f"Resposta ideal: {ex['answer']}\n(Fonte: {ex.get('source','')})"
        )
    parts.append("=== FIM DOS EXEMPLOS ===\n")
    return "\n".join(parts)


# Prompt do BASELINE SEM RAG: o modelo responde pelo CONHECIMENTO PRÓPRIO (mesma
# tarefa — responder, citar, recusar se inseguro — mas SEM contexto recuperado).
# É o contraste que mostra o valor da ferramenta.
NO_RAG_SYSTEM = (
    "Você é um assistente especializado em legislação e normas de saúde e "
    "governança de IA no Brasil. Responda à pergunta do usuário com base no seu "
    "próprio conhecimento. Cite as fontes (leis, resoluções, artigos) que "
    "sustentam a resposta; se não tiver certeza suficiente, responda EXATAMENTE: "
    "'Não encontrei informações suficientes nas fontes recuperadas para responder "
    "com segurança.' Inclua a seção '## Fontes utilizadas' quando citar."
)


class RagRunner:
    def __init__(self, generator_spec: dict, corpus: Dict[str, dict],
                 context_top_k: int = 4):
        self.llm = LLMClient(
            provider=generator_spec["provider"], model=generator_spec["model"],
            temperature=generator_spec.get("temperature", 0),
            max_tokens=generator_spec.get("max_tokens", 2048),
            reasoning_effort=generator_spec.get("reasoning_effort"))
        self.corpus = corpus
        self.context_top_k = context_top_k

    def answer(self, question: str, retrieved_ids: Sequence[str],
               icl: bool = False, n_shots: int = 2) -> dict:
        ctx_ids = list(retrieved_ids)[:self.context_top_k]
        context = _format_context(ctx_ids, self.corpus)
        system = CHAT_SYSTEM_PROMPT + (_load_icl_block(n_shots) if icl else "")
        user = f"Contexto recuperado:\n{context}\n\nPergunta do usuário: {question}"
        raw = self.llm.chat(user, system=system)
        return {
            "answer": strip_reasoning(raw),
            "context_chunk_ids": ctx_ids,
            "context_text": context,
            "icl": icl,
        }

    def answer_no_rag(self, question: str, icl: bool = False, n_shots: int = 2) -> dict:
        """Baseline SEM RAG: responde sem contexto recuperado (conhecimento próprio)."""
        system = NO_RAG_SYSTEM + (_load_icl_block(n_shots) if icl else "")
        raw = self.llm.chat(f"Pergunta do usuário: {question}", system=system)
        return {"answer": strip_reasoning(raw), "context_chunk_ids": [],
                "context_text": "", "icl": icl}

    def run_dataset(self, qa: Sequence[dict], rankings: Dict[str, List[str]],
                    icl: bool, n_shots: int = 2, no_rag: bool = False) -> List[dict]:
        """Gera respostas p/ todas as perguntas. Se no_rag=True, ignora o ranking e
        responde sem contexto (baseline da LLM sozinha)."""
        from tqdm import tqdm
        out = []
        desc = f"[gen {'no_rag' if no_rag else 'rag'} icl={icl}]"
        for r in tqdm(qa, desc=desc, leave=False):
            if no_rag:
                res = self.answer_no_rag(r["question"], icl=icl, n_shots=n_shots)
            else:
                res = self.answer(r["question"], rankings.get(r["qid"], []),
                                  icl=icl, n_shots=n_shots)
            out.append({**res, "qid": r["qid"], "question": r["question"],
                        "reference_answer": r.get("reference_answer", ""),
                        "question_type": r.get("question_type", ""),
                        "should_answer": bool(r.get("qrels"))})
        return out
