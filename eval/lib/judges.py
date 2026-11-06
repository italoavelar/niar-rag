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

CRITÉRIOS CONDICIONAIS (não-aplicável = "na"). Dois tipos de condição, para que
nenhum critério seja pontuado quando lhe falta o insumo que ele mede:
  • SEM REFERÊNCIA → correctness, completeness exigem resposta-referência;
    nas perguntas irrespondíveis (should_answer=False) não há referência, logo "na".
  • SEM CONTEXTO → faithfulness, hallucination medem sustentação NO CONTEXTO; no
    cenário sem-RAG (sem context_chunk_ids) não há contexto, logo "na". (Marcado
    pela Etapa 04 na leitura, por ausência de contexto no registro.)
  • RESPOSTA É RECUSA → answer_relevance mede o quanto a resposta responde ao
    perguntado; uma RECUSA não responde nem foge do tema — é indefinida, logo "na".
    Detectado pela forma da RESPOSTA DO SISTEMA (frase-âncora de recusa), não pelo
    tipo da pergunta: uma alucinação numa pergunta irrespondível TEM conteúdo e
    recebe answer_relevance normal.
Itens "na" ficam fora das médias e da concordância.

NOTA SOBRE O CENÁRIO SEM-RAG. Em perguntas respondíveis sob sem-RAG, o bloco de
contexto chega vazio DE PROPÓSITO (o modelo responde por conhecimento
paramétrico). O prompt NÃO afirma mais "esta pergunta tem resposta no contexto"
nesse caso — afirmá-lo com contexto vazio contradizia o juiz e o fazia abandonar
correctness/completeness com "na" indevido. correctness/completeness dependem da
RESPOSTA-REFERÊNCIA (que existe nos dois cenários), não do contexto recuperado.

ANÁLISE. κ de Cohen super-penaliza dados ordinais; para os critérios ternários a
Etapa 04 reporta κ ponderado JUNTO COM Spearman ρ/Kendall. Ref: discussão em
Yan (idem) e UMbrela (idem). O 04_agreement já calcula κ, α e Spearman.
"""
from __future__ import annotations

import unicodedata
from typing import Dict, List, Tuple

from tqdm import tqdm

from .llm_clients import LLMClient

CONTEXT_CHAR_LIMIT = 12000

# ── Tipo de escala por critério ─────────────────────────────────────────────
# "bin"  → 0 ou 1
# "tern" → 1, 2 ou 3
# criterios condicionais (só com referência): correctness, completeness
# Versão da RUBRICA, gravada em cada bloco de notas. A Etapa 04 agrupa por
# ela e se recusa a misturar escalas: um judged_*.jsonl da rubrica 1-5 e um da
# mista nao sao comparaveis, e sem marcador a mistura e silenciosa — os dois
# tem os mesmos nomes de juiz e o mesmo formato de arquivo.
RUBRICA = "mista-v2"            # bin 0/1 + tern 1-3, CoT antes da nota

BINARY = {"citation", "refusal", "hallucination"}
TERNARY = {"faithfulness", "answer_relevance", "completeness", "correctness"}
CONDICIONAIS = {"correctness", "completeness"}   # "na" quando não há referência

# answer_relevance é indefinido quando a RESPOSTA DO SISTEMA é uma recusa:
# uma recusa não responde ao perguntado nem foge do tema. Condição por FORMA da
# resposta (não pelo tipo da pergunta), para não "na-ar" uma alucinação que, numa
# pergunta irrespondível, de fato tem conteúdo a ser medido.
SEM_SENTIDO_SE_RECUSA = {"answer_relevance"}

# Núcleo da frase-padrão de recusa emitida pelos geradores (qwen3 e gpt-oss usam
# a mesma frase). Casamento ROBUSTO: normalizamos a resposta (minúsculas, sem
# acento) e procuramos este núcleo, tolerando variações de pontuação, acento e
# caixa. Se a grafia real divergir, ajuste SÓ esta constante.
FRASE_RECUSA_NUCLEO = "nao tenho informacoes suficientes"


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
    "respaldo claro; 1 = pouco ou nada sustentado pelo contexto. "
    "[Se NÃO houver contexto recuperado, use \"na\".]\n"
    "- answer_relevance (responde ao perguntado): 3 = responde exatamente o que "
    "foi perguntado; 2 = responde em parte ou tangencia o tema; 1 = foge do tema. "
    "[Se a RESPOSTA DO SISTEMA for uma recusa a responder, use \"na\".]\n"
    "- completeness (cobertura do essencial): 3 = cobre todos os pontos essenciais "
    "da resposta-referência; 2 = cobre parte; 1 = omite o essencial. "
    "[Se não houver resposta-referência, use \"na\".]\n"
    "- correctness (acerto factual vs. referência): 3 = concorda com a "
    "resposta-referência; 2 = parcialmente correta (mistura acerto e erro); 1 = "
    "factualmente errada. [Se não houver resposta-referência, use \"na\".]\n"
)


def _normaliza(s: str) -> str:
    """minúsculas + sem acento, para casamento robusto de frase."""
    s = (s or "").lower()
    s = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in s if not unicodedata.combining(ch))


def _eh_recusa(answer: str) -> bool:
    """True se a RESPOSTA DO SISTEMA é uma recusa a responder, pela frase-âncora.
    Determinístico e independente do juiz — roda antes do julgamento, então o
    rejulgamento/--completar reconstrói a mesma condição."""
    return FRASE_RECUSA_NUCLEO in _normaliza(answer)


def _tem_contexto(record: dict) -> bool:
    """True se o registro traz contexto recuperado (cenário RAG). No sem-RAG o
    contexto é vazio por desenho, então faithfulness/hallucination são indefinidos.
    Usa context_chunk_ids como sinal primário (o texto pode vir truncado)."""
    if record.get("context_chunk_ids"):
        return True
    return bool((record.get("context_text") or "").strip())


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


def _escala_txt(record: dict, c: str) -> str:
    """Formato de score anunciado ao juiz. Inclui "na" quando o critério é
    condicional E a condição de não-aplicabilidade vale para ESTE registro, para
    o juiz saber que pode (e deve) usar "na" naquele caso."""
    if c in BINARY:
        return "0 ou 1"
    if c in TERNARY:
        aceita_na = (
            c in CONDICIONAIS
            or (c == "faithfulness" and not _tem_contexto(record))
            or (c in SEM_SENTIDO_SE_RECUSA and _eh_recusa(record.get("answer", "")))
        )
        return '1, 2, 3 ou "na"' if aceita_na else "1, 2 ou 3"
    return "?"


def _prompt(record: dict, criteria: List[str]) -> str:
    # Afirmação de respondibilidade: no sem-RAG (sem contexto) NÃO afirmamos que a
    # pergunta tem resposta no contexto — afirmá-lo com contexto vazio contradiz o
    # juiz. correctness/completeness são avaliados contra a RESPOSTA-REFERÊNCIA.
    tem_ctx = _tem_contexto(record)
    if record.get("should_answer"):
        if tem_ctx:
            answerability = ("Esta pergunta TEM resposta no contexto (o sistema "
                             "DEVERIA responder).")
        else:
            answerability = (
                "Esta pergunta É respondível. NÃO há contexto recuperado neste "
                "cenário (avaliação sem recuperação): avalie correctness e "
                "completeness contra a RESPOSTA-REFERÊNCIA (presente abaixo), não "
                "contra o contexto. Deixe faithfulness/hallucination como \"na\" "
                "(não há contexto para sustentar).")
    else:
        answerability = ("Esta pergunta NÃO tem resposta no contexto (o sistema "
                         "DEVERIA recusar). Não há resposta-referência: marque "
                         "correctness e completeness como \"na\".")
    # formato JSON pedido: cada critério com reasoning + score; os condicionais
    # aceitam "na" quando a condição de não-aplicabilidade vale para o registro.
    campos = []
    for c in criteria:
        campos.append(
            f'"{c}": {{"reasoning": "<análise breve>", "score": <{_escala_txt(record, c)}>}}')
    fields = ", ".join(campos)
    ref = record.get("reference_answer", "") or "(não se aplica — pergunta sem resposta)"
    ctx = record.get("context_text", "")[:CONTEXT_CHAR_LIMIT]
    ctx_bloco = ctx if ctx.strip() else "(sem contexto recuperado neste cenário)"
    return (
        f"{_ANCHORS}\n"
        f"REGRA: raciocine em 'reasoning' ANTES de escolher 'score'. Atenha-se ao "
        f"contexto e à referência; não use conhecimento externo.\n\n"
        f"{answerability}\n\n"
        f"PERGUNTA:\n{record['question']}\n\n"
        f"RESPOSTA-REFERÊNCIA (gold):\n{ref}\n\n"
        f"CONTEXTO RECUPERADO (única base permitida):\n"
        f"{ctx_bloco}\n\n"
        f"RESPOSTA DO SISTEMA (avaliar):\n{record.get('answer','')}\n\n"
        f"Responda SOMENTE JSON no formato:\n{{{fields}}}"
    )


def _parse_score(c: str, v, record: dict):
    """Converte o score bruto para o tipo certo da escala do critério.
    Retorna int válido, a string 'na' (condição de não-aplicabilidade) ou None (erro).

    Condições de 'na' (resolvidas pelo REGISTRO, não pela nota do juiz, para que o
    rejulgamento/--completar reconstrua a mesma decisão):
      • correctness/completeness sem resposta-referência;
      • faithfulness/hallucination sem contexto (cenário sem-RAG);
      • answer_relevance quando a RESPOSTA DO SISTEMA é uma recusa.
    """
    tem_ref = bool((record.get("reference_answer") or "").strip())
    # (1) condicional sem referência → não-aplicável
    if c in CONDICIONAIS and not tem_ref:
        return "na"
    # (2) sustentação no contexto, sem contexto → não-aplicável
    if c in ("faithfulness", "hallucination") and not _tem_contexto(record):
        return "na"
    # (3) relevância de uma recusa → não-aplicável (pela forma da resposta)
    if c in SEM_SENTIDO_SE_RECUSA and _eh_recusa(record.get("answer", "")):
        return "na"
    # juiz pode ter devolvido "na" textual num caso legítimo acima; aqui, se ainda
    # devolveu "na" fora de condição, trata-se como valor válido de não-aplicável.
    if isinstance(v, str) and v.strip().lower() in ("na", "n/a", "não se aplica", "nao se aplica"):
        return "na"
    try:
        x = int(round(float(v)))
    except (TypeError, ValueError):
        return None
    # FORA DE ESCALA E REJEITADO, nao saturado. Saturar era perigoso: os juizes
    # desta rubrica sao os MESMOS modelos que rodaram a rubrica 1-5, e um juiz
    # que respondesse "3" por habito num critetio binario viraria 1 = perfeito.
    # Isso inflaria justamente hallucination, citation e refusal. Devolver None
    # tira o item da conta — perder o item e honesto, contar errado nao e.
    if c in BINARY:
        return x if x in (0, 1) else None
    if c in TERNARY:
        return x if x in (1, 2, 3) else None
    return None


def judge_answer(client: LLMClient, record: dict,
                 criteria: List[str]) -> dict:
    try:
        data = client.chat_json(_prompt(record, criteria), system=JUDGE_SYS)
    except Exception as e:
        return {"_error": str(e), "_rubrica": RUBRICA,
                **{c: None for c in criteria}}
    out, just, fora = {}, {}, {}
    for c in criteria:
        v = data.get(c) if isinstance(data, dict) else None
        bruto = v.get("score") if isinstance(v, dict) else v
        if isinstance(v, dict):
            just[c] = v.get("reasoning", v.get("justification", ""))
        out[c] = _parse_score(c, bruto, record)
        # guarda o valor cru quando ele foi rejeitado, para o descarte nao ser
        # invisivel: a Etapa 04 conta estes casos e os reporta.
        if out[c] is None and bruto is not None:
            fora[c] = bruto
    out["_justifications"] = just
    out["_rubrica"] = RUBRICA
    if fora:
        out["_fora_escala"] = fora
    return out


def run_judging(answers: List[dict], judges: List[Tuple[str, LLMClient]],
                criteria: List[str]) -> List[dict]:
    """Anota cada resposta com os scores de cada juiz: record['scores'][judge][crit].

    Devolve CÓPIAS. Mutar `answers` no lugar faria uma segunda passada de
    julgamento herdar as notas da primeira sem aviso."""
    out = []
    for rec in tqdm(answers, desc="[juízes]", leave=False):
        scores = {nome: judge_answer(cli, rec, criteria) for nome, cli in judges}
        out.append({**rec, "scores": scores})
    return out
