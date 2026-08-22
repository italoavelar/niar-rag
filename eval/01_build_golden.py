#!/usr/bin/env python3
"""
01_build_golden.py
══════════════════
Constrói o GOLD SET de QA ancorado no corpus (jurídico-saúde / governança de IA).

Para cada pergunta gera-se também o *qrel* (chunk-fonte = grau 2, vizinhos = grau 1),
o que torna possível medir RECUPERAÇÃO (nDCG, GeoRisk) — algo que nenhum dataset
público entrega, pois os julgamentos precisam apontar para os SEUS chunk_ids.

Tipos de pergunta gerados (robustez):
  • factual       — respondível por 1 chunk
  • multi_hop     — exige 2+ chunks do mesmo documento
  • comparative   — relaciona documentos/normas distintos
  • unanswerable  — fora de escopo (qrels vazio) → testa RECUSA correta na geração

Saídas:
  data/golden_qa.jsonl          gold set completo
  data/queries.csv, qrels.csv   derivados (compatíveis com a engine de retrieval)
  data/golden_qa_review.xlsx    para curadoria humana (valide ~15-20%)
  data/icl_examples.json        exemplos few-shot p/ a Etapa 03 (ICL)

Não usamos dataset público: os benchmarks PT-BR levantados não cobrem o domínio
(jurídico-saúde + governança de IA) nem a tarefa. Ver Seção 6.2 do documento.

Uso:
  python 01_build_golden.py                 # usa config.yaml
  python 01_build_golden.py --smoke          # só os 3 primeiros chunks (teste rápido)
  python 01_build_golden.py --max-chunks 50  # amostra
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.common import (load_config, load_corpus, save_jsonl, set_seed, setup_io,
                        resolve, ensure_dir, neighbors_same_doc, get_env)
from lib.llm_clients import build_client
from lib.retrievers import detect_lang


# ── Prompts de geração ───────────────────────────────────────────────────────

SYS = ("Você é um especialista em direito da saúde e governança de IA no Brasil, "
       "criando um benchmark de avaliação para um sistema RAG. O PÚBLICO que usa "
       "esse RAG são médicos com projetos de IA responsável e auditores de IA "
       "responsável — eles NÃO são juristas e têm pouca familiaridade com jargão "
       "jurídico. Gere perguntas que ESSES perfis realmente fariam (conformidade, "
       "ética, governança, proteção de dados/LGPD, risco, responsabilidade, "
       "transparência — aplicados à prática clínica e à auditoria de IA), focando no "
       "que o profissional precisa SABER ou FAZER para estar em conformidade ou "
       "auditar (não em tecnicismos de doutrina jurídica), em "
       "linguagem acessível, evitando jargão jurídico; quando um termo técnico for "
       "inevitável, formule de modo compreensível para um não-jurista. NUNCA use "
       "expressões meta como 'segundo o texto', 'de acordo com o trecho' ou 'no "
       "documento acima' — a pergunta deve fazer sentido sozinha. "
       "ATENÇÃO: o acervo é BILÍNGUE — muitos trechos estão em INGLÊS (guidance "
       "internacional: WHO, FDA, GDPR, NIST, OECD, UNESCO). Mesmo quando o trecho "
       "estiver em inglês, a PERGUNTA e a RESPOSTA-REFERÊNCIA devem ser SEMPRE em "
       "PORTUGUÊS DO BRASIL — é a língua do público-alvo (o sistema precisa responder "
       "em PT a partir de fontes em EN). Responda SEMPRE em JSON válido.")


def prompt_factual(text: str, n: int) -> str:
    return (f"Trecho de um documento jurídico/normativo:\n\"\"\"\n{text}\n\"\"\"\n\n"
            f"Gere {n} pergunta(s) cuja resposta esteja CONTIDA E COMPLETA neste trecho. "
            f"Para cada uma, dê a resposta-referência (concisa e fiel ao trecho) e a "
            f"dificuldade (easy|medium|hard).\n"
            f'Formato: {{"pairs":[{{"question":"...","answer":"...","difficulty":"easy"}}]}}')


def prompt_multihop(text_a: str, text_b: str) -> str:
    return (f"Dois trechos do MESMO documento.\nTRECHO A:\n\"\"\"\n{text_a}\n\"\"\"\n\n"
            f"TRECHO B:\n\"\"\"\n{text_b}\n\"\"\"\n\n"
            f"Gere 1 pergunta cuja resposta exija COMBINAR informação dos DOIS trechos "
            f"(não respondível por apenas um). Dê a resposta-referência.\n"
            f'Formato: {{"question":"...","answer":"...","difficulty":"medium"}}')


def prompt_comparative(text_a: str, title_a: str, text_b: str, title_b: str) -> str:
    return (f"Trecho de '{title_a}':\n\"\"\"\n{text_a}\n\"\"\"\n\n"
            f"Trecho de '{title_b}':\n\"\"\"\n{text_b}\n\"\"\"\n\n"
            f"Gere 1 pergunta que RELACIONE ou COMPARE o que dizem os dois documentos "
            f"sobre um ponto em comum. Dê a resposta-referência que integre ambos.\n"
            f'Formato: {{"question":"...","answer":"...","difficulty":"hard"}}')


def prompt_unanswerable(themes: list[str], n: int) -> str:
    return (f"O corpus trata de: {', '.join(themes)}. "
            f"Gere {n} pergunta(s) que pareçam plausíveis na área de saúde/direito/IA "
            f"mas que NÃO possam ser respondidas por esse corpus (ex.: legislação de "
            f"outros países, dados estatísticos específicos, casos clínicos individuais, "
            f"temas fora do escopo). A resposta-referência deve ser exatamente: "
            f"\"Não encontrei informações suficientes nas fontes recuperadas para "
            f"responder com segurança.\"\n"
            f'Formato: {{"questions":["...","..."]}}')


# ── Construção ───────────────────────────────────────────────────────────────

def _norm_q(q: str) -> str:
    return " ".join(q.lower().split())


def _order_key(corpus, cid):
    """Ordem do chunk dentro do documento (página, bloco). Os IDs seguem
    nomeDocumento_p{página}_c{bloco}, então chunks do mesmo doc ficam ordenados."""
    m = corpus[cid]["metadata"]
    try:
        return (int(m.get("page", 0)), int(m.get("chunk", 0)))
    except (TypeError, ValueError):
        return (0, 0)


def build(cfg: dict, args) -> None:
    gcfg = cfg["golden"]
    corpus = load_corpus(resolve(cfg["paths"]["corpus"]))
    print(f"Corpus: {len(corpus)} chunks")

    # chunks elegíveis (texto suficiente)
    eligible = [cid for cid, d in corpus.items()
                if len(d["text"]) >= gcfg.get("min_chunk_chars", 400)]
    print(f"Chunks elegíveis (>= {gcfg.get('min_chunk_chars',400)} chars): {len(eligible)}")

    if args.smoke:
        eligible = eligible[:3]
    elif args.max_chunks:
        eligible = eligible[:args.max_chunks]
    elif gcfg.get("sample_chunks"):
        random.shuffle(eligible)
        eligible = eligible[:int(gcfg["sample_chunks"])]

    gen = build_client(gcfg["generator"])
    records: list[dict] = []
    seen: set[str] = set()
    qn = 0

    def add(question, answer, qtype, qrels, source_docs, difficulty):
        nonlocal qn
        key = _norm_q(question)
        if not question or key in seen:
            return False
        seen.add(key)
        qn += 1
        # idioma do(s) chunk-fonte: define o estrato (PT→PT vs PT→EN cross-lingual).
        # A pergunta é sempre em PT (persona); o chunk pode estar em EN.
        langs = {detect_lang(corpus[c]["text"]) for c in qrels}
        source_lang = ("en" if langs == {"en"} else
                       "pt" if langs == {"pt"} else
                       ("mixed" if langs else None))
        records.append({
            "qid": f"q{qn:04d}",
            "question": question.strip(),
            "reference_answer": (answer or "").strip(),
            "question_type": qtype,
            "source_lang": source_lang,
            "difficulty": difficulty,
            "theme": ", ".join(sorted({corpus[c]["metadata"].get("theme", "")
                                       for c in qrels})) if qrels else "fora-de-escopo",
            "source_docs": sorted(source_docs),
            "qrels": qrels,
        })
        return True

    # 1) Factual ----------------------------------------------------------------
    if gcfg["types"].get("factual", True):
        n_per = gcfg.get("questions_per_chunk", 1)
        target = gcfg.get("n_factual", 40)
        pool = list(eligible); random.shuffle(pool)
        print(f"\n[1/4] Factual — alvo {target}")
        made = 0
        for cid in pool:
            if made >= target:
                break
            try:
                data = gen.chat_json(prompt_factual(corpus[cid]["text"], n_per), system=SYS)
                pairs = data.get("pairs", []) if isinstance(data, dict) else []
            except Exception as e:
                print(f"  ! chunk {cid}: {e}")
                continue
            neigh = neighbors_same_doc(cid, corpus, window=1)
            qrels = {cid: gcfg["grade_source_chunk"]}
            for nb in neigh:
                qrels.setdefault(nb, gcfg["grade_neighbor_chunk"])
            for p in pairs:
                if made >= target:
                    break
                if add(p.get("question"), p.get("answer"), "factual", dict(qrels),
                       {corpus[cid]["metadata"]["source"]}, p.get("difficulty", "medium")):
                    made += 1

    # 2) Multi-hop --------------------------------------------------------------
    by_doc = defaultdict(list)
    for cid in eligible:
        by_doc[corpus[cid]["metadata"]["source"]].append(cid)

    if gcfg["types"].get("multi_hop", True):
        pools = [v for v in by_doc.values() if len(v) >= 2]
        target = gcfg.get("n_multi_hop", 0)
        print(f"\n[2/4] Multi-hop — alvo {target}")
        made = attempts = 0
        max_attempts = max(20, target * 6)
        while made < target and pools and attempts < max_attempts:
            attempts += 1
            pool = random.choice(pools)
            # chunks do mesmo documento vão ao LLM EM ORDEM (página, bloco)
            a, b = sorted(random.sample(pool, 2), key=lambda c: _order_key(corpus, c))
            try:
                d = gen.chat_json(prompt_multihop(corpus[a]["text"], corpus[b]["text"]), system=SYS)
            except Exception as e:
                print(f"  ! {a}/{b}: {e}"); continue
            if add(d.get("question"), d.get("answer"), "multi_hop",
                   {a: gcfg["grade_source_chunk"], b: gcfg["grade_source_chunk"]},
                   {corpus[a]["metadata"]["source"]}, d.get("difficulty", "medium")):
                made += 1

    # 3) Comparative ------------------------------------------------------------
    if gcfg["types"].get("comparative", True) and len(by_doc) >= 2:
        target = gcfg.get("n_comparative", 0)
        print(f"\n[3/4] Comparativa — alvo {target}")
        docs = list(by_doc.keys())
        made = attempts = 0
        max_attempts = max(20, target * 6)
        while made < target and attempts < max_attempts:
            attempts += 1
            da, db = random.sample(docs, 2)
            a, b = random.choice(by_doc[da]), random.choice(by_doc[db])
            try:
                d = gen.chat_json(prompt_comparative(
                    corpus[a]["text"], corpus[a]["metadata"].get("title", da),
                    corpus[b]["text"], corpus[b]["metadata"].get("title", db)), system=SYS)
            except Exception as e:
                print(f"  ! {a}/{b}: {e}"); continue
            if add(d.get("question"), d.get("answer"), "comparative",
                   {a: gcfg["grade_source_chunk"], b: gcfg["grade_source_chunk"]},
                   {da, db}, d.get("difficulty", "hard")):
                made += 1

    # 4) Unanswerable -----------------------------------------------------------
    if gcfg["types"].get("unanswerable", True):
        themes = sorted({d["metadata"].get("theme", "") for d in corpus.values() if d["metadata"].get("theme")})
        target = gcfg.get("n_unanswerable", 0)
        ref = ("Não encontrei informações suficientes nas fontes recuperadas "
               "para responder com segurança.")
        print(f"\n[4/4] Inanswerable — alvo {target}")
        made = attempts = 0
        while made < target and attempts < 8:
            attempts += 1
            try:
                d = gen.chat_json(prompt_unanswerable(themes, min(target - made, 8)), system=SYS)
            except Exception as e:
                print(f"  ! unanswerable: {e}"); continue
            for q in (d.get("questions", []) if isinstance(d, dict) else []):
                if made >= target:
                    break
                if add(q, ref, "unanswerable", {}, set(), "medium"):
                    made += 1

    # ── Persistência ───────────────────────────────────────────────────────--
    out = resolve(cfg["paths"]["golden_qa"])
    save_jsonl(records, out)
    print(f"\n✓ {len(records)} perguntas → {out}")
    _dump_derived(cfg, records)
    _dump_review_xlsx(cfg, records, corpus)
    _dump_icl(cfg, records, corpus)
    _summary(records)


def _dump_derived(cfg, records):
    """queries.csv + qrels.csv (só perguntas respondíveis, p/ retrieval)."""
    q_path = resolve(cfg["paths"]["queries_csv"])
    r_path = resolve(cfg["paths"]["qrels_csv"])
    ensure_dir(q_path.parent)
    answerable = [r for r in records if r["qrels"]]
    with open(q_path, "w", newline="", encoding="utf-8") as qf:
        w = csv.writer(qf); w.writerow(["QueryId", "Query"])
        for r in answerable:
            w.writerow([r["qid"], r["question"]])
    with open(r_path, "w", newline="", encoding="utf-8") as rf:
        w = csv.writer(rf); w.writerow(["QueryId", "ChunkId", "Relevance"])
        for r in answerable:
            for cid, g in r["qrels"].items():
                w.writerow([r["qid"], cid, g])
    print(f"✓ derivados: {q_path.name} ({len(answerable)} q) + {r_path.name}")


def _dump_review_xlsx(cfg, records, corpus):
    try:
        import pandas as pd
    except Exception:
        print("! pandas ausente — pulando .xlsx de curadoria"); return
    rows = []
    for r in records:
        ctx = "\n---\n".join(corpus[c]["text"][:500] for c in list(r["qrels"])[:2])
        rows.append({"qid": r["qid"], "tipo": r["question_type"],
                     "pergunta": r["question"], "resposta_ref": r["reference_answer"],
                     "tema": r["theme"], "chunks_fonte": ", ".join(r["qrels"].keys()),
                     "contexto_fonte": ctx,
                     "APROVAR(S/N)": "", "correcao": ""})
    path = resolve(cfg["paths"]["golden_review"])
    ensure_dir(path.parent)
    pd.DataFrame(rows).to_excel(path, index=False, engine="openpyxl")
    print(f"✓ curadoria → {path.name} (valide ~15-20% e marque APROVAR)")


def _dump_icl(cfg, records, corpus):
    """Seleciona exemplos few-shot (factual de alta qualidade) p/ a Etapa 03."""
    import json
    examples = []
    for r in records:
        if r["question_type"] == "factual" and r["reference_answer"]:
            cid = max(r["qrels"], key=r["qrels"].get)
            examples.append({"question": r["question"],
                             "context": corpus[cid]["text"],
                             "answer": r["reference_answer"],
                             "source": corpus[cid]["metadata"].get("title", "")})
        if len(examples) >= max(4, cfg["generation"]["icl"]["n_shots"]):
            break
    path = resolve("data/icl_examples.json")
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(examples, f, ensure_ascii=False, indent=2)
    print(f"✓ ICL few-shot → {path.name} ({len(examples)} exemplos)")


def _summary(records):
    from collections import Counter
    c = Counter(r["question_type"] for r in records)
    print("\nResumo por tipo:")
    for t, n in c.items():
        print(f"  {t:<13} {n}")


def main():
    setup_io()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--smoke", action="store_true", help="só 3 chunks (teste rápido)")
    ap.add_argument("--max-chunks", type=int)
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))
    build(cfg, args)


if __name__ == "__main__":
    main()
