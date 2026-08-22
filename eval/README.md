# Pipeline de Avaliação RAG — NIAR (jurídico-saúde / governança de IA)

Avalia o RAG (`niar-rag-prototype`) de ponta a ponta: **recuperação** (Cenários
A/B/C) e **geração** (com/sem ICL, julgada por 2 LLMs), com métricas robustas e
significância estatística.

## Visão geral

```
gold set ancorado (01)  ──►  recuperação A/B/C (02)  ──►  geração ×ICL (03)
   │  perguntas + qrels         nDCG, GeoRisk            Qwen-32b + 2 juízes
   │  + respostas-referência                                   │
   └────────────────────────────────────────────────►  concordância (04)
                                                               │
                                              relatório + significância (05)
```

| Etapa | Script | O que faz | Métricas |
|------|--------|-----------|----------|
| 01 | `01_build_golden.py` | Gera QA ancorado nos chunks (factual/multi-hop/comparativo/inanswerable) + qrels automáticos + respostas-referência; exporta `.xlsx` p/ curadoria | — |
| 02 | `02_retrieval_eval.py` | **A** BM25 (bilíngue) · **B** Dense (Gemini, BGE-m3) · **C** RRF | **nDCG@k**, **GeoRisk**, Recall, MRR, MAP (+ por idioma) |
| 03 | `03_generation_eval.py` | RAG (×ICL) → respostas → 2 juízes pontuam | faithfulness, correctness, citação, recusa… |
| 04 | `04_agreement.py` | Concordância entre juízes | κ ponderado, Krippendorff α, Spearman, %±1 |
| 05 | `05_report.py` | Matriz comparativa + significância | randomização pareada |

## Decisões deste pipeline (travadas com o usuário)

- **Gold set:** gerado por uma LLM **independente** (Claude, ou outra que não seja avaliada) e ancorado no corpus. **Sem dataset público** — os benchmarks PT-BR levantados não cobrem o domínio nem a tarefa (ver `data/README.md`). O gabarito tem papel duplo: `qrels` → recuperação; `reference_answer` → geração.
- **Cenário B (dense):** compara **Gemini** (proprietário, 3072d) vs **BGE-m3** (aberto, multilíngue, ~560d/1024) — *viabilidade de migração p/ embedding aberto*. Ambos multilíngues porque o corpus é bilíngue (PT + EN). O jua-4B (PT-jurídico) foi abandonado (ver `docs`, seção 10.1).
- **Corpus bilíngue:** ~65% EN (WHO, GDPR, FDA, NIST…) + ~35% PT. Métricas de recuperação reportadas **por idioma** (PT→PT vs PT→EN cross-lingual); BM25 é **language-aware**.
- **Juízes:** **Llama-3.3-70B** + **gpt-oss-120b** (ambos ≠ gerador Qwen, p/ medir concordância sem viés de auto-preferência).
- **ICL:** few-shot só na **geração** (não afeta a recuperação).

## Instalação

```bash
# a partir da raiz do RAG, com o venv ativo
pip install -r eval/requirements-eval.txt
```

Variáveis no `.env` da raiz: `GROQ_API_KEY`, `CEREBRAS_API_KEY`, `GOOGLE_API_KEY`
(o código aceita o apelido `GOOGLE_GENAI_API_KEY`), `QDRANT_URL`, `QDRANT_API_KEY`.

## Como rodar

```bash
cd niar-rag-prototype

# 0) (uma vez) teste o núcleo de métricas — não usa rede
python eval/lib/metrics.py

# 1) gold set  (--smoke = só 3 chunks p/ validar o fluxo barato)
python eval/01_build_golden.py --smoke
python eval/01_build_golden.py            # completo
#    → valide eval/data/golden_qa_review.xlsx antes de confiar nos números

# 2) recuperação (A/B/C)   --limit p/ debug; --rebuild recomputa caches dense
python eval/02_retrieval_eval.py

# 3) geração + juízes      (custo de API — comece com --limit)
python eval/03_generation_eval.py --limit 30
python eval/03_generation_eval.py         # tudo

# 4) concordância entre juízes
python eval/04_agreement.py

# 5) relatório final (results/report.md + results/report.xlsx)
python eval/05_report.py
```

### Embedding aberto (BGE-m3) — roda em CPU

O BGE-m3 é leve (~560M) e roda em **CPU**, sem GPU nem Colab. Na 1ª execução do `02`
ele embute o corpus localmente e guarda os vetores em cache
(`results/indexes/dense_bge_m3_<N>.npz`); recomputa só com `--rebuild`. O caminho
Colab/`embed_offline.py` continua existindo, mas é **opcional** (só valeria a pena
para um modelo grande de GPU, como o jua-4B que foi abandonado).

## Notas importantes (lidas do seu código)

1. **Corpus atual: 34 documentos / 4.780 chunks** (bilíngue). Ao mudar o corpus,
   reindexe o Qdrant (`build_vectorstore.py`) e **regenere o gold set** (os chunk_ids
   mudam com o re-chunking).
2. **Chave Google:** produção lê `GOOGLE_GENAI_API_KEY`, mas o `.env` define
   `GOOGLE_API_KEY`. O `eval/lib/common.py` trata o apelido; confirme que o
   embedding de produção também está pegando a chave certa.
3. **Sem o corte `score_threshold=0.60`** na avaliação: A e B são buscas independentes
   no corpus inteiro, cada uma retornando top-`retrieval.top_k` (100), para que
   Recall/nDCG façam sentido. O corte de 0.60 é só de produção.
4. **GeoRisk** compara cada sistema contra o *pool* {A, B, C}; quanto mais sistemas
   no pool, mais informativa a métrica. α em `metrics.georisk_alpha` (Dinçer 2014).
5. **BGE-m3** (~560M) roda em CPU; os vetores ficam em cache
   (`results/indexes/dense_bge_m3_<N>.npz`) — só recomputa com `--rebuild`.

Tudo é dirigido por [`config.yaml`](config.yaml) (modelos, k, cenários, ICL, juízes,
α, seed) para reprodutibilidade.
