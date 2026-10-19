# Julgamentos da rubrica 1–5 — arquivo histórico

Esta pasta guarda a rodada de avaliação de geração de **31/08/2026**. Ela saiu de
`results/generation/` em 07/10/2026, quando a rubrica dos juízes mudou.

**Não misture estes arquivos com os da rubrica nova.** Eles têm o mesmo formato e
podem ter os mesmos nomes de juiz, então a mistura não daria erro — daria número
errado em silêncio.

## O que tem aqui

| | |
|---|---|
| rodada | 31/08/2026 |
| juízes | `llama` e `gptoss` — dois, no Groq |
| escala | **1 a 5 uniforme** em todos os sete critérios |
| cenários | `A_bm25_mt`, `B_dense_gemini`, `C_fusion`, `no_rag`, cada um com `icl0` e `icl1` |
| arquivos | 7 `judged_*.jsonl` (bruto) + 7 `scenario_*.csv` + 1 `.xlsx` |

## Por que saiu do lugar

Três motivos, em ordem de importância:

1. **A escala mudou.** A rubrica atual (`RUBRICA = "mista-v2"` em
   `eval/lib/judges.py`) usa binário 0/1 em `citation`, `refusal` e
   `hallucination`, e ternário 1–3 nos outros quatro. κ ou α calculado sobre
   notas 1–5 somadas a notas 0/1 não mede concordância nenhuma.

2. **O painel mudou.** `llama-3.3-70b-versatile` foi descontinuado no Groq (404
   em 29/09/2026) e a rubrica nova roda com outro painel. Lote julgado por
   juízes diferentes não é comparável.

3. **Um dos braços morreu.** O `A_bm25_mt` — BM25 com tradução — saiu do
   repositório em 30/09: o ranking dele era de 31/08, anterior à migração de
   ids, e tinha 0% dos ids vivos no corpus de 6.134 trechos.

O `04_agreement.py` hoje separa por rubrica sozinho: cada bloco de notas carrega
`_rubrica`, e registro sem o campo é da escala 1–5 por eliminação. Ele avisa e
reporta o que descartou. Arquivar é cinto e suspensório — a proteção existe, mas
não depender dela é mais limpo.

## Como ler estes arquivos, se precisar

O exportador aceita o diretório, então a planilha consolidada sai daqui sem mover
nada de volta:

    python eval/tools/export_judged_xlsx.py \
      --gen-dir eval/results/generation/arquivo-rubrica-1-5 \
      --out /tmp/rubrica_1_5.xlsx

Para a concordância, **não** aponte o `04_agreement.py` para cá esperando
comparar com a rodada nova. Os números desta pasta só se comparam entre si.

## O que estes números disseram

Vale guardar a leitura, porque ela motivou a rubrica nova: nos 700 registros, o α
de Krippendorff médio ficou em **0,379** — concordância fraca —, e `citation` deu
**−0,160**, pior que o acaso. O critério mais objetivo dos sete era o que os
juízes mais discordavam. É exatamente o que a escala binária tenta corrigir.
