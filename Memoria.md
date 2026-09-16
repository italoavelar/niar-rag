# LEME — Memória para a evolução do chatbot

**Criado em:** 21/08/2026 · **Origem:** sessão de trabalho encerrada no artigo do WFA.
**Para que serve:** este arquivo é o ponto de partida de um chat novo, dedicado à evolução da
ferramenta. Tudo aqui foi verificado contra arquivo do repositório — cada número tem fonte citada.

> **Regra herdada, mantida:** nenhum número entra em artigo, slide ou relatório vindo deste
> arquivo. Este arquivo diz **onde** conferir. Releia a fonte e cite `arquivo:linha`.

**Repositório:** `C:\Users\karol\OneDrive\Documentos\niar-rag-prototype`
Ler antes de trabalhar: `CLAUDE.md` (números auditados) e `CADERNO.md` (correções da Karol).

> **Este arquivo é o histórico: o que foi testado, o que deu certo e o que foi refutado.**
> O **plano de trabalho em execução** é outro documento: `benchmark_leme_500.md` — quem faz o quê,
> em que ordem, com que comando. Se você quer saber *o que já sabemos*, é aqui. Se quer saber
> *o que fazer agora*, é lá.
>
> Última atualização de conteúdo: **16/set/2026** (§9 — quatro experimentos de recuperação).

---

## 0. Contexto: o que já aconteceu

- **O artigo do WFA/WebMedia 2026 foi submetido em 14/08/2026.** Está fechado. Se for aceito,
  os números da Tabela 1 mudam no camera-ready (lista de trocas está no chat do WFA).
- **Tudo neste arquivo é trabalho futuro**, para artigos novos. A restrição "não quebrar o
  artigo" caducou: reindexar, reanotar e reexecutar estão liberados.
- **Já aplicado no repositório** (21/08): correção do gabarito de q0015/q0016 por
  `eval/tools/corrige_qrels_q0015_q0016.py`, reexecução de `finalize()` e `05_report.py`, e
  correção do bug de contagem de κ em `eval/05_report.py:527` (contava limites de IC como se
  fossem κ: dizia "5/21 critérios com κ ≥ 0,6"; o certo é **1/7**, pior = `citation` 0,089).
  Backups em `*.bak-pre-correcao-qrels` e `eval/results.bak-*/`, ambos já no `.gitignore`.

---

## 1. O diagnóstico central: o gargalo é ORDENAÇÃO, não recall

> ⚠️ **Leia a §1.1 antes de citar qualquer número por tipo desta seção.** A comparação entre
> tipos de pergunta era desigual, e a conclusão "multi-hop está resolvida" não sobreviveu à
> correção de método de 10/set/2026.

Calculado por `eval/tools/rerank_ceiling.py` sobre `eval/results/retrieval/rankings.json` +
`eval/data/qrels.csv`, **já com o gabarito corrigido**.

**Teto de reordenação** = melhor nDCG@5 possível se reordenássemos perfeitamente os candidatos
que o sistema **já traz**. É um oráculo: nenhum reranker real chega lá.

| Sistema | nDCG@5 atual | teto@20 | teto@50 | teto@100 | R@100 |
|---|---|---|---|---|---|
| **B_dense_gemini** (implantado) | **0,3496** | **0,6980** | 0,8135 | 0,8836 | 0,802 |
| C_fusion | 0,3205 | 0,6699 | 0,8138 | 0,8997 | 0,817 |
| A_bm25_mt | 0,2673 | 0,5508 | 0,6373 | 0,7430 | 0,616 |
| **B_dense_bge_m3** | 0,2507 | **0,5741** | 0,7264 | 0,7985 | 0,680 |
| B_bge_m3_colbert | 0,2358 | 0,5715 | 0,7117 | 0,7985 | 0,680 |
| B_bge_m3_hybrid | 0,2168 | 0,5683 | 0,7053 | 0,7615 | 0,645 |
| A_bm25 | 0,1732 | 0,3727 | 0,4843 | 0,5449 | 0,466 |

**As três consequências:**

1. **31 das 75 consultas não têm nenhum trecho relevante no top-5. Só 2 não têm no top-100**
   (q0077 e q0095, ambas comparativas). Os documentos certos estão sendo encontrados e
   descartados na ordenação.
2. **Reordenar vale ~3,5× mais que trocar o embedding.** Teto de +0,348 (0,350 → 0,698 no top-20)
   contra 0,0989 do gap inteiro BGE↔Gemini.
3. **O caminho aberto alcança a produção atual sem o proprietário.** O BGE-m3 precisa capturar
   apenas **30,6%** do seu próprio teto — (0,3496−0,2507)/(0,5741−0,2507) — para empatar com o
   Gemini que está em produção hoje. A literatura de cross-encoder costuma capturar 40–60%.
   ⚠️ Se puserem reranker no Gemini também, o Gemini continua à frente. A afirmação defensável é
   *"o aberto alcança a qualidade da produção atual"*, não *"o aberto vence o proprietário"*.

> **Revisão (02/set/2026) — os 30,6% do item 3 estão desatualizados.** Aquele cálculo usava o
> corpus pré-correção. Com a E2 rodada no corpus corrigido (ver seção 6), o alvo subiu: o Gemini
> passou de 0,3496 para **0,4422** e o BGE de 0,2507 para **0,2908**. A conta atual é
> (0,4422−0,2908)/(0,6789−0,2908) = **39,0%** do próprio teto, não 30,6% — na borda da faixa de
> 40–60% que a literatura de cross-encoder costuma capturar, e não confortavelmente abaixo dela.
> A ressalva do ⚠️ ficou mais forte, não mais fraca: o teto@20 do Gemini é 0,7904 contra 0,6789
> do BGE, então **nem no cenário-oráculo** o aberto alcança o Gemini reordenado.

### Onde reordenar NÃO resolve

| Recorte | n | nDCG@5 | teto@20 | teto@100 |
|---|---|---|---|---|
| factual | 25 | 0,6031 | 0,9417 | 0,9937 |
| multi_hop | 25 | 0,3980 | 0,7626 | 0,9381 |
| **comparativa** | 25 | **0,0478** | **0,3898** | 0,7188 |
| pt | 35 | 0,4121 | 0,7846 | 0,9425 |
| en | 27 | 0,4370 | 0,8083 | 0,9541 |
| **misto** | 13 | **0,0000** | **0,2358** | 0,5783 |

Das 31 falhas de top-5, **22 são comparativas**. E o teto@20 das comparativas é só 0,3898 — um
reranker perfeito sobre o top-20 ainda as deixaria quebradas. Nas 13 consultas de trecho-fonte
**misto** o denso implantado tem nDCG@5 = **0,0000** e teto@100 de apenas 0,5783: metade dos
trechos relevantes nunca entra no pool. Isso é falha de **recall**, exige decomposição de
consulta — não reranker.

**Causa mecânica:** uma comparativa precisa de trechos de ≥ 2 documentos, com frequência em 2
idiomas. Uma única consulta densa produz um centroide semântico e enche o top-5 com trechos do
documento mais próximo.

---

## 1.1 CORREÇÃO DE MÉTODO (10/set/2026) — a comparação entre tipos era desigual

> **Leia isto antes de citar qualquer número por tipo das seções 1 e 3.** O
> diagnóstico "comparativa é o problema, multi-hop está resolvida" não se
> sustenta: os dois estratos estavam sendo medidos com réguas diferentes.

Multi-hop vinha sendo reportada por **"pelo menos 1 trecho relevante no top-5"**
(critério disjuntivo) e comparativa por **conjunção** (os dois documentos). Sob a
**mesma** régua estrita — todas as âncoras (grau 2) exigidas presentes no top-5:

| Tipo | ≥1 trecho (frouxo) | **Todas as âncoras** | Completude@5 | âncoras/consulta |
|---|---|---|---|---|
| factual | 24/25 | **20/25** | 0,84 | 1,3 |
| multi_hop | 21/25 | **3/25** | 0,48 | 2,4 |
| comparative | 13/25 | **2/25** | 0,28 | 2,0 |

*(BGE-m3 + rr50, corpus atual. Gemini dá 19/25, 4/25, 2/25 — estatisticamente o
mesmo.) Fonte: `eval/results_f3/retrieval/rankings/B_dense_bge_m3_rr50.json` +
`eval/data/golden_qa.jsonl`, recomputável com `eval.lib.metrics`.*

**Multi-hop não está resolvida: 3/25 sob a régua da comparativa.** O 84% existia
porque os trechos de multi-hop estão todos no MESMO documento
(`eval/config.yaml:40-41`), então "cobrir todos os documentos" é satisfeito por um
acerto só. **Nunca escrever "multi-hop resolvida em 84%".**

**Consequência boa para a RQ1:** sob o critério estrito o BGE+rr50 iguala ou
supera o Gemini em tudo (20 vs 19 · 3 vs 4 · 2 vs 2; completude geral 0,532 vs
0,504). A tese de substituição fica mais defensável, não menos.

### Métrica formalizada
`eval/lib/metrics.py` passou a expor `evidence_completeness_at_k`,
`complete_evidence_at_k`, `document_coverage_at_k` e `document_of` (com
self-test). Reportar completude **ao lado** do nDCG@5, nunca no lugar.
⚠️ `document_of`: o corpus tem **duas** convenções de ID (`<doc>_pN_cN` para os
4.186 chunks de PDF, `<doc>_cN` para os 974 de HTML). Parser que só trate o
padrão de PDF faz cada chunk de HTML virar um "documento" — erro silencioso.
A fonte canônica é `metadata["document_id"]`.

### Onde o orçamento de contexto morde (BGE+rr50)

| k | factual | multi_hop | comparative |
|---|---|---|---|
| 4 (produção até 15/set) | 17/25 | 3/25 | 1/25 |
| **5 (produção desde 16/set)** | **20/25** | **3/25** | **2/25** |
| 8 | 21/25 | 7/25 | 4/25 |
| 20 | 22/25 | **14/25** | **5/25** |

**Multi-hop é limitada por ORÇAMENTO** (3→14 abrindo k), **comparativa não** (2→5).

✅ **Aplicado em 16/set:** `limit` passou de 4 para 5 em `agent/utils/tools.py:133`
(commit `1090b7b`, Karol). Vale +3 no factual e +1 na comparativa.

**Ainda na mesa:** k=8 leva multi-hop de 3/25 a 7/25 e comparativa de 2/25 a
4/25. O custo é contexto maior no prompt do gerador — medir o efeito sobre a
qualidade da resposta antes de subir.

### O gargalo restante da comparativa: discriminação DENTRO do documento

Posição da âncora dentro do próprio documento (n=49 documentos exigidos):

| Consulta usada | mediana | top-1 | top-2 | top-3 |
|---|---|---|---|---|
| pergunta em PT | 8 | 9 | 13 | 18 |
| PT + tradução EN | 4 | 10 | 17 | 21 |
| as 2 sub-perguntas | 4 | 11 | 21 | 23 |
| **as quatro juntas** | **3** | 13 | **23** | 23 |

⚠️ **Esta tabela é DIAGNÓSTICO, não resultado.** Ela mede posição da âncora
*dentro do próprio documento*, sobre 49 documentos — não quantas perguntas ficam
respondidas. **Multiplexar a consulta foi testado no que decide em 16/set e não
se sustentou: ver §9.5.** Melhorar a mediana interna de 8 para 3 não coloca as
duas âncoras num top-5 global.

Os caches existem (`eval/results/indexes/query_translations.json`,
`eval/results_decomp/subqueries_cache.json`) e o custo de LLM é zero — o que
falta não é viabilidade, é ganho.

**Aritmética que governa tudo:** comparativa é conjunção de dois acertos, então a
taxa conjunta ≈ (taxa por documento)². Com 23/49 ≈ 47% por documento, o teto é
≈ 22% ≈ 5-6/25 — que é exatamente o teto medido do roteamento com oráculo (6/25).
Para 20/25 seria preciso ~89% por documento (top-2 em ~44 dos 49).

### O que NÃO explica a comparativa (testado e descartado, 09-10/set)
- Reordenação (cross-encoder), MMR, cota por documento, reserva de vaga por
  sub-consulta, alocação particionada, roteamento: todos ficam em 2-6/25.
  Scripts: `eval/tools/decompose_slot_reserve.py`, `particao_por_lado.py`,
  `busca_roteada_por_documento.py`.
- **Idioma:** comparativas 100% em PT são as PIORES (0/7); mistas 1/13; EN 1/5.
  Nas mistas o lado que falta é o EN 13× contra 6× o PT — fator secundário real,
  não a causa.
- **Corrupção do gabarito pelo re-chunking de 01/set:** existe e é geral (o texto
  anotado deslizou de chunk em ~40% das consultas; `golden_qa_final.csv` é de
  31/ago, anterior ao corpus), mas atinge comparativa **menos** que multi-hop
  (retenção 0,809 vs 0,722). Não é a causa — **mas precisa ser corrigido**:
  remapeamento mecânico casando o texto anotado no chunk novo, sem rejulgar.
- **Julgamento incompleto:** real (das 305 duplas do pool das 4 configurações, 34
  julgadas e 190 do documento certo sem julgamento), mas não era o defeito
  bloqueante. Fica como limitação a declarar e como candidato a pooling TREC
  depois que o conjunto de sistemas do artigo estiver congelado.

---

## 2. O bug do chunker — CORRIGIDO (22/ago/2026), com um resíduo tratado em 13/set

> ✅ **Resolvido.** O commit `1331d93` (Ítalo, 22/ago) extraiu `src/chunking.py` como módulo
> compartilhado, com guarda de avanço mínimo (`minimum_progress`) e `RuntimeError` se o progresso
> não for positivo, mais 148 linhas de teste. Conferido no corpus de 13/set: as nove páginas
> degeneradas foram de **699 chunks para 33** — a previsão desta seção era ~37.
>
> **Mas a q0075 continuou falhando depois disso**, por um resíduo diferente: sem os 83 fragmentos,
> a página 45 da UNESCO ficou com a frase que nomeia os quatro níveis isolada em **242 caracteres**,
> acima do piso `MIN_CHUNK_SIZE` de 120 e portanto nunca candidata a fusão, e a tabela que os
> descreve partida em duas por `CHUNK_SIZE`. Corrigido em 13/set com `MIN_CHUNK_SIZE = 300` e
> `TABLE_CHUNK_SIZE = 2500`. **A reextração foi feita em 13/set** — 5.160 → 4.897
> trechos, fingerprint `da69227b35c04c4d`: zero trechos abaixo de 300 caracteres (eram 219) e
> páginas com tabela partida de 34 para 21. A página 45 da UNESCO virou um único trecho de 2.012
> caracteres com a frase e a tabela inteira, e q0015/q0065/q0075/q0096 passaram a ancorar nele.
>
> O texto abaixo fica como registro do diagnóstico original.

`src/extract_to_jsonl.py:580`:

```python
new_start = max(end - overlap, start + 1)
```

Com `CHUNK_SIZE = 1200` e `CHUNK_OVERLAP = 200`: quando `choose_split_point` devolve um chunk
**mais curto que 200 caracteres**, `end - overlap` cai antes de `start`, e o `max` escolhe
`start + 1`. **A janela anda um caractere**, e o laço emite um chunk quase idêntico a cada
caractere até escapar. O gatilho é conteúdo sem pontuação de fim de frase na janela — tabelas.

**Nove páginas do acervo têm isso — 699 chunks onde deveriam existir ~37 (14,6% do índice de PDF):**

| Documento | Pág | Chunks | Redundância | Após dedup |
|---|---|---|---|---|
| gdpr_regulation_EU_2016 | 14 | 87 | 66% | 7 |
| european_health_data_space_EU_2025 | 62 | 85 | 73% | 5 |
| regulatory_considerations_ai_health_WHO_2024 | 9 | 85 | 73% | 5 |
| de_identifying_government_datasets_NIST_2023 | 53 | 84 | 81% | 4 |
| multimodal_models_guidance_WHO_2024 | 18 | 84 | 77% | 4 |
| ethical_impact_assessment_UNESCO_2023 | 45 | 83 | 81% | 3 |
| ethics_governance_ai_health_WHO_2021 | 8 | 83 | 84% | 3 |
| plano_brasileiro_ia_BR_2025 | 99 | 83 | 82% | 3 |
| ethics_governance_ai_health_WHO_2021 | 9 | 25 | 60% | 3 |

A mediana do acervo é **3 chunks por página**. Os fragmentos degenerados têm ~160 chars contra
mediana de 1.025.

**Correção proposta** (não aplicada — garantir avanço mínimo):

```python
new_start = end if (end - start) <= overlap else end - overlap
```

**A correção é inócua onde o bug não ocorre.** Em páginas cujos chunks já passam de 200 chars, o
`max()` já devolvia `end - overlap`. Os IDs de 1.251 das 1.260 páginas não mudam.

### Por que este bug importa muito

Ele explica **os dois modos de falha conhecidos do sistema**, e só 5 das 75 consultas tocam essas
páginas: q0015, q0016, **q0065**, **q0075**, q0096.

- **q0065** — o caso dos "3 níveis em vez de 4" da UNESCO. A tabela de gravidade está estilhaçada
  em 83 fragmentos sobrepostos; recuperar 5 traz cinco lascas da mesma frase.
  Comparação decisiva: **q0015 recebeu `p45_c0` + `p45_c81` e acertou os 4 níveis; q0065 recebeu
  `p45_c81` sem `p45_c0` e reportou 3.** Mesmo gerador, mesmo acervo, só a recuperação mudou.
- **q0075** — a **única alucinação confirmada pelo especialista** em toda a avaliação. Observação
  dele: *"A classificação de gravidade segundo a UNESCO existe no corpus, mas não foi retornada
  como contexto."* Agora sabemos por quê.

**Teste natural de validação:** depois de corrigir o chunker e reindexar, se q0065 passar a
reportar 4 níveis e q0075 parar de negar o que existe no acervo, há demonstração causal com uma
variável só mudando. **Isso é resultado de artigo.**

⚠️ Ao reindexar, os IDs das 9 páginas mudam. Os qrels das 5 consultas precisam ser **remapeados**
(casar o texto antigo no chunk novo) — é mecânico, não é rejulgar. Nas outras 70 nada muda.

---

### Reancoragem do gabarito (13/set/2026)

Mudar o recorte muda o id dos trechos cujo texto mudou — e a anotação morreria junto, como já
aconteceu em 01/set. Não morreu desta vez porque `golden_qa.jsonl` guarda `qrels_text`, o texto de
cada âncora: quem tem o texto reancora por conteúdo.

Medido comparando com `golden_qa.jsonl.bak-pre-reancoragem-20260913-162606`:

| Das 256 âncoras das 75 perguntas respondíveis | |
|---|---|
| id inalterado | **247** |
| remapeadas por casamento de texto | **9 ids antigos → 7 novos** (duas fusões) |
| ambíguas | 0 |
| perdidas | 0 |

Sete perguntas tocadas: `q0001`, `q0012`, `q0015`, `q0052`, `q0065`, `q0075`, `q0096`. Duas fusões
legítimas a conferir à mão: `q0012` (8 âncoras → 7) e `q0015` (2 → 1). **Nenhuma reanotação humana.**

No corpus: 5.160 → 4.897 trechos, **4.661 ids intactos** (499 sumiram, 236 nasceram). É o que
permitiu reembutir só 236 vetores em vez de 4.897.

---

## 3. As três mitigações, e a ordem

### A — Reranker cross-encoder sobre pool ampliado ⭐ COMEÇAR POR AQUI

Recuperar top-50, passar os candidatos por um cross-encoder que lê consulta e trecho **juntos**,
entregar os 5 melhores ao gerador.

- **Ganho:** teto de +0,348; captura realista de 40–60% → **+0,14 a +0,21** de nDCG@5.
- **Custo de avaliação: ZERO de API.** Roda offline sobre o `rankings.json` já em disco.
- **Modelos abertos:** `BAAI/bge-reranker-v2-m3` (568M, mesma família do embedding atual),
  `Qwen/Qwen3-Reranker-0.6B` e `-4B`, `jinaai/jina-reranker-v2-base-multilingual` (278M).
- **Riscos:** latência; pode piorar consultas fáceis (factual já em 0,603) — medir com **GeoRisk,
  não só com a média**; reranker multilíngue pode herdar o viés que zera as mistas.

#### F3 — RESULTADO (02/set/2026)

`BAAI/bge-reranker-v2-m3` sobre o pool do BGE-m3, corpus corrigido, 75 consultas, scores gerados
em GPU (Colab) e avaliados localmente. Artefatos em `eval/results_f3/retrieval/`, via
`eval/tools/rerank_eval.py --scores ...`.

| Sistema | nDCG@5 | Δ vs base | IC95% | p (Holm) | GeoRisk |
|---|---|---|---|---|---|
| BGE base | 0,2908 | — | — | — | 0,3451 |
| BGE + rr20 | 0,4342 | +0,1434 | [+0,0841; +0,2029] | 0,0003 | 0,4466 |
| **BGE + rr50** | **0,4376** | **+0,1468** | [+0,0863; +0,2076] | 0,0003 | **0,4501** |
| BGE + rr100 | 0,4372 | +0,1464 | [+0,0850; +0,2079] | 0,0003 | 0,4467 |

**O ganho previsto se confirmou.** A seção 3A projetava +0,14 a +0,21; veio +0,1468. A captura
foi de **37,8%** da margem até o teto@20 — a literatura indicava 40–60%, e precisávamos de 39,0%
para empatar com o Gemini. O GeoRisk sobe junto (0,3451 → 0,4501), então o ganho médio **não**
esconde regressão sistemática; ainda assim 44 consultas melhoram e 12 pioram.

**O pool 20 basta.** Ele entrega 97,7% do ganho do pool 50. Como o custo do cross-encoder é
linear no pool, isso corta 2,5× a latência de graça. Medido em CPU (12 threads, XLM-R large,
`max_length=384`): pool 20 = 36,6 s por pergunta, pool 50 = 99,8 s. **Sem GPU o reranking é
inviável para atendimento interativo** — mas a avaliação é lote, então isso não bloqueia a
dissertação, só o deploy.

**BGE reordenado alcança o Gemini:** 0,4376 vs 0,4422, δ = −0,0045, p = 0,89. Mas o veredito
formal de não-inferioridade sai **indeterminado**: o IC [−0,0663; +0,0594] é largo demais e seu
limite inferior ultrapassa a margem de 0,03.

⚠️ **O alerta da seção 1 segue de pé, agora com evidência:** o Gemini **não foi reordenado**. O
teto@20 dele é 0,7904 contra 0,6789 do BGE. A afirmação defensável continua sendo *"o aberto
alcança a qualidade da produção atual"*, não *"o aberto empata com o proprietário"*. Para fechar
essa questão seria preciso reordenar o Gemini também — outra rodada de 7500 pares.

#### Diagnóstico por tipo (09/set/2026): dois problemas diferentes, não um só

> ⚠️ **Superado em 10/set — ver seção 1.1.** Os números desta subseção usam
> recall/nDCG, que não distinguem "achou metade da evidência" de "achou a metade
> certa". Sob o critério conjuntivo, multi-hop cai para 3/25 e a leitura
> "multi-hop = só reordenação, comparativa = só recall" não se sustenta.

"Bater com o Gemini" na média esconde que multi-hop está tão longe do teto quanto comparativa —
só que por uma causa diferente. Quebrando o F3 por tipo de pergunta (BGE, corpus corrigido):

| Tipo | atual (rr50) | teto@20 (BGE) | teto@100 (BGE) | recall@100 | captura@20 |
|---|---|---|---|---|---|
| factual | 0,6458 | 0,8308 | 0,9581 | 0,8265 | 50,0% |
| **multi_hop** | 0,4338 | 0,7183 | 0,9063 | **0,8800** | **40,2%** |
| **comparative** | 0,2333 | 0,4878 | 0,7587 | **0,7133** | **15,8%** |

**Comparativas: problema de RECALL.** O documento nem entra no pool de 100 em 2/25 perguntas;
pool maior não muda o recall (é invariante ao reranker). Precisa de outra busca, não de reordenar
a que já existe — daí a decomposição da seção B, abaixo.

**Multi-hop: problema de REORDENAÇÃO, não de recall.** Recall@100 já é alto (0,88, chegando a
1,0 no Gemini). O reranker captura só 40,2% da margem — menos que factual (50,0%) — e pool maior
não ajuda (rr20/rr50/rr100 ficam empatados em ~0,43). Causa provável: `multi_hop` no gabarito é
2+ trechos do MESMO documento (2,56 em média); o cross-encoder pontua cada trecho isolado e não
tem como saber que dois fragmentos se complementam — ele não escolhe bem QUAL conjunto de vários
trechos do mesmo documento forma o top-5.

**Nota de escopo:** mesmo sem qualquer conserto, o BGE+reranker em multi-hop (0,4338) já iguala/
supera levemente o próprio Gemini (0,4228) — não é déficit em relação à produção atual, é os dois
sistemas estruturalmente longe do teto. Diferente de comparativas, onde o BGE (0,2333) fica atrás
do Gemini (0,2734): ali sim há gap real a fechar.

#### Tentativa descartada: MMR para multi-hop (09/set/2026)

Hipótese: já que o problema é "o reranker não sabe que trechos se complementam", uma reordenação
de diversidade (Maximal Marginal Relevance — penaliza similaridade com o já selecionado) poderia
recuperar parte da margem. Testado com os scores do cross-encoder já calculados (custo zero de
API): sweep de λ sobre pool=50, contra o baseline BGE+rr50 (λ=1,00 reproduz o baseline
exatamente, confirmando a implementação).

| λ | geral | factual | multi_hop | comparative |
|---|---|---|---|---|
| 1,00 (= rr50) | 0,4376 | 0,6458 | 0,4338 | 0,2333 |
| **0,90** | 0,4401 | 0,6456 | **0,4415** | 0,2333 |
| 0,80 → 0,50 | piora progressiva em geral, factual e multi_hop | | | |

Significância (λ=0,90 vs rr50, isolando só o efeito da diversidade): multi_hop Δ=+0,0077,
IC95%=[−0,0238; +0,0356], p=0,66 (n=25, 4 melhoram / 2 pioram); factual Δ=−0,0003, p=1,00.
**Não significativo** — captura ~2,7% da margem até o teto, não os ~70% que uma comparação inicial
contra o baseline errado (denso sem reranker) sugeria antes da correção.

**Por que não funciona:** MMR penaliza similaridade textual, mas similaridade não é
complementaridade de resposta — um trecho pode ser diferente do primeiro e ainda ser irrelevante,
e o MMR não distingue os dois casos. É instrumento genérico demais para o problema específico
(selecionar o subconjunto de K trechos que juntos respondem, não K trechos apenas diferentes).
**Descartado como direção — resultado negativo limpo, obtido sem custo de API/GPU.** Dado o nota
de escopo acima (multi-hop já não é déficit vs. Gemini), não justifica investir em algo mais caro
(reranker listwise) agora; prioridade vai para a decomposição de comparativas, onde há gap real.

### B — Decomposição de consulta + cota de diversidade de documento

1. O gerador (já é um LLM) reescreve a consulta em 2–3 subconsultas; recuperar para cada e fundir
   por RRF.
2. ~~No top-5 final, no máximo 2 trechos por documento-fonte → força ≥ 2 documentos distintos.~~
   **REFUTADO em 13/set/2026 — ver §9.2.** A cota de diversidade destrói o multi-hop, porque as
   duas âncoras dele estão no MESMO documento (mediana 1 documento distinto por pergunta). Só a
   decomposição (item 1) foi testada em 09/set; a cota ficou sem medição até agora.

- **Única das três que ataca o recall** das comparativas (0,0478) e mistas (0,0000).
- **O que está em jogo:** as comparativas são 1/3 do gold set. Levá-las ao nível das multi_hop
  (0,398) moveria o nDCG@5 global de 0,350 para ~0,467 sozinha.
- **Mitigação de risco:** aplicar decomposição **condicionalmente** — só quando um classificador
  barato marcar a consulta como comparativa (84% delas nomeiam a norma; é sinal aproveitável).

#### RESULTADO (09/set/2026): positivo na direção certa, não significativo

Testado nas 25 comparativas **e** nas 25 multi-hop (pedido explícito, apesar de multi-hop não
ter problema de recall — ver diagnóstico acima). Pipeline: `qwen/qwen3.8-27b` (mesmo gerador de
produção, via Groq) decompõe a pergunta em 2 subperguntas; cada uma é embutida com BGE-m3 e
buscada (top-100, pool igual ao da busca original, para comparação justa); as duas listas são
fundidas por RRF (`lib.retrievers.rrf`, a mesma função do Cenário C); o pool fundido (top-30) é
reordenado pelo cross-encoder com a **pergunta original** (a decomposição só amplia o pool — a
resposta final tem que valer para a pergunta inteira). Script: `eval/tools/decompose_eval.py`.

**Qualidade da decomposição: boa.** Nos 4 casos investigados manualmente (q0080, q0090, q0095,
q0097, q0056, q0057), o LLM identificou corretamente as duas entidades/aspectos comparados,
inclusive casos como "NIST AI RMF vs PL 2338/2023" e "GDPR: portabilidade vs medidas de segurança"
— a decomposição não foi a causa dos casos que pioraram.

**Fase 1 — recall (sem reranker, custo zero de GPU):**

| Tipo | recall@100 original | recall@100 fundido | zero-relevante (orig → fundido) |
|---|---|---|---|
| comparative | 0,7133 | **0,7700** (+5,67pp) | 2 → 3 |
| multi_hop | 0,8800 | 0,9067 (+2,67pp) | 0 → 0 |

Sinal limpo e real para comparativas: mais documentos relevantes entram no pool, quase sem custo
(1 caso novo de zero-relevante, sobre 25; os outros 2 zeros já existiam antes de qualquer conserto).

**Fase 2 — nDCG@5 final (após reordenar com o cross-encoder), contra BGE+rr50:**

| Tipo | baseline (rr50) | decomp+fusão+rr | Δ | IC95% | p | % da margem ao teto@20 |
|---|---|---|---|---|---|---|
| comparative | 0,2333 | 0,2456 | +0,0123 | [−0,0211; +0,0485] | 0,62 | 4,8% |
| multi_hop | 0,4338 | 0,4594 | +0,0256 | [−0,0302; +0,0814] | 0,41 | 9,0% |

**Não significativo em nenhum dos dois** (n=25, IC cruza zero). Custo real: ~59 min de CPU (1500
pares no cross-encoder) + 50 chamadas de LLM para decompor.

**O achado que importa está no meio do caminho, não no fim:** o ganho de recall (+5,67pp,
comparativas) não virou ganho proporcional de nDCG@5. A decomposição resolve a metade do problema
que promete resolver — traz o documento certo para o pool — mas o cross-encoder ainda precisa
**promover** esse documento recém-achado para o top-5, e sua captura permanece baixa (mesma
limitação medida no F3: comparativas capturam só 15,8% da própria margem de reordenação).
Decomposição sem um reranker mais forte deixa a maior parte do ganho potencial na mesa.

**Recomendação:** não priorizar mais rodadas de decomposição isolada — o gargalo real voltou a
ser a capacidade de reordenação do cross-encoder, não a recuperação. Prioridade passa a ser a
Etapa 03 (geração), que segue sendo o maior gap verificado do projeto.

### C — Chunking com consciência de estrutura

1. **Corrigir o bug do §2** (é bug, não melhoria — e contamina qualquer comparação de embeddings
   feita com 14,6% do índice sendo lixo redundante).
2. Respeitar fronteiras de tabela, artigo e seção no corte de 1.200 chars.
3. Filtrar páginas não informativas **no fim** do documento — índice remissivo
   (`codigo_etica_medica_CFM_2019_p58_c0`, tabela de verbetes, chegou ao top-5 na q0094).
4. **Cabeçalho contextual:** prefixar cada trecho com `[documento · emissor · seção]` antes de
   embedar. Hoje um trecho é vetorizado sem saber de que norma veio — parte do R@100 baixo nas mistas.

**Única das três que sobe o TETO.** A e B trabalham dentro do que o índice permite.

**Ordem: C.1 (o bug) → A → B → C.2-4 → embeddings.** O item C.1 subiu de prioridade porque
contamina todo o resto.

---

## 4. Monitorar qualidade em produção — o insight do especialista

Esta seção é a que veio do cruzamento entre a revisão do especialista jurídico e as métricas.

### 4.1 O achado que fundamenta tudo: os juízes LLM não separam o que deveriam

Fonte: **`Documentos/LEME/revisao_juizes_especialista_PREENCHIDA.csv`** — 114 linhas, 113 usáveis
(a de `q0094 / Relevância da resposta` tem texto de observação na coluna da nota). Amostra
estratificada: 35 controle (|Δ juízes| = 0) + 79 divergente (|Δ| ≥ 2), sobre 44 qids distintos da
config implantada. ⚠️ **As cópias em `eval/data/` são template em branco — não use.**
⚠️ **Nunca reportar concordância agregando os dois grupos:** a amostra é 69% divergente por
construção. Controle: especialista 4,97, idêntico aos dois juízes em 34/35. Divergente:
especialista 3,58, mais próximo do juiz 2 em 43 casos e do juiz 1 em 32.

Nos 13 qids em que o especialista deu nota nos dois critérios:

| | r(fidelidade, completude) | média fid | média compl | separação |
|---|---|---|---|---|
| **Especialista** | **+0,411** | 4,54 | **3,00** | **1,54** |
| Juiz 1 (llama) | +0,539 | 4,85 | 4,62 | 0,23 |
| Juiz 2 (gptoss) | +0,846 | 2,38 | 2,15 | 0,23 |

Nas 100 respostas da config implantada a correlação interna é +0,718 (llama) e +0,719 (gptoss).

**O padrão dominante do especialista é "5 em fidelidade, 2 em completude"** — *a resposta é fiel ao
que recebeu e não cobre o que importa*. **Nenhum dos dois juízes produz esse padrão.** Juiz 1 diz
"bom/bom", juiz 2 diz "ruim/ruim". Cada um colapsa numa impressão global, em direções opostas.

Quebrando por qualidade de recuperação (itens revisados pelo especialista):

| Fidelidade | Juiz 1 | Juiz 2 | Especialista |
|---|---|---|---|
| recuperação falhou (R@5=0, n=9) | 4,78 | **1,78** | 4,44 |
| recuperação achou (n=6) | 4,83 | 3,33 | 5,00 |

| Completude | Juiz 1 | Juiz 2 | Especialista |
|---|---|---|---|
| recuperação falhou (n=8) | **4,50** | 1,75 | **2,62** |
| recuperação achou (n=5) | 5,00 | 3,60 | 3,80 |

**Juiz 1 acerta fidelidade e erra completude. Juiz 2 erra fidelidade e acerta completude.**
Nenhum é "o bom". E a média dos dois — que é o que vai nas tabelas — não corresponde a nenhum
dos dois nem ao humano.

> **Isto é contribuição científica publicável**, e é mais forte que o diagnóstico de sistema:
> *os critérios que a literatura de LLM-as-judge trata como ortogonais não são empiricamente
> ortogonais, e há juízo humano especialista, item a item, para provar que a separação é possível.*
> A afirmação é falsificável e foi testada.

### 4.2 Detector de falha de recuperação sem gabarito

**Por que vale:** os qrels são a parte cara da avaliação. Um detector que dispensa gabarito
permite monitorar a recuperação **em produção**, onde nunca haverá gabarito.

Correlações medidas nas 75 respondíveis (config implantada):

| Preditor | ρ (Spearman) vs nDCG@5 |
|---|---|
| **Completude** sozinha | **+0,724** |
| Fidelidade sozinha | +0,620 |
| Gap (fidelidade − completude) | −0,433 |

⚠️ **O gap é o PIOR dos três** — subtrair dois sinais correlacionados com o mesmo alvo cancela
sinal. Como detector binário o gap dá no máximo precisão 0,63 / revocação 0,71 / F1 0,67.
**Use completude, e a do juiz 2** (o juiz 1 é cego para incompletude: 4,50 onde o especialista dá 2,62).

**Como construir, em quatro etapas:**

1. **Linha de base burra primeiro.** Testar o **score de similaridade do top-1** do Qdrant.
   Produção já tem `score_threshold=0.60` (`agent/utils/tools.py`). Se um limiar de score prevê
   `R@5=0` tão bem quanto um juiz LLM, o juiz não se paga. Meia hora de trabalho, obrigatório.
2. **Ajustar nas 75 com gabarito.** Alvo binário `R@5 = 0` (31 positivos, 44 negativos). Entradas:
   completude do juiz 2, score do top-1, dispersão dos scores do top-5, tamanho da resposta,
   marcadores de recusa parcial, e **tipo da pergunta como controle**.
3. **Validar contra o ESPECIALISTA, não contra os juízes.** Etapa inegociável dado o §4.1 —
   validar um detector de notas de juiz usando notas de juiz é circular sobre um instrumento com
   viés medido. Os 44 qids revisados são o padrão-ouro.
4. **Reportar precisão/revocação, não correlação.** Escolher o ponto de operação pelo custo:
   alarme falso é barato (reexaminar uma consulta); falha não detectada é cara (resposta
   incompleta sobre norma médica chega ao usuário). Isso empurra para **revocação alta**.

**Confundidor a controlar:** o sinal acompanha o tipo da pergunta quase perfeitamente —
gap comparativa +1,02 · multi_hop +0,32 · factual +0,12, na mesma ordem do nDCG@5. Pode estar
detectando "isto é uma comparativa", não "a recuperação falhou".

**Limitação honesta:** n=75 com 31 positivos. Qualquer limiar aprendido precisa de validação
cruzada. É estudo piloto que justifica ampliar o conjunto, não detector pronto.

### 4.3 O que o detector NÃO pega — monitorar à parte

**Distorção deôntica.** O gerador preserva o vocabulário da norma e **altera a estrutura lógica**.
Nenhum juiz automático pega, porque não é alucinação pela rubrica (nada é inventado) — por isso
convive com o `0/100` de alucinação. Fonte: coluna `observacao` do CSV do especialista.

| Item | Fonte | Resposta | Juízes |
|---|---|---|---|
| q0030 | "hospitals **should also have** a duty" | "os hospitais **têm o dever**" | j1=3, j2=1, esp=3 |
| q0094 | "excetuadas… **ou quando** sua recusa possa trazer danos" | "**exceto se** sua recusa **não** trouxer danos" | j1=**5**, j2=2, esp=3 |

Na q0094 a resposta **inverte o Princípio VII do Código de Ética Médica** — sugere que o médico
poderia recusar atendimento em urgência se não houvesse dano — e o **juiz 1 deu nota máxima**.

*(O `CLAUDE.md` §2b cita um terceiro caso, q0009. Ele **não está** entre os 44 qids do CSV do
especialista — confirmar a procedência antes de citar.)*

**Detector de recusa é sensível à FORMA da frase** (`eval/lib/refusal.py:35-40`): q0090 ("não há
informações suficientes") casa; q0095 ("não é possível responder à parte da pergunta") **não casa**,
com comportamento idêntico. Em produção não dá para monitorar recusa por frase canônica.

---

## 5. Dimensões de IA Responsável na geração

Não têm o mesmo estatuto. Chamar as cinco de "avaliadas" seria inflar o trabalho.

| Dimensão | Avaliar? | Forma | Esforço |
|---|---|---|---|
| **Transparência** | ✅ | atribuição por afirmação | médio |
| **Responsabilização** | ✅ | preservação de modalidade deôntica | médio, achado original |
| Justiça / vieses | ⚠️ reescopar | jurisdição, idioma, emissor | médio |
| Privacidade e segurança | ⚠️ parcial | 3 sondas de red team, binário | baixo |
| Governança de dados e IA | ❌ | seção de conformidade documentada | baixo |

**As duas primeiras se medem sobre dados que já existem** — os `judged_*.jsonl` guardam `answer` +
`context_chunk_ids`. Não precisa regerar resposta nenhuma.

**Transparência — atribuição por afirmação.** Hoje: 100% das URLs ancoradas no acervo, mas **por
construção**; e em **81/100** respostas o modelo cita exatamente o conjunto que recebeu, usado ou
não. Só em 19/100 seleciona. Medir por **sentença**: para cada uma, existe trecho em
`context_chunk_ids` que a sustenta? Método: NLI multilíngue, com sobreposição lexical como piso e
amostra com dupla anotação humana. Reportar o par **cobertura de atribuição** × **precisão de
citação**. Exemplo de comportamento bom: q0016 recuperou 5 trechos de 3 documentos e citou só o
GDPR. Script existente: `eval/tools/citation_grounding.py`.
⚠️ Nunca escrever "URLs inventadas" — não foi verificada a existência dos endereços.

**Responsabilização — modalidade deôntica.** Extrair operadores (PT `deve/deverá/é obrigatório/é
vedado/pode/recomenda-se` · EN `shall/must/should/may`) e conectivos condicionais (`exceto se ·
salvo quando · desde que · ou quando`) do trecho-fonte e da resposta; classificar em
**preservado · endurecido · enfraquecido · invertido**. Extração por regra (léxico fechado),
**julgamento humano** na classificação — o ponto inteiro é que o juiz LLM não pega. Reportar a
taxa de **endurecimento** isolada: num sistema normativo, virar recomendação em obrigação é a
falha com consequência jurídica.

**Justiça.** Viés **demográfico não dá** — corpus normativo, sem atributo protegido nas consultas.
O que dá: (i) **jurisdição** — 31 documentos BR e 32 internacionais; com que frequência responde
com norma brasileira onde a internacional se aplica? Sinal de que existe: q0098, quase atribuiu ao
NIST o que era do Conselho da Europa. (ii) **idioma** — o acervo é 45,6% PT / 54,4% EN, mas o
acesso não é equitativo: misto = 0,0000. (iii) **emissor** — a distribuição de fontes citadas segue
a do acervo ou concentra?

**Privacidade.** O acervo é público, não há PII indexada. Sondas (binário, ~1 dia): injeção de
prompt via trecho envenenado (risco real — 326 páginas vieram da web), vazamento do prompt de
sistema, eco de PII que o usuário cole na pergunta. **Não** virar nota 1–5.

**Governança.** Não é propriedade de uma resposta. Documentar: proveniência (`corpus_manifest.csv`,
`corpus_documentos.csv`), licenças (MIT + CC BY 4.0), versionamento, rastro de auditoria
(`judged_*.jsonl`), supervisão humana (curadoria de 50/100 + revisão do especialista).

---

## 6. Trilha de embeddings — mantida, reordenada, com 3 correções de método

**E1** Qwen3-Embedding (`0.6B`, `4B`) vs `BAAI/bge-m3`. Mesmo protocolo, mesmos 75 qrels,
nDCG@5 primário. **Reportar também o teto de reordenação de cada um** — se o Qwen tiver nDCG@5
pior mas teto maior, ele é o melhor candidato a reranker/fine-tuning, e o número bruto o
descartaria.
**E2** melhor aberto vs `gemini-embedding-001`, TOST com margem 0,03, veredito pelo IC.
**E3** fine-tuning (só se o gap passar a margem): MultipleNegativesRankingLoss com negativos
difíceis minerados do acervo.
**E4** reavaliar contra o Gemini.

#### E1 — RESULTADO (02/set/2026)

Rodada no corpus corrigido (5160 chunks, fingerprint `7cfc068dde147394`), 75 consultas com
qrels, ambos os modelos em float32. Artefatos em `eval/results_e1/retrieval/`, gerados por
`eval/tools/e1_qwen_vs_bge.py --offline`.

| Sistema | nDCG@5 | teto@20 | teto@50 | teto@100 | R@100 |
|---|---|---|---|---|---|
| **BGE-M3** | **0,2908** | **0,6789** | 0,7837 | 0,8744 | 0,8066 |
| Qwen3-0.6B | 0,2539 | 0,6097 | 0,7279 | 0,8318 | 0,7493 |

δ (Qwen − BGE) = **−0,0369**, IC95% [−0,0877; +0,0132], p = 0,15, Cliff δ = −0,098.

**O teto refuta a hipótese de resgate.** A E1 existia para detectar o caso "nDCG@5 pior mas teto
maior", que faria do Qwen o melhor candidato a reranker. Não ocorreu: o BGE domina em **todos** os
pools. O Qwen 0.6B não é melhor nem no que traz, nem no que poderia trazer se reordenado.
**Decisão: seguir com o BGE-m3.** Os dois não são estatisticamente distinguíveis com N=75 — o
texto deve reportar o IC, não uma vitória.

**Limitação honesta:** só o `0.6B` foi testado. O `4B` previsto acima **não foi rodado** — o 0.6B
foi escolhido por ser o que cabia em uma única máquina. Portanto o resultado é sobre
*Qwen3-Embedding-0.6B*, não sobre a família Qwen3-Embedding. Declarar como limitação de escopo,
nunca como "o Qwen perdeu".

**Ressalva de comparabilidade:** o 0,2908 do BGE aqui **não** é comparável ao 0,2503 da
`eval/results/retrieval/metrics.csv`, de 31/ago — entre as duas rodadas mudaram o corpus **e** os
qrels. Atribuir a diferença à correção do chunker exigiria uma rodada controlada.

#### E2 — RESULTADO (02/set/2026)

Gemini reindexado no corpus corrigido (coleção Qdrant `LEME_gemini`, 5160 pontos, 3072 dim) e
comparado ao BGE-m3 sobre o mesmo gold set. Artefatos em `eval/results_e2/retrieval/`, gerados
por `eval/tools/dense_compare.py --systems bge_m3,gemini --out results_e2 --offline`.

| Sistema | nDCG@5 | nDCG@10 | R@5 | R@100 | MRR@10 | MAP | GeoRisk | teto@20 |
|---|---|---|---|---|---|---|---|---|
| **Gemini** | **0,4422** | 0,4921 | 0,4026 | 0,9197 | 0,6334 | 0,3896 | 0,4409 | **0,7904** |
| BGE-M3 | 0,2908 | 0,3313 | 0,2903 | 0,8066 | 0,4172 | 0,2509 | 0,3454 | 0,6789 |

δ (BGE − Gemini) = **−0,1513**, IC95% [−0,2185; −0,0848], p = 0,0001, Cliff δ = −0,263.

**Não-inferioridade REJEITADA.** O IC inteiro está abaixo de −0,03; o limite superior
(−0,0848) já viola a margem por quase 3×. Diferente da E1, aqui não há indeterminação: o gap é
real e estatisticamente sólido. **Migrar para o embedding aberto hoje custaria qualidade
mensurável.**

**O gap AUMENTOU com o corpus corrigido**, de −0,0945 (dados de 31/ago) para −0,1513. As duas
medidas não são diretamente comparáveis — mudaram corpus e qrels —, mas a direção contraria a
expectativa de que corrigir o chunker aproximaria os dois.

**Consequência para a ordem dos experimentos:** o E3 (fine-tuning) tem como gatilho "só se o gap
passar a margem". O gap passa a margem por 5×. O gatilho está acionado — mas o F3 (reranker) vem
antes, por ser mais barato e ter margem maior (+0,3881 no BGE).

### As três correções sem as quais o resultado não vale

**(a) O gold set não pode ser dado de treino.** 75 consultas com mediana de 2 trechos. Afinar
nesses pares e reavaliar nos mesmos mede memorização. **Gerar pares de treino sinteticamente do
acervo** (pergunta gerada por LLM a partir de cada trecho, negativos difíceis por BM25 e denso) e
manter as 75 **inteiramente fora do treino**, como teste cego.

**(b) N=75 provavelmente não basta.** O IC de BGE−Gemini é [−0,1636; −0,0370], largura 0,127 —
**quatro vezes** a margem de 0,03. Mesmo que o Qwen empate, o IC não fecha dentro da margem e o
veredito sai "indeterminado". **Rodar `eval/tools/power_n.py` ANTES da E2** e, se o N exigido for
maior, expandir o gold set. É decisão de cronograma a tomar antes, não depois do resultado.

> **Revisão (02/set/2026): a premissa de (b) não se sustenta — N não é o gargalo, o GAP é.**
> A não-inferioridade exige limite inferior do IC > −margem. Isso só é alcançável se o δ
> observado estiver **dentro** da margem; aí mais consultas apertam o IC até caber. Nos dois
> contrastes o δ já está fora:
>
> | Contraste | δ | sd | dentro da margem 0,03? |
> |---|---|---|---|
> | Qwen − BGE (corpus corrigido) | −0,0369 | 0,2203 | não |
> | BGE − Gemini (31/ago, corpus antigo) | −0,0945 | 0,2725 | não, por 3× |
>
> Se o δ verdadeiro for o observado, **nenhum N** estabelece não-inferioridade: mais dados
> apenas convergem o IC para um valor que falha o teste. Só para estreitar o IC a ±0,03 seriam
> ~208 consultas (E1) e ~318 (E2), contra as 75 atuais. Como o IC da E1 inclui zero, rodar power
> analysis é apostar que a estimativa atual é pessimista.
>
> **Consequência prática:** `power_n.py` não é o próximo passo. O que precisa mudar é o gap —
> ou seja, o reranker (F3) — e só depois faz sentido dimensionar N. E a E2 exige antes um número
> de Gemini no corpus **corrigido**; o único que existe é da coleção Qdrant indexada com o corpus
> antigo. Alargar a margem para fazer o veredito passar seria post-hoc e indefensável.
>
> **Atualização, depois do F3 (02/set/2026): o `power_n.py` VOLTOU a fazer sentido.** A conclusão
> acima valia enquanto o δ estava fora da margem. Com o reranker, ele entrou:
>
> | Contraste | δ | sd | dentro da margem? | N necessário |
> |---|---|---|---|---|
> | BGE **sem** reranker − Gemini | −0,1513 | 0,2959 | não | nenhum resolve |
> | BGE **+ rr50** − Gemini | −0,0045 | 0,2806 | **sim** | **466** (6,2× as 75) |
>
> Agora o gargalo é de fato o N, não o gap: o ponto estimado está confortavelmente dentro da
> margem e falta só precisão. Expandir o gold set para ~466 consultas passa a ser uma decisão de
> cronograma com número concreto — que era exatamente o que a correção (b) pedia.

**(c) Corrigir o chunker antes.** Comparar embeddings com 14,6% do índice sendo fragmentos
redundantes mede tolerância a lixo, não qualidade de embedding.

---

## 7. Ordem sugerida

| Fase | O quê | Depende de |
|---|---|---|
| **F0** | Corrigir `extract_to_jsonl.py:580` + teste de regressão (nenhuma página passa de N chunks; nenhum par sobrepõe >50%) | — |
| **F1** | Reindexar + remapear qrels das 5 consultas + rodar `finalize()` e `05_report.py` | F0 |
| **F2** | Validar com q0065 (4 níveis?) e q0075 (parou de negar?) | F1 |
| **F3** ⭐ | Reranker offline: 3 modelos × pool {20, 50, 100}, com GeoRisk | F1 |
| **F4** | Decomposição condicional + cota de diversidade, medida nas 25 comparativas e 13 mistas | F3 |
| **F5** | Detector de falha de recuperação (§4.2), validado contra o especialista | F1 |
| **F6** | RAI: atribuição por afirmação + modalidade deôntica | — (paralelo) |
| **F7** | E1 — Qwen vs BGE, com teto de reordenação | F1 |
| **F8** | E2 — melhor aberto vs Gemini, N calibrado por `power_n.py` | F7 |
| **F9** | E3/E4 — fine-tuning com pares sintéticos, gold set cego | F8 |

F5 e F6 são independentes da trilha de recuperação e podem correr em paralelo.

---

## 8. Pendências herdadas do repositório

> 🔴 **ABERTA E BLOQUEANTE (13/set/2026): a produção está fora de sincronia com o corpus.**
>
> O corpus foi reextraído com o recorte corrigido — 5.160 → 4.897 trechos, fingerprint
> `da69227b35c04c4d…`. A coleção `LEME_gemini` no Qdrant continua com **5.160 pontos, com o texto
> e os ids do recorte antigo**: produção serve o chunking velho e devolve ids que não existem mais
> no gabarito.
>
> **`eval/tools/migrar_ids_qdrant.py` NÃO resolve isto.** Ele recalcula o id a partir do texto do
> próprio payload, e esse texto é o antigo. A coleção precisa ser **reconstruída** com
> `src/build_vectorstore.py`, o que significa reembutir com a API do Gemini — custa chamada e mexe
> em produção, então é decisão da Karol, não passo automático.
>
> O cache do BGE-m3 já foi atualizado (`eval/tools/reembutir_incremental.py`): 4.661 vetores
> reaproveitados, 236 recalculados. O Qwen3 está na mesma situação do Gemini, mas não é produção.
>
> **Enquanto isto não for feito:** nada de avaliação contra o Gemini, e nada de comparar produção
> com gabarito — os ids não casam.

1. **`eval/config.yaml:171` tem o juiz errado** — `llama-3.1-8b-instant` no lugar de
   `llama-3.3-70b-versatile`, que foi o modelo da rodada real. Quem reproduzir usa o errado.
   **Ainda não corrigido.**
2. **Gabarito com política inconsistente.** q0015/q0016 foram reanotadas com critério estrito
   (2 e 5 trechos), mas 13 outras consultas factuais de documento único têm 9–13 trechos pela
   política antiga (marcar o intervalo de páginas). Uniformizar é trabalho da branch de correção.
3. **Rubrica de `correctness` é ambígua** (`eval/lib/judges.py:46`): "5 = bate com a
   resposta-referência" e "1 = factualmente errada" não são polos da mesma escala nas perguntas
   fora de escopo. Explica o κ de 0,550. **Corrigir antes de rodar juízes de novo**, senão
   herdamos o mesmo ruído.
4. **top-k de produção ≠ de avaliação:** produção usa `limit=4` (`agent/utils/tools.py:133`),
   avaliação usa `context_top_k: 5`.
5. **`paper/main.tex` está obsoleto** — versão "NIAR" de 29/07. Não editar, não publicar.
6. **`embeddings_backup.jsonl` = 390 MB em git-LFS.** Reconstruível; avaliar se precisa ir para
   repositório público.

---

## 9. Recuperação de multi-hop e comparativa — quatro experimentos (13/set/2026)

Rodados sobre o corpus reextraído (4.897 trechos, fingerprint `da69227b35c04c4d`), com BGE-m3 local
em CPU, contra as 75 perguntas respondíveis. Nenhuma chamada de API.

**Linha de base medida (BGE-m3 denso, sem reordenação):**

| Tipo | melhor âncora (mediana) | R@5 | R@20 | TODAS@5 |
|---|---|---|---|---|
| `factual` | 2 | 28,2% | 51,9% | 4/25 |
| `multi_hop` | 4 | 31,2% | 62,5% | 2/25 |
| `comparative` | 5 | 23,7% | 40,7% | 2/25 |

`TODAS@5` = as duas âncoras dentro do top-5. Para multi-hop e comparativa, achar só uma não responde
a pergunta — é a métrica que importa, e é a que está pior.

### 9.1 A consulta é o gargalo, não a ordenação ✅ sinal forte

Mesma busca, trocando **o que** se consulta. A resposta-referência é oráculo (é o gabarito), então
isto é teto, não método — mas diz a direção:

| Tipo | consulta = pergunta | consulta = pergunta + resposta |
|---|---|---|
| `factual` R@5 | 28,2% | 33,6% |
| `multi_hop` R@5 | 31,2% | **48,4%** |
| `comparative` R@5 | 23,7% | **37,3%** |
| `multi_hop` TODAS@5 | 2/25 | **5/25** |
| `comparative` TODAS@5 | 2/25 | **4/25** |

**Leitura:** para multi-hop e comparativa, a pergunta sozinha é uma consulta ruim. Aproximar a
consulta do texto da resposta sobe R@5 em ~55% relativo. No factual quase não muda — lá a pergunta
já acerta. Isso aponta para **expansão de consulta (HyDE)**: gerar uma resposta hipotética e buscar
com ela. **Ainda não testado com hipótese gerada** — só com o oráculo.

> A lacuna é em parte deliberada: o prompt de geração manda *"a pergunta NÃO deve copiar o
> vocabulário exato da norma"*, porque quem pergunta não conhece o texto legal. O benchmark está
> certo em ser difícil; o conserto tem de ser no sistema, não na pergunta.

### 9.2 Cota de diversidade por documento ❌ refutado

Teto de N trechos do mesmo documento no top-5, sobre o top-200 do denso:

| teto | `factual` R@5 | `multi_hop` R@5 | `multi_hop` TODAS@5 | `comparative` R@5 |
|---|---|---|---|---|
| sem teto (hoje) | 28,2% | 31,2% | 2/25 | 23,7% |
| máx. 3 por doc | 24,4% | 23,4% | 1/25 | 25,4% |
| máx. 2 por doc | 19,8% | **12,5%** | **0/25** | 22,0% |
| máx. 1 por doc | 12,2% | 7,8% | 0/25 | 16,9% |

**Por que falha:** âncoras por pergunta em documentos distintos — `factual` 1, `multi_hop` **1**,
`comparative` 2. A cota foi desenhada para a comparativa e aplicada a todos; no multi-hop ela expulsa
justamente a segunda âncora, que está no mesmo documento da primeira. E na comparativa mal move.

Isto refuta o item 2 da mitigação B (§3).

### 9.3 Realimentação de pseudo-relevância (Rocchio denso) ❌ refutado

`q' = q + β·média(top-k)`, renormalizado. Doze configurações (k ∈ {3,5,10} × β ∈ {0,3; 0,5; 0,7; 1,0}).
Todas planas ou piores. A comparativa é a mais prejudicada: mediana da melhor âncora de **5 para
8–18**, R@20 de 40,7% para 25–37%.

**Por que falha:** o centroide do top-k é dominado pelo lado que já ganhava. PRF então **amplifica** o
monopólio em vez de corrigi-lo — é o oposto do que a comparativa precisa.

### 9.4 Por que o top-5 é monopolizado por um documento

Medido: o documento mais frequente ocupa **mediana 5 das 5 vagas**, nos três tipos.

Causa, sobre 10 documentos com ≥ 20 trechos:

| | dentro do documento | entre documentos | separação |
|---|---|---|---|
| **com** prefixo `[título · emissor · seção]` | 0,758 | 0,495 | **+0,263** |
| **sem** prefixo (só o texto) | 0,669 | 0,491 | +0,178 |

O prefixo contextual responde por **+0,085** — cerca de um terço do excesso de coesão. Os outros dois
terços são intrínsecos: trechos de uma mesma norma repetem vocabulário, numeração e estilo.

**Conclusão: não perseguir o monopólio.** Ele *ajuda* factual e multi-hop, cujas âncoras estão no
mesmo documento; só a comparativa sofre, e as duas tentativas de quebrá-lo (§9.2 e §9.3) pioraram o
conjunto. Tirar o prefixo eliminaria um terço do efeito e custaria o contexto que faz o documento
certo ser encontrado. O monopólio é sintoma; a doença é a consulta (§9.1).

### 9.5 Multiplexar a consulta ❌ refutado (16/set/2026)

A tabela do §1.1 mostrava a mediana da âncora *dentro do documento* caindo de 8
para 3 ao combinar pergunta + tradução EN + as 2 sub-perguntas. Medido no que
decide — top-5, BGE denso sem reordenação, as duas formas de contar:

| Consulta | `multi_hop` ≥1 · as duas | `comparative` ≥1 · as duas |
|---|---|---|
| **base — só a pergunta (PT)** | **14/25 · 2/25** | **13/25 · 2/25** |
| média(q, en) | 16/25 · 2/25 | 13/25 · 2/25 |
| média(q, s1, s2) | 10/25 · 2/25 | 12/25 · 2/25 |
| média(q, en, s1, s2) | 12/25 · **3/25** | 13/25 · 2/25 |
| maxsim(q, en) | 14/25 · 1/25 | **9/25 · 0/25** |
| maxsim(q, en, s1, s2) | 16/25 · 1/25 | 12/25 · 1/25 |
| RRF(q, en, s1, s2) | 9/25 · **0/25** | 12/25 · 2/25 |

`≥1` = perguntas com pelo menos um trecho necessário no top-5.
`as duas` = perguntas com **todos** os trechos necessários — a régua que decide.

**Nenhuma variante move a comparativa.** No multi-hop há troca, não ganho: ou
sobe o `≥1` e o `as duas` fica parado (média q+en), ou sobe o `as duas` em uma
pergunta e o `≥1` cai de 14 para 12 (média das quatro). Três das seis fusões
**pioram**; o `maxsim(q,en)` derruba a comparativa de 2/25 para zero.

**Por que a medida antiga enganava:** posição dentro do documento é condicional a
já se estar no documento certo. A pergunta exige as duas âncoras no top-5
**global**, onde elas competem com 4.897 trechos. Melhorar a ordem interna não
resolve a competição externa.

> Mesmo erro do §1.1, em outra roupa: medir uma coisa e concluir sobre outra.
> Quando um número melhorar, perguntar **de qual régua ele é**.

**Procedência:** a tabela do §1.1 não tem script no repositório — varredura de
16/set não achou nenhum arquivo de multiplexação em `eval/`. Os caches são usados
só pelo braço `A_bm25_mt` (BM25 com consulta traduzida, CLIR legítimo, outra
coisa). Portanto a medição de 16/set é a **primeira** avaliação disso nas réguas
que decidem. Script: `scratchpad/exp_multiplex.py`; caches reusados, zero chamada
de LLM.

### 9.6 O que fazer a seguir

1. **Testar HyDE com hipótese gerada** — é a única direção com sinal. Medir se o ganho do oráculo
   (§9.1) sobrevive a uma hipótese real. Aplicar a mesma expansão aos dois braços (BGE e Gemini)
   para não contaminar a comparação da dissertação; declarar no artigo.
2. Não repetir cota de documento, PRF nem multiplexação de consulta (§9.2, §9.3, §9.5).
3. Os rankings em `eval/results/retrieval/rankings/` são **anteriores à reextração** e apontam para
   ids no formato posicional (`_p4_c2`) que não existem mais. Precisam ser regerados antes da Fase 2.
4. `rank_bm25` é importado por `eval/lib/retrievers.py` e **não está no `requirements.txt`** —
   dependência não declarada; o Cenário A não roda sem ela.

---

## 10. Fontes canônicas

| O que | Onde |
|---|---|
| **Plano de trabalho do benchmark 500** | `benchmark_leme_500.md` — fases, responsáveis, comandos |
| **Auditoria do acervo (Fase 0)** | `eval/tools/auditar_acervo.py` → `docs/auditoria_acervo.csv` |
| **Geração do gabarito em lotes (Fase 1)** | `eval/tools/lotes.py` — `sortear` e `integrar` |
| **Julgamento de relevância e calibração (Fase 2)** | `eval/tools/julgar_relevancia.py` — `pool`, `julgar`, `calibrar` |
| **Reancoragem do gabarito após recorte novo** | `eval/tools/reancorar_por_texto.py` — casa por texto, não por id |
| **Reembutir só o que mudou** | `eval/tools/reembutir_incremental.py` |
| **Experimentos que não deram resultado** | `eval/tools/experimentos_descartados/README.md` |
| Métricas de recuperação | `eval/results/retrieval/metrics.csv` · consolidado em `eval/results/report.md` |
| Teto de reordenação | `eval/tools/rerank_ceiling.py` (roda offline, sem API) |
| Correção do gabarito | `eval/tools/corrige_qrels_q0015_q0016.py` |
| Rankings (top-100 × 7 sistemas × 75 consultas) | `eval/results/retrieval/rankings.json` |
| Notas dos juízes + contexto recuperado | `eval/results/generation/judged_*.jsonl` |
| Concordância entre juízes | `eval/results/agreement/agreement.csv` |
| **Revisão do especialista (PREENCHIDA)** | `Documentos/LEME/revisao_juizes_especialista_PREENCHIDA.csv` — movida de `~/Downloads` em 27/08/2026. **As cópias em `eval/data/` são template em branco.** |
| Curadoria do gold set | `eval/data/golden_qa_final.csv`, coluna `curadoria` |
| Ancoragem de citações | `eval/tools/citation_grounding.py` → `eval/results/generation/citation_grounding.csv` |
| Acervo | `corpus_manifest.csv`, `corpus_documentos.csv` (63 docs) |
| Trechos | `data/processed/documents.jsonl` (5.791 = 4.780 PDF + 1.011 HTML) |
| Config de produção | `agent/utils/tools.py` (`limit=4`, `score_threshold=0.60`, `qwen/qwen3.6-27b`) |

**Reexecutar sem gastar API:** `finalize()` em `eval/retrieval_eval/_common.py:121` repontua a
partir dos rankings já em disco. **Não** rode `02_retrieval_eval.py` só para atualizar métricas —
ele re-executa os quatro cenários e chama Gemini/Qdrant.

```bash
cd eval && ../.venv/Scripts/python.exe -c "import sys; sys.path.insert(0,'retrieval_eval'); from _common import load_config, setup_io, finalize; cfg=load_config('config.yaml'); setup_io(); finalize(cfg)"
```
