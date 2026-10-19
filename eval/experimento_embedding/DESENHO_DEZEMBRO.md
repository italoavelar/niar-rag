# Artigo de dezembro (JBCS) — desenho do experimento

> Complementa `benchmark_leme_500.md`, que é o plano do artefato do KDD (fev/2027).
> Este documento é só o artigo de dezembro, e o escopo é deliberadamente menor:
> o recorte está fechado, o corpus está congelado, e a única pergunta aberta é
> **qual empilhamento de recuperação**.

## A tese

**O empilhamento aberto alcança o proprietário.** BGE-m3 mais
`BAAI/bge-reranker-v2-m3` — os dois abertos, os dois locais, zero de API —
alcançam `gemini-embedding-001`, que é pago e fechado.

Se a tese se sustentar, a consequência prática é que o LEME pode rodar sem
depender de API paga e sem enviar consulta de usuário para fora. Isso é argumento
de soberania de dado em sistema de saúde, não só de custo — e é o que faz a tese
valer um artigo em vez de uma nota técnica.

## O que já está decidido e não se remede aqui

| | valor | de onde vem |
|---|---|---|
| recorte | 1.200 caracteres, sobreposição 240 | experimento fechado em 22/09 |
| corpus | 54 documentos, 6.134 trechos | Fase 0 concluída |
| perguntas | acervo de 392; subconjunto de 300, 75 por tipo, semente 2026 | `dados/acervo.json` |
| qrels | 520 linhas, 267 perguntas, só grau 2 | derivado das citações por `gerar_qrels.py` |

**Os braços rodam sobre as 392, não sobre o subconjunto.** Custa 92 consultas a
mais e torna a recuperação imune à composição do subconjunto — que já mudou uma
vez, quando a correção de tipo por dispersão remexeu o multi_hop. O subconjunto
vira filtro na análise.

O tamanho de recorte **não é variável deste artigo**. Ele foi medido, T1200
venceu T400 em 6 de 6, e reabrir isso aqui misturaria dois artigos. O que varia é
o embedding e o que vem depois dele.

## Os braços

| # | braço | aberto? | custo por consulta | papel |
|---|---|---|---|---|
| 1 | BGE-m3 denso | sim | CPU | o desafiante |
| 2 | gemini-embedding-001 denso | **não** | API paga | o titular, como está em produção |
| 3 | **BGE-m3 + reranker** | sim | CPU | **a tese** |

**BM25 e fusão RRF saíram do artigo** (27/09). O artigo é sobre aberto contra
proprietário; um terceiro paradigma só divide a atenção. As corridas ficam em
disco caso um revisor peça a linha de base léxica — `A_bm25` e `C_fusion`, as
duas com 392 consultas.

Os dois números que justificam a saída, e que valem uma nota de rodapé se
alguém perguntar: o BM25 sem tradução tira **0,000 de 68** no estrato em inglês,
e a fusão RRF(BM25 + Gemini) **piora** o Gemini de 0,493 para 0,360 (38
vitórias a 8, p < 0,0001) — dar peso igual a um braço cego em um terço das
perguntas não funde, contamina. O `config.yaml` chama esse cenário de "fusão de
deploy"; hoje ela seria um rebaixamento.

**Por que o Gemini NÃO leva reranker** (decidido em 27/09). A comparação não é
entre dois modelos em condições iguais; é contra **o sistema que está em produção
hoje**, que não tem reranker. A objeção "por que só um lado ganhou o reforço"
some quando o artigo declara essa moldura: a linha de base é a configuração
implantada. Isso precisa estar escrito, não subentendido.

**Por que o BM25 vai sem tradução** (mesma data). A ablação CLIR já saiu no
artigo anterior e produção usa BM25 cru. O preço é uma frase que o artigo **não
pode** escrever: "o denso supera o léxico". O que os dados sustentam é "supera a
linha de base léxica implantada" — sem tradução, o BM25 fica com Junta@5 de
**0,014** no estrato EN, e o revisor diria, com razão, que ele perdeu por não
falar inglês, não por ser léxico.

## O ponto que decide a medição

**Reordenar um pool não acrescenta documento nenhum.** Está escrito no
`rerank_eval.py` e é aritmética, não achismo: se o pool tem 100 trechos,
`recall@100` depois do reranker é *idêntico* ao de antes. O reranker só mexe em
ordem.

Consequência dura para a tese:

- **Anunciar a tese em `recall@100` é impossível** — lá o reranker é
  matematicamente incapaz de mudar nada, e o artigo diria "não houve ganho" sobre
  uma métrica que não podia ter ganho.
- **A tese se mede em k pequeno**: `Junta@5`, `nDCG@5`, `MRR`. É onde o reranker
  puxa para o top-5 o trecho que o denso deixou na posição 20.

E uma simplificação que o experimento de recorte nos deu de graça: lá, comparar
em k fixo era inválido, porque T400 devolvia 1.600 caracteres no top-5 e T3500
devolvia 14.750. **Aqui todos os braços devolvem trechos do mesmo corpus
1.200/240**, então k fixo já é orçamento fixo. O confundimento que quase inverteu
a decisão de recorte não se aplica a este artigo — e vale dizer isso no texto,
porque é a pergunta natural de quem leu o outro experimento.

## Métricas

**Primária: `Junta@5`** — a pergunta só conta como acerto se **todas** as suas
citações vierem no top-5. É a métrica de completude de evidência: com "≥1 trecho"
a multi-hop pulou de 12% para 84% sem o sistema ter melhorado nada.

Secundárias: `Recall@5` e `Recall@10` (≥1 citação), `nDCG@5`, `MRR`, `P@5`.

**GeoRisk, obrigatório.** A média esconde regressão: um reranker pode subir o
agregado e piorar consultas que já iam bem. Sem GeoRisk o artigo afirma melhora
onde houve troca.

**Recusa**, nas 75 irrespondíveis — elas não têm trecho relevante, mas são
recuperadas de propósito: o teste é se o gerador recusa **tendo material
plausível em mãos**. As colunas de métrica de recuperação ficam vazias nelas, não
zero: nDCG sem documento relevante é indefinido.

## Quando a incompletude do qrels morde, e quando não

| métrica | o pool da Fase 2 muda o resultado? |
|---|---|
| `Junta@k`, `Recall@k` | **não** — se a pergunta precisa do trecho X e o braço trouxe X, é acerto, seja o que for julgado |
| `nDCG`, `MAP`, `P@5` | **sim** — não julgado conta zero, e o braço que acha um trecho bom não citado é punido |

Medição de 27/09 com `medir_melhor_trecho.py`: em **6,5%** das perguntas
respondíveis existe, no documento certo e sem anotação, um trecho com sinal mais
forte que o anotado — contra **48% (36 de 75)** medidos em 13/09. Nas famílias em
que a âncora foi escolhida por leitura e não por pontuação, fica entre 4,4% e
12%.

Isso **não** dispensa o pool para nDCG; reduz a urgência. A leitura honesta é:
liderar com `Junta@5`, que é pool-independente, e publicar nDCG só depois do pool
da Fase 2, ou com a incompletude declarada e quantificada.

## Estatística

- `Junta@k` é binário e pareado → **McNemar exato**.
- `nDCG` e `MRR` → **randomização pareada** mais IC bootstrap, com **Cliff's
  delta** de tamanho de efeito. Tudo em `eval/lib/stats.py`.
- Correção para testes múltiplos: **Benjamini-Hochberg em famílias separadas** —
  "aberto vs proprietário", "com vs sem reranker", "lexical vs denso" — com
  **Bonferroni reportado ao lado**. Bonferroni é conservadora demais aqui porque
  os braços são correlacionados (mesmas perguntas, mesmo corpus), mas trocar de
  correção depois de a primeira falhar é exatamente o que um revisor cobra;
  reportar as duas resolve.

## As três leituras que o artigo precisa ter

1. **Aberto contra proprietário** — a tese.
2. **PT contra EN.** A pergunta é sempre em português e 70 das 300 apontam para
   documento em inglês. É onde o BM25 deve sofrer e onde o denso multilíngue
   justifica o custo. O braço 2 (tradução PT→EN) existe para que o BM25 apareça
   na sua melhor versão, não como espantalho.
3. **Por tipo.** A comparativa é a única que obriga **dois documentos** no mesmo
   top-k — o ranking pode se entupir com cinco trechos ótimos de um lado só.

## Limitação a declarar, e que também é uma leitura

**286 das 377 perguntas do acervo foram escritas por LLM em sessão**, e o campo
`origem` registra a procedência de cada uma. Isso precisa estar no artigo, e a
forma de tratá-lo não é pedir desculpa: é **cortar o resultado por `origem`**. Se
o ranking dos braços for o mesmo nas famílias escritas por humano e nas escritas
por LLM, a procedência não contaminou a conclusão, e isso é um resultado
reportável. Se divergir, é achado — e aí a conclusão vale para o subconjunto
humano.

As outras duas, já registradas: `difficulty` não é comparável entre famílias, e o
acervo somado é enviesado para janela 1.200, o que o impede de comparar tamanhos
de recorte (não é problema aqui, porque o recorte é fixo).

## Resultado (28/09/2026)

225 perguntas respondíveis, k=5. Números em `resultados/bracos_dezembro.json`.

| braço | Junta@5 | Compl@5 | Recall@5 | nDCG@5 | MRR@10 | CobDoc |
|---|---|---|---|---|---|---|
| **BGE+reranker** | **0,538** | 0,679 | 0,822 | 0,593 | 0,667 | 0,662 |
| Gemini | 0,489 | 0,645 | 0,809 | 0,546 | 0,621 | 0,636 |
| BGE-m3 | 0,387 | 0,547 | 0,716 | 0,442 | 0,517 | 0,573 |

| comparação | vit | der | p | Bonferroni |
|---|---|---|---|---|
| BGE → BGE+reranker | 3 | 37 | <0,0001 | <0,0001 |
| BGE vs Gemini | 9 | 32 | 0,0004 | 0,0013 |
| Gemini vs BGE+reranker | 13 | 24 | **0,0989** | 0,2966 |

**A tese se sustenta na forma "alcança", não "supera".** p = 0,0989 é
indistinguibilidade; o ponto estimado favorece o aberto em 4,9 pontos. Um
empilhamento inteiramente aberto, local, com zero de API, fica estatisticamente
indistinguível do proprietário em produção.

Por dispersão, onde o reranker agiu: `d=2..20` foi de **0,222 para 0,593**.
`recall@100` idêntico nos quatro pools — o ganho é de ordem, como tinha de ser.

**Limitações medidas, a declarar:**
- Em inglês o proprietário ainda ganha (0,588 contra 0,544); em português o
  aberto ganha (0,766 contra 0,670). A vantagem residual é cross-lingual.
- 38 consultas pioraram contra 131 que melhoraram. GeoRisk subiu de 0,4267 para
  0,5354, então o saldo é positivo ajustado por risco — mas as 38 vão no texto.
- Comparativa continua o buraco: os dois lados em 13 de 75. Teto de reordenação
  0,640 — 27 das 75 não têm o material no pool, e nenhuma reordenação as alcança.

## Auditoria do gabarito das comparativas (29/09) — teste de robustez

As 25 comparativas do piloto foram relidas uma a uma. **16 tinham defeito:**

| defeito | nº | exemplo |
|---|---|---|
| estrutural: a pergunta nomeia duas normas, o gabarito cita uma | 4 | q0093 afirmava provisões da OMS sem nenhuma citação da OMS |
| a citação não sustenta a resposta | 8 | q0092 citava *"a seção seguinte descreve as informações que os patrocinadores devem fornecer"* para sustentar "sistema de qualidade" |
| limítrofe: frase adjacente à que sustenta | 3 | q0097 citava "ao avaliar o nível de segurança"; a lista está na frase anterior |
| citação intrusa | 1 | q0086 tinha uma citação que é evidência de q0094 |

Reparadas por `corrigir_comparativas_piloto.py`, que roda depois do conversor e é
idempotente. **Efeito no resultado: nenhum.** Geral 0,542 → 0,538, comparativa
0,187 → 0,173, McNemar idêntico casa por casa.

Isso vale como seção do artigo: 16 perguntas corrigidas, quatro delas
estruturalmente inválidas, e a conclusão não se moveu. Note que o reparo tornou
o gabarito **mais exigente** — as 4 estruturais eram artificialmente fáceis,
porque cobravam um documento só.

Dois dos defeitos são detectáveis por script e viraram checagem: comparativa com
citações de um documento só (4 das 25 do piloto; **zero** das 50 geradas), e
citação ausente do documento declarado.

## Diversificação do ranking: resultado negativo

Testado em 28/09 como teto por documento (1, 2 e 3 trechos) sobre os três
braços. **Piora em todos os tetos e todos os braços**: no BGE+reranker, 0,542 →
0,471 com teto 2. O MMR, varrido antes, dá a mesma troca (λ=0,7: comparativa
0,233 → 0,255, geral 0,438 → 0,427), e o teto por documento já havia sido
refutado em 13/09.

**A causa é conflito de requisito, não ajuste de parâmetro.** A comparativa quer
dois documentos no top-5; a multi-hop quer duas regiões do MESMO documento. Um
teto torna a segunda impossível por construção. Contagem no teto 2: comparativa
ganha 7 e perde 2, factual perde 6, multi-hop perde 15.

Vale como seção do artigo: explica POR QUE a comparativa é difícil.

## Execução

- [x] Corpus congelado — 54 documentos, 6.134 trechos
- [x] Acervo e subconjunto de 300 regerados; `documentos` corrigido nas 21 comparativas do piloto
- [x] `qrels.csv` derivado — 441 linhas, 252 perguntas, só grau 2
- [x] `golden_dezembro.jsonl` + `queries_dezembro.csv` exportados; `config.yaml` apontado
- [x] `qdrant_collection` corrigido para `leme_gemini` (estava `LEME_gemini`, e o Qdrant diferencia caixa)
- [ ] Índice BGE completado para o corpus de 54 documentos
- [ ] Braços 1 a 4 rodados sobre as 300
- [ ] Braços 5 e 6 (reranker) sobre os pools dos braços 3 e 4
- [ ] Braço 7 (fusão) sobre o melhor denso
- [ ] Significância, GeoRisk e as três leituras
- [ ] Pool da Fase 2 — necessário só se o artigo publicar nDCG
