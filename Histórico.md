# Histórico do LEME — o que mudou desde o artigo do WFA

> **Para a equipe.** Este documento conta, em ordem, o que foi feito no LEME desde a submissão do
> artigo do WFA (14/08/2026): o que quebrou, o que foi consertado, o que testamos e não deu certo, e
> o que ainda falta testar.
>
> **Os três documentos e para que serve cada um:**
>
> | Documento | Serve para |
> |---|---|
> | **`Histórico.md`** (este) | entender o que já aconteceu e por quê |
> | `benchmark_leme_500.md` | saber **o que fazer agora** — fases, responsáveis, comandos |
> | `Memoria.md` | o registro técnico completo, com fonte de cada número |
>
> Os números aqui vêm do `Memoria.md`, que por sua vez cita arquivo e linha. **Para citar em artigo
> ou slide, vá à fonte** — este documento é para entender, não para copiar número.

## Como tudo é medido — leia antes de qualquer número

Sem isto, dois números deste documento podem parecer contraditórios sendo os dois corretos.

### Os dois graus dos qrels, e por que a diferença é enorme

| Grau | O que é | Quem decidiu |
|---|---|---|
| **2** | o trecho **de onde a pergunta nasceu** — o gerador leu ele para escrever a pergunta | o gerador, e a revisão humana |
| **1** | vizinho **posicional**: todos os trechos das páginas ±1 do trecho-fonte | **ninguém** — é auto-adicionado por `neighbors_same_doc(window=1)` |

O grau 1 não é julgamento de relevância, é heurística de paginação. E ele é adicionado **só nas
factuais**:

| tipo | âncoras por pergunta (grau≥1) | só grau 2 | total de grau 1 |
|---|---|---|---|
| `factual` | **5** | **1** | 100 |
| `multi_hop` | 2 | 2 | 5 |
| `comparative` | 2 | 2 | 10 |

**Toda métrica conjuntiva usa `min_grade=2`.** O `complete_evidence_at_k`
(`eval/lib/metrics.py:142`) já faz isso por padrão; script que leia `g['qrels']` direto pega os dois
graus e fica errado. Tamanho do estrago:

| tipo | Joint Recall@5 exigindo grau≥1 | exigindo grau 2 |
|---|---|---|
| `factual` | 4/25 | **17/25** |
| `multi_hop` | 2/25 | 2/25 |
| `comparative` | 2/25 | 2/25 |

**11 das 25 factuais exigem mais de 5 âncoras de grau≥1** — uma delas exige 11. Num top-5 elas são
impossíveis por construção, e o "4/25" media defeito de anotação, não recuperação. Multi-hop e
comparativa são imunes porque quase não têm grau 1.

> ⚠️ **Números de factual em relatórios anteriores a 16/09 precisam ser reconferidos** — se foram
> calculados sobre todos os graus, estão subestimados.

### As duas formas de contar, para multi-hop e comparativa

| Forma | O que conta | Para que serve |
|---|---|---|
| **≥1** | perguntas que trazem **pelo menos um** trecho-fonte | mede se a busca *chegou perto* |
| **Joint Recall@k** | perguntas que trazem **todos** os trechos-fonte | mede se a pergunta **foi respondida** |

A segunda é a que decide. Metade da evidência não responde a pergunta. Foi confundir as duas que
produziu o *"multi-hop resolvida em 84%"* — era o "≥1"; pela régua que decide eram 3 de 25.

**"Joint Recall@k" é o nome da literatura** para o que o código chama de `complete_evidence_at_k`.
Adotamos o nome, não uma métrica nova — é o termo que um revisor do KDD reconhece. Vantagem extra:
como só verifica se as âncoras **conhecidas** estão no conjunto, é robusta ao julgamento incompleto
dos qrels, ao contrário do nDCG.

### O denominador

Quando um número aparece como `x/50`, são **as 50 perguntas multi-evidência** (25 multi-hop + 25
comparativas). Quando aparece como `x/25`, é um tipo só. As factuais nunca entram nas tabelas de
multi-evidência.

### O tamanho do contexto

Comparação entre estratégias só vale **no mesmo tamanho de contexto**. Uma união de top-5 + top-5
entrega até 10 trechos; compará-la com o top-5 de uma consulta única é trapaça. Por isso todas as
tabelas de comparação trazem a coluna "trechos".

---

## Onde estávamos

O artigo do WFA foi submetido em **14/08/2026** e está fechado. A partir daí a restrição "não quebrar
o artigo" caducou.

A tese da dissertação é uma só: **o embedding aberto (BGE-m3) pode substituir o proprietário
(gemini-embedding-001) que está em produção.** Tudo abaixo existe para responder isso com honestidade.

### ⚠️ São DOIS artigos, com prazos e escopos diferentes — não confundir

| | **JBCS** | **KDD 2027** |
|---|---|---|
| prazo | **dezembro/2026** | ciclo 2, **fevereiro/2027** |
| trilha | artigo de pesquisa | **Datasets & Benchmarks** |
| o que é | a pesquisa: o que medimos, o que descobrimos sobre medir recuperação de evidência distribuída | **o artefato**: o benchmark de 500 perguntas com julgamento de pool, publicado para outros usarem |
| depende de | o que já está medido | Fases 0, 1 e 2 concluídas |

O `benchmark_leme_500.md` é o plano do **artefato do KDD**. O JBCS de dezembro é outro texto, e o que
o sustenta está principalmente nas Partes 1 a 4 deste documento — que já estão medidas.

**Consequência prática de prazo:** o KDD tem fevereiro, o JBCS tem dezembro. Experimento novo de
arquitetura ou de chunking não cabe em dezembro e não precisa caber — ele pertence ao segundo artigo,
medido contra a régua que o primeiro constrói.

---

## Parte 1 — O que estava quebrado, e foi consertado

### 1.1 O chunker partia páginas em pedaços inúteis

**O problema.** Nove páginas do acervo viraram 699 trechos — pedaços de poucas dezenas de caracteres,
sem conteúdo utilizável. Corrigido pelo Ítalo em 22/08: as mesmas nove páginas passaram a gerar
**33 trechos**.

### 1.2 Os identificadores de trecho mudavam quando o corpus era reprocessado

**O problema, e ele era grave.** O id de um trecho era a posição dele: `documento_p4_c2` = página 4,
trecho 2. Quando o corpus era reprocessado, o trecho da posição 2 passava a ser outro texto — **o id
continuava existindo, apontando para outra coisa**. A validação não pegava, porque ela conferia se o
id *existe*, não se ele aponta para o mesmo texto.

Isso já tinha acontecido: o re-chunking de 01/09 deslocou o texto anotado em **cerca de 40% das
perguntas**, silenciosamente.

**O conserto.** O id passou a ser derivado do **conteúdo**: `documento_<hash do texto>`. Agora:

- remover um documento **não toca** nos ids dos outros;
- mudar o recorte muda o id **apenas** dos trechos cujo texto mudou;
- se o texto mudar, o id muda — e a anotação aponta para **nada**, falha barulhenta, em vez de
  apontar para outro texto, falha silenciosa.

### 1.3 Multi-hop e comparativa estavam sendo medidas com réguas diferentes

**O problema.** Multi-hop vinha sendo reportada por "**pelo menos 1** trecho relevante no top-5" e
comparativa por "**os dois** documentos". Réguas diferentes, comparação inválida.

Sob a mesma régua estrita — todas as âncoras exigidas presentes no top-5:

| Tipo | régua frouxa (≥1 trecho) | **régua estrita (todas)** |
|---|---|---|
| `factual` | 24/25 | **20/25** |
| `multi_hop` | 21/25 | **3/25** |
| `comparative` | 13/25 | **2/25** |

**"Multi-hop resolvida em 84%" era ilusão.** Os trechos de multi-hop estão todos no mesmo documento,
então "cobrir os documentos" era satisfeito por um acerto só. Sob a régua certa: **3 de 25**.

**A consequência é boa para a tese:** sob o critério estrito o BGE+reranker **iguala ou supera** o
Gemini nos três tipos (20 vs 19 · 3 vs 4 · 2 vs 2). A tese de substituição fica mais defensável.

### 1.4 O recorte partia a resposta ao meio — o caso q0075

**O caso.** A pergunta era sobre como a UNESCO classifica a gravidade dos impactos. A resposta certa
são quatro níveis. O recorte produziu **três trechos**: a frase que nomeia os quatro níveis isolada
em 242 caracteres, e a tabela que os descreve **partida em duas**.

Nenhum dos três chegava ao top-5. O sistema respondeu que *"a UNESCO não estabelece classificação de
gravidade"* — negativa falsa. O especialista deu **nota 1, marcando como alucinação**.

Não era alucinação. Era o recorte.

**O conserto (13/09).** Piso de fragmento de 120 → **300** caracteres, e teto próprio para tabela
(**2.500**), separado do teto da prosa. Resultado medido:

| | antes | depois |
|---|---|---|
| trechos no corpus | 5.160 | **4.897** |
| trechos abaixo de 300 caracteres | 219 | **0** |
| páginas com tabela partida | 34 | **21** |

A página 45 da UNESCO virou **um único trecho de 2.012 caracteres** com a frase e a tabela inteira.
As quatro perguntas que dependiam dela agora ancoram nele.

### 1.5 A anotação sobreviveu à mudança de recorte, sem ninguém reanotar

Mudar o recorte muda ids. A anotação morreria junto — só que o gabarito guarda `qrels_text`, o
**texto** de cada âncora. Quem tem o texto reancora por conteúdo:

| Das 256 âncoras das 75 perguntas respondíveis | |
|---|---|
| id inalterado | **247** |
| remapeadas por casamento de texto | **9 → 7** (duas fusões) |
| ambíguas | 0 |
| **perdidas** | **0** |

**Nenhuma reanotação humana.** Duas fusões legítimas ficaram para conferência à mão (`q0012`, `q0015`).

> ⚠️ **Defeito separado, achado no caminho: o grau 1 das factuais.** O
> `build_gold_manual.py` auto-adiciona grau 1 a todos os trechos das páginas vizinhas
> (`neighbors_same_doc`, janela 1) — **só nas factuais**. Resultado: a factual tem **mediana de 5
> âncoras, sendo 1 de grau 2**; multi-hop e comparativa têm 2 e 2. Esses vizinhos nunca foram
> julgados por ninguém: são heurística de paginação, não relevância.
>
> Toda métrica conjuntiva precisa usar `min_grade=2` — o `complete_evidence_at_k`
> (`eval/lib/metrics.py:142`) já faz isso por padrão. Nas multi-hop e comparativas a diferença é
> pequena (5 e 10 âncoras de grau 1 no total); **na factual ela é enorme** e qualquer número
> conjuntivo de factual calculado sobre todos os graus está errado.

---

## Parte 2 — O que testamos e funcionou

### 2.1 Qwen3-Embedding-0.6B vs BGE-m3 → fica o BGE

| Sistema | nDCG@5 | teto@20 | R@100 |
|---|---|---|---|
| **BGE-M3** | **0,2908** | **0,6789** | 0,8066 |
| Qwen3-0.6B | 0,2539 | 0,6097 | 0,7493 |

δ = −0,0369, IC95% [−0,0877; +0,0132], p = 0,15.

O experimento existia para achar o caso *"nDCG pior mas teto maior"*, que faria do Qwen um bom
candidato a reranker. **Não ocorreu** — o BGE domina em todos os pools. Os dois não são
estatisticamente distinguíveis com N=75, então o texto reporta o intervalo, não uma vitória.

> **Limitação honesta:** só o `0.6B` foi testado; o `4B` não rodou. O resultado é sobre
> *Qwen3-Embedding-0.6B*, **nunca** "o Qwen perdeu".

### 2.2 O orçamento de contexto — metade do ganho já aplicada

Perguntas com **todos** os trechos no top-k (BGE + reranker):

| k | `factual` | `multi_hop` | `comparative` |
|---|---|---|---|
| 4 — o que a produção usava | 17/25 | 3/25 | 1/25 |
| **5 — o que a produção usa hoje** | **20/25** | **3/25** | **2/25** |
| 8 | 21/25 | 7/25 | 4/25 |
| 20 | 22/25 | **14/25** | 5/25 |

**Multi-hop é limitada por orçamento** (3 → 14 abrindo k); comparativa não (2 → 5).

✅ **Aplicado em 16/09:** a produção passou de `limit=4` para `limit=5` (`agent/utils/tools.py:133`,
commit `1090b7b`). Isso já vale +3 perguntas no factual e +1 na comparativa.

**Ainda na mesa:** subir para 8 levaria multi-hop de 3/25 para 7/25 e comparativa de 2/25 para 4/25 —
mais que o dobro em ambos. O custo é contexto maior no prompt do gerador, que precisa ser medido
contra o risco de diluir a resposta.

*(O item que ficava aqui — multiplexar a consulta — foi testado em 16/09 e não se sustentou.
Ver 3.4.)*

---

## Parte 3 — O que testamos e NÃO funcionou

> Esta parte vale tanto quanto a anterior. O valor de um experimento negativo é impedir que alguém
> gaste a mesma semana refazendo o mesmo caminho. Cada item abaixo já foi tentado — **não tente de
> novo sem um motivo novo.**

> ⚠️ **Cuidado: estes experimentos foram reportados em réguas diferentes**, e comparar número de um
> com número de outro é erro. Cada linha abaixo diz qual régua usou. São três:
>
> | Régua | Significa | Base (comparativas) |
> |---|---|---|
> | **A — os dois trechos** | as duas âncoras no top-5 | **2/25** |
> | **B — os dois documentos** | cada documento exigido representado por *algum* trecho relevante no top-5 | **13/25** |
> | **C — posição da âncora** | onde a âncora cai *dentro do próprio documento* | mediana **8** |
>
> A régua **A** é a que decide se a pergunta foi respondida. A **B** é mais frouxa: ter um trecho
> qualquer do documento certo não é ter o trecho que responde. A **C** é diagnóstica — condicional a
> já se estar no documento certo, não diz nada sobre o top-5 global.

| O que foi tentado | Régua | Resultado |
|---|---|---|
| Reordenação por cross-encoder | **A** | fica em **2–6/25** |
| MMR (varredura de λ) | **A** | não move |
| Reserva de vaga por sub-consulta | — (quantas seleções mudaram) | **no-op** — 22 das 25 idênticas à base |
| Partição 3+2 por lado *(ver abaixo)* | **B** | **piorou: 13/25 → 11/25** |
| Roteamento por documento, com roteador-oráculo | **A** | teto de **6/25** |
| Decomposição + fusão RRF | recall@100, depois nDCG@5 | recall **+5,7pp**; nDCG@5 não muda (p = 0,62) |
| `section_path` no prefixo do embedding | **C** | piorou: mediana **8** com a seção, **6** sem |
| Cortar por unidade normativa (artigo, seção) | — (viabilidade) | **inviável** — só 34 dos 63 documentos têm marcador regular |

> **O que foi a "partição 3+2 por lado".** O pipeline decompõe a pergunta comparativa em duas
> sub-perguntas — uma por lado — justamente para escapar do efeito de "centroide", em que uma busca
> única enche o top-5 com um documento só. Só que depois ele **desfaz a decomposição**: funde as duas
> listas por RRF e repontua tudo contra a **pergunta original**. A competição volta a ser global e
> volta a ser ganha pelo lado dominante.
>
> A partição não desfazia: cada sub-pergunta buscava com o seu próprio vetor, e as 5 vagas eram
> **repartidas a priori — 3 para um lado, 2 para o outro —, sem competição entre eles**. Custo quase
> zero. A ideia era obrigar os dois documentos a aparecer.
>
> **Piorou: 13/25 → 11/25**, na régua B (os dois documentos representados). Reservar vaga garante
> que o segundo documento entre, mas gasta vagas com trechos ruins desse documento — e tira vagas de
> trechos bons do primeiro. **Forçar presença não é o mesmo que encontrar o trecho certo**, e é por
> isso que nem na régua frouxa ela ganhou.

### 3.1 Cota de diversidade por documento — refutada em 13/09

Era metade de uma mitigação recomendada no `Memoria.md` e **nunca tinha sido medida**. Teto de N
trechos do mesmo documento no top-5:

| teto | `factual` R@5 | `multi_hop` R@5 | `multi_hop` "as duas @5" |
|---|---|---|---|
| sem teto (hoje) | 28,2% | 31,2% | 2/25 |
| máx. 2 por doc | 19,8% | **12,5%** | **0/25** |

**Por que falha:** as duas âncoras do multi-hop estão no **mesmo documento**. A cota foi desenhada
para a comparativa e, aplicada a todos, expulsa justamente a segunda âncora do multi-hop.

### 3.2 Realimentação de pseudo-relevância (Rocchio) — refutada em 13/09

Doze configurações, todas planas ou piores. A comparativa é a mais prejudicada: mediana da melhor
âncora de **5 para 8–18**.

**Por que falha:** o centroide do top-k é dominado pelo lado que já ganhava, então o método
**amplifica** o desequilíbrio em vez de corrigi-lo.

### 3.3 Por que o top-5 vem quase todo do mesmo documento

Medido: o documento mais frequente ocupa **mediana 5 das 5 vagas**, nos três tipos.

| | dentro do documento | entre documentos |
|---|---|---|
| **com** o prefixo `[título · emissor · seção]` | 0,758 | 0,495 |
| **sem** o prefixo | 0,669 | 0,491 |

O prefixo responde por cerca de **um terço** do excesso de coesão; os outros dois terços são
intrínsecos — trechos da mesma norma repetem vocabulário, numeração e estilo.

**Conclusão: não perseguir isso.** O monopólio *ajuda* factual e multi-hop, cujas âncoras estão no
mesmo documento. Só a comparativa sofre, e as duas tentativas de quebrá-lo (3.1 e 3.2) pioraram o
conjunto. **O monopólio é sintoma; a doença é a consulta.**

### 3.4 Multiplexar a consulta — refutada em 16/09

O `Memoria.md` registrava que combinar a pergunta com a tradução em inglês e com as duas
sub-perguntas levava a posição da âncora **dentro do próprio documento** de mediana 8 para 3. Parecia
ganho de graça: os dois caches já existem, custo zero de LLM.

**Testado no que decide, não se sustenta.** Top-5, BGE denso sem reordenação:

| Consulta | `multi_hop` ≥1 · as duas | `comparative` ≥1 · as duas |
|---|---|---|
| **base — só a pergunta** | **14/25 · 2/25** | **13/25 · 2/25** |
| média(q, en) | 16/25 · 2/25 | 13/25 · 2/25 |
| média(q, en, s1, s2) | 12/25 · **3/25** | 13/25 · 2/25 |
| maxsim(q, en) | 14/25 · 1/25 | **9/25 · 0/25** |
| RRF(q, en, s1, s2) | 9/25 · **0/25** | 12/25 · 2/25 |

**Nenhuma variante move a comparativa.** No multi-hop, ou sobe o "≥1" e o "as duas" fica parado, ou
sobe o "as duas" em uma pergunta e o "≥1" cai. Metade das fusões piora — o `maxsim` derruba a
comparativa de 2/25 para **zero**.

**Por que a medida antiga enganava.** Ela era *posição da âncora dentro do próprio documento*, sobre
49 documentos — grandeza de diagnóstico, não resultado por pergunta. Melhorar a mediana de 8 para 3
dentro do documento **não coloca as duas âncoras num top-5 global**, que é o que a pergunta exige.

> É o mesmo erro de 1.3, em outra roupa: medir uma coisa e concluir sobre outra. Quando um número
> melhorar, pergunte **de qual das duas réguas ele é**.

---

## Parte 4 — O achado que orienta o próximo passo

Mesma busca, trocando apenas **o que** se consulta:

| Tipo | consulta = a pergunta | consulta = pergunta + resposta |
|---|---|---|
| `factual` R@5 | 28,2% | 33,6% |
| `multi_hop` R@5 | 31,2% | **48,4%** |
| `comparative` R@5 | 23,7% | **37,3%** |

Usar a resposta-referência é **oráculo** — é o gabarito, não é método. Mas diz a direção com clareza:
**para multi-hop e comparativa, a pergunta sozinha é uma consulta ruim.** Aproximá-la do texto da
resposta sobe o acerto em cerca de 55% relativo. No factual quase não muda — lá a pergunta já acerta.

> A lacuna é **em parte deliberada**: o prompt de geração manda que a pergunta *não copie o
> vocabulário da norma*, porque quem pergunta não conhece o texto legal. O benchmark está certo em
> ser difícil. **O conserto tem de ser no sistema, não na pergunta.**

Isso converge com 2.3 (multiplexar a consulta) — as duas evidências apontam para o mesmo lugar.

### 4.1 O primeiro sinal positivo: recuperação por cobertura de evidências (16/09)

**A ideia, e a mudança conceitual que a torna precisa.** Não é "decompor a pergunta" — é **decompor
as necessidades de evidência**. A diferença decide o desenho:

| | O que assume | O que faz com os resultados |
|---|---|---|
| decomposição de **pergunta** | as partes são sub-perguntas a responder | funde por RRF e **repontua contra a pergunta inteira** |
| decomposição de **necessidade de evidência** | "para responder isto, o recuperador precisa achar A **e** B" | busca cada necessidade nos próprios termos e **une**, sem repontuar |

Repontuar contra a pergunta inteira é coerente se você acha que está decompondo a *pergunta* — mas é
justamente o que **desfaz** a decomposição: a competição volta a ser global e volta a ser vencida
pelo lado dominante. Foi isso que matou o `decompose_eval.py`.

**O resultado, comparado em tamanho de contexto igual** (o controle que decide — uma união de 10+10
não pode ser comparada com um top-5):

| contexto | consulta única | união das 2 necessidades |
|---|---|---|
| ~5–6 trechos | top-5 → **4/50** | top-3 cada (5,8) → **7/50** |
| ~10 trechos | top-10 → **8/50** | top-5 cada (9,4) → **9/50** |
| ~15 trechos | top-16 → **13/50** | top-8 cada (14,7) → **15/50** |
| ~19 trechos | top-20 → **14/50** (20 trechos) | top-10 cada → **19/50** (18,2 trechos) |

A união ganha nas quatro faixas, e na melhor delas entrega **mais cobertura com menos contexto**.
Por tipo, no melhor ponto: `multi_hop` 9/25 → **12/25**, `comparative` 5/25 → **7/25**. Latência
praticamente idêntica (322 → 327 ms) — as necessidades estão em cache e o produto de matriz domina.

**Mas ainda não dá para provar.** Teste pareado de McNemar entre união(10+10) e única(top-20):
7 discordantes a favor contra 2, **p = 0,18**. Com n=50 não há poder estatístico.

Quanto faltaria: **15 discordantes** na mesma proporção dariam p = 0,035 — o que pede ~83 perguntas
multi-evidência. **O benchmark de 500 terá 250**, e nele esse efeito sairia em p ≈ 0,0002.

> **É o melhor argumento que apareceu para priorizar o benchmark.** O experimento não falhou; a
> amostra atual é que não consegue decidir.

⚠️ **Leia o 4.2 antes de agir sobre isto.** O ganho acima está concentrado nas perguntas cujas duas
âncoras já estavam ao alcance. Num teto realista de 10 trechos (5 por necessidade) a decomposição
acrescenta **uma** pergunta — de 8/50 para 9/50. Ela só se destaca quando o contexto cresce além de
15 trechos, e mesmo lá p = 0,18.

**Duas ressalvas.**

- **O detector é oráculo.** Usamos o rótulo `question_type` do gabarito para escolher quais perguntas
  decompor. O número acima é o **teto** do que um detector real entregaria.
- **O MRR da comparativa piora** (0,279 → 0,184). Os trechos certos entram no conjunto, mas não no
  topo, porque a união intercala as duas necessidades. Para Joint Recall não importa; para a resposta
  final, se o gerador for sensível à ordem, pode importar.

Script: `eval/tools/recuperacao_por_cobertura.py`.

---

### 4.2 Onde o problema realmente está — e por que quase nada funciona (16/09)

**Primeiro, como ler os números desta parte.** O denominador é sempre **as 50 perguntas
multi-evidência** (25 multi-hop + 25 comparativas). As factuais não entram: elas precisam de um
trecho só e já vão bem. "Trechos" é o tamanho do contexto entregue ao gerador.

**Subir o orçamento NÃO resolve.** Joint Recall com consulta única, conforme se abre o k:

| k | `multi_hop` | `comparative` |
|---|---|---|
| **5 — produção** | 2/25 | 2/25 |
| 10 | 4/25 | 4/25 |
| 20 | 9/25 | 5/25 |
| 50 | 17/25 | 8/25 |
| 100 | 18/25 | **13/25** |

Para a comparativa chegar a 13/25 é preciso **k=100** — contexto impossível para geração. Dobrar de
5 para 10 compra **duas perguntas**.

> Eu havia escrito que "o orçamento é o lever principal". **Estava errado**, e a Karol apontou. O
> orçamento compra pouco e satura.

**O que realmente bloqueia: a âncora mais difícil está longe demais.** Posição da pior âncora de cada
pergunta, na consulta única:

| | além do top-20 | além do top-100 |
|---|---|---|
| `multi_hop` | 16/25 | **7/25** |
| `comparative` | 20/25 | **12/25** |

Nas comparativas as piores âncoras estão em 151, 157, 180, 247, 258, 324, 337, 344, 422, 574, 922 e
**1.272**. Como Joint Recall é conjunção, **a âncora mais fraca governa o resultado**: ter uma em
primeiro lugar não vale nada se a outra está em 1.272.

Isso explica de uma vez por que reranking sobre top-100 não move a comparativa (não há o que
promover), por que o orçamento satura, e por que a cota e o PRF falharam.

**E a decomposição também não resolve esses casos.** Das 19 perguntas com âncora além do top-100, a
busca por necessidade traz a âncora para o top-10 em **1**. Nas comparativas, **zero de 12**.

> **São dois problemas diferentes, e estávamos tratando como um:**
>
> | | quantas | o que resolve |
> |---|---|---|
> | as duas âncoras ao alcance, mas a consulta única só pega uma | 31/50 | **decomposição por necessidade** (4.1) |
> | uma âncora fora de alcance (pos. 100–1272) | **19/50** | nada do que foi testado |

**Parte do segundo grupo é anotação ruim, não falha de busca.** Lendo os casos: a `q0077` pergunta
sobre *direitos das pessoas sobre seus dados* e tem uma âncora sobre **sistemas de armas autônomas**;
a `q0099` pergunta sobre *uso secundário de dados* e duas das três âncoras são sobre **parcerias
público-privadas**. Medindo o suporte da âncora à resposta-referência por similaridade, a tendência
existe mas é modesta (mediana 0,698 nas alcançadas vs 0,659 nas distantes; 40% das distantes abaixo
do limiar).

**Conclusão:** não dá para afirmar que as 19 sejam erro de anotação, mas dá para afirmar que
**algumas são, e que decidir isso exige julgamento humano sobre o pool** — que é exatamente a Fase 2.
Otimizar recuperação contra um gabarito onde parte das âncoras não responde a pergunta é otimizar
contra ruído.

---

### 4.3 O que estávamos medindo errado — e por que isso NÃO é contribuição

Depois de refutar cota, PRF, multiplexação, teto e roteamento, e de ver todo método parar em 2–6/25,
a pergunta deixou de ser *"que método falta?"* e passou a ser **"a busca está falhando, ou a medida
está?"**.

**O teste.** Para cada âncora de grau 2 fora do top-k: o top-k contém um trecho **do mesmo documento
exigido** que sustenta a resposta-referência tão bem quanto a âncora?

| | top-5 | top-10 |
|---|---|---|
| âncoras de grau 2 que ficaram de fora | 78 | 68 |
| **o top-k já trazia evidência equivalente** | **38 (49%)** | **39 (57%)** |
| falha real de recuperação | 40 (51%) | 29 (43%) |

Cerca de **metade dos "erros" são acertos não julgados**. Isso explica o platô de 2–6/25 de todo
método, o teto de 6/25 do roteador-oráculo e a inércia do reranking: a régua tinha teto.

> **Limitação:** "sustenta a resposta" foi medido por similaridade do BGE-m3 com a
> resposta-referência — o mesmo modelo avaliado, com risco de circularidade. Indicativo, não
> veredito. O veredito é julgamento humano/LLM sobre o pool, que é a Fase 2.

#### ⚠️ Isto é higiene de método, não descoberta — e o artigo precisa tratar assim

Os quatro problemas que corrigimos **já eram conhecidos da área**. Corrigi-los era necessário; vendê-los
como achado não sobreviveria a um revisor:

| o que corrigimos | por que não é novidade |
|---|---|
| "≥1 trecho" superestima multi-hop | avaliação conjuntiva de evidência é padrão desde o **HotpotQA (2018)**, que reporta *Joint EM/F1* sobre *supporting facts* exatamente por isso; 2WikiMultihopQA e MuSiQue seguem a mesma linha |
| grau 1 contaminando a métrica | defeito da **nossa** anotação (`neighbors_same_doc`), não fenômeno |
| 49% de erros não julgados | **incompletude de pool** é assunto clássico de RI; é a razão de a metodologia TREC exigir declarar a profundidade do pool |
| seis métodos indistinguíveis | consequência dos anteriores |

**Nós estávamos medindo errado; a área não.** No artigo, isso entra em **Método** — *"seguimos a
prática estabelecida para avaliação de evidência conjuntiva (HotpotQA; pooling TREC)"* — em uma
página, com citação. Nunca em Contribuições.

*(Referências citadas de memória: confirmar nos artigos antes de submeter.)*

#### O que sobra, e é real

- **O artefato.** Não há benchmark de RAG sobre normas de saúde e governança de IA em **português**,
  bilíngue PT/EN, com evidência anotada e pool julgado. Trilha de *Datasets & Benchmarks* não exige
  novidade metodológica — exige artefato útil, documentado e reusável. É o **KDD**.
- **Achados de domínio**, medidos aqui: trechos da mesma norma têm coesão de **0,758** contra 0,495
  entre normas, e isso monopoliza o top-5 (§3.3); a âncora mais fraca das comparativas cai entre as
  posições 100 e 1.272 (§4.2); 44% do acervo tem página faltando (Parte 5).
- **A comparação BGE × Gemini** neste domínio — a tese, e o **JBCS**.
- **Os resultados negativos**, com mecanismo: seis estratégias refutadas e o porquê de cada uma.

#### A decisão que fica

**Nenhum experimento novo de recuperação antes de o pool ser julgado.** Não porque os métodos sejam
ruins — porque **não há como distinguir um bom de um ruim** com metade dos erros sendo falsos erros.
Os já feitos estão registrados com script e se reavaliam em uma tarde quando a régua funcionar.

---

## Parte 4.5 — O que o LEME É (17/09)

Até aqui o documento mede **o que o LEME recupera**. Esta seção olha **o que ele é** — foi a primeira
vez que alguém leu a arquitetura em vez do resultado, e apareceram duas coisas.

### O LEME é agentic na arquitetura e tradicional no comportamento

O grafo em `agent/agent.py` tem ciclo:

```
chat_node → (o LLM pediu tool?) → tools_chat → chat_node → ...
                                → END
```

O LLM **pode** decidir se busca, **escreve a própria query** (logo, pode reformular) e **pode buscar
de novo** depois de ver o resultado. Recuperação é ferramenta, não passo fixo: isso é agentic RAG por
definição.

Não existe: planejamento, decomposição, reflexão/verificação, múltiplos agentes, classificação por
complexidade, reranking.

### A contradição: a iteração existe e está desligada por instrução

O `CHAT_SYSTEM_PROMPT` (`agent/utils/prompt.py`) **nunca menciona a ferramenta**. Não diz quando
buscar, nem que se pode buscar de novo. E diz o contrário:

> *"Se a informação não estiver disponível ou for insuficiente, informe claramente: 'Não encontrei
> informações suficientes nas fontes recuperadas para responder com segurança.'"*

**Quando a recuperação falha, o prompt manda desistir — não manda buscar outra vez.** A capacidade de
iterar existe no grafo e está desligada por instrução.

É um **CRAG invertido**: em vez de verificar a suficiência e reagir, verifica e capitula.

É a mudança mais barata do projeto inteiro — uma frase no prompt liga a iteração que o código já
sustenta — e é testável com o benchmark.

### O `score_threshold` que a avaliação nunca simulou

```python
limit=5,
score_threshold=0.60      # agent/utils/tools.py:134
```

A produção descarta qualquer trecho abaixo de 0,60 de similaridade. **A avaliação roda sem esse
corte** — então estamos medindo um sistema diferente do que está no ar.

Medido no BGE (a produção usa Gemini, então os valores absolutos diferem — o formato do risco é que
importa): **33% dos trechos-fonte de grau 2 ficam abaixo de 0,60**, e **4 das 75 perguntas voltariam
com zero trechos**.

Um corte duro em cosseno descarta evidência em silêncio. Antes de mexer nele, a avaliação precisa
passar a simulá-lo — senão não se sabe o que ele custa.

---

## Parte 5 — O estado do acervo hoje

⏳ **A Fase 0 está EM ANDAMENTO.** O Ítalo e o Matheus já passaram uma vez pelos 63 documentos
(commit `dd976ef`) e **seguem conferindo** — os números abaixo são o estado atual da planilha, não o
veredito final. Podem mudar até eles fecharem.

Estado em 16/09:

| Classificação | Destino | Docs | Trechos | Perguntas afetadas |
|---|---|---|---|---|
| `integro` | manter | **42** | 2.692 — 55,0% | 61 |
| `truncado` | recoletar | **12** | **2.173 — 44,4%** | **28** |
| `nao_e_o_documento` | remover | **8** | 30 — 0,6% | 0 |
| `nao_e_o_documento` | recoletar | 1 | 2 — 0,0% | 0 |

**A revisão humana está mais severa que o palpite do script**, e na direção certa: o mecânico
sugeria 8 truncados (33,4% do acervo), a passagem humana já marcou **12 (44,4%)**. Se o número se
mantiver, quase metade do corpus precisa ser recoletada.

**As 28 perguntas afetadas são o custo real.** São perguntas do gabarito atual ancoradas em
documentos que vão ser recoletados — quando o texto mudar, elas passam pela reancoragem (1.5). Como
o id vem do conteúdo, só muda o que precisa mudar.

**Os 8 marcados para remover são baratos:** 30 trechos em 4.897, nenhuma pergunta ancorada. Saem
sem quebrar nada.

> **Enquanto a Fase 0 não fechar, a Fase 1 não começa.** É a lista de documentos aprovados que diz
> sobre o que as 400 perguntas novas podem ser geradas.

> ⚠️ **A coluna `licenca` está vazia nos 63.** É ela que decide o que pode ser redistribuído junto
> com o benchmark publicado — perguntas, qrels e identificadores são nossos; o texto de normas com
> direito autoral só entra se a licença permitir. **Sem isso não há artefato publicável no KDD.**

**O caso que mostra o tamanho do problema.** No Código de Ética Médica faltam 48 páginas, e 12 das 14
ausentes na faixa dos artigos são **pares** — defeito sistemático, não sorteio. O efeito é
verificável: o Art. 77 está na página 37 e o Art. 80 na 39; **a página 38 não existe no corpus**, e é
onde estão os Art. 78 e 79. O mesmo com a página 40 e os Art. 88 a 91.

**Seis artigos do Código de Ética Médica não estão no acervo.** Uma pergunta sobre o Art. 78 o LEME
não erra por falha de busca — o texto nunca foi indexado.

---

## Parte 6 — O problema que o benchmark existe para resolver

Hoje **só conta como relevante o trecho de onde a pergunta nasceu**. Quando a busca devolve outro
trecho que responde igualmente bem, a métrica conta como erro.

Medindo o que os três braços de busca produzem hoje:

| Tipo | duplas no pool | julgadas | não julgadas, **mas do documento certo** |
|---|---|---|---|
| `factual` | 255 | 40 | 124 |
| `multi_hop` | 282 | 27 | 159 |
| `comparative` | 294 | **13** | **145** |

Nas comparativas, **13 de 294** têm julgamento. E não é só omissão: em **36 das 75** perguntas existe,
no documento certo e sem anotação, um trecho que casa com a resposta melhor que a âncora anotada.

É isso que o plano do `benchmark_leme_500.md` corrige: 500 perguntas e julgamento sobre o pool real.

---

## Parte 7 — O que ainda falta testar

**Recuperação:**

1. **Confirmar a recuperação por cobertura no benchmark de 500** (4.1) — o efeito existe e é
   consistente, mas p = 0,18 com n=50. Não aplicar em produção antes disso.
2. **Detector de pergunta multi-evidência** — hoje o experimento usa o rótulo do gabarito. Sem um
   detector real, o método não sai do papel. Vale tentar um sintático antes de um LLM.
3. **HyDE com hipótese gerada** — ainda com sinal (Parte 4), mas agora **em segundo plano**: ataca a
   lacuna de vocabulário, enquanto a cobertura ataca a estrutura da pergunta. As duas precisam de LLM
   em produção, e a cobertura já tem resultado medido.
4. **Subir o top-k de produção de 5 para 8** (2.2) — leva multi-hop de 3/25 para 7/25. Medir junto o
   efeito de mais contexto sobre a qualidade da resposta gerada.
5. **Reranker mais forte (F3)** — o gargalo voltou a ser reordenação, não recuperação.
6. **Fine-tuning do BGE (E3)** — o gatilho está acionado (o gap passa a margem por 5×), mas o F3 vem
   antes por ser mais barato. Regra: **o gold set não pode ser dado de treino** — gerar pares
   sintéticos e manter as 75 como teste cego.

**Benchmark:** as três fases do `benchmark_leme_500.md`.

---

## Parte 8 — O que está bloqueado

> 🔴 **A produção está fora de sincronia com o corpus.** A coleção `LEME_gemini` no Qdrant continua
> com **5.160 pontos do recorte antigo**, enquanto o corpus tem 4.897. Produção serve o chunking
> velho e devolve ids que não existem mais no gabarito.
>
> Reconstruir exige reembutir com a API do Gemini — **custa chamada e mexe em produção**, então é
> decisão da Karol. **Enquanto não for feito:** nada de avaliação contra o Gemini, e nada de comparar
> produção com gabarito. Os ids não casam.
>
> O cache do BGE-m3 **já está atualizado** (4.661 vetores reaproveitados, 236 recalculados).

Outras três, menores:

- **Os rankings dos três braços estão obsoletos** — apontam para ids no formato posicional antigo.
  Precisam ser regerados antes da Fase 2.
- **`rank_bm25` não está no `requirements.txt`**, mas é importado pelo código. O braço BM25 não roda.
- **`eval/config.yaml` aponta o juiz errado** (`llama-3.1-8b-instant` no lugar do `3.3-70b` que rodou
  de verdade). Quem reproduzir usa o modelo errado.

---

## A tese, hoje

| Sistema | nDCG@5 |
|---|---|
| Gemini (produção) | **0,4422** |
| BGE-m3 (aberto) | 0,2908 |

δ = −0,1513, IC95% [−0,2185; −0,0848], p = 0,0001.

**A não-inferioridade está rejeitada** com a margem de 0,03: o intervalo inteiro está abaixo dela.
**Migrar para o embedding aberto hoje custaria qualidade mensurável**, e o texto precisa dizer isso.

Mas o quadro tem duas ressalvas que mudam a leitura, e as duas precisam entrar no artigo:

1. **Sob a régua estrita de completude (1.3), o BGE+reranker iguala ou supera o Gemini nos três
   tipos.** O nDCG@5 e a completude de evidência não contam a mesma história.
2. **O julgamento está incompleto** (Parte 6). Medir com qrels incompletos favorece sistematicamente
   quem foi usado para construí-los. É exatamente isso que o benchmark de 500 existe para corrigir —
   e é por isso que ele vem antes de qualquer conclusão final sobre a tese.
