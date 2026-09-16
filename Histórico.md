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

## Como ler os números de multi-hop e comparativa

Estes dois tipos precisam de **mais de um trecho** para serem respondidos, então um número só não
diz nada. Sempre que aparecerem aqui, aparecem nas duas formas:

| Forma | O que conta | Para que serve |
|---|---|---|
| **≥1** | perguntas que trazem **pelo menos um** dos trechos necessários | mede se a busca *chegou perto* |
| **as duas** | perguntas que trazem **todos** os trechos necessários | mede se a pergunta **foi respondida** |

**A que decide é a segunda.** Metade da evidência não responde a pergunta: se o multi-hop precisa do
Art. 5 e do Art. 12 e a busca traz só o Art. 5, a resposta sai incompleta ou errada.

Foi confundir as duas que produziu o erro descrito em 1.3 — *"multi-hop resolvida em 84%"* era o
"≥1"; pela régua que decide era 3 de 25.

---

## Onde estávamos

O artigo do WFA foi submetido em **14/08/2026** e está fechado. A partir daí a restrição "não quebrar
o artigo" caducou, e o trabalho passou a ser preparar o próximo: um benchmark publicável, mirando o
**KDD 2027, trilha Datasets & Benchmarks**.

A tese da dissertação é uma só: **o embedding aberto (BGE-m3) pode substituir o proprietário
(gemini-embedding-001) que está em produção.** Tudo abaixo existe para responder isso com honestidade.

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

1. **HyDE com hipótese gerada** — é a única direção com sinal (Parte 4). Falta medir se o ganho do
   oráculo sobrevive a uma hipótese real. Aplicar a mesma expansão aos dois braços, para não
   contaminar a comparação da tese.
2. **Subir o top-k de produção de 5 para 8** (2.2) — leva multi-hop de 3/25 para 7/25. Medir junto o
   efeito de mais contexto sobre a qualidade da resposta gerada.
3. **Reranker mais forte (F3)** — o gargalo voltou a ser reordenação, não recuperação.
4. **Fine-tuning do BGE (E3)** — o gatilho está acionado (o gap passa a margem por 5×), mas o F3 vem
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
