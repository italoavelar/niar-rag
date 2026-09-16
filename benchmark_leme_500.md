# Benchmark 500: cobertura total do acervo

> **LEME · niar-rag · plano de trabalho**
>
> **Executores:** Ítalo, Matheus, João e Helen ·
> **Alvo:** KDD 2027, trilha *Datasets & Benchmarks*, ciclo 2 (fev/2027)
>


Expandir o gabarito de 100 para 500 perguntas, cobrindo os documentos do acervo, e anotar a
relevância dos trechos que os três braços de busca realmente recuperam.

## Onde cada um atua

| Pessoa | Fase | O que faz | Começa |
|---|---|---|---|
| **Ítalo** | 0, depois 2 | Auditoria do acervo: decidir, documento a documento, se o que está indexado é o documento inteiro. Depois, executa metade da busca e do julgamento | **agora** |
| **Matheus** | 0, 1 e 2 | Auditoria junto com o Ítalo. Depois revisa por amostra 60 das 400 perguntas novas, e serve de referência humana em ~80 duplas do julgamento | **agora** |
| **João** | 1, depois 2 | Gera as 400 perguntas novas em sessão do Claude, lote a lote. Depois executa a outra metade da busca e do julgamento | quando a Fase 0 fechar |
| **Helen** | 2 | Escolhe o juiz LLM e **confere se ele é confiável** — um portão antes do lote grande e uma auditoria depois | no piloto da Fase 2 |

**A ordem importa e não pode ser invertida:** a Fase 0 fecha a lista de documentos; a Fase 1 gera
perguntas só sobre documentos aprovados; a Fase 2 julga o que a busca traz para essas perguntas.
Começar a Fase 1 antes de a Fase 0 fechar significa gerar perguntas sobre documentos que podem sair
do acervo.

---

## Por que agora

O gabarito atual tem 100 perguntas e toca **13 dos 63 documentos** do acervo. Os outros 50 nunca foram consultados por nenhuma pergunta — incluindo os maiores do corpus. Na prática, boa parte do
que o sistema indexa nunca foi avaliada.

Há um segundo problema, independente e mais grave: hoje **só é considerado relevante o trecho de onde a pergunta nasceu**. Quando a busca devolve outro trecho que responde igualmente bem, a métrica conta como erro. Medindo o pool que os três braços produzem hoje (top-5 de cada, sobre as 100 perguntas atuais):

| Tipo | Duplas únicas no pool | Julgadas | Não julgadas, **mas do documento certo** |
|---|---|---|---|
| `factual` | 255 | 40 | 124 |
| `multi_hop` | 282 | 27 | 159 |
| `comparative` | 294 | 13 | **145** |

Nas comparativas, **13 de 294** duplas têm julgamento. Metade do pool vem do documento correto e é contada como erro apenas porque ninguém olhou.

E não é só omissão de julgamento. Medindo se existe, no documento certo e **sem anotação**, um trecho que casa com a própria resposta-referência melhor que a âncora anotada: **36 das 75 perguntas
respondíveis**. A anotação escolheu um trecho entre vários que respondem — e os outros contam como erro. A Fase 2 corrige isso por construção, para o que os braços alcançam.

> **Resultado esperado.** Um benchmark de **500 perguntas** com julgamento de relevância sobre o que os três braços realmente recuperam — não só sobre o trecho de origem. É o artefato que sustenta as conclusões do artigo e que pode ser publicado junto com ele.

---

## Regras que valem para todas as fases

### Congelar o corpus antes de começar

Os identificadores de trecho mudam quando o corpus é reprocessado, e as anotações morrem junto. Isso já aconteceu: o re-chunking de 01/09 deslocou o conteúdo por trás dos IDs em cerca de 40% das perguntas — o ID continuou existindo, apontando para outro texto, e a validação não pegou porque ela confere se o ID *existe*.

A Fase 0 corrigiu a causa (ID por conteúdo) e é a **única janela** para mexer no acervo: depois que a Fase 1 começar, ninguém roda `src/extract_to_jsonl.py` nem `src/extract_html_to_jsonl.py` até o
benchmark fechar.

Registrem no cabeçalho de cada entrega o fingerprint devolvido por `corpus_embedding_fingerprint()`.
O atual, **pós-reextração de 13/09**, é:

```
da69227b35c04c4d6b037f6e954e41fa943e13de3384759db7ced8e5c169dc84
```

### Regras de conteúdo

- Pergunta e resposta-referência **sempre em português**, mesmo quando o trecho está em inglês — é a língua de quem usa o sistema.
- **Nada de expressões meta:** "segundo o texto", "no trecho acima". A pergunta tem que fazer sentido sozinha, sem o documento à vista.
- Para saber o documento de um trecho, leiam `metadata["document_id"]` — **não tentem deduzir do nome
  do ID.** (O corpus usava `<doc>_pN_cN` para PDF e `<doc>_cN` para HTML; hoje o ID termina em hash e não carrega mais essa informação.)
- O trecho que originou a pergunta permanece anotado como **grau 2** em todas as etapas. A Fase 2 só acrescenta julgamentos; nunca remove esse.
- Toda anotação guarda **o texto do trecho**, não só o id. O `golden_qa.jsonl` tem o campo
  `qrels_text` (preenchido nas 75 perguntas respondíveis atuais); as 400 novas e os julgamentos da
  Fase 2 precisam preenchê-lo também. É o que permite reancorar por casamento de texto se o corpus
  mudar um dia — sem isso, uma anotação órfã vira reanotação.

---

## Fase 0 — Auditoria do acervo

**Ítalo + Matheus** · *bloqueia a Fase 1*

Nem tudo que está indexado é documento normativo. Nove entradas do corpus são páginas de catálogo do
site da ISO, não as normas: o texto indexado de `iso_42001_2023`, por exemplo, contém *"Edition 1
2023-12 Read sample · CHF 225 Add to cart · Shipping costs not included"*. São de 2 a 10 trechos
cada, todas apontando para `iso.org/standard/NNNNN.html`.

Se a regra de cobertura mandar gerar perguntas para todo documento, o João vai produzir perguntas
sobre número de edição e preço em franco suíço. Isso entra no benchmark, vai para revisão do KDD, e é
o tipo de coisa que um revisor encontra.

### Suspeitos a verificar, um por um

As nove ISO — `iso_22989_2022`, `iso_23894_2023`, `iso_24368_2022`, `iso_25237_2017`,
`iso_27001_2022`, `iso_27701_2025`, `iso_38507_2022`, `iso_42001_2023`, `iso_5338_2023` — e mais
estes, todos pequenos demais para o que dizem ser:

| Documento | Trechos | Observação |
|---|---|---|
| `ai_principles_OECD_` | 3 | |
| `rnds_MS_` | 3 | |
| `relatorio_pesquisa_CEP_` | 4 | |
| `lei_13787_BR_2018` | 4 | |
| `tcud_CEP_2015` | 5 | |
| `certificacao_SBIS_` | 6 | |
| `lei_6681_BR_1979` | 6 | |
| `eca_BR_1990` | 10 | um estatuto inteiro em 10 trechos |
| `dpia_ICO_` | 10 | |

### Tarefas

**1. Rodar o preparador da auditoria.** A parte mecânica — contar, medir, procurar marcador — está
pronta. Dois comandos:

```bash
python eval/tools/auditar_acervo.py --csv docs/auditoria_acervo.csv
python eval/tools/auditar_acervo.py --dump docs/auditoria/
```

O primeiro gera a planilha com uma linha por documento e os sinais lado a lado. O segundo grava o
**texto indexado** de cada documento num `.txt` — é esse texto que precisa ser julgado, não o PDF
original: o que o sistema conhece do documento é só o que foi indexado.

**A pergunta da Fase 0 não é "que documento é esse".** O acervo foi montado a dedo; todo mundo já
sabe que `iso_42001_2023` deveria ser a ISO/IEC 42001. A pergunta é outra: **o que foi indexado é o
documento inteiro?** O que o LEME conhece de uma norma é só o texto que entrou no corpus — se a
extração perdeu páginas, nenhum modelo de embedding recupera o que não está lá.

Os sinais na planilha, e o peso real de cada um:

| Coluna | Significa | O que separa | Peso no acervo |
|---|---|---|---|
| `paginas_ausentes` | buracos na sequência de páginas indexadas | extração que falhou vs. documento completo | **8 docs, 1.638 trechos (33,4%), 24 perguntas** |
| `sinal_loja` | marcadores de e-commerce (*Add to cart, CHF, Read sample*) | página do site da ISO vs. a norma | 8 docs, 29 trechos (0,6%), 0 perguntas |
| `sinal_deontico` | verbos de obrigação (*deve, deverá, é vedado, shall, must*) | norma tem; ficha de produto não | apoia o anterior |
| `sinal_dispositivo` | referências a artigo/seção (*Art., §, Section, Clause, Anexo*) | norma tem; página de catálogo não | apoia o anterior |
| `inicio` / `fim` | primeiros e últimos 160 caracteres, para bater o olho | — | — |

A ordem da tabela é a ordem da importância. As oito páginas da ISO são 0,6% do acervo e não
sustentam nenhuma pergunta: despacham-se em minutos. O que exige trabalho é o truncamento.

**Por que o truncamento é o que importa — o caso do `codigo_etica_medica_CFM_2019`.** Faltam 48
páginas. Mapa da faixa dos artigos (`.` indexada, `X` ausente, páginas 4 a 51):

```
.XXXX...X...X...X.....X.....X.....X.X.X...X.X...
456789012345678901234567890123456789012345678901
```

Doze das catorze ausentes são **pares** — defeito sistemático, não sorteio. E o efeito é verificável:

- Art. 77 está na página 37, Art. 80 na 39. **A página 38 não existe no corpus** — é onde estão os
  Art. 78 e 79.
- Art. 87 na página 39, Art. 92 na 41. **A página 40 não existe** — Art. 88, 89, 90 e 91.

**Seis artigos do Código de Ética Médica não estão no acervo.** Uma pergunta sobre o Art. 78 não é
falha de recuperação: o texto não foi indexado. Se ela entrar nas 500, mede a extração e não o
recuperador — e BGE e Gemini erram igual, pelo mesmo motivo, o que esvazia a comparação que é a tese.

E é justamente aqui que a máquina para: das 48 páginas ausentes do CFM, 31 são um bloco contíguo
(52 a 82) que, pela posição — entre a Exposição de Motivos e a Composição do CFM —, é quase
certamente o índice remissivo, e perder índice remissivo não custa nada. As páginas 38 e 40 custam
seis artigos. **O script não distingue as duas coisas; quem abre o PDF distingue.**

A coluna `sugestao` é palpite explícito, para ordenar a fila — **não é veredito**. Quem assina a classificação é quem revisa.

**2. Revisar a fila, de cima para baixo.** A execução de 13/09, sobre o acervo **já reextraído**, deu:

Ordenada por quanto cada fila pesa, não por nome de categoria:

| Fila | Docs | Trechos | Perguntas | O que provavelmente é |
|---|---|---|---|---|
| `TRUNCADO? (paginas ausentes)` | 8 | **1.638** (33,4%) | **24** | WHO, UNESCO, CFM, plano brasileiro de IA e outros — conferir, página a página, se o buraco é página em branco, página em imagem sem OCR, ou coleta interrompida. **É aqui que está o trabalho.** |
| `NAO E O DOCUMENTO (catalogo/loja)` | 8 | 29 (0,6%) | 0 | as páginas da ISO — texto com preço e "Add to cart". Decisão rápida: sai do acervo ou vira nota de rodapé na descrição do corpus |
| `SUSPEITO (pequeno demais)` | 7 | 33 (0,7%) | 0 | `predetermined_change_control_FDA_2023`, `certificacao_SBIS_`, `tcud_CEP_2015`, `relatorio_pesquisa_CEP_`, `imdrf_good_ml_practice_2025`, `rnds_MS_`, `ai_principles_OECD_` |
| `SUSPEITO (sem marca de norma)` | 1 | 13 (0,3%) | 0 | `transparency_ml_devices_FDA_2024` |
| `parece integro` | 39 | 3.184 (65,0%) | 75 | conferência rápida: abrir o `.txt`, ler início e fim |

A coluna *Perguntas* conta pares pergunta–documento: uma comparativa com âncoras em dois documentos
conta nos dois, e as **25 perguntas `unanswerable`** não contam em lugar nenhum porque não têm âncora
por construção. Em perguntas distintas: das 75 respondíveis, **22 tocam documento truncado e 14 têm
todas as âncoras dentro de um**.

Os 16 documentos das três filas do meio somam **75 trechos em 4.897 e nenhuma pergunta** — são meia
manhã de trabalho somados. Os 8 truncados sustentam um terço do acervo e 22 das 75 perguntas
respondíveis do gabarito atual.

**São 24 documentos que pedem atenção real**, não 63. O resto é bater o olho.

Para cada linha, preencher quatro colunas:

- `classificacao` — `integro` \| `truncado` \| `nao_e_o_documento`
- `destino` — `manter` \| `recoletar` \| `remover`
- `licenca` — o texto pode ser redistribuído? `sim` \| `nao` \| `so metadados`
- `observacao` — o que você viu que justifica a decisão

**Critério de destino.** Recoletar quando a fonte permitir — a ISO/IEC 22989, por exemplo, é de
download gratuito e pode entrar de verdade. Remover quando não houver como obter o texto: é melhor um
acervo de 50 documentos íntegros do que de 63 com nove fichas de produto.

**Divisão sugerida:** Ítalo pega as três primeiras filas (16 documentos, os que provavelmente saem ou voltam); Matheus pega a de truncamento e a conferência rápida (47, mas quase toda de bater o olho). Cada um preenche sua parte da mesma planilha.

**Remover documento não quebra o resto.** Isso mudou com o id por conteúdo: o id é o hash do texto daquele trecho (`src/chunk_id.py`), então tirar o documento X não altera um único id do documento Y.
Somem apenas os ids do documento removido.

O que fazer com as perguntas que apontavam para ele: **nada de apagar.** O `eval/tools/reancorar_por_texto.py` marca cada uma com `status:
invalida_documento_removido` e as deixa fora do `qrels.csv` — que é o arquivo que a engine de
recuperação lê. A pergunta continua no `golden_qa.jsonl`, com o motivo registrado, e o número de
perguntas efetivas do benchmark fica auditável em vez de sumir no histórico.

> ⚠ Remoção e recoleta acontecem **agora**, antes da Fase 1 — não porque quebrem o acervo, mas porque mudam o que existe para ser perguntado e recuperado. Um documento que entra depois da Fase 2 nunca é julgado por ninguém.

**3. Completar a procedência.** A planilha já traz `fonte` (a URL ou o arquivo de origem) e `tipo`. Falta acrescentar, para cada documento que fica: **data de coleta** e a decisão de `licenca`. Essa tabela é insumo direto da submissão ao KDD — o critério de acessibilidade e a seção de ética saem dela, e é ela que decide o que pode ser redistribuído junto com o benchmark.

**4. Trocar o identificador de trecho por um baseado no conteúdo.** ✅ **CONCLUÍDO em 12/09/2026.**

O id agora é `<document_id>_<12 hex do sha256 do texto normalizado>`, gerado por `src/chunk_id.py` e
usado pelos dois extratores. Corpus, `qrels.csv`, `golden_qa.jsonl`, os caches de embedding e o
payload da coleção `LEME_gemini` em produção já foram migrados e conferidos — o texto
embutido não mudou, só o rótulo.
> [!WARNING]
Ainda é necessário atualizar o gemini em produção com a correção das tabelas (só depois da conferência do Italo e Matheus) **Karol fazer**

O que isso compra, e por que não vai precisar ser refeito:

- Mesmo texto, mesmo ID — sempre, independentemente do que aconteça aos vizinhos.
- Documento novo só acrescenta. Nada do que já existe é renumerado.
- Reexecutar a extração sem mudar parâmetros reproduz exatamente os mesmos IDs.
- Se o recorte mudar de verdade, o ID muda — e isso é o comportamento certo: a anotação passa a
  apontar para nada (falha barulhenta) em vez de apontar para outro texto (falha silenciosa).

O hash é do **texto cru**, nunca do texto com prefixo de contexto. É o que garante que enriquecer metadado depois não invalide anotação nenhuma.

**5. Corrigir dois defeitos de recorte — piso de fragmento e tabela partida.** ✅ **CONCLUÍDA pela
Karol em 13/09/2026** — código aplicado, corpus reextraído, gabarito reancorado. Os números abaixo
são os **medidos depois de rodar**, não mais a previsão.

O caso que expôs isso é a **q0075**, onde o especialista deu nota 1 por alucinação. A pergunta é
*"a UNESCO … classifica a gravidade dos impactos — o que exige e como classifica?"*, e a resposta
correta são quatro níveis. Na página 45 do documento da UNESCO, o recorte produziu três trechos:

| Trecho | Conteúdo | Tamanho |
|---|---|---|
| `…_d0afb10b4a79` | *"…a continuum of four Gravity level: Moderate/Minor, Serious, Critical and Catastrophic"* | 242 chars |
| `…_7a5369df5e89` | `[TABELA]` Catastrophic + Critical | 964 chars |
| `…_0c61e31bbd29` | `[TABELA]` Serious + Moderate/minor | 859 chars |

A frase que responde inteiro ficou sozinha, e a tabela que a detalha foi cortada ao meio. Nenhum dos
três chega ao top-5. O sistema recebeu o prefácio do documento e respondeu que *"a UNESCO não
estabelece classificação de gravidade"* — negativa falsa. Somados, os três dão 2.065 caracteres:
caberiam num único trecho.

**Nunca foi regressão.** `merge_undersized_chunks` sempre fez o que promete — o defeito era o
**valor** do piso: com 120, a frase de 242 chars nunca foi candidata a fusão. Os dois ajustes e o
que deram de fato:

| Ajuste | Previsto | **Medido após rodar** |
|---|---|---|
| `MIN_CHUNK_SIZE` de **120 → 300** (`src/chunking.py:18`) | 4.965 trechos; 4.789 ids intactos | 5.160 → **4.897** trechos; **219 → 0** abaixo de 300 chars; **4.661 ids intactos** (499 sumiram, 236 nasceram) |
| `TABLE_CHUNK_SIZE = 2500` (`src/chunking.py:27`) | 13 das 24 tabelas inteiras | páginas com tabela partida **34 → 21**; trechos nelas **132 → 51** |

As 11 tabelas que continuariam partidas são as grandes do `justice_data_governance_OECD_2024`
(4.200 a 6.600 caracteres) — tabelas de dados linha a linha, onde partir é legítimo. As que importam
são as tabelas-definição curtas, como a da UNESCO.

**Custo real na anotação, medido comparando `golden_qa.jsonl` com o backup
`.bak-pre-reancoragem-20260913-162606`:**

| Das 256 âncoras das 75 perguntas respondíveis | |
|---|---|
| id inalterado, nada a fazer | **247** |
| remapeadas por casamento de texto | **9 ids antigos → 7 novos** (duas fusões) |
| ambíguas (texto em mais de um trecho novo) | **0** |
| não encontradas, exigindo revisão humana | **0** |

Sete perguntas são tocadas: `q0001`, `q0012`, `q0015`, `q0052`, `q0065`, `q0075`, `q0096`. Em
`q0012`, duas âncoras de grau 1 caem no mesmo trecho novo e as duas linhas de qrels viram uma —
conferir à mão, é o único caso que não é mecânico. **Nenhuma reanotação humana.**

Resultado concreto na página 45 da UNESCO, que é o caso da q0075:

```
antes    d0afb10b4a79   242 chars  "…continuum of four Gravity level: Moderate/Minor, Serious,
                                    Critical and Catastrophic…"
         7a5369df5e89   964 chars  [TABELA] Catastrophic + Critical
         0c61e31bbd29   859 chars  [TABELA] Serious + Moderate/minor

agora    dda90a2b9133  2012 chars  a frase com os QUATRO níveis + a tabela inteira
         02d3aa37f72d   880 chars  [TABELA] Scope
```

Saiu melhor que o previsto: em vez de 1.208 chars com metade da tabela, o `dda90a2b9133` traz
**2.012 chars com a frase e a tabela de gravidade completa** — porque o teto de tabela e o piso
agiram juntos. `q0015`, `q0065`, `q0075` e `q0096` ancoram todas nele agora. A q0075 é respondível
por um único trecho.

**Código:** `MIN_CHUNK_SIZE = 300` (`src/chunking.py:18`) e `TABLE_CHUNK_SIZE = 2500`
(`src/chunking.py:27`); `pack_structured_units` passou a usar o teto próprio para tabela. **Os 65
testes da suíte passam** (35 deles nos dois arquivos de recorte), incluindo dois novos:
`test_table_below_the_ceiling_stays_whole` e
`test_lead_sentence_merges_into_the_table_it_introduces` — este último cobre justamente o caminho que
faltava, fundir prosa dentro de um trecho `[TABELA]…[/TABELA]`.

Dois testes tiveram a expectativa corrigida: `test_unesco_page_45_table_is_atomic_and_not_duplicated`
exigia `len(table_chunks) > 1`, ou seja, travava o sintoma que o nome do teste prometia evitar.

**Ordem de execução — o que já rodou e o que falta:**

| | Passo | Estado |
|---|---|---|
| 1 | Remover e recoletar documentos (auditoria da Fase 0) | ⏳ **adiado de propósito** — ver nota abaixo |
| 2 | `src/extract_to_jsonl.py` + `src/extract_html_to_jsonl.py` com os parâmetros novos | ✅ 13/09 16:21 — 5.160 → 4.897 trechos |
| 3 | `eval/tools/reancorar_por_texto.py --aplicar` | ✅ 13/09 16:26 — 247 inalteradas, 9→7 remapeadas, 0 perdidas |
| 4 | Reembutir o que mudou (`eval/tools/reembutir_incremental.py`) | ✅ **BGE** 13/09 17:16 — `da69227b35c04c4d`, 4.897 vetores, cobre o corpus exato. ❌ **Qwen3** ainda em `1d3a1532afeba86f` com 5.160 vetores |
| 5 | Atualizar o payload da `LEME_gemini` no Qdrant | 🔴 **bloqueado** — `migrar_ids_qdrant.py` não serve mais (ele renomeia id, e aqui o conteúdo mudou). Precisa de `build_vectorstore.py`, que hoje casa o backup por **índice posicional** e falha com `ValueError`; tem de passar a casar por conteúdo para reaproveitar os 4.661 vetores Gemini já pagos |
| 6 | Registrar o fingerprint novo e **congelar** | ⏳ depois do passo 1 |

**Nota sobre a inversão do passo 1.** A ordem original mandava auditar e só então reextrair. Foi
invertido de propósito: o time audita já a versão corrigida do corpus, em vez de gastar o trabalho
duas vezes sobre um recorte que se sabia errado. O preço é que **se a auditoria mandar remover ou
recoletar documento, os passos 2 a 4 rodam de novo** — o que é barato, porque o id vem do conteúdo:
só os trechos do documento mexido mudam, e a reancoragem já provou ser mecânica.

> **Entrega:** planilha de procedência com os 63 documentos classificados + corpus reprocessado com
> o recorte corrigido, com o fingerprint novo registrado.
> **Pronto quando:** a lista de documentos que a Fase 1 deve cobrir está fechada e o corpus não muda
> mais até o benchmark fechar.

---

## Fase 1 — Geração das 400 perguntas novas

**João** · *precede a Fase 2*

Complementar o gabarito existente até 500 perguntas, mantendo a proporção atual de 25% por tipo. As
100 atuais permanecem como estão.

| Tipo | Hoje | Gerar | Total | O que caracteriza |
|---|---|---|---|---|
| `factual` | 25 | 100 | 125 | respondível por 1 trecho |
| `multi_hop` | 25 | 100 | 125 | 2+ trechos do mesmo documento |
| `comparative` | 25 | 100 | 125 | 1 trecho em cada um de 2 documentos |
| `unanswerable` | 25 | 100 | 125 | fora do escopo do acervo |

O tamanho não é arbitrário: a análise de poder do próprio projeto estimou ~466 consultas para fechar
o teste de não-inferioridade entre o embedding aberto e o proprietário. Com **375 respondíveis** das
500, a conta fecha com folga — e é isso que transforma o benchmark em instrumento capaz de sustentar
a conclusão do artigo, não só em volume maior.

### 1. O prompt corrigido — já atualizado em `eval/01_build_golden.py`

As personas antigas ("médicos… e auditores de IA responsável") foram substituídas pelos três perfis.
O texto abaixo é o mesmo que está no código, para quem for trabalhar direto no Claude. Cole como
instrução de sistema e trabalhe **um documento por vez**.

```text
Você é um especialista em direito da saúde e governança de IA no Brasil, criando um benchmark de avaliação para um sistema RAG.

O PÚBLICO que usa esse RAG são TRÊS perfis. Nenhum deles é jurista, e todos têm POUCA familiaridade com jargão jurídico:
1) PROFISSIONAL DA SAÚDE (médico, enfermeiro, gestor clínico) — pergunta o que precisa fazer, registrar ou garantir para estar em conformidade no atendimento e no uso de IA na prática assistencial.
2) DESENVOLVEDOR (quem constrói, integra ou mantém o sistema de IA) — pergunta o que o software precisa cumprir, documentar, validar, registrar ou monitorar, e o que muda conforme o nível de risco.
3) PESQUISADOR (acadêmico, membro de comitê de ética, cientista de dados) — pergunta sobre exigências de projeto, ética em pesquisa, consentimento, uso secundário de dados, compartilhamento e publicação.

Gere perguntas que ESSES perfis realmente fariam (conformidade, ética, governança, proteção de dados/LGPD, risco, responsabilidade, transparência), focando no que a pessoa precisa SABER ou FAZER — não em tecnicismos de doutrina jurídica. Use linguagem acessível; quando um termo técnico for inevitável, formule de modo compreensível para quem não é jurista. Uma pergunta NÃO deve copiar o vocabulário exato da norma: quem pergunta não conhece o texto legal, descreve a situação com as próprias palavras.

Em 'persona', indique qual dos três perfis faria a pergunta, usando exatamente um destes valores: saude | desenvolvedor | pesquisador. Varie os três perfis ao longo do conjunto.

NUNCA use expressões meta como 'segundo o texto', 'de acordo com o trecho' ou 'no documento acima' — a pergunta deve fazer sentido sozinha, para quem nunca viu o documento.

ATENÇÃO: o acervo é BILÍNGUE — muitos trechos estão em INGLÊS (guidance internacional: WHO, FDA, GDPR, NIST, OECD, UNESCO). Mesmo quando o trecho estiver em inglês, a PERGUNTA e a RESPOSTA-REFERÊNCIA devem ser SEMPRE em PORTUGUÊS DO BRASIL — é a língua do público-alvo (o sistema precisa responder em PT a partir de fontes em EN).

REGRA DE CONTENÇÃO — a mais importante de todas:
A RESPOSTA-REFERÊNCIA deve estar INTEIRAMENTE contida nos trechos fornecidos. Não complete com conhecimento próprio, não generalize além do que está escrito, não afirme quantidades ('são quatro níveis', 'há três requisitos') que o texto não enumere por extenso. Se o trecho descreve dois itens, a resposta fala de dois — ainda que você saiba que existem mais.
Para cada afirmação da resposta, inclua em 'evidencia' a citação LITERAL do trecho que a sustenta, copiada caractere a caractere. Se você não conseguir citar, a afirmação não pode estar na resposta.
Se os trechos não bastarem para uma resposta completa e autossuficiente, devolva a lista vazia e NÃO gere a pergunta. Descartar é barato; uma pergunta irrespondível contamina o benchmark inteiro.

Responda SEMPRE em JSON válido.
```

Cada pergunta passa a sair com o campo `evidencia` — a citação literal que sustenta cada afirmação da resposta. É o que torna auditável se a resposta excedeu o trecho, e barateia a revisão do
Matheus: conferir citação contra texto é mais rápido e menos subjetivo que julgar se a resposta "está certa".

O campo `persona` entra no registro de cada pergunta e permite estratificar os resultados por perfil depois — inclusive para verificar se o sistema atende pior a um dos três. **As 100 perguntas atuais não têm esse campo** (conferido: 0 de 100); as 400 novas terão (Analisar se da para colocar nelas e como).

### 2. O que muda no modo de construir o gabarito

Antes de gerar mais 400 no mesmo molde: **dois defeitos reais** e **uma tentação a evitar**.

O critério para separá-los é um só — uma mudança é legítima quando torna a medida **mais válida**, e ilegítima quando torna o sistema **mais bonito**. Corrigir pergunta impossível por construção é a primeira; filtrar pergunta difícil é a segunda.

**a) Os qrels são os trechos sorteados, por construção.** O script anota o trecho-fonte com grau 2 e
os vizinhos imediatos com grau 1 (`eval/01_build_golden.py:204-206`); no multi-hop e na comparativa,
anota exatamente os dois trechos que foram ao LLM (`:235`, `:257`). **Nunca há busca.** O gerador vê
1 ou 2 trechos de 5.160 e não tem como saber que outro responde igual ou melhor. É por isso que em
36 das 75 perguntas existe um trecho melhor não anotado.

*Isto não se conserta com prompt.* É a razão de existir a Fase 2: gerar tópicos e depois julgar o
pool são as duas metades do protocolo padrão de benchmark de recuperação, não um remendo.

**c) A resposta-referência podia exceder a evidência.** ✅ **corrigido em 13/09.** Nada conferia que o
que a resposta afirma está nos trechos. Na `q0075` a resposta diz "quatro níveis" e o trecho anotado
tem dois — o modelo completou de memória, e a pergunta ficou impossível por construção: nem a
recuperação perfeita a satisfaz.

A regra de contenção e o campo `evidencia` entraram no prompt (ver item 1 acima), e os três prompts
de tipo passaram a aceitar recusa explícita: `{"pairs": []}` na factual, `{"question": null}` na
multi-hop e na comparativa. Descartar é barato; pergunta irrespondível contamina o benchmark.

Uma última mudança em `add()`: toda pergunta nova já sai com `qrels_text` preenchido, como as 75
atuais. Não é mais tarefa manual.

### 3. Forçar a cobertura dos documentos aprovados na Fase 0

O gerador hoje amostra trechos livremente até bater a meta por tipo — sem nenhuma garantia de que
todo documento seja visitado. Trocar por uma alocação explícita: **todo documento aprovado na
auditoria recebe ao menos uma pergunta factual**, e as vagas restantes são distribuídas
proporcionalmente ao número de trechos, com teto para que os documentos grandes não dominem.

A meta de cobertura é a lista que sai da Fase 0, que pode ser menor que 63. Cobertura total de um
acervo auditado vale mais que cobertura nominal de um acervo com lixo.

**Orçamento:** 375 perguntas respondíveis geram cerca de 500 vagas de documento (comparativas ocupam
duas), para ~63 documentos — cerca de 8 por documento. Dá para garantir cobertura e ainda distribuir
o excedente por tamanho.

### 4. Gerar, revisar e integrar — passo a passo do João

O gerador precisa ser um LLM **independente dos modelos avaliados**: não pode ser Gemini (embedding),
Qwen (gerador do RAG) nem Llama/gpt-oss (juízes). **Claude atende a esse requisito** — não é nenhum
dos três. Por isso gerar em sessão do Claude é caminho legítimo, e não atalho.

> ⚠ O gabarito atual **não** foi produzido por `01_build_golden.py` — veio de sessão manual
> integrada por `eval/tools/build_gold_manual.py` (`eval/config.yaml:23-26`). O procedimento abaixo
> formaliza esse mesmo modo de trabalho, com as travas que faltavam.

#### A regra que não pode cair

**Quem sorteia o trecho é o script. O Claude nunca escolhe.**

Se o João colar o acervo e pedir *"escolha bons trechos e gere perguntas"*, o modelo seleciona o que
ele próprio entende bem — e o benchmark passa a medir o que o gerador gosta, não o que o acervo
exige. É a mesma contaminação que derrubou o emparelhamento por similaridade: **item escolhido pelo
modelo não serve para avaliar modelo.** O sorteio é aleatório, com semente registrada, e o Claude
recebe os trechos já escolhidos, sem poder trocar.

Corolário prático: **nunca colar o corpus inteiro na sessão.** O Claude vê 1 ou 2 trechos por
pergunta e mais nada.

#### Passo 0 — pré-requisitos

- [ ] Fase 0 fechada: existe a lista de documentos **aprovados** (pode ser menor que 63).
- [ ] Corpus congelado, `fingerprint` registrado.
- [ ] `eval/data/golden_qa.jsonl` com as 100 atuais intacto — as novas são **acrescentadas**, o
      arquivo nunca é regerado do zero.

#### Passo 1 — sortear o lote (script, com semente)

```bash
python eval/tools/lotes.py sortear --tipo factual --n 10 --semente 42
```

Saem dois arquivos por lote, em `eval/lotes/`:

| Arquivo | Para que serve |
|---|---|
| `lote_01_factual.md` | **cole este arquivo inteiro** na sessão — já traz a instrução de sistema, a instrução do tipo, o formato de resposta e os trechos numerados, **sem os ids** |
| `lote_01_factual.ids.json` | o manifesto: qual número corresponde a qual `chunk_id`. Não mexer |

O `.md` não leva id de propósito: o Claude não precisa dele e, se levar, tende a devolvê-lo
transcrito errado. O casamento número → id é feito pelo script na integração.

O sorteio respeita a alocação de cobertura do item 3 (todo documento aprovado recebe ao menos uma
factual). **Não há filtro de marcador normativo antes de gerar** — filtrar trecho por verbo deôntico
ou número de artigo foi testado e barraria 40% do acervo, incluindo as âncoras de 8 perguntas que
hoje existem. Quem decide se o trecho sustenta pergunta é a regra de contenção, na hora da geração.

#### Passo 2 — colar na sessão do Claude

Abra uma sessão **por lote** e cole o `.md` inteiro. É uma única colagem: o arquivo já contém a
instrução de sistema, a instrução do tipo, o formato de resposta exigido e os trechos.

**Não há prompt para copiar à mão.** O script lê a constante `SYS` direto de
`eval/01_build_golden.py` e a escreve no `.md`. Se alguém mudar o prompt no código, o próximo lote
já sai com o novo — e prompt de sessão nunca diverge de prompt de código, que é o que faria as 400
novas deixarem de ser comparáveis entre si.

O Claude responde com um array JSON. Salve **exatamente como veio** em
`eval/lotes/lote_01_factual.out.json` (a cerca ```` ```json ```` em volta, se vier, é removida na
integração).

#### Passo 3 — o que o João digita e o que o script deriva

É daqui que vem a maioria dos erros. O Claude devolve **cinco campos**; todo o resto é calculado:

| Campo | Vem do Claude | Derivado pelo script |
|---|---|---|
| `question`, `reference_answer`, `evidencia`, `difficulty`, `persona` | ✅ | |
| `qid` | | numeração contínua a partir de `q0101` |
| `qrels`, `qrels_text` | | do manifesto `.ids.json` |
| `source_docs`, `source_lang`, `theme` | | do corpus |
| `procedencia` | | nome do lote + semente, gravado em cada pergunta |

**João nunca digita `chunk_id` à mão.** Foi exatamente assim que o gabarito antigo se descasou do
corpus.

#### Passo 4 — recusa é resultado, não falha

O prompt manda o Claude devolver `{"pairs":[]}` (factual) ou `{"question":null}` (multi-hop e
comparativa) quando os trechos não sustentam pergunta autossuficiente. **Isso é para acontecer.**

Se um lote inteiro vier sem nenhuma recusa, desconfie: ou os trechos eram excepcionalmente bons, ou
o modelo está forçando pergunta. Descartar é barato; pergunta irrespondível contamina o benchmark
inteiro — foi o que a `q0075` custou.

A **taxa de recusa por tipo** é número a reportar no artigo. Anotar lote a lote.

#### Passo 5 — salvar e integrar

Salvar a resposta do Claude como `eval/lotes/lote_01_factual.out.json` e rodar primeiro a conferência,
que **não escreve nada**:

```bash
python eval/tools/lotes.py integrar --lote lote_01_factual --conferir
```

Ela checa cinco coisas:

1. Todo `chunk_id` do manifesto existe no corpus — aborta e lista os inválidos se não.
2. **Cada string de `evidencia` aparece literalmente no trecho correspondente.** Esta é a trava que
   torna a falha da `q0075` impossível: se o Claude afirmou *"quatro níveis"* e o trecho só enumera
   dois, a citação não casa e o registro é rejeitado.
3. Nenhuma pergunta repetida (normalizada) contra as já existentes.
4. Multi-hop traz citação **dos dois** trechos; comparativa idem. Sem isso não é multi-hop de verdade.
5. A pergunta não contém expressão meta (*segundo o texto*, *no documento acima*, *conforme o trecho*).

Só depois de a conferência passar limpa:

```bash
python eval/tools/lotes.py integrar --lote lote_01_factual --aplicar
```

O `--aplicar` **aborta** se houver qualquer item rejeitado — ou você corrige e regera, ou passa
`--forcar` para gravar só os aceitos. Ele faz backup do `golden_qa.jsonl` antes de escrever e
regenera `queries.csv` e `qrels.csv` no fim.

#### Passo 6 — os 10 primeiros à mão, antes do resto

Este caminho está sendo estreado agora. **Gerar um lote de 10 factuais, integrar, e ler os 10
registros finais no `golden_qa.jsonl`** antes de seguir. Conferir em cada um: a resposta cabe no
trecho; a pergunta se sustenta sem ver o documento; a `persona` combina com o que foi perguntado; a
linguagem não é de jurista. Se algo estiver sistematicamente torto, o conserto é no prompt — e sai
muito mais barato agora que depois de 400.

#### Passo 7 — `unanswerable` tem um passo a mais

As 100 `unanswerable` não saem de trecho nenhum, então a trava da `evidencia` não se aplica. Em
troca, elas exigem uma verificação que as outras não exigem: **confirmar que o acervo realmente não
responde.**

O `--conferir` avisa disso e **não bloqueia**: conferir se o acervo responde exige rodar a pergunta
contra o índice, e isso é trabalho da Fase 2. Aqui a checagem é sua, à mão, sobre as 25 de cada
lote. Uma *unanswerable* que o acervo responde é um falso negativo plantado no gabarito — pior que
não ter a pergunta.

A `reference_answer` é sempre exatamente: *"Não encontrei informações suficientes nas fontes
recuperadas para responder com segurança."*

#### Passo 8 — registrar a procedência

Metade disso o script já faz: cada pergunta sai com o campo `procedencia` (lote + semente), e o
manifesto de cada lote guarda a semente e a data. Os arquivos de `eval/lotes/` ficam versionados —
são o registro de o que foi oferecido ao modelo.

O que **só o João sabe** e precisa escrever à mão em `eval/lotes/PROCEDENCIA.md`: modelo e versão do
Claude usados, taxa de recusa por tipo, e o `fingerprint` do corpus vigente. Sem isso a geração não é
reproduzível — e o artigo precisa poder dizer como o gabarito nasceu.

#### Quantos lotes dá isso

| Tipo | Perguntas | Trechos por pergunta | Lotes de 10 |
|---|---|---|---|
| `factual` | 100 | 1 | 10 |
| `multi_hop` | 100 | 2 (mesmo doc) | 10 |
| `comparative` | 100 | 2 (docs diferentes) | 10 |
| `unanswerable` | 100 | 0 | 4 lotes de 25 |

Cerca de **34 lotes**. Com a recusa esperada, sortear ~15% a mais de trechos que a meta.

#### Resumo dos comandos

```bash
# 1. sortear (uma vez por lote)
python eval/tools/lotes.py sortear --tipo factual --n 10 --semente 42

# 2. colar eval/lotes/lote_01_factual.md na sessão do Claude
# 3. salvar a resposta em eval/lotes/lote_01_factual.out.json

# 4. conferir — não escreve nada
python eval/tools/lotes.py integrar --lote lote_01_factual --conferir

# 5. gravar
python eval/tools/lotes.py integrar --lote lote_01_factual --aplicar
```

Trocar `--tipo` por `multi_hop`, `comparative` ou `unanswerable`, e **mudar a semente a cada lote**
(ela vai registrada no manifesto e é o que torna o sorteio reproduzível).

> ✅ **`eval/tools/lotes.py` está pronto e testado (13/09/2026).** Um arquivo, dois subcomandos.
> Testado de ponta a ponta: sorteio dos quatro tipos, as cinco validações, a recusa de gravar com
> item rejeitado, o backup automático e a regeração dos derivados. O gabarito voltou a 100 registros
> depois do teste.

### Os 50 documentos sem nenhuma pergunta hoje

Lista completa, por tamanho. É o alvo de cobertura da Fase 1.

| Documento | Trechos |
|---|---|
| `cp_BR_1940` | 282 |
| `plano_brasileiro_ia_BR_2025` | 273 |
| `de_identifying_government_datasets_NIST_2023` | 265 |
| `justice_data_governance_OECD_2024` | 230 |
| `multimodal_models_guidance_WHO_2024` | 222 |
| `regulatory_considerations_ai_health_WHO_2024` | 175 |
| `resolucao_2306_CFM_2022` | 166 |
| `global_digital_health_strategy_WHO_2020_2027` | 113 |
| `cdc_BR_1990` | 88 |
| `imdrf_samd_risk_framework_2014` | 76 |
| `lei_8080_BR_90` | 76 |
| `ai_ml_discussion_paper_FDA_2021` | 64 |
| `cf_BR_1988` | 63 |
| `resolucao_1627_CFM_2001` | 60 |
| `resolucao_2336_CFM_2023` | 56 |
| `resolucao_2454_CFM_2026` | 54 |
| `regulamento_interno_CEP_2024` | 51 |
| `decreto_44045_BR_1958` | 44 |
| `imdrf_samd_clinical_evaluation_2017` | 41 |
| `ai_framework_convention_council_europe_2024` | 36 |
| `imdrf_ml_medical_devices_terms_2022` | 35 |
| `resolucao_UFMG_01_2022` | 32 |
| `aiml_samd_action_plan_FDA_2021` | 28 |
| `resolucao_1821_CFM_2007` | 28 |
| `lei_3268_BR_1957` | 19 |
| `transparency_ml_devices_FDA_2024` | 14 |
| `resolucao_2_ANPD_2024` | 14 |
| `dpia_ICO_` | 10 |
| `iso_27001_2022` | 10 |
| `eca_BR_1990` | 10 |
| `good_ml_practice_guiding_principles_FDA_2021` | 9 |
| `resolucao_2424_CFM_2025` | 8 |
| `predetermined_change_control_FDA_2023` | 7 |
| `resolucao_1638_CFM_2002` | 6 |
| `iso_42001_2023` | 6 |
| `lei_6681_BR_1979` | 6 |
| `certificacao_SBIS_` | 6 |
| `imdrf_good_ml_practice_2025` | 5 |
| `tcud_CEP_2015` | 5 |
| `relatorio_pesquisa_CEP_` | 4 |
| `rnds_MS_` | 4 |
| `lei_13787_BR_2018` | 4 |
| `iso_24368_2022` | 3 |
| `ai_principles_OECD_` | 3 |
| `iso_27701_2025` | 3 |
| `iso_25237_2017` | 3 |
| `iso_38507_2022` | 2 |
| `iso_22989_2022` | 2 |
| `iso_5338_2023` | 2 |
| `iso_23894_2023` | 2 |

> Documentos de 2 a 6 trechos (as ISO, em especial) comportam **uma pergunta factual cada**, não
> mais. Não force multi-hop onde não há dois trechos que se complementem.

### 5. Revisão humana por amostra — **Matheus**

Revisar pelo menos **60 das 400** (estratificadas por tipo e persona) verificando: a resposta está
mesmo contida no trecho indicado; a pergunta se sustenta sem ver o documento; a linguagem é a do
perfil, não a de um jurista. Registrar na coluna `curadoria`, como no conjunto atual — essa taxa de
aprovação vira número reportado no artigo.

> **Entrega:** `eval/data/golden_qa.jsonl` com 500 registros. **Derivados:** `queries.csv`, `qrels.csv`.
> **Pronto quando:** todo documento aprovado na Fase 0 aparece em ao menos uma pergunta.

---

## Fase 2 — Busca nos três braços e anotação de relevância

**João + Ítalo** (execução, ~188 perguntas cada) · **Helen** (juiz e calibração) ·
**Matheus** (referência humana) · *depois da Fase 1*

Para cada pergunta, rodar os três braços de recuperação, tomar o top-5 de cada um e anotar a
relevância de cada trecho devolvido, com um LLM juiz. Isso é **pooling**: o que qualquer um dos
braços colocou no topo passa a ter julgamento, e "não julgado = irrelevante" deixa de ser uma
suposição cega.

**Os três braços**

- BM25 — `eval/retrieval_eval/scenario_A_bm25.py`
- BGE-m3 (denso) — `scenario_B_dense_bge.py`
- Fusão RRF — `scenario_C_fusion.py`

**Escala de relevância**

- **2** — responde diretamente a pergunta
- **1** — contribui, mas não basta sozinho
- **0** — não contribui

### 1. Dividir o trabalho

Só as **375 perguntas respondíveis** entram aqui — as 125 `unanswerable` não têm trecho relevante por
definição. Ficam cerca de **188 perguntas para cada um**, divididas de forma estratificada por tipo e
persona (não pela ordem do arquivo), para que os dois lotes sejam comparáveis.

Os dois lotes podem ser inteiramente disjuntos — o que garante padrão comum é **o juiz ser o mesmo
modelo, com o mesmo prompt e a mesma temperatura**. Fixem isso por escrito antes de começar.

### 2. Profundidade: top-5 por braço (decidido em 13/09, revisando o top-3)

A decisão anterior era top-3. A medição a derrubou. Tomando os 36 trechos que respondem à pergunta
melhor que a âncora anotada (ver "Por que agora") e perguntando quantos já apareceriam no pool:

| Profundidade do pool | Trechos capturados | Duplas únicas por pergunta |
|---|---|---|
| top-3 | 9 de 36 | 6,3 a 7,2 |
| **top-5** | **11 de 36** | **10,2 a 11,8** |
| top-10 | 18 de 36 | — |

O top-3 captura um quarto do problema que a Fase 2 existe para resolver. O top-5 custa cerca de
4 duplas a mais por pergunta e é o mínimo defensável. Para 375 perguntas respondíveis: cerca de
**4.200 duplas no total, ~2.100 por pessoa**. É volume de execução, não de leitura — quem julga é o
LLM, e isso são algumas horas desacompanhadas e poucos dólares de API.

> **O pooling não conserta tudo, e isso precisa estar escrito no artigo.** Mesmo no top-10, metade
> dos trechos que respondem melhor continua fora do pool — porque nenhum braço os alcança. O caso
> exemplar é a q0075: a frase que responde está na **posição 33** do BGE e fora do pool de 100 no
> BM25 e na fusão. Esses casos são de chunking, e se resolvem na Fase 0, não aqui.

> Registrem a profundidade do pool no artigo: com pool de profundidade 5, um sistema futuro que traga
> algo na 6ª posição terá achados não julgados. É limitação normal de pooling, **desde que
> declarada**.

**O que não dá para fazer é anotar só o que o BGE traz.** O artigo compara BM25, BGE e fusão; julgar
apenas o pool de um deles torna os julgamentos sistematicamente favoráveis a esse — tudo que os
outros acham e ele não vira "irrelevante" por omissão, não por avaliação. Seria a mesma incompletude
que estamos corrigindo, agora enviesada a favor da conclusão que queremos defender. É o erro clássico
de pooling, e é o primeiro lugar onde um revisor de trilha de benchmark olha.

### 3. Escolher o juiz — **Helen**

Precisa ser independente do que está sendo avaliado — não pode ser o Gemini (embedding em produção)
nem o Qwen (gerador). Llama-3.3-70B e gpt-oss-120b já estão integrados e foram usados na avaliação de
geração; qualquer um serve como juiz primário.

Julgar dupla a dupla, com a pergunta e o trecho, **sem mostrar ao juiz qual braço trouxe o trecho nem
se ele é o trecho de origem** — senão o julgamento fica contaminado.

A Helen fixa por escrito, **antes de qualquer lote rodar**: modelo, versão, prompt e temperatura do
juiz primário, e o mesmo do juiz secundário. O que garante que os lotes do João e do Ítalo sejam
comparáveis não é eles combinarem entre si — é o juiz ser literalmente o mesmo.

### 4. Calibração do juiz — **Helen**, em dois momentos

Quem julga é o LLM; João e Ítalo só executam. Rodar a mesma dupla nos dois computadores daria o mesmo
resultado e não mediria nada. **O que precisa ser medido é se o juiz é confiável** — e esse é o
trabalho da Helen.

**Entrada 1 — o portão, antes do lote grande.**

João e Ítalo rodam um **piloto de ~100 duplas** e entregam para a Helen. O trecho de onde a pergunta
nasceu entra no lote como qualquer outro, sem marcação, e é relevante **por construção**: se o juiz
der 0 nele, o erro é do juiz, não do trecho.

A Helen calcula a taxa de erro nesse controle e decide se libera as ~4.200 duplas. **O limiar é
fixado por escrito antes de olhar o resultado** — sugestão: *mais de 10% de nota 0 nas âncoras
reprova o juiz*. Fixar depois é escolher o número que aprova o que já se rodou.

Reprovou, troca-se o modelo ou o prompt e repete-se o piloto. É o passo que impede gastar o lote
inteiro com um juiz que não serve.

**Entrada 2 — a auditoria, depois do lote grande.**

- **Segundo juiz numa amostra.** ~300 duplas estratificadas por tipo, braço e nota do primeiro juiz,
  passadas por um segundo LLM independente. A Helen calcula **κ de Cohen** entre os dois. É esse
  número que autoriza usar o julgamento automático no restante — e é ele que um revisor do KDD vai
  procurar.
- **Referência humana.** ~80 duplas revisadas pelo Matheus, concentradas onde os dois juízes
  divergem. Serve para dizer qual dos dois está mais perto de uma pessoa.
- **Taxa de erro no controle de âncora**, agora sobre o lote completo.

Os três números vão para o artigo. Sem eles, o julgamento automático é afirmação sem prova.

### 5. Passo a passo da Helen

A ferramenta é `eval/tools/julgar_relevancia.py`, com três subcomandos. Os dois primeiros são do João
e do Ítalo; o terceiro é dela.

#### Antes de tudo — o que precisa estar pronto

- [ ] Fase 1 fechada: `golden_qa.jsonl` com as 500 perguntas.
- [ ] **Rankings regerados.** Os que estão em `eval/results/retrieval/rankings/` são anteriores à
      reextração de 13/09 e apontam para ids que não existem mais (`_p4_c2`, formato posicional).
      O subcomando `pool` **recusa rodar** sobre eles e diz isso — é falha barulhenta, não silenciosa.
- [ ] Helen escreve a **ficha do juiz** antes de qualquer lote: modelo, versão, prompt, temperatura,
      do primário e do secundário.

#### Passo 1 — João e Ítalo montam o pool

```bash
python eval/tools/julgar_relevancia.py pool --profundidade 5
```

Junta o top-5 dos três braços numa lista de duplas únicas em `eval/results/pool_fase2.csv`. **As
âncoras entram no pool mesmo que nenhum braço as tenha trazido** — sem elas não há controle.

O CSV guarda `origem` (que braços trouxeram) e `e_ancora`, mas **nada disso entra no prompt do juiz**.
Ele vê pergunta e trecho, e mais nada.

#### Passo 2 — o piloto, que é o portão da Helen

```bash
python eval/tools/julgar_relevancia.py julgar --juiz llama --piloto 100
```

Julga 100 duplas, **metade âncoras** — é isso que dá denominador para a taxa de controle. Escreve em
`eval/results/julgamentos_llama.csv`.

#### Passo 3 — Helen decide se libera o lote grande

```bash
python eval/tools/julgar_relevancia.py calibrar --julgamentos eval/results/julgamentos_llama.csv
```

Saída:

```
1) CONTROLE DE ÂNCORA  (254 âncoras julgadas)
   nota 2:  142  (55.9%)
   nota 1:  100  (39.4%)
   nota 0:   12  ( 4.7%)

   taxa de erro (nota 0 em âncora): 4.7%   limiar: 10%
   >>> APROVADO <<<
```

O limiar padrão é **10% de nota 0 em âncora**, e muda com `--limiar`. **Fixe o valor por escrito antes
de olhar o resultado** — escolher o limiar depois é escolher o número que aprova o que já rodou.

Reprovou, troca-se o modelo ou o prompt e repete-se o piloto. É o passo que impede gastar as ~4.200
duplas com um juiz que não serve.

#### Passo 4 — o lote grande (João e Ítalo)

```bash
python eval/tools/julgar_relevancia.py julgar --juiz llama
```

Retoma de onde parou se cair no meio: ele lê o que já foi julgado e só faz o que falta.

#### Passo 5 — Helen fecha com o segundo juiz

```bash
python eval/tools/julgar_relevancia.py julgar --juiz gptoss --piloto 300
python eval/tools/julgar_relevancia.py calibrar \
    --julgamentos eval/results/julgamentos_llama.csv \
    --segundo     eval/results/julgamentos_gptoss.csv \
    --divergentes eval/results/divergentes_para_matheus.csv
```

Sai o **κ de Cohen** entre os dois juízes, com a leitura (fraca / moderada / forte / quase perfeita),
e o arquivo de duplas divergentes com a coluna `nota_humana` em branco — é a amostra que o **Matheus**
preenche, concentrada exatamente onde os dois juízes discordam.

> **Entrega da Helen:** o relatório de calibração com os três números — taxa de erro no controle de
> âncora (piloto e lote completo), κ entre os dois juízes, e a ficha do juiz. Sem eles o julgamento
> automático é afirmação sem prova, e é o primeiro lugar onde um revisor do KDD olha.

### 6. Consolidar

Gerar o `qrels.csv` final unindo: os trechos de origem (grau 2, preservados) e todos os julgamentos
novos de grau 1 e 2. Registrar **quantos relevantes por pergunta havia antes e depois** — é esse
número que mostra o tamanho do problema que o pooling corrigiu.

> **Entrega:** `qrels.csv` expandido + planilha de julgamentos por dupla (João e Ítalo).
> **Entrega da Helen:** o relatório de calibração — κ entre os dois juízes, taxa de erro no controle
> de âncora (piloto e lote completo), e a ficha do juiz (modelo, versão, prompt, temperatura).
> **Reportar no artigo:** esses três números mais relevantes por pergunta antes/depois do pooling.

---

## O que a trilha do KDD exige

O KDD tem uma trilha de **Datasets & Benchmarks**, com dois ciclos por ano e revisão com o mesmo
rigor das trilhas de pesquisa. O ciclo 1 do KDD 2027 fechou em 26/07/2026; o ciclo 2 tem prazo em
**fevereiro de 2027** (datas exatas ainda não publicadas). É esse o alvo.

**Formato:** 8 páginas de conteúdo, template ACM em duas colunas, revisão single-blind, no máximo
duas submissões por autor por ciclo.

| Critério | O que a chamada pede | De onde sai |
|---|---|---|
| **Accessibility** | Dados acessíveis "sem exigir pedidos pessoais"; código e ferramentas abertos e documentados | Repositório público + licença + DOI arquival (Zenodo). **Falta definir — decisão da Karol** |
| **Quality & Documentation** | Descrição clara de coleta, curadoria e organização; metadados e pré-processamento | Fases 1 e 2 produzem isso **se o protocolo for registrado enquanto roda**, não depois |
| **Impact** | "Datasets originais devem idealmente vir com resultados empíricos que comprovem o valor dos dados" | Os três braços já avaliados sobre o benchmark novo — mais o achado de completude |
| **Ethics & Fairness** | Privacidade, consentimento, viés e potencial de uso indevido | Acervo é normativo e público, sem dados pessoais — mas isso precisa estar escrito |

### O que ainda não existe

- **Licença e repositório público do benchmark, com DOI arquival.** Sem isso não há badge de artefato
  e o critério de acessibilidade fica em aberto. A tabela de procedência da Fase 0 decide o que pode
  ser redistribuído: perguntas, qrels e identificadores de trecho são de vocês; o texto de normas com
  direito autoral (a ISO/IEC 42001 custa CHF 225; já a ISO/IEC 22989 é de download gratuito) só entra
  se a licença permitir.
- **Documentação do protocolo** — como as perguntas foram geradas, com que modelo, com que prompt,
  que fração passou por revisão humana e com que taxa de aprovação; como a relevância foi julgada,
  por qual modelo, com que concordância. **Escrevam enquanto rodam.** Reconstruir isso depois é o que
  mais custa.
- **Declaração de ética** — origem dos documentos, ausência de dados pessoais, riscos de uso indevido
  de um benchmark jurídico-sanitário.

---

## O que pode dar errado

- **Reprocessar o corpus no meio do caminho** invalida tudo que já foi anotado. É o risco mais caro
  da lista e já se materializou uma vez.
- **Perguntas em linguagem de jurista.** Se a pergunta usa o vocabulário exato da norma, a busca
  acerta por casamento lexical e o benchmark fica fácil demais — deixa de medir o que interessa.
- **Documentos pequenos forçados a multi-hop.** Uma ISO de 2 trechos não sustenta pergunta que exige
  combinar dois trechos complementares.
- **Juiz não calibrado.** Por isso o controle da etapa 4: sem ele, não há como distinguir "a busca
  falhou" de "o juiz julgou mal" (**Helen analisar essa parte**).
- **Pool preso aos braços atuais.** Um sistema novo, criado depois, vai recuperar trechos que ninguém
  julgou. Declarar no artigo quais braços compuseram o pool e em que profundidade.
- **Esperar que o pooling conserte o que é recorte.** Metade dos trechos que respondem melhor que a
  âncora está fora do alcance dos três braços em qualquer profundidade razoável. Se a etapa 5 da
  Fase 0 não for feita, esses casos continuam contando como falha de busca — e não são.
- **Protocolo reconstruído de memória.** Se ninguém anotar qual modelo, qual prompt e qual
  temperatura foram usados enquanto o trabalho corre, isso terá que ser reconstruído na hora de
  escrever — e é exatamente o que a revisão vai cobrar em detalhe.

---

*Registro canônico do projeto, com fonte por número: `niar-rag/Memoria.md`.*