# LEME — Recuperação: o que foi testado, o que funciona, o que vai para produção

**Data:** 30/09/2026 · **Corpus:** 6.134 trechos, 54 documentos

**População deste relatório:** o subconjunto **balanceado** de 300 perguntas —
**75 factuais, 75 multi-hop, 75 comparativas e 75 irrespondíveis** —, decidido em
30/09. As 225 respondíveis são a base de toda tabela de recuperação; as 75
irrespondíveis entram só na seção de abstenção, porque não têm trecho relevante
por definição. O acervo completo tem 393 perguntas e fica como base de fevereiro.

O balanceamento **não é detalhe de apresentação: ele muda a conclusão do
artigo.** A seção 8 mostra o quanto, e por quê.

> **A tese, em uma linha:** nesta população, o empilhamento aberto **alcança** o
> proprietário (p = 0,0989), e **ganha em português** (p = 0,0352). "Supera" só
> vale no acervo completo de 267, que não é a população escolhida — ver a seção 8.

Cada tabela sai de um script reexecutável; a seção 9 diz qual produz qual.
Nenhum número aqui foi copiado de relatório anterior.

---

## Resumo e decisão

**Na qualidade, o empilhamento aberto (BGE-m3 + `bge-reranker-v2-m3`) fica à
frente do `gemini-embedding-001`. Na latência em CPU, ele perde de longe.** A
recomendação de produção depende de haver GPU, e essa condicional é nova neste
relatório — a versão de 29/09 trazia uma latência 12 vezes otimista.

| | Complete Evidence@5 | custo por consulta |
|---|---|---|
| BGE-m3 sozinho | 0,387 | ~0,1 s de CPU |
| Gemini (produção hoje) | 0,489 | API paga, ~1–2 s |
| BGE-m3 + reranker, pool 20 | 0,502 | **~50 s de CPU** |
| **BGE-m3 + reranker, pool 100** | **0,538** | **~250 s de CPU** (ou GPU) |

- **Com GPU:** trocar para BGE-m3 + reranker, pool 100. Ganha 4,9 pontos sobre o
  Gemini e zera a conta de API.
- **Sem GPU:** **manter o Gemini.** O aberto só passa o proprietário a partir do
  pool 20, e pool 20 custa ~50 segundos de CPU por consulta — não é interativo.

Acrescentar **recusa automática** quando a melhor nota do reranker ficar abaixo
de **0,574**: recusa 68,3% das perguntas sem resposta no acervo, ao custo de
11,9% de recusa indevida.

---

## 1. O que o sistema faz

Uma pergunta em português entra, o sistema recupera trechos de 54 normas
(brasileiras e internacionais, em português e inglês) e um gerador responde a
partir deles. Este relatório trata **só da recuperação** — se os trechos certos
chegam. A qualidade da resposta gerada é outra etapa, e a auditoria da seção 7 é
o portão de entrada dela.

### O recorte: 1.200 caracteres com 240 de sobreposição

Decidido por experimento em 22/09 com 4 condições de recorte × 91 perguntas × 3
orçamentos de contexto, numa amostra de 15 documentos. A comparação é a
**orçamento de caracteres fixo**, não a k fixo: o que produção paga é o contexto
enviado ao gerador, não o número de trechos.

| condição | orç. 2.000 | orç. 5.000 | orç. 10.000 | teto |
|---|---|---|---|---|
| T400 (janela 400) | 0,484 | 0,593 | 0,681 | **0,802** |
| **T1200 (adotado)** | **0,648** | **0,835** | 0,868 | 1,000 |
| T3500 | 0,484 | 0,703 | 0,879 | 1,000 |
| parent-child 400/3500 | 0,626 | 0,824 | 0,912 | 1,000 |
| parent-child 400/1200 | 0,604 | 0,813 | 0,901 | 1,000 |

Complete Evidence, modelo Gemini, de
`eval/experimento_embedding/resultados/recorte_resultados.json`. O arquivo em
disco hoje guarda só as linhas do Gemini — as do BGE foram sobrescritas por uma
rodada posterior, e os testes de significância abaixo estão registrados em
`analisar_recorte.py`.

- **T1200 vence T400 em 6 de 6 comparações** (2 modelos × 3 orçamentos), p entre
  0,000 e 0,011, sobrevivendo a Benjamini-Hochberg. O motivo não é busca: o T400
  tem **teto de 0,802** porque a fronteira de 400 caracteres parte a evidência
  antes de qualquer busca acontecer. É impossibilidade de recorte.
- **T1200 vence T3500 direcionalmente nas 6**, significativo em 3, mas **não
  sobrevive à correção para testes múltiplos**. A formulação honesta é
  "direcionalmente superior em todas as seis, sem significância após correção".
- **Parent-child não ganha do T1200** — 0 de 6 significativas. Na variante filho
  400 / pai 1.200, em que os pais são literalmente os trechos do T1200, perde nos
  dois orçamentos apertados. *Small-to-big não entrega nada neste corpus.*
- **Cuidado de medição que quase inverteu a decisão:** a k fixo, o top-5 do T400
  devolve ~1.600 caracteres e o do T3500 ~14.750. A k fixo o parent-child marcava
  92% contra 68% do T1200 e a conclusão seria "adote parent-child". Normalizado
  por orçamento, dá 67% contra 67%.

**A LIMITAÇÃO DESTE EXPERIMENTO, e ela é séria.** As 91 perguntas são **56
factuais e 35 multi-hop — zero comparativa e zero irrespondível**. O recorte foi
escolhido numa população que não contém os dois tipos em que o sistema mais
falha hoje, e nenhum número acima diz o que o recorte faz com pergunta que cruza
dois documentos.

Pior: cada pergunta nasceu de uma passagem de **600, 1400 ou 3000** caracteres, e
**a vantagem do T1200 se concentra na janela mais próxima do seu próprio
tamanho.** Medido em 05/10 pelo corte por janela de `analisar_recorte.py`
(Gemini, orçamento de 5.000):

| condição | janela 600 (n=29) | janela 1400 (n=33) | janela 3000 (n=29) |
|---|---|---|---|
| T400 | 72% | 58% | 48% |
| **T1200** | 86% | **88%** | **76%** |
| T3500 | 72% | 73% | 66% |
| parent-child 400/3500 | **97%** | 76% | **76%** |
| T1200 sem sobreposição | **97%** | 82% | 72% |

Na janela de 1400 o T1200 é o melhor. Na de **600 ele perde por 11 pontos** para
o parent-child e para a variante sem sobreposição, e no orçamento de 10.000 na
janela de 3000 ele perde também para o T3500 (76% contra 86%). O padrão repete
nos três orçamentos, o que é mais difícil de atribuir a ruído do que uma célula
isolada seria — mas cada janela tem n ≈ 30, uma pergunta vale ~3 pontos, e
**nenhum teste por janela foi feito**.

**O que isso muda na prática:** o 1.200 continua sendo a escolha defensável no
agregado e é o que está em produção, mas a afirmação "1.200 é o melhor recorte"
não sobrevive ao corte por janela. A formulação honesta é *"no agregado da
população medida, nenhum recorte bate o 1.200 com significância; decomposto pela
janela de origem da pergunta, a vantagem não é uniforme"*.

#### A RÉPLICA FOI FEITA — 06/10, e a decisão sobrevive

Quatro condições reindexadas sobre os **54 documentos** (não 15) e medidas nas
**225 respondíveis** (75 factuais, 75 multi-hop, 75 comparativas), com BGE-m3 em
GPU do Colab. 70.222 trechos embutidos. Scripts: `preparar_recorte_225.py`,
`colab_recorte_225.py` e o mesmo `analisar_recorte.py`, agora com `--dados`,
`--resultados` e `--agrupar-por tipo` — rodar o mesmo código nas duas populações é
o que torna a comparação entre elas válida.

| condição | orç. 2.000 | orç. 5.000 | orç. 10.000 | teto |
|---|---|---|---|---|
| T400 | 17% | 25% | 33% | 62% |
| **T1200** | **24%** | **33%** | **41%** | 93% |
| T3500 | 15% | 27% | 33% | 100% |
| parent-child 400/3500 | 21% | 29% | 40% | 100% |

**O T1200 é o melhor nos três orçamentos.** A objeção de que ele havia sido
escolhido numa população sem comparativas **não derruba a escolha**.

Os níveis NÃO são comparáveis aos do experimento de setembro (84% contra 33% a
5.000), por dois motivos de desenho: o corpus foi de 15 para 54 documentos, o que
multiplica os distratores, e a população passou a conter 75 comparativas que quase
tudo erra. O que se compara entre os dois experimentos é a ORDEM, não o nível.

**Mas a vitória é uma vitória de pergunta factual, e só.** McNemar dentro de cada
estrato, n = 75 em cada, no orçamento de 5.000:

| estrato | T1200 × T400 | T1200 × T3500 | T1200 × PC |
|---|---|---|---|
| **factual** | 53 vs 36, **p = 0,0033** | 53 vs 39, **p = 0,0094** | 53 vs 47, p = 0,26 |
| multi-hop | 19 vs 16, p = 0,55 | 19 vs 22, p = 0,63 | 19 vs 19, p = 1,00 |
| comparativa | 2 vs 5, p = 0,45 | 2 vs 0, p = 0,50 | 2 vs 0, p = 0,50 |

O padrão repete nos três orçamentos: na factual o T1200 ganha dos dois tamanhos
fixos com significância em **todos** eles (p de 0,0009 a 0,0094); na multi-hop e
na comparativa **nenhum par é significativo em nenhum orçamento**.

**A leitura, que é melhor do que "a escolha estava errada":** nas comparativas
todas as condições ficam entre 0 e 13 de 75, e a diferença entre elas é ruído. O
tamanho do trecho **não é a alavanca** daquele estrato — nem a favor nem contra o
1.200. Ter medido com as comparativas dentro não mudaria a decisão, porque dentro
delas não há decisão a tomar. O que muda é a frase: o 1.200 é a melhor escolha
medida, e o que a sustenta é a pergunta factual.

**O parent-child empata com o T1200 em tudo** — nenhuma comparação significativa
em nenhum orçamento nem em nenhum estrato. É o mesmo resultado de setembro, agora
num corpus 3,6 vezes maior e com os tipos difíceis presentes.

#### Por dispersão da evidência: é aqui que o mecanismo aparece

O `recorte_resultados.json` da réplica guarda o acerto **por pergunta**, então os
cortes abaixo não custaram GPU nenhuma. Acertos no orçamento de 5.000:

| recorte | mesmo trecho (28) | vizinho (19) | d=2..20 (27) | **disperso, d>20 (107)** | 1 citação (44) |
|---|---|---|---|---|---|
| T400 | 9 | 7 | 7 | 8 | 26 |
| **T1200** | **23** | 11 | 9 | 3 | **28** |
| T3500 | 17 | 10 | 11 | 2 | 21 |
| parent-child | 19 | 10 | 10 | 1 | 26 |

**Quando as duas citações caem no mesmo trecho, o T1200 arrasa o T400: 23 contra
9, 0 a 14 nos discordantes, p = 0,0001** — e o padrão repete nos três orçamentos
(p = 0,0034 / 0,0001 / 0,0010). O mecanismo é aritmético, não estatístico: um
trecho de 1.200 contém as duas citações e um de 400 as parte.

**Nas 107 perguntas de evidência dispersa — quase metade do conjunto — nenhum
recorte se distingue.** Todas as condições ficam entre 1 e 13 acertos, e o T400,
que fica à frente em pontos no orçamento de 5.000 (8 contra 3), não resiste ao
teste (7 a 2, p = 0,18) nem mantém a vantagem a 10.000 (13 contra 12, p = 1,00).

**A síntese:** o tamanho do trecho é alavanca **onde a evidência já está
reunida**, e não é alavanca nenhuma onde ela se espalha — que é justamente onde o
sistema falha. Isto amarra o resultado do recorte ao eixo de dificuldade que a
seção 5 estabelece: dispersão, não tipo.

#### Por que o T400 parece ganhar na comparativa — cobertura contra conversão

Na coluna da comparativa o T400 tem 5 acertos contra 2 do T1200, e a diferença
não passa no teste (5 a 2, p = 0,45). Mas o motivo da aparência é instrutivo e
foi medido. Uma comparativa é impossível se o conjunto devolvido não contiver
**todos os documentos** que ela exige, então contamos isso, no orçamento de
5.000:

| recorte | unidades devolvidas (mediana) | documentos distintos (mediana) | traz TODOS os documentos exigidos | acerta a pergunta | converte |
|---|---|---|---|---|---|
| T400 | 16 | 3 | **42 de 75** | 5 | 12% |
| T1200 | 5 | 2 | 18 de 75 | 2 | 11% |
| T3500 | 2 | 1 | 6 de 75 | 0 | 0% |
| parent-child | 2 | 1 | 6 de 75 | 0 | 0% |

**O T400 domina a cobertura: 42 contra 18, com 24 a 0 nos discordantes e
p < 0,000001** — nunca perde. É aritmética de orçamento: em 5.000 caracteres
cabem 16 trechos de 400 e só 5 de 1.200, e 16 trechos alcançam 3 documentos
distintos contra 2.

**E a taxa de conversão é a mesma: 12% contra 11%.** Dos 42 que o T400 cobre, ele
responde 5. O T400 não é melhor na comparativa — ele tem uma base de cobertura
duas vezes maior e a mesma conversão ruim.

**Isto localiza o problema da comparativa, e o lugar não é o recorte.** Trazer os
documentos certos não é o gargalo: o gargalo é caber as citações **inteiras** no
orçamento depois de trazê-los. Nenhum tamanho da grade resolve as duas coisas ao
mesmo tempo — trecho pequeno alcança mais documentos e parte a citação (teto de
62%), trecho grande preserva a citação e alcança menos documentos. É o mesmo
impasse que a decomposição de consulta tentou atacar pela montagem do top-5, e
que também não cedeu.

**Por idioma, o mesmo padrão.** Acertos a 5.000, com McNemar T400 × T1200:

| recorte | inglês (68) | português (94) | duas línguas (63) |
|---|---|---|---|
| T400 | 18 | 34 | 5 |
| **T1200** | **27** | **45** | 2 |
| T3500 | 22 | 39 | 0 |
| parent-child | 20 | 46 | 0 |

Em português o T1200 ganha do T400 com p = 0,0072 a 2.000 e 0,0347 a 5.000. Nas
63 perguntas de duas línguas — quase todas comparativas cruzando norma brasileira
com internacional — nenhum recorte se distingue. Um detalhe para fevereiro: a
10.000 em português o parent-child passa à frente (59 contra 51, 11 a 3,
p = 0,0574), o único lugar em toda a réplica onde ele chega perto de ganhar.

Ficaram fora da réplica o PC1200 e as três variantes de sobreposição: mais 57 mil
vetores de GPU para reconfirmar dois achados que ninguém contestou. O
`--condicoes` do `preparar_recorte_225.py` os inclui quando houver motivo.

**A sobreposição não faz diferença medível.** Com tamanho fixo em 1.200,
sobreposição 0, 120, 240 e 400 não se distinguem (p = 0,690 / 1,000 / 0,581
contra o padrão 240).

## 2. Como medimos

**Complete Evidence@5** é a métrica principal: a pergunta só conta como acerto se
**todas** as citações que ela exige aparecerem nos 5 primeiros trechos. No código
a função se chama `complete_evidence_at_k`; relatórios anteriores a chamavam
"Junta@5" — é a mesma coisa.

É severa de propósito. A métrica frouxa — "veio ao menos um trecho relevante" —
dá 0,822 onde a severa dá 0,538 no mesmo braço. Num sistema que cita norma, meia
evidência é resposta errada.

Três réguas, lado a lado, porque confundi-las já custou meses aqui:

| régua | definição | BGE+reranker |
|---|---|---|
| **Complete Evidence@5** | 1 se **todas** as citações chegaram | **0,538** |
| **Completude@5** | **fração** das citações que chegaram | 0,679 |
| **Recall@5 (frouxo)** | 1 se **ao menos uma** chegou | 0,822 |

Uma citação vale se **qualquer** trecho que a contenha entrar no top-k, porque a
sobreposição de 240 caracteres faz a mesma citação aparecer em dois trechos
vizinhos. Exigir os dois puniria o sistema por um detalhe do recorte — afetava
15,6% das perguntas antes de ser corrigido.

### O nome da métrica, conferido na literatura (30/09/2026)

Esta métrica já se chamou **Junta@5** (jargão nosso, intraduzível) e, por uns
dias, **Strict Recall@5**. **"Strict Recall" está errado para o nosso caso, e
erra na direção mais perigosa.**

Em RI, a dupla *strict* / *relaxed recall* vem de Karpukhin et al. (2020), o
artigo do DPR, e distingue **como se julga um acerto**, não quantas evidências
se exigem:

| termo na literatura | o que exige |
|---|---|
| **strict recall@k** | o trecho-ouro **identificado** tem de estar no top-k |
| **relaxed recall@k** | basta a *string* da resposta aparecer em algum trecho |

O nosso casamento é deliberadamente do tipo **relaxed**: aceitamos **qualquer**
trecho que contenha a citação, porque a sobreposição de 240 caracteres faz a
mesma citação aparecer em dois vizinhos. Chamar isso de "strict" promete
exatamente o que não fazemos — um revisor leria "casamento exato de identificador
de trecho" e nos cobraria por isso.

**O ancoradouro certo é a literatura de multi-hop**, onde "todas as evidências no
top-k" é métrica estabelecida sob dois nomes:

- **Retrieval EM** / **Passage Exact Match (P-EM)** — "whether both gold passages
  are included in the retrieved passages" (Xiong et al., 2021, *Multi-Hop Dense
  Retrieval*); usada também em Baleen e em Beam Retrieval.
- **Supporting-Fact EM** — a formulação do HotpotQA.

A diferença entre a nossa e a deles é só de generalidade: elas fixam *duas*
passagens e nós admitimos **de uma a quatro** citações, em k variável.

**Recomendação: `Complete Evidence@k`**, que é como a função já se chama no código
(`complete_evidence_at_k`). É autoexplicativo, não colide com termo alheio, e não
reivindica prioridade sobre nada. Defina no primeiro uso, em uma frase, dizendo
as duas metades:

> *Complete Evidence@k: a consulta conta como acerto apenas se **todas** as
> citações exigidas aparecerem no top-k. Uma citação é considerada recuperada se
> **qualquer** trecho que a contenha estiver no top-k — casamento no nível da
> citação, não do trecho, porque a sobreposição do recorte duplica citações.
> É o Retrieval EM / passage exact match da literatura de multi-hop (Xiong et
> al., 2021), generalizado para k e para um número variável de evidências.*

Com essa frase, "estrito na pergunta, frouxo no casamento" fica dito, e é
justamente a combinação que um revisor cobraria se ficasse implícita.

*Se preferir manter "Strict Recall@k" por já estar em slides ou rascunho, então é
obrigatório dizer na definição que o casamento é no nível da citação — senão o
nome contradiz o método.*

**O gabarito não é anotado em identificador de trecho.** Cada pergunta guarda o
**texto literal** da norma que a responde, e os qrels são derivados localizando
esse texto no corpus em uso. Trocar o recorte e rodar de novo dá o gabarito
correto para o recorte novo — é o que torna o benchmark reutilizável.

## 3. Resultados

### Os quatro braços, 225 perguntas balanceadas

| braço | Complete Evidence@5 | Completude@5 | frouxo@5 | CE@10 | CE@100 (teto) |
|---|---|---|---|---|---|
| Qwen3-Embedding-0.6B | 0,356 | 0,486 | 0,627 | 0,436 | 0,791 |
| BGE-m3 | 0,387 | 0,547 | 0,716 | 0,498 | 0,836 |
| Gemini | 0,489 | 0,647 | 0,813 | 0,604 | **0,924** |
| **BGE-m3 + reranker** | **0,538** | **0,679** | **0,822** | **0,636** | 0,836 |

**O Qwen entrou em 05/10 e fecha a escolha do embedding aberto.** Antes, o BGE-m3
era o aberto por decisão; agora é por medição. Os dois rodam no **mesmo regime** —
matriz local, busca exata, mesmo corpus, mesmo fingerprint —, então a comparação
entre eles é a mais limpa da tabela, sem a ressalva de índice que vale contra o
Gemini (ver seção 8).

O teto do Qwen é **0,791** contra 0,836 do BGE: o pool dele contém a evidência
com menos frequência, então nem reordenar salvaria a diferença.

`CE@100` é o teto: a fração de perguntas cuja evidência inteira está nos 100
primeiros. Nenhuma reordenação passa disso. O teto do BGE+reranker é idêntico ao
do BGE-m3 porque o reranker reordena **exatamente** o top-100 do BGE — conferido
nas 225. Reordenar não acrescenta documento.

**BM25 e fusão RRF saíram desta tabela** (decisão de 30/09). São resultado do
artigo do WebMedia, não são candidatos a produção aqui, e ocupavam espaço numa
comparação que é sobre aberto contra proprietário. Os números ficam na seção 4,
onde sustentam a recomendação de não adotá-los. O BM25 medido é o **sem
tradução** (`eval/config.yaml:99`) — é a linha de base léxica implantada. O
ranking traduzido em disco (`A_bm25_mt.json`) é de **31/08**, anterior à migração
de ids de 12/09, e está morto: não use.

### McNemar exato, Complete Evidence@5

| comparação | vitórias | derrotas | discordantes | p |
|---|---|---|---|---|
| **Qwen3-0.6B vs BGE-m3** | **8** | **15** | **23** | **0,2100** |
| Qwen3-0.6B vs Gemini | 9 | 39 | 48 | < 0,0001 |
| Qwen3-0.6B vs BGE+reranker | 5 | 46 | 51 | < 0,0001 |
| BGE-m3 vs Gemini | 9 | 32 | 41 | 0,0004 |
| BGE-m3 vs BGE+reranker | 3 | 37 | 40 | < 0,0001 |
| **Gemini vs BGE+reranker** | **13** | **24** | **37** | **0,0989** |

"Vitórias" = a primeira acerta e a segunda não. Só os pares **discordantes**
entram: uma pergunta que os dois braços acertam não informa nada sobre qual é
melhor, e é por isso que o n aqui é menor que 225.

**Entre os dois abertos a diferença não é significativa** (8 a 15, p = 0,2100).
O ponto estimado favorece o BGE-m3 em 3,1 pontos e todos os cortes apontam na
mesma direção, mas a formulação defensável é "o BGE-m3 fica à frente, sem
significância" — não "o BGE-m3 supera o Qwen".

**Na população balanceada, a frase certa é "alcança", não "supera".** O ponto
estimado favorece o aberto em 4,9 pontos, mas p = 0,0989 sustenta
indistinguibilidade, não superioridade. Escrever "supera" entrega ao revisor a
única brecha do resultado. **Ver a seção 8: no acervo completo o mesmo par dá
p = 0,0055, e a diferença é a composição do benchmark.**

### Por idioma

| estrato | n | Qwen3-0.6B | BGE-m3 | Gemini | BGE+reranker | Gemini × BGE+rr | p |
|---|---|---|---|---|---|---|---|
| português | 94 | 48 (0,511) | 52 (0,553) | 63 (0,670) | **72 (0,766)** | 3 a 12 | **0,0352** |
| inglês | 68 | 29 (0,426) | 30 (0,441) | **40 (0,588)** | 37 (0,544) | 8 a 5 | 0,5811 |
| duas línguas | 63 | 3 (0,048) | 5 (0,079) | 7 (0,111) | **12 (0,190)** | 2 a 7 | 0,1797 |

**Como ler.** `n` é quantas perguntas do estrato entraram — 94 com pergunta e
evidência em português, 68 em inglês, 63 que cruzam as duas línguas (típico da
comparativa: uma norma brasileira e uma internacional). As células dos braços são
**quantas dessas n o braço acertou**, com a fração entre parênteses: 72 de 94 =
0,766. A coluna `Gemini × BGE+rr` conta os **pares discordantes** na ordem
*Gemini acerta e o aberto não* **a** *o aberto acerta e o Gemini não* — "3 a 12"
quer dizer que o aberto ganhou 12 perguntas que o Gemini perdeu e perdeu 3 que o
Gemini ganhou. O `p` é o McNemar exato desse par.

**É em português que o aberto ganha, e ali ele ganha com significância**
(p = 0,0352). **Em inglês o proprietário fica à frente** (40 contra 37), sem
significância. Quem operar em inglês não deve esperar ganho da troca.

### Por tipo de pergunta

| tipo | n | Qwen3-0.6B | BGE-m3 | Gemini | BGE+reranker | Gemini × BGE+rr | p |
|---|---|---|---|---|---|---|---|
| factual | 75 | 60 (0,800) | 61 (0,813) | 66 (0,880) | **70 (0,933)** | 2 a 6 | 0,2891 |
| multi-hop | 75 | 16 (0,213) | 20 (0,267) | 35 (0,467) | **38 (0,507)** | 8 a 11 | 0,6476 |
| comparativa | 75 | 4 (0,053) | 6 (0,080) | 9 (0,120) | **13 (0,173)** | 3 a 7 | 0,3438 |

**Onde os dois abertos se separam.** Na factual eles empatam (60 contra 61); a
distância abre na multi-hop (16 contra 20) e na comparativa (4 contra 6). O Qwen
não é pior em achar *um* trecho — é pior em trazer *os dois*, e isso aparece no
`frouxo@5`: 0,627 contra 0,716, a maior diferença entre eles em qualquer coluna.

Nenhum estrato de tipo, isolado, distingue o aberto do proprietário. A
comparativa é o buraco, e é estrutural: em **27 das 75** o trecho necessário não
está nem entre os 100 recuperados — nenhuma reordenação alcança.

### Por dispersão da evidência

Dispersão = distância, em trechos do corpus, entre a primeira e a última citação
de uma pergunta. É propriedade do documento, não do rótulo.

| dispersão | n | Qwen3-0.6B | BGE-m3 | Gemini | BGE+reranker |
|---|---|---|---|---|---|
| 1 citação só | 44 | **0,773** | 0,727 | 0,841 | **0,909** |
| d = 0 (mesmo trecho) | 28 | 0,857 | 0,964 | 0,964 | **1,000** |
| d = 1 (vizinho) | 19 | 0,421 | 0,526 | 0,632 | **0,684** |
| d = 2 a 20 | 27 | 0,222 | 0,222 | 0,481 | **0,593** |
| d > 20 (disperso) | 107 | 0,075 | 0,112 | 0,196 | **0,224** |

Repare na primeira linha: **o Qwen é o melhor dos dois abertos quando a pergunta
exige uma citação só** (0,773 contra 0,727) e desaba conforme a evidência se
espalha, até 0,075 em `d > 20`. É a confirmação mais direta de que a dispersão,
e não o tipo da pergunta, é o eixo que separa os sistemas — e serve de aviso: um
modelo escolhido por desempenho em pergunta factual pode ser o pior justamente
onde o acervo é mais difícil.

Medido na população balanceada de 225 em 30/09
(`analisar_bracos.py --gabarito eval/data/golden_dezembro.jsonl`). Três leituras:

- **A dispersão é o eixo de dificuldade, não o tipo.** De 1,000 em `d = 0` a
  0,224 em `d > 20`, com os três braços na mesma ordem: a distância entre as
  citações explica o desempenho melhor que o rótulo da pergunta.
- **O reranker ganha onde o material já estava no pool.** A faixa `d = 2..20`
  quase triplica (0,222 → 0,593) — eram trechos recuperados e mal ordenados, que
  é exatamente o que reordenar conserta.
- **`d > 20` é onde todo mundo colapsa**, e são 107 das 225 perguntas — quase
  metade. É onde vive a comparativa, e nenhum dos três braços passa de 0,224.
  Também é onde `d = 0` expõe o problema de rotulagem que corrigimos: ali o
  BGE+reranker acerta **todas as 28**, então uma pergunta "multi-hop" cujas
  citações caem no mesmo trecho não mede nada.

### Onde ainda cabe ganho

| | |
|---|---|
| Complete Evidence@5 hoje | 0,538 (121 de 225) |
| **teto: tudo o que está no pool de 100** | **0,836 (188 de 225)** |
| perguntas com tudo no pool e **não** no top-5 | **67** |
| perguntas sem o material no pool | 37 |

Os três números fecham: 121 + 67 + 37 = 225. O teto de um top-5 coincide com o
`CE@100` porque **nenhuma pergunta do acervo exige mais de 4 citações** — então
cabe sempre em cinco vagas.

Há **67 perguntas** em que a evidência inteira já foi recuperada e o corte em 5
deixou algo fora. Esse é o orçamento de qualquer trabalho de ordenação — e
ninguém, nem nós nem as técnicas conhecidas que testamos, conseguiu sacá-lo.

## 4. O que testamos e NÃO funcionou

Esta seção existe para ninguém repetir o caminho. **Leia antes de propor
intervenção de recuperação.**

### Nesta rodada (dezembro), corpus de 6.134 trechos

| tentativa | resultado |
|---|---|
| **BM25 sem tradução** | 0,232 geral e **0,000 de 86** no estrato em inglês — cego a um terço do corpus |
| **Fusão RRF (BM25 + Gemini)** | **piora** o Gemini de 0,521 para 0,393. Peso igual a um braço cego em um terço das perguntas contamina, não funde |
| **Teto de trechos por documento** | troca comparativa por multi-hop em qualquer valor. Geral 0,596 → 0,529 com teto 3 |
| **MMR (λ 0,9 a 0,5)** | move uma pergunta. 0,596 → 0,542 no melhor caso |
| **Recusar trechos vizinhos** | **pior de todas**: multi-hop cai de 38 para 30. Motivo: 17 multi-hop exigem exatamente dois trechos vizinhos |
| **Decomposição + score global** | 13 de 75 — idêntico à consulta única |
| **Decomposição + RRF** | 6 de 75 — pior que a base |

*As cinco primeiras linhas foram medidas na população de 267 do acervo completo,
antes do balanceamento; as duas de decomposição, nas 75 comparativas. Nenhuma
chega perto de mudar de sinal com a troca de população, mas o número exato é o da
época.*

A razão de a diversificação falhar sempre é **conflito de requisito**, não ajuste
de parâmetro: a comparativa precisa de dois documentos no top-5, a multi-hop
precisa de duas regiões do **mesmo** documento. Um teto por documento torna a
segunda impossível por construção.

### Rodada anterior (WebMedia), corpus de 5.160 trechos

Em `eval/tools/experimentos_descartados/`, cada script com a hipótese e o número
que a derrubou no cabeçalho. **Os números são da época** — o acervo foi
reextraído em 13/09 e os ids mudaram; para reexecutar, refaça a medição.

| hipótese | veredito e número |
|---|---|
| preencher `section_path` no prefixo de embedding melhora a recuperação | **não** — posição mediana 8 com a seção, 6 sem (n = 12) |
| cortar por unidade normativa (artigo, seção) em vez de janela fixa | **inviável** — só 34 de 63 documentos têm marcador regular, e os 29 sem são justamente NIST/WHO/UNESCO/OECD/FDA/ISO; 49% das unidades ficariam abaixo de 300 caracteres |
| reservar 3+2 vagas do top-5, uma partição por lado | **não** — 13/25 caiu para 11/25 |
| identificar os dois documentos e buscar dentro de cada um | **teto de 6/25 mesmo com roteador-oráculo** |
| decompor a consulta e reservar vaga por sub-pergunta | **no-op** — 22 das 25 seleções idênticas à base |
| decomposição de consulta + fusão RRF | recall sobe, top-5 não muda (p = 0,62) |
| pergunta + tradução EN + as 2 sub-perguntas numa consulta só | **não** — nenhuma variante move a comparativa |
| teto de N trechos por documento no top-5 | **não** — derruba multi-hop de 31,2% para 12,5% |
| PRF/Rocchio denso | **não** — piora a comparativa em **todas** as 12 configurações |

**Duas destas foram redescobertas por acidente nesta rodada** — o teto por
documento e a decomposição com reserva de vaga. O custo foi de dias.

### Três réguas circulando, e o erro de comparar entre elas

| régua | significa | base (comparativas, época do WebMedia) |
|---|---|---|
| **A — os dois trechos** | as duas âncoras no top-5 | 2/25 |
| **B — os dois documentos** | cada documento exigido com *algum* trecho relevante | 13/25 |
| **C — posição da âncora** | onde ela cai *dentro do próprio documento* | mediana 8 |

A régua **A** é a que decide se a pergunta foi respondida. A **B** é frouxa: um
trecho qualquer do documento certo não é o trecho que responde. A **C** é
diagnóstica e **não diz nada sobre o top-5 global** — foi confiar nela que fez a
multiplexação de consulta parecer ganho por meses.

## 5. O que funcionou

### Reranker cross-encoder — e o que ele custa de verdade

`BAAI/bge-reranker-v2-m3`, aberto, sobre o pool que o BGE já trouxe:

| pool | Complete Evidence@5 | acertos de 225 | pares | CPU por consulta |
|---|---|---|---|---|
| sem reranker | 0,387 | 87 | 0 | — |
| 10 | 0,444 | 100 | 10 | ~25 s |
| 20 | 0,502 | 113 | 20 | **~50 s** |
| 50 | 0,520 | 117 | 50 | ~125 s |
| 100 | **0,538** | **121** | 100 | **~250 s** |

> **CORREÇÃO IMPORTANTE.** A versão anterior deste relatório dizia "~4 s de CPU"
> para o pool 20 e "~20 s" para o pool 100. **Está errado por cerca de 12×.**
> Medido em 30/09 nesta máquina: **100 pares em 250 s, ou 2,499 s por par**
> (`completar_ranking.py`, que imprime a taxa). Máquina: Intel i7-1260P, 12
> núcleos / 16 threads, 15,7 GB, `torch 2.13.0+cpu`, `batch_size=16`,
> `max_length=384`, sem quantização. O número bate com a observação de 28 horas
> para os 39.200 pares da rodada completa, que a latência antiga contradizia.
>
> Uma implantação otimizada (ONNX, int8, mais threads) plausivelmente ganha 2–4×,
> o que põe o pool 20 em 12–25 s. **Ainda não é interativo**, então a conclusão
> "precisa de GPU para uso interativo" é robusta a essa otimização.

Piora **3** consultas e melhora **37**. `CE@100` fica idêntico em todos os pools,
o que confirma que o ganho é de ordem.

### Decompor a pergunta sem saber o tipo dela — o resultado novo

Esta é a peça que faltava. A decomposição que funcionou em 29/09 (comparativa
13 → 21 de 75) só era aplicável porque **alguém já sabia que a pergunta era
comparativa**: o prompt anunciava "esta pergunta compara duas normas". Produção
não sabe. Enquanto a decisão de decompor vier do rótulo do gabarito, o ganho é
truque de oráculo, não método.

Em 30/09 o `gpt-oss-120b` recebeu **só o enunciado** das 225 respondíveis — sem o
tipo, sem o gabarito — e decidiu sozinho se a pergunta pede uma informação ou
duas. Uma chamada de LLM por consulta, nenhuma GPU.

| o gabarito exige | o modelo disse 1 parte | o modelo disse 2 partes |
|---|---|---|
| 1 citação (44 perguntas) | **40** | 4 |
| 2 ou mais (181 perguntas) | 19 | **162** |

**Concordância: 202 de 225 = 89,8%**, a partir do texto da pergunta apenas. E o
erro é **assimétrico no lado barato**:

- **4 perguntas divididas sem precisar** — custo: uma vaga do top-5 gasta com um
  lado que não existe.
- **19 não divididas quando precisava** — custo: nenhum. Cai de volta na consulta
  única, que é o sistema de hoje.

O que o modelo decidiu, por tipo:

| tipo | n | 1 parte | 2 partes | % dividida |
|---|---|---|---|---|
| factual | 75 | 49 | 26 | 35% |
| multi-hop | 75 | 7 | 68 | **91%** |
| comparativa | 75 | 3 | 72 | **96%** |

**Isto dissolve a objeção "produção não sabe o tipo" — para a DECISÃO.** Decidir
se a pergunta pede uma informação ou duas é inferível do enunciado, com ~90% de
acerto e com o erro caindo do lado inofensivo. A seção seguinte mostra que isso
não basta: o que o oráculo também dava, e que o enunciado não entrega, é o
**conteúdo** do corte.

#### O sistema sempre-ligado, medido — e o resultado é negativo

As 166 perguntas que o modelo dividiu tiveram os dois pools reranqueados em GPU
(16.600 pares). Nas 225:

| montagem | acertos | CE@5 | discordantes | p |
|---|---|---|---|---|
| consulta única reranqueada | 121 de 225 | **0,538** | — | — |
| alternância estrita | 124 de 225 | 0,551 | 13 a 16 | 0,7111 |
| reserva de vaga | 120 de 225 | 0,533 | 17 a 16 | 1,0000 |
| score global | 114 de 225 | 0,507 | 19 a 12 | 0,2810 |

**A decomposição sempre-ligada não move a métrica.** A melhor montagem ganha 3
perguntas em 225 e não resiste a um teste. Restrito às 166 divididas, 74 → 77.
Juntar os dois lados por nota é **pior** que não dividir (74 → 67): a nota do
cross-encoder não é comparável entre consultas diferentes.

Por tipo, com a alternância estrita:

| tipo | n | dividida | única | decomposta | ganho | disc. | p |
|---|---|---|---|---|---|---|---|
| factual | 75 | 26 | 70 | 67 | **−3** | 3 a 0 | 0,2500 |
| multi-hop | 75 | 68 | 38 | 42 | +4 | 5 a 9 | 0,4240 |
| comparativa | 75 | 72 | 13 | 15 | +2 | 5 a 7 | 0,7744 |

Na factual a decomposição **nunca ganha e perde três** — 3 a 0 nos discordantes.
São as 26 que o modelo dividiu sem precisar, e a consulta única já acertava todas
as 26 no top-5. É custo puro, e é o argumento mais forte para o decompositor
errar para o lado de não dividir.

#### Por que o ganho não veio: não é a decisão, é o corte

A comparação que isola a causa: o experimento de 29/09 — o do oráculo, cujo
prompt anunciava "esta pergunta compara duas normas" — rodado **no mesmo
orçamento de 50+50**, nas 72 comparativas em que as duas versões existem:

| | acertos | McNemar contra a consulta única |
|---|---|---|
| consulta única | 13 de 72 | — |
| decomposição **cega** | 15 de 72 | 7 a 5, p = 0,7744 |
| decomposição **com o tipo** | **21 de 72** | **10 a 2, p = 0,0386** |

E o oráculo contra a cega, direto: **6 a 0, p = 0,0312** — ganha seis e não perde
nenhuma.

**Saber que a pergunta é comparativa vale 6 perguntas em 72, com significância.
A versão cega não captura nada disso.** O oráculo dá 21 tanto em 50+50 quanto em
100+100, então não é orçamento; e os dois chegam ao mesmo teto de material (52 de
75 contra 51 de 72), então não é recuperação. É montagem.

**O mecanismo NÃO está identificado, e vale dizer isso em vez de inventá-lo.** A
hipótese natural — que a versão cega corta a pergunta "pelo lado errado", gerando
duas sub-perguntas quase iguais — foi medida e **não se sustenta**: a semelhança
textual mediana entre sub1 e sub2 é **0,74 no oráculo contra 0,66 na cega**, ou
seja, as do oráculo são *mais* parecidas entre si. Filtrar os cortes
quase-idênticos (não dividir quando a semelhança passa de ~0,45) leva o sistema
de 121 para 126 em 225, mas o limiar foi escolhido na mesma amostra e p = 0,18.

Olhando as 6 perguntas uma a uma, só uma tem defeito nomeável: em q0076 a
versão cega **perdeu um aspecto exigido** — a pergunta pede "transparência e
responsabilização" e a sub-pergunta 1 ficou só com transparência. Nas outras o
texto cego é igual ou melhor e perde mesmo assim: em q0096 e c044 ele nomeia o
instrumento com mais precisão que o oráculo, e em c011 o oráculo escreve
literalmente "a Norma X" e "a Norma Y" — e ganha.

A única diferença sistemática que aparece é o **hedge**: 17 das 144
sub-perguntas cegas trocam o nome da norma por "legislação brasileira",
"normativa vigente" ou "legislação aplicável", contra 3 das 150 do oráculo. Não
foi possível ligar isso às perguntas que viraram, então fica como pista, não como
causa.

Os tetos, na mesma população dos dois lados (só as divididas, pool de 100 da
consulta única contra a união 50+50 da decomposta):

| tipo | n | teto da única | teto da decomposta | leitura |
|---|---|---|---|---|
| factual | 26 | 1,00 | 1,00 | dividir não tem o que acrescentar |
| multi-hop | 68 | 0,87 | **0,78** | dividir **perde** material, e ainda assim ganha +4 |
| comparativa | 72 | 0,65 | **0,71** | dividir **acha** material — o único lugar onde acha |

A comparativa é o caso claro: **47 das 72 têm o material no pool da própria
consulta única, e só 13 chegam ao top-5.** O oráculo entrega 21 do mesmo
material. O gargalo é ordem, não recuperação.

**O próximo passo decorre do que o oráculo tinha e a versão cega não: a
identificação dos dois LADOS.** O decisor cego já marca 96% das comparativas como
"duas partes", então ele reconhece a estrutura; o que falta é obrigá-lo a dizer
*quais* são os dois lados antes de escrever as sub-perguntas, em vez de partir o
enunciado direto. Isso é uma segunda passada do mesmo LLM, não uma rodada de GPU,
e é testável sem tocar no gabarito. **Mas é hipótese**: a inspeção das 6
perguntas não explica por que o oráculo ganha, e um experimento assim pode voltar
sem ganho.

### Abstenção pela nota do reranker

A nota separa respondível de irrespondível com **AUC 0,845** no acervo completo
(mediana 0,960 contra 0,458).

*O que é AUC.* É a probabilidade de que uma pergunta **respondível** sorteada ao
acaso receba nota maior que uma **irrespondível** sorteada ao acaso (empates
contam meio). 0,5 significa que a nota não distingue nada — o mesmo que jogar
moeda; 1,0 é separação perfeita. **0,845 quer dizer que, em 84,5% dos pares
sorteados, a respondível tem nota mais alta.** A vantagem do AUC é não depender
de limiar: resume a ordenação inteira. É por isso que ele vem *ao lado* de um
limiar específico, e não em vez dele — o AUC diz que existe sinal, o limiar diz
onde cortar.

| | acervo completo (268 + 125) | subconjunto balanceado (225 + 75) |
|---|---|---|
| AUC | **0,845** | 0,845 |
| limiar escolhido na metade A | **0,574** | 0,368 |
| recusa correta, medida na metade B | **68,3%** | 55,3% |
| recusa indevida, medida na metade B | **11,9%** | 3,5% |

**Use a coluna do acervo completo**: ela tem 125 irrespondíveis contra 75, e o
limiar é estimado em metades de 62/63 em vez de 37/38. O subconjunto serve de
verificação de sanidade — o AUC é o mesmo (0,8448 contra 0,8447), o que mostra que
o sinal não depende da composição; só a escolha do corte fica mais ruidosa com
menos dados.

O critério foi "maior recusa correta com perda ≤ 10%", escolhido na metade A e
medido na B, com semente fixa. Na metade A o limiar de 0,574 dava 58,1% de recusa
correta; na B, os 68,3% relatados. A diferença é **negativa** (−10,2 pontos):
não há sinal de sobreajuste, a metade B saiu por acaso mais fácil.

Isso substitui o `score_threshold = 0,60` que havia em produção e foi removido em
26/09. Aquele era similaridade de cosseno — sem escala absoluta, nunca medido, e
a avaliação nunca o aplicava, então o número media um sistema e a produção servia
outro. Este é a saída de um modelo treinado para julgar o par, com limiar
escolhido e validado em metades separadas.

## 6. Recomendação de produção

**A decisão depende de uma pergunta de infraestrutura, não de qualidade: existe
GPU?**

### Com GPU
1. **Recuperação:** BGE-m3 denso + `bge-reranker-v2-m3`, **pool 100**. Strict
   Recall@5 de 0,538 contra 0,489 do Gemini, zero de API, e a consulta do usuário
   não sai da máquina.
2. **Abstenção:** recusar quando a melhor nota do reranker < 0,574.

### Sem GPU
1. **Manter o Gemini.** 0,489 a ~1–2 s de latência de API bate qualquer ponto de
   operação do aberto em CPU: o pool 20 só chega a 0,502 gastando ~50 s.
2. Se a saída de dados para a API for inaceitável por política, a alternativa é
   **BGE-m3 sem reranker** — 0,387, ~0,1 s, local. É uma perda de 10 pontos
   assumida em troca de privacidade, e precisa ser decidida como tal.

### Em qualquer caso
3. **Recorte:** manter **1.200/240**. `CHUNK_OVERLAP = 0` daria índice 28% menor
   (2.254 contra 3.146 trechos na amostra do experimento) sem perda medida — mas
   **mantemos a sobreposição**, e a razão é o limite do teste: ele foi feito com
   perguntas sintéticas, cuja citação foi extraída de dentro de um trecho e
   portanto raramente cai numa fronteira. Pergunta real de usuário não tem esse
   viés e pode pedir justamente o que atravessa a borda. A economia é de custo de
   índice, não de qualidade, e não vale correr o risco antes de medir com
   consulta de produção.
4. **Não adotar:** BM25, fusão RRF, diversificação do top-5.

**Uma alternativa de teto mais alto que não foi medida.** O `CE@100` do Gemini é
**0,924** contra 0,836 do BGE: o pool do Gemini contém a evidência com mais
frequência, e o reranker não conserta o que não está no pool. Um arranjo "pool do
Gemini + reranker aberto" teria teto **9 pontos** maior que o recomendado aqui.
Ficou fora **da comparação do artigo** de propósito, para a tese comparar contra
a produção como ela é — mas para produção pode haver ganho real, ao preço de
manter a dependência de API. Fechar esse número custa um lote de Colab.

**Risco operacional — resolvido em 30/09.** O `src/build_vectorstore.py` chamava
`recreate_collection` na coleção que a produção serve: apagava tudo e só então
começava a subir os 6.134 pontos, de modo que qualquer queda no meio deixava o
índice parcial, funcionando e sem aviso. Agora a indexação sobe numa coleção
**versionada pelo fingerprint do corpus**, verifica contagem, conjunto de ids e
perfil de embedding, e só então move o **apelido** servido — operação atômica no
Qdrant. Se algo falhar, o apelido continua na coleção antiga e a produção não
percebe. Ids de ponto passaram a ser `uuid5` do id do trecho, o que torna o
upload retomável. A migração ainda não foi executada; ver a seção 9.

## 7. O que está aberto

| item | custo | expectativa |
|---|---|---|
| ~~prompt novo para o decompositor~~ **descartado em 05/10** | — | o teto da linha inteira é 130 de 225, **p = 0,1360**: o melhor caso concebível não passa no teste. O resíduo é problema de SELEÇÃO — escolher por pergunta daria 137 (p < 0,0001), e falta o sinal |
| auditoria humana dos top-5 | ~3 h de especialista por lote | fecha a lacuna do gabarito derivado |
| reranker mais rápido em CPU (ONNX/int8, ou `bge-reranker-base`) | 1 dia de engenharia | é o que decide se o aberto serve sem GPU |
| "pool do Gemini + reranker aberto" | 1 lote de Colab | teto 0,924 em vez de 0,836 |
| pool de reranking mais fundo (200, 500) | GPU | pouco: o teto do BGE é 0,836 e o pool de 100 já alcança |
| as 37 perguntas sem material no pool | desconhecido | é falha de recuperação, não de ordem |

**A auditoria humana está pronta para começar.** `montar_auditoria.py` gerou uma
amostra de **78 perguntas respondíveis** em 3 lotes, estratificada por tipo ×
acerto (metade o sistema acerta, metade erra — auditar só acertos mediria
satisfação), com os 5 trechos de cada uma **em ordem embaralhada** e sem nota do
modelo, sem marcação de gabarito e sem resposta-referência na planilha do juiz. O
lote 1 (30 perguntas, 150 julgamentos) já é amostra válida.

**Só respondíveis, decidido em 30/09.** A primeira versão incluía irrespondíveis
como controle e estava errada: para dizer "estes 5 trechos não respondem porque o
acervo não tem a resposta", o especialista teria de garantir que a resposta não
está em nenhuma das 54 normas — o que não se verifica lendo 5 trechos. Ele
responderia "não" por padrão, sem informação, ou "sim" por engano num trecho só
topicamente parecido. A abstenção já é validada por número, em metades separadas,
contra 125 irrespondíveis escritas de propósito; não precisa de juiz humano. O
que precisa de juiz é o que a planilha agora pergunta: **para uma pergunta que
tem resposta no acervo, o contexto recuperado basta?**

**Anotação de pool (julgamento de relevância).** Medimos se o gabarito derivado
subconta. Das 108 perguntas que o BGE+reranker erra no acervo completo, entre
**1 e 20** têm no top-5 um trecho não anotado que passa em dois indicadores
independentes (nota do reranker ≥ a do pior trecho anotado, e ≥ 30–60% dos termos
distintivos da resposta-referência). Lendo os 9 candidatos mais fortes, **2 são
falso negativo real** — o que põe o Complete Evidence@5 em 0,603 em vez de 0,596
naquela população. O envelope mecânico vai de 0,599 a 0,670, e a leitura indica
que o extremo otimista não se sustenta. Subestimação de 1 a 2 pontos, que não
muda nenhuma conclusão.

Um dos dois casos é instrutivo e estrutural: **normas citam normas.** O Espaço
Europeu de Dados de Saúde reafirma a regra de portabilidade do GDPR, o PL 2338
remete à LGPD, o Decreto 44.045 repete as penas da Lei 3.268. O mesmo fato existe
em vários documentos, e gabarito ancorado na citação de **um** documento
subconta por construção. Isso não se resolve mexendo no recorte — só julgando o
pool, e é a razão mais forte para fazê-lo no artefato do KDD.

## 8. Limitações a declarar

### A composição do benchmark decide a conclusão do artigo

Esta é a limitação mais importante, e é também o achado metodológico mais forte
do estudo. **Os mesmos sistemas, os mesmos rankings, a mesma métrica — e o
veredito muda com a mistura de tipos:**

| população | factual | multi-hop | comparativa | n | Gemini × BGE+rr | p | conclusão |
|---|---|---|---|---|---|---|---|
| **balanceada (a do artigo)** | 75 | 75 | 75 | 225 | 13 a 24 | **0,0989** | **alcança** |
| acervo completo | 118 | 75 | 75 | 267 | 14 a 34 | **0,0055** | **supera** |

A causa é direta e verificável: a vantagem do empilhamento aberto sobre o Gemini
é significativa **só no estrato factual** — no acervo completo, 109 contra 95,
3 a 17, p = 0,0026. Cortar a factual de 118 para 75 remove 43 perguntas
justamente de onde a vitória estava, e o agregado cai abaixo da significância.

Consequências que o artigo tem de assumir:

1. **Declarar a população antes de declarar o resultado.** "O aberto supera o
   proprietário" e "o aberto alcança o proprietário" são as duas verdadeiras,
   sobre misturas diferentes do mesmo acervo. Misturar as duas não é.
2. **Para o artefato do KDD, isto é requisito de projeto, não nota de rodapé.**
   Um benchmark cuja mistura de tipos determina o vencedor precisa publicar a
   mistura e, de preferência, obrigar o relato por estrato.
3. O balanceamento é a escolha conservadora e é a adotada. A regra de seleção
   está em `montar_subconjunto.py`: amostra estratificada proporcional em
   origem × idioma, semente 2026, **cega ao resultado** — os estratos são
   propriedades do dado (quem escreveu, em que língua), nunca a nota de nenhum
   braço. Sortear 75 de 118 olhando o acerto seria escolher a conclusão.

### A comparativa tem duas naturezas diferentes no acervo

As 25 vindas do gabarito original **nomeiam as duas normas no enunciado** ("Como
o NIST e a UNESCO tratam…"); a maioria das 50 escritas depois não nomeia. Medido
por detector de acrônimo de emissor e número de instrumento
(`comparativa_nomeia_normas.py`):

| | n | nomeia as duas | nomeia uma | nomeia nenhuma |
|---|---|---|---|---|
| piloto | 25 | 25 (100%) | 0 | 0 |
| geradas depois | 50 | 11 (22%) | 14 (28%) | 25 (50%) |

As primeiras pressupõem que o usuário já sabe o que comparar; as segundas
perguntam pelo assunto. São tarefas distintas, e o benchmark do KDD deve dizê-lo.

**A afirmação de que "a decomposição só é possível na primeira" foi testada e não
se sustenta.** Ganho da alternância estrita sobre a consulta única, por classe:

| classe | n | única | intercalado | ganho | discordantes | p |
|---|---|---|---|---|---|---|
| nomeia as duas | 36 | 4 | 6 | +2 | 1 a 3 | 0,6250 |
| **nomeia uma** | 14 | 0 | **5** | **+5** | 0 a 5 | 0,0625 |
| nomeia nenhuma | 25 | 9 | 10 | +1 | 1 a 2 | 1,0000 |

O maior ganho está na classe que nomeia **uma** norma, não na que nomeia as duas.
Leitura mecanicista: nomear uma norma dá ao decompositor uma âncora e deixa o
outro lado a ser inferido do assunto, e é aí que a alternância tem o que fazer —
ela força o lado não nomeado para dentro do top-5 em vez de deixá-lo perder no
score. Nenhuma classe atinge p < 0,05 e todos os n são pequenos, então a
conclusão defensável é **"medimos e não encontramos apoio para a afirmação"**,
não "o contrário é verdade". O detector é grosso: acha "NIST" e "2338", não
entende paráfrase ("a lei brasileira de IA"), e portanto **subestima** quantas
nomeiam.

### O tipo de pergunta depende do recorte

Uma pergunta cujas duas citações caem no mesmo trecho de 1.200 caracteres é
factual na prática, mesmo que o enunciado pareça multi-hop. Havia **42 assim**,
todas rotuladas multi-hop; todo sistema acertava 0,964 nelas — poder
discriminativo zero. Corrigido em 27/09, e a correção **aumentou** a diferença
medida entre os modelos, de 7,2 para 10,2 pontos. A dispersão deve ser reportada
em caracteres, que é propriedade do documento, para que cada sistema derive a
própria distribuição de tipos.

A 75ª multi-hop foi escrita em 30/09 para fechar o alvo balanceado, com três
critérios declarados **antes** de procurar âncora: documento ainda não usado nas
15 anteriores, grande o bastante para folga de dispersão, e evidência em inglês
(o estrato estava 41 pt contra 33 en). Saiu do documento da OMS sobre ética e
governança, capítulo 4 contra capítulo 7 — **dispersão de 311 trechos**, a maior
do acervo.

### Auditoria do gabarito: 16 de 25 comparativas tinham defeito

As 25 comparativas do piloto foram relidas uma a uma em 29/09.

**Estrutural (4)** — a pergunta nomeia duas normas e o gabarito cita uma só, de
modo que a resposta afirma coisas sobre a norma ausente sem nenhuma citação:

| qid | o que faltava |
|---|---|
| q0081 | faltava o PL 2338; a citação do NIST era da função GOVERN e a resposta falava de MAP |
| q0085 | faltava a Lei 14.874; a resposta afirmava atribuições do CEP sem citação |
| q0093 | faltava a OMS; a resposta afirmava provisões para vulneráveis sem citação |
| q0099 | faltava a OMS |

**De conteúdo (8)** — a citação existe mas é outra frase, às vezes adjacente à
certa, e não sustenta o que a resposta diz:

| qid | o que estava errado |
|---|---|
| q0078 | a citação da LGPD era sobre exigir consentimento; a resposta fala da dispensa |
| q0082 | a citação da ANPD só falava em reparação; a resposta afirmava inversão do ônus e regresso |
| q0083 | a citação da FDA era um marcador solto |
| q0084 | a citação da OMS era sobre implementação por stakeholders; a resposta fala de responsividade |
| q0092 | a citação da FDA era moldura pura ("a seção seguinte descreve…") e a resposta fala de sistema de qualidade |
| q0095 | a mesma citação-moldura da FDA |
| q0097 | a citação do GDPR era "ao avaliar o nível"; a lista de medidas está na frase anterior |
| q0098 | a citação do NIST era sobre consequências graves; a resposta fala de notificação |

**Limítrofe (3)** — âncora adjacente à que sustenta a resposta: q0076 (a frase
"um sistema transparente não é necessariamente preciso" estava fora da janela),
q0094 (faltava a parte das escolhas do paciente), q0096 (a resposta fala de falta
de métricas consensuais e a citação era sobre contexto).

**Citação intrusa (1)** — q0086 tinha 4 citações e a quarta era evidência de
q0094.

Todas reparadas por `corrigir_comparativas_piloto.py`, que roda **depois** do
conversor e é idempotente. Efeito no resultado: praticamente nenhum
(0,542 → 0,538 no subconjunto), e o McNemar ficou idêntico caso a caso.

#### Por que o reparo não moveu o número

Isto precisa ser dito, porque "reparei 16 e não mudou nada" soa como reparo
inútil. Não é: o número **não podia** mudar, e a razão é mensurável
(`explicar_reparo.py`).

Posição, no ranking do BGE+reranker, da citação exigida que decide o acerto:

| | |
|---|---|
| perguntas reparadas que acertam no top-5 | **1 de 16** |
| mediana da posição da última citação exigida | **73** |
| mínimo / máximo | 5 / 94 |
| perguntas com alguma citação **fora do pool de 100** | **8 de 16** |

Um corte em 5 só vê posição ≤ 5. Com a citação decisiva em mediana 73, **trocar
qual citação é exigida não atravessa o corte**: o alvo antigo estava fora e o
novo também. Três fatores se somam:

1. **O estrato já estava no chão.** A comparativa marca 13 de 75 (0,173). Um item
   que valia 0 e continuou 0 não move média nenhuma — a métrica tem pouca
   resolução ali.
2. **O reparo endureceu a exigência.** Os 4 estruturais foram de 1 para 2
   citações; numa métrica conjuntiva isso só mantém ou piora. Antes do reparo
   essas 4 eram artificialmente **fáceis** — e com uma citação só,
   `montar_acervo.py` as retipava para `factual`. Só q0086 afrouxou (4 → 3
   citações). É por isso que a comparativa caiu de 0,187 para 0,173 em vez de
   subir: a previsão inicial de que subiria estava errada.
3. **O que foi reparado não é o que a métrica mede.** O defeito era a resposta
   afirmar mais do que a citação sustenta — isso é defeito do **alvo da
   geração**. Complete Evidence@5 só pergunta se os trechos exigidos voltaram.

Um padrão lateral que corrobora de forma independente o achado da decomposição:
na maioria das 16, **uma citação está entre as posições 1 e 5 e a outra entre 60
e 94, ou fora do pool**. É exatamente o mecanismo "o segundo lado perde para o
quinto trecho do primeiro", encontrado por outro caminho.

**O valor do reparo é de robustez, e vale escrever como tal:** 16 perguntas
corrigidas, o alvo da geração agora é confiável, e a conclusão do artigo não
mudou. A maioria dos artigos não faz esse teste.

### A procedência das perguntas

**Todas as 393 perguntas do acervo foram escritas por LLM em sessão**, com a
citação extraída do corpus e conferida à mão. Não há perguntas escritas por
humano. A diferença entre as famílias é **o nível de auditoria**, não a autoria:

- as 100 do **gabarito original (piloto)** tiveram uma **amostra auditada por
  especialista** — 114 casos de divergência entre dois juízes LLM, revistos à mão
  (`revisao_juizes_especialista_PREENCHIDA.csv`, 13/08/2026), sobre a qualidade
  da **resposta**, não sobre a relevância do trecho;
- as 25 comparativas do piloto foram, além disso, **relidas integralmente** em
  29/09 (acima);
- as demais famílias tiveram a citação conferida na geração, sem auditoria
  independente.

O campo `origem` registra a procedência de cada uma, e o resultado **deve ser
reportado cortado por ela** — se uma categoria for melhor que outra, a primeira
hipótese a descartar é quem escreveu, não o tipo.

**A auditoria de recuperação que falta** é a da seção 7, e é a primeira vez que
um especialista vai julgar **o trecho recuperado**, não a resposta gerada.

### Os braços não usam o mesmo tipo de índice

| braço | onde a busca acontece | vizinho mais próximo |
|---|---|---|
| Gemini | coleção do Qdrant em produção | **aproximado** (HNSW) |
| BGE-m3 | matriz local em memória | exato |
| Qwen3-Embedding | matriz local em memória | exato |

O Gemini é medido contra o Qdrant **de propósito** — é a produção como ela é, e
foi essa a escolha de desenho. A consequência é que ele usa busca aproximada
enquanto os abertos usam exata, e **isso favorece levemente os abertos**: um
vizinho que o HNSW deixa escapar do top-100 é um acerto que o Gemini perde e que
a busca exata teria encontrado.

**Medido em 05/10** por `reteste_gemini.py`, reembutindo as 225 consultas e
comparando a busca do Qdrant com a busca exata sobre os mesmos vetores:

| corte | Qdrant (HNSW) | exata local | diferença |
|---|---|---|---|
| top-5 | 110 | 111 | +1 |
| top-10 | 136 | 137 | +1 |
| top-20 | 167 | 169 | +2 |
| top-100 | 208 | **213** | **+5** |

CE@5 de 0,489 contra 0,493 — **0 a 1 nos discordantes, p = 1,0000**, e o top-5 é
idêntico em 221 das 225. **A conclusão do artigo não depende deste confundidor.**

Onde a aproximação custa é no **pool**: CE@100 sobe de 0,924 para 0,947, e as
cinco perguntas que a busca exata acha e o HNSW perde são **todas comparativas**
(c008, c027, c044, c046, c049). Pergunta que precisa de duas evidências em
documentos diferentes vive no fundo do pool, que é justamente o que a busca
aproximada poda. Consequência prática: o experimento "pool do Gemini + reranker
aberto" da seção 7 deve usar a busca exata, ou perde 5 de teto antes de começar.

**E o braço reproduz exato.** Reembutindo as consultas hoje e reconsultando a
coleção de produção, o ranking volta idêntico: primeiro lugar igual em 225 de
225, top-5 na mesma ordem em 225 de 225, top-100 com o mesmo conjunto em 225 de
225, Jaccard@100 médio 1,0000, acerto 110 contra 110. O 0,489 deste relatório é
reproduzível hoje, e a coleção de produção não derivou.

**O braço exato está construído**, e custou 225 chamadas de embedding. Os
vetores de documento saem de `embeddings_context_v1_backup.jsonl` (7.265 vetores
de 3.072 dimensões) e as consultas ficaram em `dense_gemini_queries.npz`, então
refazer não paga de novo.

> **Uma armadilha achada no caminho, que vale para qualquer reaproveitamento de
> vetor.** A junção entre o backup e o corpus atual só fecha pelo **texto de
> embedding reconstruído**: por id do trecho casam 2.105 de 6.134, pelo
> `embedding_text_fingerprint` gravado no arquivo casam **0 de 6.134**, e pelo
> texto reconstruído casam **6.134 de 6.134**. O id falha por causa da migração
> para id por conteúdo. O fingerprint falha porque a receita de
> `build_embedding_text` mudou depois de 22/09 **sem que o rótulo do perfil
> mudasse** — o backup diz `context-v1` e o código de hoje também, e os hashes
> divergem para um registro de texto e metadados idênticos. É a mesma classe de
> falha silenciosa que a validação local flagrou no Qwen, e aqui ela está no
> rótulo que deveria protegê-la.

**Por que o Qwen ficou local e não no Qdrant**, já que existe um cluster dele: o
caminho do Qdrant na avaliação (`QdrantDenseRetriever`) **não valida o corpus** —
ele busca e lê `payload['id_original']`, sem conferir fingerprint. O caminho
local valida, e foi essa validação que flagrou que os vetores do Qwen eram do
corpus antigo de 5.160 trechos. Usar o caminho sem conferência logo no modelo
cujo índice se sabe desalinhado seria trocar uma falha visível por uma
silenciosa. Some-se a isso que manter o Qwen local o deixa no mesmo regime do
BGE-m3, que é o outro aberto — e a comparação entre os dois é o que interessa.

## 9. Reprodutibilidade

Tudo em `eval/experimento_embedding/`:

```
montar_acervo.py               junta as famílias, confere coerência de tipo
montar_subconjunto.py          a vista balanceada 75/75/75, semente 2026
gerar_qrels.py                 qrels derivado das citações
exportar_gabarito.py           acervo → formato que o pipeline lê
gerar_multihop.py              as 16 multi-hop de evidência dispersa
completar_ranking.py           ranqueia só as perguntas novas (e mede a latência)
analisar_bracos.py             todas as métricas e cortes, com McNemar
tabelas_relatorio.py           as tabelas da seção 3
teto_reordenacao.py            quanto cabe de ganho só reordenando
selecao_diversa.py             MMR, teto por documento, vizinhança
diversificar.py                teto por documento, por tipo
decompor_tudo.py               parte a pergunta SEM ver o tipo (seção 5)
medir_decomposicao.py          pools, reranking e as três montagens
fundir_subperguntas.py         a versão de 29/09, com o tipo conhecido
comparativa_nomeia_normas.py   decomposição × a pergunta nomear as normas
abstencao_e_idioma.py          AUC da abstenção e corte por idioma
fechamento_dezembro.py         limiar validado em metades separadas
falsos_negativos.py            o gabarito subconta? quanto?
explicar_reparo.py             por que o reparo do gabarito não moveu o número
montar_auditoria.py            planilha de auditoria humana dos top-5
```

Planilha de auditoria em `eval/experimento_embedding/auditoria/`.

### O lote de Colab da decomposição — executado em 05/10

Para refazer, na ordem:

```bash
python eval/experimento_embedding/medir_decomposicao.py --pools
```

```bash
python eval/experimento_embedding/preparar_rerank_subperguntas.py --fonte cega --aplicar
```

Sobe `eval/results/indexes/rerank_entrada_decomposicao_cega.json` no Colab com o
`colab_rerank.py`, e **na célula 2 escreve
`SAIDA = "rerank_scores_decomposicao_cega.json"`**. Sem isso o nome sai do modelo
e colide com `rerank_scores_bge_reranker_v2_m3.json`, que são as notas da
consulta única e é lido por cinco scripts da análise. O arquivo que desce vai
para `eval/results/indexes/`. Depois:

```bash
python eval/experimento_embedding/medir_decomposicao.py
```

O `--rerank` do `medir_decomposicao.py` faz o mesmo em CPU, em blocos de 2.000
pares e gravando a cada bloco, mas leva ~11,5 h. O oráculo de 29/09, para a
comparação de orçamento, sai de `fundir_subperguntas.py`.

### Indexação: a migração que falta executar

O `src/build_vectorstore.py` foi reescrito em 30/09 (seção 6). Hoje
`leme_gemini` é uma **coleção real** com 6.134 pontos; o padrão novo quer que ela
seja um **apelido** apontando para uma coleção versionada. A migração é uma vez
só e apaga a coleção antiga **depois** de a nova estar verificada:

```bash
python src/build_vectorstore.py --listar
```

```bash
python src/build_vectorstore.py --upload
```

```bash
python src/build_vectorstore.py --promover --assumir-nome
```

O `--upload` não chama a API do Gemini: exige que o backup em
`data/processed/embeddings_context_v1_backup.jsonl` esteja completo e falha em
vez de gastar sem avisar. Depois da migração, cada reindexação é `--upload`
seguido de `--promover`, e o rollback é `--promover --para <coleção anterior>`.

**Não executado ainda** — precisa de janela combinada, porque o passo
`--assumir-nome` apaga o índice antigo.
