# Desenho da análise — auditoria humana da recuperação

Escrito em 26/11/2026, **antes** de cruzar qualquer nota humana com o gabarito.
Até este ponto só foi verificado o preenchimento das planilhas. As decisões
abaixo foram fechadas com a Karol nesta data e não mudam depois de ver os
resultados. Qualquer desvio entra na seção "Desvios", com data e motivo.

## Dados

| planilha | quem | o que tem |
|---|---|---|
| `auditoria/AUDITORIA_RECUPERACAO.xlsx` | especialista 1 (anotador 1), membro do projeto | 78 perguntas, 390 trechos com nota 0–3; 75 vereditos (3 em branco, no lote 2) |
| `auditoria_anotadores/AUDITORIA_RECUPERACAO_anotador_2.xlsx` | anotador 2, externo ao projeto | as 30 perguntas do lote 1 (150 trechos) com nota; 29 vereditos |

- **Não haverá 3º anotador.** A planilha `_anotador_3.xlsx` fica fora.
- As linhas das planilhas são ligadas a `trecho_id` por `(qid, trecho)` através de
  `auditoria_trechos.csv` e `auditoria_trechos_anotador_2.csv`. Conferido:
  as 390 linhas estão alinhadas, e as duas únicas diferenças de texto são de
  codificação (`\x96` lido pelo Excel como `_x0096_`).
- As 30 perguntas do anotador 2 são exatamente o lote 1 do especialista 1.

## Gabarito automático usado na comparação

O acerto automático é **recalculado sobre os 5 trechos exibidos**, com o gabarito
atual (`eval/data/golden_dezembro.jsonl`, campo `qrels_grupos`, casamento no nível
da citação): a pergunta é acerto quando cada grupo de citação tem ao menos um
trecho entre os 5. A CHAVE só marca, trecho a trecho, se ele está no gabarito
(`esta_no_gabarito`); o acerto por pergunta precisa dos grupos de citação, por isso
é recalculado. Conferido em 26/11: a marcação da CHAVE coincide com o gabarito
atual nos 390 trechos (100 no gabarito, 290 fora, nenhuma divergência).

## Análise 1 — concordância entre os dois anotadores (lote 1)

- **Estatística:** κ de Cohen ponderado (pesos quadráticos), decidido pela Karol.
  - nota do trecho (0–3): 150 pares;
  - veredito "os 5 bastam?" (N < PARCIAL < S): perguntas com veredito dos dois.
- **IC 95%:** bootstrap por pergunta (os 5 trechos de uma pergunta não são
  independentes), 10.000 reamostras, semente 2026.
- Reportar junto: concordância bruta, matriz de confusão e as distribuições
  marginais. Isso permite ver o paradoxo do κ, se uma classe dominar.

## Análise 2 — o especialista 1 é mais favorável que o anotador independente?

Motivo: o especialista 1 faz parte do projeto, e o anotador 2 não.

- **Teste da nota:** diferença pareada na nota média por pergunta (especialista 1
  menos anotador 2), 30 perguntas, Wilcoxon de postos sinalizados, bilateral.
- **Teste do veredito:** teste do sinal nos pares discordantes (especialista 1
  mais favorável vs. menos favorável), bilateral.
- **Regra de decisão, fixada aqui:**
  - **nenhum dos dois testes com p < 0,05:** a análise principal das seções 3 e 4
    usa as **78 perguntas do especialista 1**; a participação dele no projeto
    entra como limitação declarada e testada;
  - **algum teste com p < 0,05 na direção "especialista 1 mais favorável":** a
    análise principal das seções 3 e 4 usa **só as 30 do anotador 2**, e as 78
    do especialista 1 são reportadas como secundárias.
- Em qualquer caso, as seções 3 e 4 são **repetidas nas 30 com o anotador 2**,
  e as duas versões vão para o relatório.

## Análise 3 — humano × gabarito automático (por pergunta)

- **Principal:** veredito S = acerto; PARCIAL e N = erro. A régua automática exige
  toda a evidência, e só S corresponde a ela.
- **Sensibilidade:** S e PARCIAL = acerto.
- Reportar: tabela 2×2 (humano × automático), concordância bruta e κ de Cohen
  (não ponderado, binário), e a tabela 3×2 sem binarizar.
- Os casos **automático erra / humano S** são candidatos a subcontagem do
  gabarito; os casos **automático acerta / humano N** são candidatos a acerto
  espúrio da métrica. Os dois grupos são listados para leitura.
- Vereditos em branco são excluídos (75 perguntas do especialista 1, 29 do
  anotador 2); nada é imputado.

## Análise 4 — trechos relevantes fora do gabarito (por trecho)

- Trecho **relevante para o humano** = nota ≥ 2 (contém parte ou o todo da
  resposta). Sensibilidade: nota 3.
- Tabela 2×2 por trecho: relevante para o humano × está no gabarito atual.
- **Falso negativo candidato do gabarito:** nota 3, fora do gabarito. Todos são
  listados com a observação do anotador, para leitura (não são contados como
  falso negativo sem leitura).
- **Gabarito sem apoio humano:** trecho no gabarito com nota 0 ou 1. Também
  listados.

## Taxas e pesos

- A amostra **não é proporcional**: ela tem cota igual de acertos e erros por tipo,
  enquanto a população tem 70/5 (factual), 38/37 (multi-hop) e 13/62
  (comparativa) acertos/erros em 75, para BGE-m3 + reranker. Qualquer **taxa**
  que pretenda descrever a população (por exemplo, "em X% das perguntas os 5
  trechos bastam") é ponderada pelo peso da célula tipo × acerto. As células
  são as da amostragem original (`montar_auditoria.py`).
- **Concordâncias** (κ) e **comparações pareadas** são reportadas sem peso,
  porque descrevem a amostra auditada, não a população.
- Tudo é reportado também **por tipo de pergunta** (factual, multi-hop,
  comparativa).

## O que esta análise não faz

- Não avalia perguntas fora de escopo (elas não estão na amostra, por desenho).
- Não substitui o Complete Evidence@5 do artigo: mede a validade dele.

## Script

`eval/experimento_embedding/analisar_auditoria.py` (a escrever), lendo as duas
planilhas e gravando `auditoria/resultados_auditoria.json` mais as listas de
casos para leitura.

## Desvios

Nenhum desvio nas análises 1 a 4. Rodadas em 26/11/2026; resultados em
`resultados_auditoria.json` e `casos_para_leitura.csv`.

**Análises exploratórias, feitas depois de ver os resultados** (não fazem parte
do desenho e devem ser reportadas como exploratórias):

- κ binário entre os anotadores com nota ≥ 2 = relevante: 0,376 (IC 95%
  0,185–0,566; 150 trechos). Motivo: a matriz da análise 1 sugeria que os dois
  usam a escala de forma diferente (o anotador 2 quase não dá nota 3, nem a
  trechos do gabarito). Binarizar não aumentou a concordância.
- Cruzamento dos dois anotadores só nos trechos fora do gabarito do lote 1
  (113 trechos): 42 recebem nota 3 do especialista 1, e o anotador 2 dá nota ≥ 2
  a 32 deles (nota 3 a 9).
