# Auditoria da geração — instruções

Obrigado por auditar. É uma planilha e uma passada por pergunta. Não abra
`auditoria_geracao_CHAVE_nao_abrir_antes.csv`: ela diz qual resposta é de qual
configuração, quais perguntas não têm resposta e qual era a resposta esperada.
Consultá-la antes invalida o julgamento.

## O que o sistema faz

Uma pergunta entra; o sistema responde em no máximo uma página, terminando com
uma seção `## Fontes utilizadas`. O acervo são 54 normas brasileiras e
internacionais, em português e inglês.

Para cada pergunta, a planilha traz **duas respostas**, em configurações
diferentes do sistema, chamadas **A** e **B**. Qual é qual você não sabe, e a
ordem é sorteada a cada pergunta.

## Uma coisa importante antes de começar

**Algumas perguntas desta planilha não têm resposta em norma nenhuma.** Foram
escritas assim de propósito, e não estão marcadas. Para essas, o comportamento
certo do sistema é dizer que não encontrou base — e responder como se soubesse é
o pior defeito possível.

Por isso a planilha separa duas coisas que costumam ser confundidas: **o que a
resposta faz** (afirma? se absteve?) e **o que ela deveria fazer** (a pergunta
tinha resposta?). A primeira você preenche por resposta; a segunda, por
pergunta.

## O que preencher, por resposta

| coluna | o que é |
|---|---|
| `o que ela afirma está certo?` | **3** tudo certo e entrega o que foi pedido · **2** mistura acerto e erro, ou entrega parte · **1** afirma coisa errada · **ABSTEVE** não afirma nada, diz que não tem informação suficiente |
| `as fontes conferem?` | **S** as normas citadas existem e sustentam o que a resposta afirma · **PARCIAL** sustentam parte, ou falta a principal · **N** citam norma que não sustenta aquilo, ou que não existe · **SEM FONTE** |
| `inventou? S/N` | **S** se a resposta afirma lei, artigo, número ou prazo que você sabe que não é aquilo |
| `observação da resposta` | o que a nota não conta |

`ABSTEVE` é **descritivo, não é nota**: não decida aí se abster-se foi certo ou
errado. Se não tiver como julgar, **deixe em branco** e diga o motivo na
observação.

## O que preencher, por pergunta

| coluna | o que é |
|---|---|
| `existe norma que responda isso?` | **SIM** existe norma, brasileira ou internacional, que responde · **NÃO** a pergunta pede algo que norma nenhuma estabelece · **NÃO SEI** você não tem como afirmar nem negar |
| `qual responde melhor?` | **A**, **B**, **EMPATE** (servem ou falham igualmente), **NENHUMA** |
| `por quê` | a norma certa, o número certo, a omissão, a invenção |

`NÃO SEI` é resposta legítima e é melhor que chute. Essa coluna é o que permite
interpretar o `ABSTEVE`: abster-se numa pergunta sem resposta é o certo;
abster-se numa que tem resposta é falhar.

E se você marcou que **não** existe norma que responda, "melhor" é a que disse
que não encontrou base — não a que respondeu bonito.

## Três coisas que mudam o resultado

- Julgue cada resposta **contra a pergunta** e contra o que a norma diz, não
  uma contra a outra. As duas podem ser 3, e as duas podem ser 1.
- **Você provavelmente vai adivinhar qual configuração é qual** em várias
  perguntas, pelo jeito de escrever. Não tem problema — mas julgue o conteúdo,
  não o palpite. A nota de cada resposta é o que mais importa; a preferência
  entre as duas é secundária, justamente porque é a mais sensível ao palpite.
- **Tamanho não é qualidade.** As duas respostas podem diferir muito em
  extensão, e a mais longa costuma ser a que inventa.

Algumas respostas não cabem na célula do Excel e começam com um aviso em
maiúsculas dizendo isso. Nesses casos leia o texto completo em
`auditoria_geracao_respostas.csv`, na mesma pergunta e na mesma letra — não
julgue pela parte que aparece.

## Ordem sugerida

Faça o **lote 1** inteiro primeiro. Cada lote é uma amostra estratificada
válida por si, com os quatro tipos de pergunta na mesma proporção.

## O que vai ser feito com isso

Duas coisas. (1) Medir a concordância entre o seu julgamento e o de um painel de
quatro modelos de linguagem que já julgou estas mesmas respostas — o painel é
barato e escala, mas precisa de uma âncora humana, e já deu problema: dois dos
quatro se recusaram a pontuar centenas de respostas alegando motivos que o
próprio enunciado contradizia. (2) Decidir, de fora, se a busca em normas
melhora a resposta ou se o modelo já respondia bem sem ela — e se ela evita que
o sistema invente quando não há o que responder.

Boa parte das perguntas respondíveis são as **mesmas** da auditoria da
recuperação, de propósito: dá para cruzar as duas e ver se, quando os 5 trechos
não bastavam, a resposta saiu errada.
