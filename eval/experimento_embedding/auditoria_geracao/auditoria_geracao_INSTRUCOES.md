# Auditoria da geração — instruções

Obrigado por auditar. É uma planilha e uma passada por pergunta. Não abra
`auditoria_geracao_CHAVE_nao_abrir_antes.csv`: ela diz qual resposta é de qual
configuração e traz a resposta-referência, e consultá-la antes invalida o
julgamento.

## O que o sistema faz

Uma pergunta entra; o sistema responde em no máximo uma página, terminando com
uma seção `## Fontes utilizadas`. O acervo são 54 normas brasileiras e
internacionais, em português e inglês.

Para cada pergunta, a planilha traz **duas respostas**, em configurações
diferentes do sistema, chamadas **A** e **B**. Qual é qual você não sabe, e a
ordem é sorteada a cada pergunta.

## O que preencher, por resposta

| coluna | o que é |
|---|---|
| `correção 1-3` | **3** correta e completa · **2** parcialmente correta, mistura acerto e erro ou omite o essencial · **1** factualmente errada, ou não entrega a resposta |
| `as fontes conferem?` | **S** as normas citadas existem e sustentam o que a resposta afirma · **PARCIAL** sustentam parte, ou falta a principal · **N** citam norma que não sustenta aquilo, ou que não existe · **SEM FONTE** |
| `inventou? S/N` | **S** se a resposta afirma lei, artigo, número ou prazo que você sabe que não é aquilo |
| `observação da resposta` | o que a nota não conta |

Uma **recusa** (`Não tenho informações suficientes para responder com
segurança`) vale **1**. Todas as perguntas desta planilha **têm** resposta nas
normas do acervo, então abster-se aqui é falhar, não é prudência.

Se não tiver como julgar, **deixe em branco** e diga o motivo na observação.

## O que preencher, por pergunta

Depois de julgar as duas, na primeira linha do bloco: **qual responde melhor?**
`A`, `B`, `EMPATE` (as duas servem igualmente, ou falham igualmente) ou
`NENHUMA`. E o motivo em `por quê`.

## Três coisas que mudam o resultado

- Julgue cada resposta **contra a pergunta** e contra o que a norma diz, não
  uma contra a outra. As duas podem ser 3, e as duas podem ser 1.
- **Você provavelmente vai adivinhar qual configuração é qual** em várias
  perguntas, pelo jeito de escrever. Não tem problema — mas julgue o conteúdo,
  não o palpite. A nota de cada resposta é o que mais importa; a preferência
  entre as duas é secundária, justamente porque ela é a mais sensível ao
  palpite.
- **Tamanho não é qualidade.** As duas respostas podem diferir muito em
  extensão, e a mais longa costuma ser a que inventa.

Algumas respostas não cabem na célula do Excel e começam com um aviso em
maiúsculas dizendo isso. Nesses casos leia o texto completo em
`auditoria_geracao_respostas.csv`, na mesma pergunta e na mesma letra — não
julgue pela parte que aparece.

## Ordem sugerida

Faça o **lote 1** inteiro primeiro. Ele já é uma amostra estratificada válida.

## O que vai ser feito com isso

Duas coisas. (1) Medir a concordância entre o seu julgamento e o de um painel de
quatro modelos de linguagem que já julgou estas mesmas respostas — o painel é
barato e escala, mas precisa de uma âncora humana, e já deu problema: dois dos
quatro se recusaram a pontuar centenas de respostas alegando motivos que o
próprio enunciado contradizia. (2) Decidir, de fora, se a busca em normas
melhora a resposta ou se o modelo já respondia bem sem ela — que é a pergunta
do artigo.

São as **mesmas perguntas** da auditoria da recuperação, nos mesmos lotes, de
propósito: dá para cruzar as duas e ver se, quando os 5 trechos não bastavam, a
resposta saiu errada.
