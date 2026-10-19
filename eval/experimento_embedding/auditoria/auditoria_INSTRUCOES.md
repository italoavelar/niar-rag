# Auditoria da recuperação — instruções

Obrigado por auditar. São duas planilhas e o trabalho é em duas passadas por
pergunta. Não abra `auditoria_CHAVE_nao_abrir_antes.csv`: ela tem o gabarito, e
consultá-la antes invalida o julgamento.

## O que o sistema faz

Uma pergunta entra; o sistema busca em 54 normas (brasileiras e internacionais,
em português e inglês) e devolve os **5 trechos** que considerou mais relevantes.
Cada trecho tem cerca de 1.200 caracteres. A pergunta é respondida **só** com
esses 5 trechos — o gerador não tem acesso a mais nada.

Estamos auditando a etapa de busca, antes de construir o gerador. A pergunta de
fundo é simples: **esses 5 trechos bastam?**

## Passada 1 — `auditoria_trechos.csv`, trecho por trecho

Cada linha é um trecho. Preencha `JUIZ_relevancia_0a3`:

| nota | quando |
|---|---|
| **0** | não tem relação com a pergunta |
| **1** | mesmo assunto, mas não contribui para a resposta |
| **2** | contém **parte** do que a resposta precisa |
| **3** | contém, sozinho, o suficiente para responder |

Duas observações que mudam o resultado:

- Julgue o trecho **contra a pergunta**, não contra os outros trechos. Não há
  cota: os 5 podem ser 0, ou os 5 podem ser 3.
- **A ordem na planilha é aleatória.** `trecho_n` é posição de exibição, não a
  posição que o sistema deu. Não tente inferir qual o sistema achou melhor.

Use `JUIZ_observacao` quando a nota não contar a história — por exemplo "é o
artigo certo mas a frase que responde está no trecho anterior", que é o defeito
mais comum que encontramos.

## Passada 2 — `auditoria_perguntas.csv`, a pergunta inteira

Depois de julgar os 5 trechos de uma pergunta, responda: **com esses 5 trechos,
é possível responder a pergunta de forma completa e correta?**

| valor | quando |
|---|---|
| **S** | sim, completa e correta |
| **PARCIAL** | dá para responder parte, ou responder com ressalva |
| **N** | não dá |

Se for `PARCIAL` ou `N`, diga em `JUIZ_o_que_falta` o que faltou: o dispositivo,
a norma, o lado da comparação, o número, o prazo.

**Todas as perguntas desta planilha têm resposta no acervo.** Você não precisa
decidir se a resposta existe em algum lugar — só se ela está nestes 5 trechos.
Quando marcar `N`, está dizendo "a busca não trouxe", não "não existe".

Os enunciados estão em português. Os trechos podem estar em português ou em
inglês, porque o acervo é bilíngue — isso é normal e não é defeito.

## Ordem sugerida

Faça o **lote 1** inteiro primeiro. Ele já é uma amostra estratificada válida, e
com ele fechamos uma primeira leitura. Os lotes 2 e 3 refinam.

## O que vai ser feito com isso

Três coisas: (1) medir concordância entre o seu julgamento e o gabarito
automático, que é derivado de citação literal e pode estar incompleto;
(2) encontrar trechos que respondem e não estão no gabarito — já achamos dois
casos, em que uma norma reafirma a regra de outra; (3) decidir se a recuperação
está boa o bastante para passarmos à geração.

Metade das perguntas sorteadas são casos em que o sistema, pelo gabarito
automático, **errou**. Isso é de propósito: auditar só os acertos mediria
satisfação, não qualidade. Não se surpreenda se vários conjuntos de 5 trechos
parecerem ruins.
