# Auditoria da recuperação — anotador_2

Obrigado por auditar. São duas planilhas e a ordem entre elas importa.

## A ordem de trabalho

1. **As 30 primeiras perguntas** de `AUDITORIA_RECUPERACAO_anotador_2.xlsx`. Estão
   no topo, numeradas na coluna `fazer` de `1 de 30` até `30 de 30`, e uma borda
   dupla vermelha marca onde elas acabam.
2. **Pare e vá para `AUDITORIA_GERACAO.xlsx`**, lote 1.
3. **Se sobrar tempo**, volte a esta planilha e siga pelas perguntas marcadas
   `extra`.

As 30 primeiras são as mesmas que o outro avaliador está julgando. A medida de
concordância entre avaliadores só existe nas perguntas que vocês dois fizeram,
então são essas que não podem ficar pela metade. As `extra` acrescentam
cobertura e podem ficar incompletas.

## O que o sistema faz

Uma pergunta entra; o sistema busca em 54 normas (brasileiras e internacionais,
em português e inglês) e devolve os **5 trechos** que considerou mais
relevantes. Cada trecho tem cerca de 1.200 caracteres. A pergunta é respondida
**só** com esses 5 trechos.

A pergunta de fundo é simples: **esses 5 trechos bastam?**

## Por trecho — `relevância 0-3`

| nota | quando |
|---|---|
| **0** | não tem relação com a pergunta |
| **1** | mesmo assunto, mas não contribui para a resposta |
| **2** | contém **parte** do que a resposta precisa |
| **3** | contém, sozinho, o suficiente para responder |

- Julgue o trecho **contra a pergunta**, não contra os outros trechos. Não há
  cota: os 5 podem ser 0, ou os 5 podem ser 3.
- **A ordem dos trechos é aleatória**, e é diferente da que os outros
  avaliadores receberam. A coluna `trecho` é posição de exibição, não a posição
  que o sistema deu.

## Por pergunta — `os 5 bastam?`

| valor | quando |
|---|---|
| **S** | sim, completa e correta |
| **PARCIAL** | dá para responder parte, ou responder com ressalva |
| **N** | não dá |

Se for `PARCIAL` ou `N`, diga em `o que falta` o que faltou.

**Todas as perguntas desta planilha têm resposta no acervo.** Quando marcar `N`,
está dizendo "a busca não trouxe", não "não existe".

Metade das perguntas sorteadas são casos em que o sistema, pelo gabarito
automático, **errou**. É de propósito.

## Uma regra

**Não compare suas notas com as de ninguém enquanto preenche.** A medida de
concordância só vale se os julgamentos forem independentes.
