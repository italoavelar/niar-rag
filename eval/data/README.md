# Por que não usamos um dataset público

**Decisão:** a avaliação (recuperação **e** geração) roda inteiramente sobre o
**gabarito ancorado** no nosso corpus (`golden_qa.jsonl`). Não há camada de dataset
público — o suporte a ela foi removido do código.

O gabarito tem **papel duplo**: os `qrels` avaliam a recuperação; as
`reference_answer` avaliam a geração. Nada externo é necessário.

## Candidatos levantados (e por que não servem)

| Candidato | Por que não serve aqui |
|---|---|
| [BR-TaxQA-R](https://huggingface.co/datasets/unicamp-dl/BR-TaxQA-R) | QA jurídico com referências, mas do domínio **tributário** — não cobre saúde nem governança de IA. |
| [legalbench.br](https://huggingface.co/datasets/celsowm/legalbench.br) | Benchmark jurídico brasileiro de tarefas variadas; **jurídico geral**, sem o recorte sanitário/IA. |
| [JurisTCU](https://arxiv.org/html/2503.08379) | Recuperação jurídica **com** qrels — porém sobre o **corpus próprio** (jurisprudência do TCU). Os julgamentos apontam para os documentos dele, não para os nossos chunks. |
| [HealthQA-BR](https://huggingface.co/datasets/Larxel/healthqa-br) | 5.632 questões de provas de licenciamento/residência: mede **conhecimento clínico em múltipla escolha**, não geração fundamentada em normas recuperadas. |
| [lavita/medical-qa-datasets](https://huggingface.co/datasets/lavita/medical-qa-datasets), [bigbio/med_qa](https://huggingface.co/datasets/bigbio/med_qa) | QA médico majoritariamente em **inglês** e em **múltipla escolha** — língua e formato incompatíveis. |

## As duas incompatibilidades de fundo

1. **Domínio** — nenhum cobre a interseção *jurídico-saúde + governança de IA*.
2. **Tarefa** — os de múltipla escolha medem o **conhecimento interno** do modelo, não a
   fidelidade de uma resposta gerada a partir de contexto recuperado do **nosso** acervo;
   os de recuperação trazem qrels contra os **próprios** corpora.

Usá-los mediria outra coisa e daria falsa sensação de validade externa. A ausência de um
benchmark adequado é justamente uma das **lacunas que motivam o trabalho** — usar um de
encaixe ruim contradiria o próprio argumento.

**Trabalho futuro:** construir ou adotar um benchmark público PT-BR de RAG jurídico-saúde.

> Justificativa completa: Seção 6.2 do documento (`docs/Pipeline_Avaliacao_RAG_NIAR.docx`).
