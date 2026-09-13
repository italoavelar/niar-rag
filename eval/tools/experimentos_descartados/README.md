# Experimentos descartados

Scripts de estratégias que foram **testadas e não deram resultado**. Estão aqui, e não em
`eval/tools/`, para não serem confundidos com ferramenta de trabalho — nenhum deles faz parte do
pipeline.

Ficam versionados de propósito: o valor de um experimento negativo é impedir que alguém gaste a
mesma semana refazendo o mesmo caminho. Cada script explica no cabeçalho a hipótese que testou e o
número que a derrubou.

| Script | Hipótese | Resultado |
|---|---|---|
| `ablacao_section_path.py` | preencher `section_path` no prefixo de embedding melhora a recuperação | **não** — posição mediana 8 com a seção, 6 sem (n=12) |
| `viabilidade_chunk_normativo.py` | cortar por unidade normativa (artigo, seção) em vez de janela fixa | **inviável** — só 34 dos 63 documentos têm marcador regular, e os 29 sem são justamente NIST/WHO/UNESCO/OECD/FDA/ISO; 49% das unidades ficariam abaixo de 300 caracteres |
| `particao_por_lado.py` | reservar 3+2 vagas do top-5, uma partição por lado da comparação | **não** — 13/25 caiu para 11/25 |
| `busca_roteada_por_documento.py` | identificar os dois documentos e buscar dentro de cada um | **teto de 6/25 mesmo com roteador-oráculo**; dentro do próprio documento a âncora fica em posição mediana 8 |
| `decompose_slot_reserve.py` | decompor a consulta e reservar vaga por sub-pergunta | **no-op** — 22 das 25 seleções idênticas à base |
| `decompose_eval.py` | decomposição de consulta + fusão RRF | recall sobe, top-5 não muda (p=0,62) |

## Antes de reexecutar qualquer um

Eles foram escritos contra o corpus de **5.160 trechos**. O acervo foi reextraído em 13/09 e tem
**4.897**; os identificadores dos trechos que mudaram são outros. Os números nos cabeçalhos são de
quando cada um rodou — se for reexecutar, refaça a medição em vez de citar o comentário.

Duas docstrings já foram corrigidas por conterem números superados:

- `decompose_eval.py` dizia que as comparativas tinham problema de recall no pool (*"2/25 sem nenhum
  relevante no pool de 100"*). Não tinham: 2/25 é o número de comparativas com **todas** as âncoras
  no top-5; no pool de 100 são 20 a 22 de 25 que trazem ao menos uma.
- `busca_roteada_por_documento.py` declarava como hipótese aberta que a âncora estaria bem colocada
  dentro do próprio documento. Foi medido e refutado.
