# Fase 0 congelada e índice Qwen

Este documento registra o estado canônico da Fase 0 e a forma de usar,
reproduzir e verificar o índice Qwen correspondente.

## Estado congelado

| Item | Valor |
|---|---:|
| Documentos ativos | 54 |
| PDFs | 34 |
| HTMLs | 20 |
| Chunks | 6.139 |
| IDs únicos | 6.139 |
| Qrels válidos | 256 |
| Modelo | `Qwen/Qwen3-Embedding-0.6B` |
| Dimensão | 1024 |
| Perfil textual | `context-v1` |

Fingerprint completo:

```text
fd5a51003b22dcb78938eb8d2333007895b896a75f210625b4a94b584a94ff3c
```

Prefixo usado na coleção:

```text
fd5a51003b22dcb7
```

Coleção Qdrant oficial:

```text
niar_rag_documents_qwen3_0_6b_context_v1_fd5a51003b22dcb7
```

A coleção foi verificada com 6.139 pontos, 6.139 `id_original` únicos e o
fingerprint global correto em todos os payloads. Ela corresponde exatamente à
ordem e ao conteúdo de `data/processed/documents.jsonl` no estado congelado.

Durante os experimentos, não reextraia nem rechunke o corpus. Qualquer alteração
de ID, texto, metadado contextual ou ordem exige um novo fingerprint, novos
embeddings e uma nova coleção versionada.

## Uso normal do RAG

Para consultar a coleção pronta não é necessário gerar novamente os 6.139
embeddings de documentos. Ainda é necessário transformar cada query com o mesmo
modelo Qwen local, usando `prompt_name="query"` e normalização, salvo quando essa
etapa estiver exposta por outro serviço ou backend.

O pipeline Qwen oferece `embed_query()` em `src/build_qwen_vectorstore.py`. O
vetor resultante tem 1.024 dimensões e pode ser enviado a `query_points()` na
coleção oficial.

O agente em `agent/utils/tools.py` ainda representa o caminho legado de
recuperação com Gemini e a coleção `niar_rag_documents`. Não aponte esse caminho
diretamente para a coleção Qwen: primeiro adapte o embedding da query e a
configuração do cliente para `QDRANT_QWEN_URL`, `QDRANT_QWEN_API_KEY` e a coleção
oficial. O gerador conversacional atual usa `GROQ_API_KEY`.

## Reprodução completa

Use Python 3.11 em um ambiente separado. A configuração validada foi:

- Python 3.11;
- PyTorch CPU;
- `transformers==4.51.3`;
- `sentence-transformers>=3.0,<6`;
- `accelerate`;
- `qdrant-client`;
- `python-dotenv`;
- `tqdm`.

Não use simplesmente `pip install -r requirements.txt` para esta tarefa. O
arquivo geral instala dependências que não são necessárias para Qwen e pode
baixar pacotes CUDA grandes.

Instalação mínima validada:

```bash
python3.11 -m venv ~/.venvs/niar-rag-qwen
source ~/.venvs/niar-rag-qwen/bin/activate

python -m pip install --upgrade pip setuptools wheel

pip install torch --index-url https://download.pytorch.org/whl/cpu

pip install \
  "transformers==4.51.3" \
  "sentence-transformers>=3.0,<6" \
  accelerate \
  qdrant-client \
  python-dotenv \
  tqdm
```

Prepare o ambiente e o nome da coleção:

```bash
source ~/.venvs/niar-rag-qwen/bin/activate

COLLECTION=niar_rag_documents_qwen3_0_6b_context_v1_fd5a51003b22dcb7
```

### 1. Gerar ou retomar o backup

```bash
PYTHONPATH=src python src/build_qwen_vectorstore.py \
  --generate-only \
  --collection "$COLLECTION"
```

`--generate-only` não acessa o Qdrant. A retomada usa:

```text
data/processed/embeddings_qwen3_0_6b_context_v1_backup.jsonl
```

Esse backup é derivado, contém vetores grandes e está ignorado pelo Git por
`data/processed/embeddings_*.jsonl`. Um prefixo existente só é reutilizado após
validar modelo, dimensão, perfil, fingerprint global, fingerprint individual,
documento e ordem. Ao terminar, os 6.139 registros são validados novamente.

### 2. Fazer upload

```bash
PYTHONPATH=src python src/build_qwen_vectorstore.py \
  --upload-only \
  --collection "$COLLECTION"
```

`--upload-only` não carrega o modelo nem gera embeddings. Ele exige o backup
completo e validado antes de criar a coleção. A coleção legada
`niar_rag_documents` é bloqueada pelo código. Não use `--recreate` no primeiro
upload da coleção versionada; essa opção só é apropriada quando a substituição
destrutiva da mesma coleção for uma decisão explícita.

### 3. Verificar a coleção

```bash
PYTHONPATH=src python src/build_qwen_vectorstore.py \
  --verify \
  --collection "$COLLECTION"
```

`--verify` é somente leitura. Ele confere:

- 6.139 pontos;
- os mesmos 6.139 `id_original` do corpus local;
- zero IDs ausentes, extras ou duplicados;
- dimensão 1.024;
- modelo e perfil corretos;
- fingerprint global completo em todos os payloads.

## Variáveis de ambiente

Crie `.env` a partir de `.env.example`. Para gerar o backup não são necessárias
credenciais Qdrant; elas são necessárias apenas para upload e verificação.

```dotenv
QDRANT_QWEN_URL=
QDRANT_QWEN_API_KEY=
QDRANT_QWEN_COLLECTION=niar_rag_documents_qwen3_0_6b_context_v1_fd5a51003b22dcb7
```

`QWEN_EMBEDDING_BACKUP_FILE` é opcional; sem ele, o pipeline usa o caminho
padrão documentado acima. `GROQ_API_KEY` é necessário somente para executar o
gerador conversacional atual.

O `.env` real está ignorado pelo Git. Nunca versione credenciais; compartilhe-as
apenas por um gerenciador de segredos ou outro canal seguro.
