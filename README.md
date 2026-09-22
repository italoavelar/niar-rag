# NIAR RAG

Protótipo de RAG para consulta a documentos regulatórios, jurídicos e técnicos
relacionados a IA responsável e saúde.

## Estado do corpus

A Fase 0 está congelada com 54 documentos e 6.139 chunks. O corpus usa o perfil
textual `context-v1` e foi indexado com `Qwen/Qwen3-Embedding-0.6B` na coleção:

`niar_rag_documents_qwen3_0_6b_context_v1_fd5a51003b22dcb7`

Fingerprint oficial:

`fd5a51003b22dcb78938eb8d2333007895b896a75f210625b4a94b584a94ff3c`

Consulte [docs/fase0_qwen.md](docs/fase0_qwen.md) para configuração mínima,
uso da coleção pronta, reprodução dos embeddings e validação do índice.

Copie `.env.example` para `.env` e preencha as credenciais localmente. O `.env`
real não deve ser versionado nem compartilhado por canais inseguros.
