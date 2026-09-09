# ============================================================================
# LEME — embeddings BGE-M3 + Qwen3 sobre o corpus corrigido
# Rodar no Google Colab com GPU:  Ambiente de execucao > Alterar tipo > T4 GPU
# ============================================================================

# ---------------------------------------------------------------- CELULA 1
# Instala as dependencias. Reinicie a sessao se o Colab pedir.
!pip -q install "sentence-transformers>=3.0" openpyxl

# ---------------------------------------------------------------- CELULA 2
# Suba DOIS arquivos do repositorio:
#   1) data/processed/documents.jsonl
#   2) src/embedding_text.py      <- reaproveitar o do repo garante que o
#                                    fingerprint seja identico ao local
from google.colab import files
print("Selecione documents.jsonl E embedding_text.py")
enviados = files.upload()
assert "documents.jsonl" in enviados, "faltou documents.jsonl"
assert "embedding_text.py" in enviados, "faltou embedding_text.py"

# ---------------------------------------------------------------- CELULA 3
import json, sys, time, hashlib
from pathlib import Path
import numpy as np
sys.path.insert(0, "/content")

from embedding_text import (
    EMBEDDING_TEXT_PROFILE,
    build_embedding_text,
    bge_cache_filename,
    corpus_embedding_fingerprint,
)

corpus = []
with open("documents.jsonl", encoding="utf-8") as f:
    for linha in f:
        if linha.strip():
            corpus.append(json.loads(linha))

entradas = [build_embedding_text(r) for r in corpus]
ids = [str(r["id"]) for r in corpus]
fingerprint = corpus_embedding_fingerprint(corpus)

print(f"chunks............: {len(corpus)}")
print(f"perfil textual....: {EMBEDDING_TEXT_PROFILE}")
print(f"fingerprint.......: {fingerprint}")
print(f"exemplo de entrada:\n{entradas[0][:220]}")

# ---------------------------------------------------------------- CELULA 4
# BGE-M3 -> .npz no formato que upload_bge_to_qdrant.py exige
from sentence_transformers import SentenceTransformer

BGE_ID, BGE_DIM = "BAAI/bge-m3", 1024
t0 = time.time()
bge = SentenceTransformer(BGE_ID, device="cuda")
vet_bge = bge.encode(
    entradas,
    batch_size=16,
    normalize_embeddings=True,      # colecao usa COSINE
    show_progress_bar=True,
    convert_to_numpy=True,
).astype(np.float32)
print(f"BGE pronto: {vet_bge.shape} em {time.time()-t0:.0f}s")
assert vet_bge.shape == (len(corpus), BGE_DIM), vet_bge.shape

nome_bge = bge_cache_filename("bge_m3", fingerprint)
np.savez_compressed(
    nome_bge,
    matrix=vet_bge,
    ids=np.array(ids, dtype=object),
    embedding_text_profile=EMBEDDING_TEXT_PROFILE,
    embedding_text_fingerprint=fingerprint,
    embedding_model=BGE_ID,
)
print("salvo:", nome_bge)

del bge
import torch, gc; gc.collect(); torch.cuda.empty_cache()

# ---------------------------------------------------------------- CELULA 5
# Qwen3-Embedding-0.6B -> .jsonl no formato que build_qwen_vectorstore.py exige
QWEN_ID, QWEN_DIM = "Qwen/Qwen3-Embedding-0.6B", 1024
t0 = time.time()
qwen = SentenceTransformer(QWEN_ID, device="cuda")
vet_qwen = qwen.encode(
    entradas,
    batch_size=16,
    normalize_embeddings=True,
    show_progress_bar=True,
    convert_to_numpy=True,
).astype(np.float32)
print(f"Qwen pronto: {vet_qwen.shape} em {time.time()-t0:.0f}s")
assert vet_qwen.shape == (len(corpus), QWEN_DIM), vet_qwen.shape

nome_qwen = "qwen3_0_6b_context_v1_backup.jsonl"
with open(nome_qwen, "w", encoding="utf-8") as saida:
    for registro, vetor in zip(corpus, vet_qwen):
        saida.write(json.dumps({
            "embedding_model": QWEN_ID,
            "embedding_dimension": QWEN_DIM,
            "embedding_text_profile": EMBEDDING_TEXT_PROFILE,
            # fingerprint POR CHUNK, como o script local espera
            "embedding_text_fingerprint": corpus_embedding_fingerprint([registro]),
            "document": registro,
            "vector": [float(x) for x in vetor],
        }, ensure_ascii=False) + "\n")
print("salvo:", nome_qwen)

# ---------------------------------------------------------------- CELULA 6
# Conferencia + planilha de registro + download automatico
import pandas as pd, zipfile, os

conf = np.load(nome_bge, allow_pickle=True)
assert str(conf["embedding_text_fingerprint"]) == fingerprint
assert [str(i) for i in conf["ids"]] == ids
assert conf["matrix"].shape == (len(corpus), BGE_DIM)
linhas_qwen = sum(1 for _ in open(nome_qwen, encoding="utf-8"))
assert linhas_qwen == len(corpus), linhas_qwen
print("conferencia OK: fingerprint, ordem dos ids e dimensoes batem")

planilha = "leme_embeddings_resumo.xlsx"
pd.DataFrame([
    {"modelo": BGE_ID,  "dimensao": BGE_DIM,  "chunks": len(corpus),
     "arquivo": nome_bge,  "formato": "npz",   "perfil": EMBEDDING_TEXT_PROFILE,
     "fingerprint": fingerprint, "destino": "eval/results/indexes/"},
    {"modelo": QWEN_ID, "dimensao": QWEN_DIM, "chunks": len(corpus),
     "arquivo": nome_qwen, "formato": "jsonl", "perfil": EMBEDDING_TEXT_PROFILE,
     "fingerprint": fingerprint, "destino": "data/processed/"},
]).to_excel(planilha, index=False)

pacote = "leme_embeddings.zip"
with zipfile.ZipFile(pacote, "w", zipfile.ZIP_DEFLATED) as z:
    for arq in (nome_bge, nome_qwen, planilha):
        z.write(arq)
print(f"pacote: {pacote} ({os.path.getsize(pacote)/1e6:.1f} MB)")

files.download(planilha)   # a planilha, separada
files.download(pacote)     # tudo junto
