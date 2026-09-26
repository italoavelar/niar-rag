# =============================================================================
# EXPERIMENTO DE RECORTE — embutir as quatro condições no Colab
#
# Rodar em: Ambiente de execução > Alterar tipo de ambiente > GPU (T4 serve)
#
# Em CPU isto leva ~7 horas; em T4, minutos. É a única razão de existir este
# script: o resto do experimento roda na máquina local.
#
# ESTE SCRIPT SÓ FAZ O BGE. O BGE-m3 é modelo local e precisa de GPU — daí o
# Colab. O Gemini é chamada de API, não precisa de GPU nenhuma, e roda na
# máquina local por `gemini_embutir.py`, que grava `emb_gemini_<condição>.npz`
# no mesmo formato. Os dois juntos dão as oito matrizes do experimento.
#
# O QUE SUBIR (5 arquivos, ~35 MB no total):
#   T400.jsonl          de eval/experimento_embedding/dados/
#   T1200.jsonl                    idem
#   T3500.jsonl                    idem
#   PC.jsonl                       idem
#   embedding_text.py   de src/   <- o do repositório, para o texto de
#                                    embedding ser idêntico ao local
#
# O QUE VOLTA (4 arquivos .npz, ~40 MB):
#   emb_T400.npz  emb_T1200.npz  emb_T3500.npz  emb_PC.npz
#
# Cada .npz baixa sozinho assim que fica pronto, ainda dentro do laço — não no
# final. Se o Colab cair no meio (e ele cai), o que já embutiu está salvo no seu
# computador e a condição seguinte é a única a refazer.
#
# ⚠️ Na PRIMEIRA baixa o navegador pergunta se aceita "vários downloads deste
#    site". Aceite — senão só o primeiro arquivo chega.
#
# Ponha os quatro em eval/experimento_embedding/resultados/.
# =============================================================================

# ------------------------------------------------------------------ CÉLULA 1
!pip -q install "sentence-transformers>=3.0"

# ------------------------------------------------------------------ CÉLULA 2
from google.colab import files
print("Selecione os 4 .jsonl e o embedding_text.py")
enviados = files.upload()
for esperado in ("T400.jsonl", "T1200.jsonl", "T3500.jsonl", "PC.jsonl",
                 "embedding_text.py"):
    assert esperado in enviados, f"faltou {esperado}"

# ------------------------------------------------------------------ CÉLULA 3
import json, sys, time
import numpy as np
import torch
sys.path.insert(0, "/content")
from embedding_text import (EMBEDDING_TEXT_PROFILE, build_embedding_text,
                            embedding_text_hash)
from sentence_transformers import SentenceTransformer

assert torch.cuda.is_available(), (
    "Sem GPU. Ambiente de execução > Alterar tipo de ambiente > T4 GPU. "
    "Em CPU isto leva ~7 horas.")
print("GPU:", torch.cuda.get_device_name(0))

modelo = SentenceTransformer("BAAI/bge-m3", device="cuda")

# ------------------------------------------------------------------ CÉLULA 4
# Embute condição por condição. O `embedding_text_hash` vai junto para que a
# máquina local possa reaproveitar vetor entre condições sem depender de id —
# T400 e PC compartilham muitos filhos idênticos, por exemplo.
#
# A ordem é do mais barato ao mais caro de propósito: se a sessão cair, você
# perde a condição mais cara, não as três baratas junto.
CONDICOES = ["T3500", "T1200", "T400", "PC"]

import os

for nome in CONDICOES:
    saida = f"emb_{nome}.npz"
    if os.path.exists(saida):
        print(f"══ {nome}: {saida} já existe, pulando")
        continue

    registros = []
    with open(f"{nome}.jsonl", encoding="utf-8") as f:
        for linha in f:
            if linha.strip():
                registros.append(json.loads(linha))

    entradas = [build_embedding_text(r) for r in registros]
    ids = [str(r["id"]) for r in registros]
    hashes = [embedding_text_hash(r) for r in registros]

    print(f"\n══ {nome}: {len(registros)} trechos")
    t0 = time.time()
    vetores = modelo.encode(entradas, normalize_embeddings=True,
                            batch_size=64, show_progress_bar=True)
    vetores = np.asarray(vetores, dtype="float32")
    dt = time.time() - t0

    normas = np.linalg.norm(vetores, axis=1)
    assert vetores.shape[0] == len(registros), "perdeu trecho no caminho"
    assert np.allclose(normas, 1.0, atol=1e-3), "vetor não normalizado"
    assert not np.isnan(vetores).any(), "vetor com NaN"

    np.savez_compressed(
        saida,
        matrix=vetores,
        ids=np.array(ids, dtype=object),
        text_hash=np.array(hashes, dtype=object),
        embedding_text_profile=np.array(EMBEDDING_TEXT_PROFILE),
        embedding_model=np.array("BAAI/bge-m3"),
    )
    print(f"   {dt/60:.1f} min | {dt/len(registros)*1000:.0f} ms/trecho | "
          f"norma {normas.min():.4f}–{normas.max():.4f} | → {saida}")

    # Baixa AGORA, não no final. Um arquivo salvo no computador não se perde
    # quando a sessão do Colab morre.
    files.download(saida)
    print(f"   ⬇ {saida} baixando")

print("\nTodas as condições prontas. Se algum download falhou, rode de novo: "
      "as condições já embutidas são puladas e só o que faltou baixa.")
