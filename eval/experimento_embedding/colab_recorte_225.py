#!/usr/bin/env python3
"""Réplica do experimento de recorte: embutir as condições em GPU do Colab.

COMO USAR
─────────
1. Colab → Ambiente de execução → Alterar tipo de ambiente → GPU (T4 serve).
2. Arraste `recorte225_pacote.json.gz` (8 MB) para o painel de arquivos do Colab
   — o ícone de pasta na barra esquerda. A célula 2 tem o upload como plano B.
3. Copie cada bloco "CÉLULA N" abaixo para uma célula, na ordem.

O QUE SOBE E O QUE DESCE
────────────────────────
Sobe    `recorte225_pacote.json.gz`, de `preparar_recorte_225.py`: o texto de
        embedding de 70.222 trechos em 4 condições de recorte, já pronto.
Desce   um `emb_<condição>.npz` por condição, de poucos KB, com a ORDEM do
        ranking de cada pergunta truncada em 300 — não as matrizes de vetores,
        que somariam ~290 MB e não são usadas pela análise.

Cada arquivo baixa sozinho assim que a condição termina, ainda dentro do laço.
Se a sessão cair (e cai), o que já desceu está salvo e só a condição seguinte
precisa refazer — rode a célula 3 de novo e ela pula as prontas.

⚠️ Na primeira baixa o navegador pergunta se aceita "vários downloads deste
   site". Aceite, senão só o primeiro arquivo chega.

Ponha os .npz em `eval/experimento_embedding/resultados_225/`.
"""

# ══════════════════════════════════════════════════════════════════════════════
# CÉLULA 1 — instalar e conferir a GPU
# ══════════════════════════════════════════════════════════════════════════════
!pip -q install "sentence-transformers>=3.0"

import torch
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available()
      else "NENHUMA — pare e troque o ambiente de execução para GPU")


# ══════════════════════════════════════════════════════════════════════════════
# CÉLULA 2 — carregar o pacote e embutir as 225 consultas
# ══════════════════════════════════════════════════════════════════════════════
import glob, gzip, json, os, time
import numpy as np

ENTRADA = "recorte225_pacote.json.gz"
if not os.path.exists(ENTRADA):
    from google.colab import files
    print(f"{ENTRADA} não está no painel. Selecione:")
    ENTRADA = next(iter(files.upload()))

with gzip.open(ENTRADA, "rt", encoding="utf-8") as f:
    pacote = json.load(f)

MODELO = pacote["modelo"]
PROF = pacote["profundidade"]
perguntas = pacote["perguntas"]
print(f"modelo       : {MODELO}")
print(f"perguntas    : {len(perguntas)}")
print(f"profundidade : {PROF}")
for cond, v in pacote["condicoes"].items():
    print(f"  {cond:7s} {len(v['textos']):6d} trechos")

from sentence_transformers import SentenceTransformer
modelo = SentenceTransformer(MODELO, device="cuda" if torch.cuda.is_available() else "cpu")

# As consultas são embutidas AQUI, no mesmo modelo e no mesmo dispositivo que os
# trechos. Embutir um lado na CPU local e o outro na GPU é pedir para a última
# casa decimal divergir entre o que se mediu e o que se serviu.
Q = modelo.encode(perguntas, normalize_embeddings=True, batch_size=32,
                  show_progress_bar=True).astype("float32")
print("consultas:", Q.shape)


# ══════════════════════════════════════════════════════════════════════════════
# CÉLULA 3 — embutir cada condição e baixar a ordem (a demorada)
# ══════════════════════════════════════════════════════════════════════════════
from google.colab import files

LOTE = 64 if torch.cuda.is_available() else 8

for cond, v in pacote["condicoes"].items():
    saida = f"emb_{cond}.npz"
    if os.path.exists(saida):
        print(f"{cond}: {saida} já existe — pulando")
        continue

    textos, ids = v["textos"], v["ids"]
    print(f"\n{cond}: {len(textos)} trechos")
    t0 = time.time()
    M = modelo.encode(textos, normalize_embeddings=True, batch_size=LOTE,
                      show_progress_bar=True).astype("float32")
    print(f"  embutido em {(time.time()-t0)/60:.1f} min")

    # A ordem é o que a análise usa; a matriz não. Truncar em PROF corta o
    # download de centenas de MB para alguns KB, e PROF=300 é fundo de sobra
    # para o maior orçamento, que gasta ~31 unidades.
    k = min(PROF, M.shape[0])
    sim = Q @ M.T
    ordem = np.argpartition(-sim, k - 1, axis=1)[:, :k]
    linhas = np.arange(ordem.shape[0])[:, None]
    ordem = ordem[linhas, np.argsort(-sim[linhas, ordem], axis=1)].astype("int32")

    np.savez_compressed(saida, ordem=ordem, ids=np.array(ids),
                        modelo=MODELO, condicao=cond, n_trechos=M.shape[0])
    print(f"  ✓ {saida}  ordem {ordem.shape}")
    files.download(saida)
    del M, sim
    import gc; gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None


# ══════════════════════════════════════════════════════════════════════════════
# CÉLULA 4 — conferir o que ficou pronto
# ══════════════════════════════════════════════════════════════════════════════
for f in sorted(glob.glob("emb_*.npz")):
    d = np.load(f, allow_pickle=True)
    print(f"{f:16s} ordem {d['ordem'].shape}  de {d['n_trechos']} trechos")
print("\nSe algum faltar, rode a CÉLULA 3 de novo — ela pula os prontos.")
