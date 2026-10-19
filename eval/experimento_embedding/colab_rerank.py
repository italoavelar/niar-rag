#!/usr/bin/env python3
"""Reranker cross-encoder em GPU do Colab.

COMO USAR
─────────
1. Colab → Ambiente de execução → Alterar tipo de ambiente → GPU (T4 serve).
2. Arraste o pacote `rerank_entrada*.json` para o painel de arquivos do Colab
   (ícone de pasta na barra esquerda). Arrastar é mais confiável que
   `files.upload()` para 5–8 MB — a célula 2 tem o upload como plano B.
3. Copie cada bloco marcado "CÉLULA N" abaixo para uma célula, na ordem.

O QUE SOBE E O QUE DESCE
────────────────────────
Sobe    um `rerank_entrada*.json`: o da consulta única
        (`preparar_rerank_colab.py`) ou `rerank_entrada_decomposicao_cega.json`
        das sub-perguntas (`preparar_rerank_subperguntas.py`).
Desce   `rerank_scores_*.json`, baixado pela célula 4.

ATENÇÃO ao nome que desce. No padrão ele sai do MODELO, e o modelo é o mesmo
nos dois pacotes: `rerank_scores_bge_reranker_v2_m3.json`. Esse arquivo é as
notas da CONSULTA ÚNICA e é lido por cinco scripts (`montar_auditoria`,
`abstencao_e_idioma`, `falsos_negativos`, `fechamento_dezembro`,
`completar_ranking`). Ao pontuar o pacote da decomposição, escreva
`SAIDA = "rerank_scores_decomposicao_cega.json"` na célula 2 — que é o nome que
`medir_decomposicao.py` espera — ou o download sobrescreve as notas daquela.

O texto de cada trecho JÁ VEM PRONTO no pacote, com o prefixo de contexto que o
pipeline local usa. Esta célula não remonta nada: se remontasse, bastaria uma
diferença de versão para o Colab pontuar um texto e a análise local medir outro.

A célula 3 grava a cada 20 blocos e RETOMA de onde parou se a sessão cair — basta
rodá-la de novo. Colab derrubar sessão longa é o modo de falha normal aqui, não o
excepcional, e perder a GPU já gasta por desconexão seria o jeito previsível de
falhar.
"""

# ══════════════════════════════════════════════════════════════════════════════
# CÉLULA 1 — instalar e conferir a GPU
# ══════════════════════════════════════════════════════════════════════════════
!pip -q install sentence-transformers

import torch
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available()
      else "NENHUMA — pare e troque o ambiente de execução para GPU")


# ══════════════════════════════════════════════════════════════════════════════
# CÉLULA 2 — carregar o pacote
# ══════════════════════════════════════════════════════════════════════════════
import glob, json, os, time

# Qual pacote pontuar. Em None, acha o único rerank_entrada*.json do painel;
# com mais de um, nomeie aqui qual é.
ENTRADA = None

# Como o resultado desce. Em None, sai pelo nome do MODELO — que é o nome
# histórico `rerank_scores_bge_reranker_v2_m3.json`, LIDO POR CINCO SCRIPTS da
# análise de consulta única. Um pacote novo tem que descer com outro nome, ou
# sobrescreve o deles.
SAIDA = None

if ENTRADA is None:
    achados = sorted(glob.glob("rerank_entrada*.json"))
    if len(achados) == 1:
        ENTRADA = achados[0]
    elif achados:
        raise SystemExit(f"mais de um pacote no painel: {achados} — "
                         "escreva em ENTRADA qual deles")

if ENTRADA is None or not os.path.exists(ENTRADA):
    from google.colab import files
    print("Pacote não está no painel. Selecione o rerank_entrada*.json:")
    ENTRADA = next(iter(files.upload()))

pacote = json.load(open(ENTRADA, encoding="utf-8"))
MODELO = pacote["modelo"]
MAX_LEN = pacote["max_length"]
perguntas = pacote["perguntas"]
trechos = pacote["trechos"]
if SAIDA is None:
    SAIDA = "rerank_scores_" + MODELO.split("/")[-1].replace("-", "_") + ".json"

print(f"entrada   : {ENTRADA}")
print(f"base      : {pacote.get('base')}")
print(f"modelo    : {MODELO}")
print(f"perguntas : {len(perguntas)}")
print(f"trechos   : {len(trechos)}")
print(f"pares     : {sum(len(v['candidatos']) for v in perguntas.values())}")
print(f"saída     : {SAIDA}")


# ══════════════════════════════════════════════════════════════════════════════
# CÉLULA 3 — pontuar (a demorada; rode de novo se a sessão cair)
# ══════════════════════════════════════════════════════════════════════════════
from sentence_transformers import CrossEncoder

scores = {}
if os.path.exists(SAIDA):
    try:
        scores = json.load(open(SAIDA, encoding="utf-8")).get("scores", {})
        print(f"retomando: {sum(len(v) for v in scores.values())} pares já pontuados")
    except Exception:
        scores = {}

fila = [(q, c) for q, v in perguntas.items()
        for c in v["candidatos"] if c not in scores.get(q, {})]
total = len(fila)
print(f"a pontuar: {total} pares\n")

if total:
    modelo = CrossEncoder(MODELO, max_length=MAX_LEN,
                          device="cuda" if torch.cuda.is_available() else "cpu")
    # O cross-encoder lê consulta e trecho JUNTOS, então o par é a unidade de
    # trabalho — não há vetor a reaproveitar, como haveria num bi-encoder.
    LOTE = 128 if torch.cuda.is_available() else 16
    t0, feitos = time.time(), 0

    for bloco, i in enumerate(range(0, total, LOTE), start=1):
        pedaco = fila[i:i + LOTE]
        notas = modelo.predict([[perguntas[q]["pergunta"], trechos[c]]
                                for q, c in pedaco],
                               batch_size=LOTE, show_progress_bar=False)
        for (q, c), nota in zip(pedaco, notas):
            scores.setdefault(q, {})[c] = float(nota)
        feitos += len(pedaco)

        if bloco % 20 == 0 or feitos >= total:
            json.dump({"modelo": MODELO, "max_length": MAX_LEN,
                       "max_candidates": pacote["max_candidates"],
                       "texto": pacote["texto"], "base": pacote["base"],
                       "n_pares": sum(len(v) for v in scores.values()),
                       "scores": scores},
                      open(SAIDA, "w", encoding="utf-8"), ensure_ascii=False)
            gasto = time.time() - t0
            resta = gasto / max(feitos, 1) * (total - feitos)
            print(f"  {feitos}/{total}  [{gasto/60:.1f} min · faltam ~{resta/60:.1f} min]")

    print(f"\nterminado em {(time.time()-t0)/60:.1f} min")


# ══════════════════════════════════════════════════════════════════════════════
# CÉLULA 4 — baixar
# ══════════════════════════════════════════════════════════════════════════════
print(f"✓ {SAIDA} — {sum(len(v) for v in scores.values())} pares, "
      f"{len(scores)} perguntas")
from google.colab import files
files.download(SAIDA)
