# ============================================================================
# LEME — Colab COMPLETO: documentos + consultas, BGE-M3 e Qwen3, em float32
# Rodar com GPU:  Ambiente de execucao > Alterar tipo de ambiente > T4 GPU
#
# Substitui colab_embeddings_leme.py (que so gerava DOCUMENTOS) e o adendo
# colab_e1_queries.py. Entrega, numa rodada so, os quatro artefatos que a
# comparacao E1 precisa, mais a planilha de registro.
#
# O QUE MUDA EM RELACAO A PRIMEIRA RODADA
#
#   1) CONSULTAS. A primeira versao nunca leu queries.csv — embutia apenas
#      `entradas`, que sao os documentos. Sem os vetores das 75 consultas nao
#      ha como ranquear nada, e por isso a E1 nao rodava.
#
#   2) DTYPE. O Qwen carregou em bfloat16 (8 bits de mantissa) e o BGE em
#      float32. Comparar um modelo quantizado com outro em precisao plena mede
#      a quantizacao junto com o modelo. Aqui os dois sao forcados a float32.
#
#   3) PROMPT DE CONSULTA. O Qwen3-Embedding e ASSIMETRICO: a consulta usa
#      prompt_name="query" e o documento vai sem prompt. O BGE-m3 e simetrico
#      e nao usa instrucao. A celula 5 verifica que o prompt teve efeito.
#
# SAIDAS (a celula 6 empacota tudo em um zip):
#   dense_bge_m3_context-v1_<fp>.npz       -> eval/results/indexes/
#   dense_bge_m3_queries.npz               -> eval/results/indexes/
#   dense_qwen3_0_6b_context-v1_<fp>.npz   -> eval/results/indexes/
#   dense_qwen3_0_6b_queries.npz           -> eval/results/indexes/
#   qwen3_0_6b_context_v1_backup.jsonl     -> data/processed/   (upload ao Qdrant)
#   leme_embeddings_resumo.xlsx            -> registro da rodada
# ============================================================================

# ---------------------------------------------------------------- CELULA 1
!pip -q install "sentence-transformers>=3.0" openpyxl

# ---------------------------------------------------------------- CELULA 2
# Suba TRES arquivos do repositorio:
#   1) data/processed/documents.jsonl
#   2) eval/data/queries.csv        <- FALTAVA na primeira rodada
#   3) src/embedding_text.py        <- garante fingerprint identico ao local
from google.colab import files
print("Selecione: documents.jsonl, queries.csv E embedding_text.py")
enviados = files.upload()
for obrigatorio in ("documents.jsonl", "queries.csv", "embedding_text.py"):
    assert obrigatorio in enviados, f"faltou {obrigatorio}"

# O .npz de documentos do BGE da 1a rodada ja esta em float32 e validado, entao
# por padrao nao o refazemos (economiza GPU). O do Qwen saiu em bfloat16 e
# precisa ser refeito. Mude para True se quiser regerar tudo do zero.
REFAZER_DOCS_BGE = False
REFAZER_DOCS_QWEN = True

# ---------------------------------------------------------------- CELULA 3
import csv, gc, json, sys, time
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

sys.path.insert(0, "/content")
from embedding_text import (EMBEDDING_TEXT_PROFILE, build_embedding_text,
                            bge_cache_filename, corpus_embedding_fingerprint)

corpus = []
with open("documents.jsonl", encoding="utf-8") as fh:
    for linha in fh:
        if linha.strip():
            corpus.append(json.loads(linha))

entradas = [build_embedding_text(r) for r in corpus]        # DOCUMENTOS
ids = [str(r["id"]) for r in corpus]
fingerprint = corpus_embedding_fingerprint(corpus)

with open("queries.csv", encoding="utf-8-sig", newline="") as fh:
    linhas = list(csv.DictReader(fh))
qids = [r["QueryId"] for r in linhas]
qtexts = [r["Query"] for r in linhas]                        # CONSULTAS

BGE_ID, QWEN_ID, DIM = "BAAI/bge-m3", "Qwen/Qwen3-Embedding-0.6B", 1024
registro = []

print(f"chunks......: {len(corpus)}")
print(f"consultas...: {len(qtexts)}")
print(f"perfil......: {EMBEDDING_TEXT_PROFILE}")
print(f"fingerprint.: {fingerprint}")
print(f"documento[0]: {entradas[0][:150]}")
print(f"consulta[0].: {qtexts[0][:150]}")


def carregar(model_id):
    """float32 explicito: sem isso o Qwen vem em bfloat16 pelo config do Hub."""
    modelo = SentenceTransformer(model_id, device="cuda",
                                 model_kwargs={"torch_dtype": torch.float32})
    print(f"  dtype interno: {modelo[0].auto_model.dtype}")
    assert modelo[0].auto_model.dtype == torch.float32, "nao carregou em float32"
    return modelo


def conferir_normas(nome, matriz):
    normas = np.linalg.norm(matriz.astype(np.float64), axis=1)
    print(f"  {nome}: shape={matriz.shape} norma[{normas.min():.6f}, {normas.max():.6f}]")
    assert np.allclose(normas, 1.0, atol=1e-4), (
        f"{nome}: normas fora de 1.0 — sinal de precisao reduzida")


def salvar_docs_npz(nome_curto, model_id, matriz):
    arquivo = bge_cache_filename(nome_curto, fingerprint)
    np.savez_compressed(arquivo, matrix=matriz.astype(np.float32),
                        ids=np.array(ids, dtype=object),
                        embedding_text_profile=EMBEDDING_TEXT_PROFILE,
                        embedding_text_fingerprint=fingerprint,
                        embedding_model=model_id)
    registro.append({"modelo": model_id, "papel": "documentos", "dtype": "float32",
                     "prompt_query": "-", "itens": len(ids), "dimensao": DIM,
                     "arquivo": arquivo, "formato": "npz",
                     "perfil": EMBEDDING_TEXT_PROFILE, "fingerprint": fingerprint,
                     "destino": "eval/results/indexes/"})
    print("  salvo:", arquivo)
    return arquivo


def salvar_queries_npz(nome_curto, model_id, matriz, prompt_name):
    arquivo = f"dense_{nome_curto}_queries.npz"
    np.savez_compressed(arquivo, matrix=matriz.astype(np.float32),
                        qtexts=np.array(qtexts, dtype=object),
                        qids=np.array(qids, dtype=object),
                        embedding_model=model_id,
                        embedding_text_profile=EMBEDDING_TEXT_PROFILE,
                        query_prompt_name=str(prompt_name))
    registro.append({"modelo": model_id, "papel": "consultas", "dtype": "float32",
                     "prompt_query": str(prompt_name), "itens": len(qtexts),
                     "dimensao": DIM, "arquivo": arquivo, "formato": "npz",
                     "perfil": EMBEDDING_TEXT_PROFILE, "fingerprint": fingerprint,
                     "destino": "eval/results/indexes/"})
    print("  salvo:", arquivo)
    return arquivo


# ---------------------------------------------------------------- CELULA 4
# BGE-M3 — simetrico: documentos e consultas sem instrucao.
print("\n=== BGE-M3 ===")
t0 = time.time()
bge = carregar(BGE_ID)

if REFAZER_DOCS_BGE:
    vet = bge.encode(entradas, batch_size=16, normalize_embeddings=True,
                     convert_to_numpy=True, show_progress_bar=True).astype(np.float32)
    conferir_normas("docs BGE", vet)
    salvar_docs_npz("bge_m3", BGE_ID, vet)
else:
    print("  documentos: pulados (o .npz da 1a rodada ja esta em float32)")

vet_bge_q = bge.encode(qtexts, batch_size=16, normalize_embeddings=True,
                       convert_to_numpy=True, show_progress_bar=True).astype(np.float32)
conferir_normas("queries BGE", vet_bge_q)
salvar_queries_npz("bge_m3", BGE_ID, vet_bge_q, None)
print(f"BGE concluido em {time.time()-t0:.0f}s")

del bge; gc.collect(); torch.cuda.empty_cache()

# ---------------------------------------------------------------- CELULA 5
# Qwen3 — ASSIMETRICO: documento sem prompt, consulta com prompt_name="query".
print("\n=== Qwen3-Embedding-0.6B ===")
t0 = time.time()
qwen = carregar(QWEN_ID)
print("  prompts registrados:", getattr(qwen, "prompts", None))

if REFAZER_DOCS_QWEN:
    vet_qwen_d = qwen.encode(entradas, batch_size=16, normalize_embeddings=True,
                             convert_to_numpy=True,
                             show_progress_bar=True).astype(np.float32)
    conferir_normas("docs Qwen", vet_qwen_d)
    salvar_docs_npz("qwen3_0_6b", QWEN_ID, vet_qwen_d)

    # .jsonl tambem: e o formato que src/build_qwen_vectorstore.py exige
    # para subir ao Qdrant (o .npz acima serve a avaliacao).
    nome_jsonl = "qwen3_0_6b_context_v1_backup.jsonl"
    with open(nome_jsonl, "w", encoding="utf-8") as saida:
        for reg, vetor in zip(corpus, vet_qwen_d):
            saida.write(json.dumps({
                "embedding_model": QWEN_ID,
                "embedding_dimension": DIM,
                "embedding_text_profile": EMBEDDING_TEXT_PROFILE,
                "embedding_text_fingerprint": corpus_embedding_fingerprint([reg]),
                "document": reg,
                "vector": [float(x) for x in vetor],
            }, ensure_ascii=False) + "\n")
    registro.append({"modelo": QWEN_ID, "papel": "documentos", "dtype": "float32",
                     "prompt_query": "-", "itens": len(ids), "dimensao": DIM,
                     "arquivo": nome_jsonl, "formato": "jsonl",
                     "perfil": EMBEDDING_TEXT_PROFILE, "fingerprint": fingerprint,
                     "destino": "data/processed/"})
    print("  salvo:", nome_jsonl)

vet_qwen_q = qwen.encode(qtexts, batch_size=16, prompt_name="query",
                         normalize_embeddings=True, convert_to_numpy=True,
                         show_progress_bar=True).astype(np.float32)
conferir_normas("queries Qwen", vet_qwen_q)
salvar_queries_npz("qwen3_0_6b", QWEN_ID, vet_qwen_q, "query")

# Prova de que o prompt foi aplicado. Se der ~1.0000, prompt_name nao teve
# efeito e os vetores de consulta do Qwen estao errados.
sem_prompt = qwen.encode(qtexts[:5], normalize_embeddings=True,
                         convert_to_numpy=True).astype(np.float32)
cos = float(np.mean(np.sum(sem_prompt * vet_qwen_q[:5], axis=1)))
print(f"  cosseno com-prompt x sem-prompt: {cos:.4f}")
assert cos < 0.999, "prompt_name='query' NAO teve efeito — investigar antes de usar"
print("  prompt de consulta confirmado")
print(f"Qwen concluido em {time.time()-t0:.0f}s")

# ---------------------------------------------------------------- CELULA 6
# Conferencias finais, planilha e download.
import os, zipfile
import pandas as pd

print("\n=== conferencia ===")
for item in registro:
    arq = item["arquivo"]
    assert os.path.exists(arq), f"ausente: {arq}"
    if item["formato"] == "npz":
        d = np.load(arq, allow_pickle=True)
        assert d["matrix"].shape == (item["itens"], DIM), (arq, d["matrix"].shape)
        if item["papel"] == "documentos":
            assert [str(i) for i in d["ids"]] == ids, f"{arq}: ordem dos ids"
            assert str(d["embedding_text_fingerprint"]) == fingerprint
        else:
            assert [str(t) for t in d["qtexts"]] == qtexts, f"{arq}: textos"
    else:
        assert sum(1 for _ in open(arq, encoding="utf-8")) == item["itens"]
    print(f"  OK  {arq}")

planilha = "leme_embeddings_resumo.xlsx"
pd.DataFrame(registro)[["modelo", "papel", "itens", "dimensao", "dtype",
                        "prompt_query", "arquivo", "formato", "perfil",
                        "fingerprint", "destino"]].to_excel(planilha, index=False)
print("\nplanilha:", planilha)
print(pd.DataFrame(registro)[["modelo", "papel", "itens", "dtype", "prompt_query"]]
      .to_string(index=False))

pacote = "leme_embeddings_completo.zip"
with zipfile.ZipFile(pacote, "w", zipfile.ZIP_DEFLATED) as z:
    for item in registro:
        z.write(item["arquivo"])
    z.write(planilha)
print(f"\npacote: {pacote} ({os.path.getsize(pacote)/1e6:.1f} MB)")

files.download(planilha)
files.download(pacote)

# ============================================================================
# NO REPOSITORIO, depois de descompactar o zip:
#
#   dense_*_queries.npz            -> eval/results/indexes/
#   dense_qwen3_0_6b_context-*.npz -> eval/results/indexes/
#   qwen3_0_6b_context_v1_backup.jsonl -> data/processed/
#       (sobrescreve o backup em bfloat16; guarde o antigo se quiser comparar)
#
#   PYTHONIOENCODING=utf-8 venv/Scripts/python.exe eval/tools/e1_qwen_vs_bge.py
#
# O conversor eval/tools/qwen_jsonl_to_npz.py deixa de ser necessario: este
# script ja entrega o .npz do Qwen pronto. Ele continua util se algum dia so o
# .jsonl for regerado.
# ============================================================================
