# ============================================================================
# LEME — F3: reranker cross-encoder na GPU + as 25 consultas que faltam
# Rodar com GPU:  Ambiente de execucao > Alterar tipo de ambiente > T4 GPU
#
# POR QUE AQUI. Um cross-encoder de 560M sobre 7500 pares leva ~5 HORAS em CPU
# (medido). Na T4 sao minutos. A pontuacao e a unica parte cara; a avaliacao
# (nDCG, GeoRisk, significancia) roda local em segundos sobre os scores.
#
# DUAS TAREFAS numa sessao so:
#   A) pontua os pares (consulta, trecho) com um ou mais rerankers
#   B) embute as 25 consultas fora-de-escopo que faltam no .npz do BGE e
#      concatena com as 75 ja existentes — os vetores antigos sao PRESERVADOS,
#      nao recalculados, para nao divergir da E1/E2 ja rodadas
#
# SAIDAS:
#   rerank_scores_<modelo>.json   -> passar em eval/tools/rerank_eval.py --scores
#   dense_bge_m3_queries.npz      -> eval/results/indexes/ (agora com as 100)
# ============================================================================

# ---------------------------------------------------------------- CELULA 1
!pip -q install "sentence-transformers>=3.0"

# ---------------------------------------------------------------- CELULA 2
# Suba CINCO arquivos do repositorio:
#   1) data/processed/documents.jsonl
#   2) eval/data/golden_qa.jsonl
#   3) src/embedding_text.py
#   4) eval/results_e2/retrieval/rankings/B_dense_bge_m3.json
#   5) eval/results/indexes/dense_bge_m3_queries.npz
from google.colab import files
print("Selecione: documents.jsonl, golden_qa.jsonl, embedding_text.py, "
      "B_dense_bge_m3.json e dense_bge_m3_queries.npz")
enviados = files.upload()
for obrigatorio in ("documents.jsonl", "golden_qa.jsonl", "embedding_text.py",
                    "B_dense_bge_m3.json", "dense_bge_m3_queries.npz"):
    assert obrigatorio in enviados, f"faltou {obrigatorio}"

# Rerankers a testar. O bge-reranker-v2-m3 e a primeira escolha da Memoria.md
# secao 3A: mesma familia do embedding em uso. O jina e o desafiante leve.
# (O Qwen3-Reranker NAO entra aqui: ele nao expoe interface de CrossEncoder,
#  pontua via logits de "yes"/"no" num LM causal e exigiria codigo proprio.)
MODELOS = [
    "BAAI/bge-reranker-v2-m3",
    "jinaai/jina-reranker-v2-base-multilingual",
]
MAX_CANDIDATOS = 100     # precisa casar com --max-candidates no script local
MAX_LENGTH = 384         # cobre 99,6% dos pares (mediana 287 tokens, medido)
BATCH = 64

# ---------------------------------------------------------------- CELULA 3
import json, sys, time
import numpy as np
sys.path.insert(0, "/content")
from embedding_text import EMBEDDING_TEXT_PROFILE, build_embedding_text

corpus = {}
with open("documents.jsonl", encoding="utf-8") as fh:
    for linha in fh:
        if linha.strip():
            d = json.loads(linha)
            corpus[str(d["id"])] = d

gold = [json.loads(l) for l in open("golden_qa.jsonl", encoding="utf-8") if l.strip()]
pergunta = {r["qid"]: r["question"] for r in gold}
com_qrel = {r["qid"] for r in gold if r.get("qrels")}

base = json.load(open("B_dense_bge_m3.json", encoding="utf-8"))
qids = [q for q in base if q in com_qrel]

pares, indice = [], []
for qid in qids:
    for cid in base[qid][:MAX_CANDIDATOS]:
        pares.append((pergunta[qid], build_embedding_text(corpus[cid])))
        indice.append((qid, cid))

print(f"corpus...: {len(corpus)} chunks")
print(f"consultas: {len(qids)} com qrels")
print(f"pares....: {len(pares)}")
print(f"perfil...: {EMBEDDING_TEXT_PROFILE}")
print(f"exemplo..: {pares[0][0][:70]!r}  x  {pares[0][1][:70]!r}")

# ---------------------------------------------------------------- CELULA 4
# TAREFA A — pontuacao dos pares, um arquivo por reranker.
import gc, re
import torch
from sentence_transformers import CrossEncoder

gerados = []
for model_id in MODELOS:
    print(f"\n=== {model_id} ===")
    t0 = time.time()
    ce = CrossEncoder(model_id, max_length=MAX_LENGTH, device="cuda",
                      trust_remote_code=True)
    scores = ce.predict(pares, batch_size=BATCH, show_progress_bar=True)
    print(f"  {len(scores)} pares em {time.time()-t0:.0f}s")

    por_query = {}
    for (qid, cid), s in zip(indice, scores):
        por_query.setdefault(qid, {})[cid] = float(s)
    assert sum(len(v) for v in por_query.values()) == len(pares)

    apelido = re.sub(r"[^a-zA-Z0-9]+", "_", model_id).strip("_")
    arquivo = f"rerank_scores_{apelido}.json"
    with open(arquivo, "w", encoding="utf-8") as saida:
        json.dump({"modelo": model_id, "max_length": MAX_LENGTH,
                   "max_candidates": MAX_CANDIDATOS,
                   "texto": "contextual", "n_pares": len(pares),
                   "scores": por_query}, saida, ensure_ascii=False)
    gerados.append(arquivo)
    print(f"  salvo: {arquivo}")

    del ce; gc.collect(); torch.cuda.empty_cache()

# ---------------------------------------------------------------- CELULA 5
# TAREFA B — as 25 consultas fora-de-escopo que faltam no .npz do BGE.
# Os 75 vetores existentes sao COPIADOS, nunca recalculados: assim a E1 e a E2
# ja rodadas continuam reproduzindo exatamente os mesmos numeros.
from sentence_transformers import SentenceTransformer

antigo = np.load("dense_bge_m3_queries.npz", allow_pickle=True)
tex_antigo = [str(t) for t in antigo["qtexts"]]
mat_antiga = antigo["matrix"].astype(np.float32)
print(f"npz atual: {mat_antiga.shape[0]} consultas")

todas = [r["question"] for r in gold]
faltando = [t for t in todas if t not in set(tex_antigo)]
print(f"faltando : {len(faltando)}")

if faltando:
    bge = SentenceTransformer("BAAI/bge-m3", device="cuda",
                              model_kwargs={"torch_dtype": torch.float32})
    novos = bge.encode(faltando, batch_size=16, normalize_embeddings=True,
                       convert_to_numpy=True,
                       show_progress_bar=True).astype(np.float32)
    n = np.linalg.norm(novos.astype(np.float64), axis=1)
    print(f"  norma[{n.min():.6f}, {n.max():.6f}]")
    assert np.allclose(n, 1.0, atol=1e-4)

    textos = tex_antigo + faltando
    matriz = np.vstack([mat_antiga, novos])
else:
    textos, matriz = tex_antigo, mat_antiga

por_texto = {r["question"]: r["qid"] for r in gold}
np.savez_compressed(
    "dense_bge_m3_queries.npz",
    matrix=matriz,
    qtexts=np.array(textos, dtype=object),
    qids=np.array([por_texto.get(t, "") for t in textos], dtype=object),
    embedding_model="BAAI/bge-m3",
    embedding_text_profile=EMBEDDING_TEXT_PROFILE,
    query_prompt_name="None",
)
print(f"salvo: dense_bge_m3_queries.npz com {matriz.shape[0]} consultas")

# confere que os 75 antigos nao mudaram
conf = np.load("dense_bge_m3_queries.npz", allow_pickle=True)
assert np.array_equal(conf["matrix"][:len(tex_antigo)], mat_antiga), "vetores antigos mudaram!"
assert [str(t) for t in conf["qtexts"]][:len(tex_antigo)] == tex_antigo
print("conferido: os 75 vetores originais estao intactos")

# ---------------------------------------------------------------- CELULA 6
import os, zipfile
pacote = "leme_f3.zip"
alvos = gerados + ["dense_bge_m3_queries.npz"]
with zipfile.ZipFile(pacote, "w", zipfile.ZIP_DEFLATED) as z:
    for arq in alvos:
        z.write(arq)
print(f"pacote: {pacote} ({os.path.getsize(pacote)/1e6:.1f} MB)")
for arq in alvos:
    print("  -", arq)
files.download(pacote)

# ============================================================================
# NO REPOSITORIO, depois de descompactar:
#
#   dense_bge_m3_queries.npz  -> eval/results/indexes/   (substitui o de 75)
#   rerank_scores_*.json      -> onde preferir; o caminho vai no --scores
#
#   PYTHONIOENCODING=utf-8 venv/Scripts/python.exe eval/tools/rerank_eval.py `
#     --rankings eval/results_e2/retrieval/rankings/B_dense_bge_m3.json `
#     --scores rerank_scores_BAAI_bge_reranker_v2_m3.json `
#     --model BAAI/bge-reranker-v2-m3 --pools 20,50,100 --out results_f3
#
# Nenhum modelo e carregado localmente com --scores.
# ============================================================================
