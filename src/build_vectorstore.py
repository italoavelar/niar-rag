"""Indexação Gemini no Qdrant, com troca de coleção por apelido (alias).

POR QUE ESTE ARQUIVO MUDOU (30/09/2026). A versão anterior chamava
`recreate_collection` na coleção que a produção serve: apagava tudo e só então
começava a subir os 6.134 pontos. Qualquer queda no meio — rede, 429, Ctrl-C —
deixava o índice PARCIAL e SEM AVISO. A busca continuava respondendo, com menos
corpus, e nada no sistema denunciava a falta. É o pior tipo de falha: silenciosa
e plausível.

O PADRÃO SEGURO, que o `build_qwen_vectorstore.py` já usava neste repositório:

    1. a coleção física tem nome VERSIONADO pelo fingerprint do corpus
       (`leme_gemini_<16 hex>`) e é criada vazia, do lado, sem tocar na servida;
    2. sobe-se o corpus inteiro nela;
    3. VERIFICA-SE: contagem, conjunto de ids, perfil de embedding no payload;
    4. só então o APELIDO estável (`leme_gemini`) passa a apontar para ela.

A troca do apelido é uma operação atômica do Qdrant. Se o passo 2 ou 3 falhar, o
apelido continua na coleção antiga e a produção não percebeu nada. A coleção
anterior fica em disco para rollback — `--promover --para <antiga>` volta atrás
em um segundo, e `--limpar` remove as velhas quando ninguém mais precisar.

Ids de ponto DETERMINÍSTICOS. Antes o id do ponto era a posição no corpus, o que
tornava o upload não-idempotente: mudar a ordem do corpus fazia um upsert
sobrescrever ponto alheio. Agora é `uuid5` do id do trecho — que já é endereçado
por conteúdo. Consequência prática: o upload virou retomável (`--upload` de novo
só manda o que falta) e a busca não muda, porque o avaliador e o agente leem o
trecho por `payload['id_original']`, nunca pelo id do ponto.

Uso:
  python src/build_vectorstore.py --listar          # leitura: coleções e apelidos
  python src/build_vectorstore.py --gerar            # só embeddings (custa API)
  python src/build_vectorstore.py --upload           # sobe na coleção nova
  python src/build_vectorstore.py --verificar        # confere sem alterar nada
  python src/build_vectorstore.py --promover         # move o apelido
  python src/build_vectorstore.py --tudo             # gerar → upload → verificar → promover
"""

from __future__ import annotations

import argparse
import json
import os
import time
import uuid
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types
from qdrant_client import QdrantClient
from qdrant_client.http import models
from tqdm import tqdm

from embedding_text import (
    EMBEDDING_TEXT_PROFILE,
    build_embedding_text,
    corpus_embedding_fingerprint,
    embedding_text_hash,
)

load_dotenv()

# APELIDO estável: é este nome que `eval/config.yaml` (embedders.gemini
# .qdrant_collection) e o agente consultam. Ele nunca é uma coleção real depois
# da primeira promoção — é um ponteiro para a coleção versionada da vez.
#
# minúsculo de propósito: o Qdrant diferencia maiúsculas, e o `LEME_gemini` que
# estava aqui como padrão devolvia 404 contra a coleção real `leme_gemini`.
COLLECTION_ALIAS = os.getenv("QDRANT_GEMINI_COLLECTION", "leme_gemini").strip()

# Mantido pelo nome antigo porque `tests/test_embedding_text.py` e o restante do
# repositório importam `COLLECTION_NAME`.
COLLECTION_NAME = COLLECTION_ALIAS

# Namespace fixo para o uuid5 dos pontos. Trocar este valor reescreve todos os
# ids, então ele é constante do formato de índice, não configuração.
NAMESPACE_PONTOS = uuid.UUID("6f9c1f52-4d2a-5b7e-9c31-0a5d8e2b7f14")

JSONL_FILE = Path("data/processed/documents.jsonl")
LEGACY_BACKUP_FILE = Path("data/processed/embeddings_backup.jsonl")
BACKUP_FILE = Path(
    os.getenv(
        "GEMINI_CONTEXTUAL_BACKUP_FILE",
        "data/processed/embeddings_context_v1_backup.jsonl",
    )
)
if BACKUP_FILE == LEGACY_BACKUP_FILE:
    raise ValueError(
        "GEMINI_CONTEXTUAL_BACKUP_FILE não pode apontar para o backup legado."
    )

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
GOOGLE_GENAI_API_KEY = os.getenv("GOOGLE_GENAI_API_KEY")

EMBED_DIM = 3072

BATCH_SIZE_EMBEDDINGS = 20
BATCH_SIZE_QDRANT = 64

# Pausa entre lotes. O padrao 15s foi calibrado para o limite por minuto do
# free tier; em plano pago da para reduzir. Um 429 nao perde trabalho: o
# backup e gravado a cada lote e a execucao retoma de onde parou.
SLEEP_BETWEEN_BATCHES = float(os.getenv("GEMINI_SLEEP_BETWEEN_BATCHES", "15"))


# Normaliza um vetor para ter norma 1 (unitário)
def normalize(vec):
    v = np.array(vec)
    norm = np.linalg.norm(v)

    if norm == 0:
        return v.tolist()

    return (v / norm).tolist()


# Carrega os documentos processados do arquivo JSONL
def load_documents():
    documents = []

    with open(JSONL_FILE, "r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                documents.append(json.loads(line))

    return documents


def build_embedding_inputs(documents: list[dict]) -> list[str]:
    """Monta as entradas contextuais sem alterar os registros originais."""
    return [build_embedding_text(document) for document in documents]


# Carrega o backup de embeddings já processados para evitar retrabalho em caso de falhas
def load_backup(documents: list[dict]):
    # O backup é indexado pelo TEXTO DE EMBEDDING, não por posição nem por
    # fingerprint.
    #
    # Antes, `load_backup` assumia que o backup era o prefixo do corpus atual, na
    # mesma ordem, e conferia `documents[index]` contra `processed_data[index]`.
    # Isso quebra sempre que o corpus é reprocessado: remover um documento ou
    # mudar o recorte desloca tudo a partir do primeiro trecho alterado, e o
    # script aborta com "fingerprint não confere" mesmo tendo em mãos milhares de
    # vetores reaproveitáveis.
    #
    # Por que NÃO dá para casar por `corpus_embedding_fingerprint`: ele hasheia o
    # **id** junto com o texto (`src/embedding_text.py:61-75`). Como os ids
    # deixaram de ser posicionais e passaram a vir do conteúdo, todo fingerprint
    # mudou — inclusive o dos trechos cujo texto é idêntico. Medido: casando por
    # fingerprint, 0 de 4.897 seriam reaproveitados.
    #
    # A chave certa é o texto que de fato produziu o vetor: `build_embedding_text`,
    # com o prefixo `[título · emissor · seção]` incluído. Se o prefixo mudou, o
    # vetor não serve — e é justo recalcular.
    por_texto = {}

    if BACKUP_FILE.exists():
        with open(BACKUP_FILE, "r", encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue
                item = json.loads(line)
                if item.get("embedding_text_profile") != EMBEDDING_TEXT_PROFILE:
                    # Falha alta, de propósito: um backup de outro perfil não é
                    # reaproveitável, e seguir em silêncio significaria recalcular
                    # o acervo inteiro com a API sem ninguém perceber a conta.
                    raise ValueError(
                        "Backup incompatível: perfil de embedding ausente ou diferente "
                        f"de {EMBEDDING_TEXT_PROFILE!r}."
                    )
                doc_backup = item.get("document") or {}
                chave = build_embedding_text(doc_backup)
                if chave:
                    por_texto[chave] = item

        aproveitaveis = sum(
            1 for doc in documents if build_embedding_text(doc) in por_texto
        )
        print(
            f"Backup {EMBEDDING_TEXT_PROFILE}: {len(por_texto)} vetores em disco, "
            f"{aproveitaveis} reaproveitáveis para os {len(documents)} trechos atuais "
            f"({len(documents) - aproveitaveis} a calcular)."
        )
    else:
        print(
            f"Nenhum backup {EMBEDDING_TEXT_PROFILE} encontrado. "
            "Iniciando do zero."
        )

    return por_texto


# Gera embeddings para os documentos usando a API Gemini e salva em backup
def generate_embeddings(documents, por_texto, *, permitir_api: bool = True):
    """Devolve um registro por documento, NA ORDEM DO CORPUS.

    Reaproveita do backup todo trecho cujo texto contextual não mudou e calcula
    só o resto. A saída segue `documents`, então o upload para o Qdrant fica
    alinhado com o corpus mesmo que o backup esteja em outra ordem.

    `permitir_api=False` é o que torna `--upload` e `--verificar` gratuitos: se
    faltar algum vetor, falha em vez de abrir a carteira sem avisar.
    """
    processed_data = []
    documents_to_process = []
    for doc in documents:
        achado = por_texto.get(build_embedding_text(doc))
        if achado is not None:
            # O vetor é reaproveitado, mas o documento vem do corpus atual e o
            # fingerprint é recalculado: ele carrega o id, que mudou.
            processed_data.append({
                **achado,
                "document": doc,
                "embedding_text_fingerprint": corpus_embedding_fingerprint([doc]),
            })
        else:
            processed_data.append(None)
            documents_to_process.append(doc)

    if not documents_to_process:
        print("Todos os embeddings já estavam no backup.")
        return processed_data

    if not permitir_api:
        raise RuntimeError(
            f"{len(documents_to_process)} de {len(documents)} trechos estão sem "
            "vetor no backup. Rode `--gerar` primeiro: esta etapa não chama a "
            "API do Gemini por conta própria."
        )

    print(f"A calcular: {len(documents_to_process)} de {len(documents)} trechos.")

    print("Inicializando cliente Gemini...")
    client = genai.Client(api_key=GOOGLE_GENAI_API_KEY)

    BACKUP_FILE.parent.mkdir(parents=True, exist_ok=True)
    novos = {}

    with open(BACKUP_FILE, "a", encoding="utf-8") as backup:
        for i in tqdm(
            range(0, len(documents_to_process), BATCH_SIZE_EMBEDDINGS),
            desc="Gerando embeddings",
        ):
            batch = documents_to_process[i:i + BATCH_SIZE_EMBEDDINGS]
            texts = build_embedding_inputs(batch)

            try:
                response = client.models.embed_content(
                    model="gemini-embedding-001",
                    contents=texts,
                    config=types.EmbedContentConfig(
                        task_type="RETRIEVAL_DOCUMENT",
                        title="Base de Conhecimento NIAR Saúde",
                    ),
                )

                for doc, emb in zip(batch, response.embeddings):
                    record = {
                        "document": doc,
                        "vector": normalize(emb.values),
                        "embedding_text_profile": EMBEDDING_TEXT_PROFILE,
                        "embedding_text_fingerprint": (
                            corpus_embedding_fingerprint([doc])
                        ),
                        # hash SÓ do texto: é por ele que se reaproveita vetor
                        # depois de uma troca de id. Ver embedding_text_hash.
                        "embedding_text_hash": embedding_text_hash(doc),
                    }

                    backup.write(json.dumps(record, ensure_ascii=False) + "\n")
                    novos[doc["id"]] = record

                time.sleep(SLEEP_BETWEEN_BATCHES)

            except Exception as error:
                print("Erro ao gerar embeddings. Progresso salvo no backup.")
                print(f"Detalhe: {error}")
                raise

    # encaixa os recém-calculados nos buracos, preservando a ordem do corpus
    for i, doc in enumerate(documents):
        if processed_data[i] is None:
            processed_data[i] = novos[doc["id"]]

    faltando = [i for i, r in enumerate(processed_data) if r is None]
    if faltando:
        raise RuntimeError(
            f"{len(faltando)} trechos ficaram sem vetor — não enviar ao Qdrant."
        )
    return processed_data


# Constrói o payload para cada ponto a ser inserido no Qdrant com os metadados
def build_payload(doc: dict) -> dict:
    metadata = doc.get("metadata", {})

    return {
        "id_original": doc.get("id", ""),
        "document_id": metadata.get("document_id", ""),
        "texto": doc.get("text", ""),
        "fonte": metadata.get("source", ""),
        "source_type": metadata.get("source_type", ""),
        "title": metadata.get("title", ""),
        "page": metadata.get("page"),
        "chunk": metadata.get("chunk", ""),
        "document_type": metadata.get("document_type", ""),
        "author": metadata.get("author", ""),
        "issuer": metadata.get("issuer", ""),
        "year": metadata.get("year", ""),
        "theme": metadata.get("theme", ""),
        "ria_dimensions": metadata.get("ria_dimensions", []),
        "source_url": metadata.get("source_url", ""),
        "section_path": metadata.get("section_path", ""),
        "embedding_text_profile": EMBEDDING_TEXT_PROFILE,
    }


# ── nomes, apelidos e cliente ───────────────────────────────────────────────

def nome_fisico(corpus_fingerprint: str) -> str:
    """Nome imutável da coleção, amarrado ao corpus que ela contém.

    Amarrar ao fingerprint é o que impede a confusão mais cara da casa: subir um
    corpus novo e, meses depois, não saber qual índice corresponde a qual
    recorte. Aqui o nome responde sozinho.
    """
    if not COLLECTION_ALIAS:
        raise ValueError(
            "QDRANT_GEMINI_COLLECTION está vazio. Defina-o no .env."
        )
    if not corpus_fingerprint:
        raise ValueError("Fingerprint do corpus não pode ser vazio.")
    return f"{COLLECTION_ALIAS}_{corpus_fingerprint[:16]}"


def conectar() -> QdrantClient:
    if not QDRANT_URL or not QDRANT_API_KEY:
        raise RuntimeError("Defina QDRANT_URL e QDRANT_API_KEY no .env.")
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=120)


def id_do_ponto(chunk_id: str) -> str:
    """uuid5 do id do trecho — determinístico, e o trecho já é endereçado por
    conteúdo. Dois efeitos: o upload fica idempotente (dá para retomar) e nunca
    sobrescreve ponto alheio quando a ordem do corpus muda."""
    return str(uuid.uuid5(NAMESPACE_PONTOS, str(chunk_id)))


def _colecoes(client: QdrantClient) -> set[str]:
    return {c.name for c in client.get_collections().collections}


def _apelidos(client: QdrantClient) -> dict[str, str]:
    return {a.alias_name: a.collection_name for a in client.get_aliases().aliases}


def listar(client: QdrantClient) -> None:
    """Leitura pura: o que existe hoje no cluster."""
    apelidos = _apelidos(client)
    print(f"apelido servido : {COLLECTION_ALIAS}")
    destino = apelidos.get(COLLECTION_ALIAS)
    if destino:
        print(f"  → aponta para : {destino}")
    elif COLLECTION_ALIAS in _colecoes(client):
        print("  → é uma COLEÇÃO REAL, ainda não um apelido "
              "(ver --assumir-nome)")
    else:
        print("  → não existe ainda")

    print("\ncoleções no cluster:")
    for nome in sorted(_colecoes(client)):
        try:
            n = client.count(nome, exact=True).count
        except Exception:
            n = "?"
        marca = "  ← servida" if apelidos.get(COLLECTION_ALIAS) == nome else ""
        print(f"  {nome:56s} {n:>8} pontos{marca}")

    if apelidos:
        print("\napelidos:")
        for a, c in sorted(apelidos.items()):
            print(f"  {a:40s} → {c}")


# ── upload, verificação e promoção ──────────────────────────────────────────

def upload_to_qdrant(processed_data, *, colecao: str,
                     recriar: bool = False, pular_existentes: bool = True):
    """Sobe o corpus numa coleção que NÃO é a servida.

    Nada aqui toca no apelido. É a propriedade que faz a queda no meio do upload
    deixar de ser incidente: a produção continua servindo a coleção antiga.
    """
    client = conectar()
    existe = client.collection_exists(colecao)

    if existe and not recriar:
        print(f"Coleção {colecao!r} já existe — completando o que falta.")
    elif existe and recriar:
        print(f"Apagando e recriando {colecao!r} (--recriar).")
        client.delete_collection(colecao)
        existe = False

    if not existe:
        client.create_collection(
            collection_name=colecao,
            vectors_config=models.VectorParams(
                size=EMBED_DIM,
                distance=models.Distance.COSINE,
            ),
        )
        print(f"Coleção {colecao!r} criada com {EMBED_DIM} dimensões.")

    pendentes = processed_data
    if pular_existentes and existe:
        alvo = [id_do_ponto(item["document"]["id"]) for item in processed_data]
        presentes: set[str] = set()
        for i in range(0, len(alvo), 256):
            fatia = alvo[i:i + 256]
            for p in client.retrieve(colecao, ids=fatia, with_payload=False,
                                     with_vectors=False):
                presentes.add(str(p.id))
        pendentes = [
            item for item in processed_data
            if id_do_ponto(item["document"]["id"]) not in presentes
        ]
        print(f"{len(presentes)} pontos já estavam lá; "
              f"{len(pendentes)} a enviar.")

    if not pendentes:
        print("Nada a enviar.")
        return colecao

    for i in tqdm(range(0, len(pendentes), BATCH_SIZE_QDRANT),
                  desc=f"Enviando para {colecao}"):
        lote = pendentes[i:i + BATCH_SIZE_QDRANT]
        pontos = [
            models.PointStruct(
                id=id_do_ponto(item["document"]["id"]),
                vector=item["vector"],
                payload=build_payload(item["document"]),
            )
            for item in lote
        ]
        # wait=True para a falha aparecer AQUI, e não como buraco silencioso.
        client.upsert(collection_name=colecao, points=pontos, wait=True)

    return colecao


def verificar(documents: list[dict], *, colecao: str) -> None:
    """Confere a coleção contra o corpus local. Não altera nada.

    É este passo que autoriza a promoção. Sem ele, "subiu" significa apenas que
    o script não levantou exceção — que é exactamente o que um upload
    interrompido também parece.
    """
    client = conectar()
    if not client.collection_exists(colecao):
        raise ValueError(f"Coleção inexistente: {colecao!r}.")

    info = client.get_collection(colecao)
    vetores = info.config.params.vectors
    dim = vetores.size if hasattr(vetores, "size") else vetores["size"]
    if dim != EMBED_DIM:
        raise ValueError(f"Dimensão incompatível: {dim}; esperado {EMBED_DIM}.")

    esperados = [str(d["id"]) for d in documents]
    if len(set(esperados)) != len(esperados):
        raise ValueError("Corpus local tem id de trecho duplicado.")

    n = client.count(colecao, exact=True).count
    if n != len(esperados):
        raise ValueError(
            f"Contagem de pontos incompatível: {n}; esperado {len(esperados)}."
        )

    achados: list[str] = []
    offset = None
    while True:
        pontos, offset = client.scroll(
            colecao, limit=256, offset=offset,
            with_payload=True, with_vectors=False,
        )
        for p in pontos:
            payload = p.payload or {}
            if payload.get("embedding_text_profile") != EMBEDDING_TEXT_PROFILE:
                raise ValueError(
                    f"Perfil de embedding incompatível no ponto {p.id}."
                )
            cid = str(payload.get("id_original") or "")
            if not cid:
                raise ValueError(f"Ponto {p.id} sem id_original.")
            achados.append(cid)
        if offset is None:
            break

    if len(achados) != len(set(achados)):
        raise ValueError("Coleção tem id_original duplicado.")
    faltam = set(esperados) - set(achados)
    sobram = set(achados) - set(esperados)
    if faltam or sobram:
        raise ValueError(
            f"Ids não correspondem ao corpus: ausentes={len(faltam)}, "
            f"extras={len(sobram)}."
        )

    print(f"✓ verificação OK — {colecao}: {len(achados)} pontos, "
          f"perfil {EMBEDDING_TEXT_PROFILE}, dimensão {dim}.")


def promover(*, colecao: str, assumir_nome: bool = False) -> None:
    """Move o apelido servido para `colecao`. Operação atômica no Qdrant.

    Este é o único ponto do arquivo que muda o que a produção vê, e ele roda
    depois da verificação. Voltar atrás é `--promover --para <coleção antiga>`.
    """
    client = conectar()
    if not client.collection_exists(colecao):
        raise ValueError(f"Coleção inexistente: {colecao!r}.")

    apelidos = _apelidos(client)
    anterior = apelidos.get(COLLECTION_ALIAS)

    # Migração de uma vez só: enquanto `leme_gemini` for uma COLEÇÃO real, o
    # Qdrant não aceita um apelido com o mesmo nome. Só se remove a coleção
    # antiga sob pedido explícito — e a esta altura a nova já passou na
    # verificação, então o dado não está em risco.
    if anterior is None and client.collection_exists(COLLECTION_ALIAS):
        if COLLECTION_ALIAS == colecao:
            raise ValueError(
                f"{COLLECTION_ALIAS!r} é a própria coleção física. Suba o corpus "
                "numa coleção versionada (--upload) antes de promover."
            )
        if not assumir_nome:
            raise ValueError(
                f"{COLLECTION_ALIAS!r} existe como COLEÇÃO real, não como "
                f"apelido. Para transformá-la em apelido de {colecao!r} é "
                f"preciso APAGAR a coleção {COLLECTION_ALIAS!r} — rode de novo "
                "com --assumir-nome. A coleção nova já está verificada, então "
                "o corpus não corre risco; o que se perde é só o índice antigo."
            )
        print(f"Apagando a coleção real {COLLECTION_ALIAS!r} para liberar o nome...")
        client.delete_collection(COLLECTION_ALIAS)

    operacoes = []
    if anterior is not None:
        operacoes.append(models.DeleteAliasOperation(
            delete_alias=models.DeleteAlias(alias_name=COLLECTION_ALIAS)
        ))
    operacoes.append(models.CreateAliasOperation(
        create_alias=models.CreateAlias(
            collection_name=colecao, alias_name=COLLECTION_ALIAS
        )
    ))
    client.update_collection_aliases(change_aliases_operations=operacoes)

    print(f"✓ {COLLECTION_ALIAS} → {colecao}")
    if anterior and anterior != colecao:
        print(f"  (antes: {anterior} — mantida em disco para rollback)")
        print(f"  rollback: python src/build_vectorstore.py --promover "
              f"--para {anterior}")


def limpar(*, manter: int = 1) -> None:
    """Remove coleções versionadas antigas, preservando a servida e as `manter`
    mais recentes. Nunca remove aquela para onde o apelido aponta."""
    client = conectar()
    servida = _apelidos(client).get(COLLECTION_ALIAS)
    candidatas = sorted(
        c for c in _colecoes(client)
        if c.startswith(f"{COLLECTION_ALIAS}_") and c != servida
    )
    a_remover = candidatas[:-manter] if manter else candidatas
    if not a_remover:
        print("Nada a limpar.")
        return
    for c in a_remover:
        print(f"apagando {c}")
        client.delete_collection(c)


# ── orquestração ────────────────────────────────────────────────────────────

def build_vectorstore(modo: str = "tudo", *, colecao: str | None = None,
                      recriar: bool = False, assumir_nome: bool = False,
                      manter: int = 1) -> None:
    if modo == "listar":
        listar(conectar())
        return
    if modo == "limpar":
        limpar(manter=manter)
        return

    print("Carregando chunks...")
    documents = load_documents()
    print(f"{len(documents)} chunks encontrados.")
    fingerprint = corpus_embedding_fingerprint(documents)
    destino = colecao or nome_fisico(fingerprint)
    print(f"fingerprint do corpus: {fingerprint[:16]}")
    print(f"coleção de destino   : {destino}")
    print(f"apelido servido      : {COLLECTION_ALIAS}")

    if modo == "promover":
        promover(colecao=destino, assumir_nome=assumir_nome)
        return
    if modo == "verificar":
        verificar(documents, colecao=destino)
        return

    if modo == "gerar":
        generate_embeddings(documents, load_backup(documents))
        print("✓ backup completo. Próximo: --upload")
        return

    if modo == "upload":
        dados = generate_embeddings(documents, load_backup(documents),
                                    permitir_api=False)
        upload_to_qdrant(dados, colecao=destino, recriar=recriar)
        verificar(documents, colecao=destino)
        print("✓ coleção pronta e verificada. Próximo: --promover")
        return

    if modo == "tudo":
        dados = generate_embeddings(documents, load_backup(documents))
        upload_to_qdrant(dados, colecao=destino, recriar=recriar)
        verificar(documents, colecao=destino)
        promover(colecao=destino, assumir_nome=assumir_nome)
        print("Indexação finalizada com sucesso.")
        print(f"Total de chunks indexados: {len(dados)}")
        return

    raise ValueError(f"Modo desconhecido: {modo!r}.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    m = ap.add_mutually_exclusive_group(required=True)
    m.add_argument("--listar", action="store_true",
                   help="Leitura: coleções, contagens e apelidos.")
    m.add_argument("--gerar", action="store_true",
                   help="Só os embeddings do Gemini (CUSTA API). Não toca no Qdrant.")
    m.add_argument("--upload", action="store_true",
                   help="Sobe o backup numa coleção versionada e verifica. "
                        "Não mexe no apelido servido.")
    m.add_argument("--verificar", action="store_true",
                   help="Confere a coleção contra o corpus local; não altera nada.")
    m.add_argument("--promover", action="store_true",
                   help="Move o apelido servido para a coleção verificada.")
    m.add_argument("--limpar", action="store_true",
                   help="Apaga coleções versionadas antigas (nunca a servida).")
    m.add_argument("--tudo", action="store_true",
                   help="gerar → upload → verificar → promover.")
    ap.add_argument("--para", dest="colecao",
                    help="Coleção física explícita; use para rollback.")
    ap.add_argument("--recriar", action="store_true",
                    help="Apaga a coleção de DESTINO antes de subir. "
                         "Nunca afeta a servida.")
    ap.add_argument("--assumir-nome", action="store_true",
                    help="Na primeira promoção: apaga a coleção real de mesmo "
                         "nome do apelido para liberá-lo.")
    ap.add_argument("--manter", type=int, default=1,
                    help="Quantas coleções antigas preservar em --limpar.")
    a = ap.parse_args()

    modo = ("listar" if a.listar else "gerar" if a.gerar else
            "upload" if a.upload else "verificar" if a.verificar else
            "promover" if a.promover else "limpar" if a.limpar else "tudo")
    build_vectorstore(modo, colecao=a.colecao, recriar=a.recriar,
                      assumir_nome=a.assumir_nome, manter=a.manter)


if __name__ == "__main__":
    main()
