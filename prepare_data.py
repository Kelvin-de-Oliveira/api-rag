"""
Prepara todos os dados necessários para a API RAG ANEEL.

Idempotente: pode ser executado quantas vezes for necessário. Cada etapa
verifica se o trabalho já foi feito antes de executar.

Fluxo:
    1. Baixa o banco vetorial (banco_chroma/) do Hugging Face, se ainda não existe.
    2. Baixa os chunks (chunks/chunks.jsonl) do Hugging Face, se ainda não existem.
    3. Aguarda o Elasticsearch estar acessível.
    4. Popula o índice 'aneel_lexical' no Elasticsearch, se ainda está vazio.

Pré-requisitos:
    - Docker Compose rodando (para o Elasticsearch estar disponível).
    - Dependências instaladas: pip install -r requirements-setup.txt
"""
import json
import math
import os
import shutil
import sys
import time
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm
from elasticsearch import Elasticsearch
from langchain_core.documents import Document
from langchain_elasticsearch import ElasticsearchStore


URL_BANCO_CHROMA = (
    "https://huggingface.co/datasets/joaopauloCand/Embeddings_RAG_ANEEL/"
    "resolve/main/banco_chroma.zip?download=true"
)
URL_CHUNKS = (
    "https://huggingface.co/datasets/joaopauloCand/Embeddings_RAG_ANEEL/"
    "resolve/main/chunks.zip?download=true"
)

PASTA_BANCO_CHROMA = Path("banco_chroma")
ARQUIVO_VERIFICACAO_CHROMA = PASTA_BANCO_CHROMA / "chroma.sqlite3"
ZIP_BANCO_CHROMA = Path("banco_chroma.zip")

PASTA_CHUNKS = Path("chunks")
ARQUIVO_CHUNKS = PASTA_CHUNKS / "chunks.jsonl"
ZIP_CHUNKS = Path("chunks.zip")

ELASTICSEARCH_URL = os.getenv("ELASTICSEARCH_URL_HOST", "http://localhost:9200")
ELASTICSEARCH_INDEX = os.getenv("ELASTICSEARCH_INDEX_NAME", "aneel_lexical")

TAMANHO_LOTE_INGESTAO = 500
TOLERANCIA_CHUNKS_FALTANTES = 0.01


def info(msg: str) -> None:
    print(f"  {msg}")


def secao(titulo: str) -> None:
    print()
    print("=" * 60)
    print(f"  {titulo}")
    print("=" * 60)


def ok(msg: str) -> None:
    print(f"  OK: {msg}")


def aviso(msg: str) -> None:
    print(f"  AVISO:  {msg}")


def erro(msg: str) -> None:
    print(f"  ERRO: {msg}")


def baixar_arquivo(url: str, destino: Path) -> bool:
    try:
        with requests.get(url, stream=True, timeout=30) as resposta:
            resposta.raise_for_status()
            tamanho_total = int(resposta.headers.get("content-length", 0))

            with open(destino, "wb") as ficheiro, tqdm(
                desc=destino.name,
                total=tamanho_total,
                unit="iB",
                unit_scale=True,
                unit_divisor=1024,
            ) as barra:
                for chunk in resposta.iter_content(chunk_size=8192):
                    if chunk:
                        barra.update(ficheiro.write(chunk))
        return True

    except requests.exceptions.RequestException as e:
        erro(f"Falha no download de {url}: {e}")
        if destino.exists():
            destino.unlink()
        return False


def extrair_zip(arquivo_zip: Path, destino: Path = Path(".")) -> bool:
    try:
        with zipfile.ZipFile(arquivo_zip, "r") as zf:
            zf.extractall(destino)
        return True
    except Exception as e:
        erro(f"Falha ao descompactar {arquivo_zip}: {e}")
        return False


# ============================================================
# FASE 1 — BANCO VETORIAL
# ============================================================
def preparar_banco_chroma() -> bool:
    secao("FASE 1 — Banco vetorial (ChromaDB)")

    if ARQUIVO_VERIFICACAO_CHROMA.exists():
        ok(f"'{PASTA_BANCO_CHROMA}/' já está presente. Pulando.")
        return True

    if not ZIP_BANCO_CHROMA.exists():
        info("Banco vetorial não encontrado. Baixando do Hugging Face...")
        info("(arquivo grande — pode demorar vários minutos)")
        if not baixar_arquivo(URL_BANCO_CHROMA, ZIP_BANCO_CHROMA):
            return False
        ok("Download concluído.")
    else:
        info(f"'{ZIP_BANCO_CHROMA}' já existe — pulando download.")

    info("Descompactando...")
    if not extrair_zip(ZIP_BANCO_CHROMA):
        return False

    if not ARQUIVO_VERIFICACAO_CHROMA.exists():
        erro(
            f"Após descompactar, '{ARQUIVO_VERIFICACAO_CHROMA}' não foi encontrado. "
            "O ZIP pode ter sido descompactado em uma estrutura inesperada."
        )
        return False
    
    ZIP_BANCO_CHROMA.unlink()
    ok(f"Banco vetorial pronto em '{PASTA_BANCO_CHROMA}/'.")
    return True


# ============================================================
# FASE 2 — CHUNKS
# ============================================================
def preparar_chunks() -> bool:
    secao("FASE 2 — Chunks (chunks.jsonl)")

    if ARQUIVO_CHUNKS.exists():
        ok(f"'{ARQUIVO_CHUNKS}' já está presente. Pulando.")
        return True

    if not ZIP_CHUNKS.exists():
        info("Chunks não encontrados. Baixando do Hugging Face...")
        if not baixar_arquivo(URL_CHUNKS, ZIP_CHUNKS):
            return False
        ok("Download concluído.")
    else:
        info(f"'{ZIP_CHUNKS}' já existe — pulando download.")

    info("Descompactando...")
    if not extrair_zip(ZIP_CHUNKS):
        return False

    if not ARQUIVO_CHUNKS.exists():
        erro(
            f"Após descompactar, '{ARQUIVO_CHUNKS}' não foi encontrado. "
            "O ZIP pode ter sido descompactado em uma estrutura inesperada."
        )
        return False

    ZIP_CHUNKS.unlink()
    ok(f"Chunks prontos em '{ARQUIVO_CHUNKS}'.")
    return True


# ============================================================
# FASE 3 — AGUARDAR ELASTICSEARCH
# ============================================================
def aguardar_elasticsearch(timeout_total: int = 120) -> Elasticsearch | None:
    secao("FASE 3 — Aguardando Elasticsearch")

    info(f"Tentando conectar em {ELASTICSEARCH_URL}...")
    inicio = time.time()
    ultimo_erro = None

    while time.time() - inicio < timeout_total:
        try:
            cliente = Elasticsearch(ELASTICSEARCH_URL, request_timeout=5)
            if cliente.ping():
                health = cliente.cluster.health()
                ok(
                    f"Elasticsearch acessível. "
                    f"Status do cluster: {health.get('status', 'unknown')}."
                )
                return cliente
        except Exception as e:
            ultimo_erro = e

        time.sleep(2)
        print("  .", end="", flush=True)

    print()
    erro(f"Timeout aguardando Elasticsearch após {timeout_total}s.")
    if ultimo_erro:
        erro(f"Último erro: {ultimo_erro}")
    aviso(
        "Verifique se o Docker Compose subiu corretamente: "
        "`docker-compose ps` e `docker-compose logs elasticsearch`."
    )
    return None


# ============================================================
# FASE 4 — INGESTÃO NO ELASTICSEARCH
# ============================================================
def contar_chunks_no_arquivo(caminho: Path) -> int:
    with open(caminho, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def gerador_de_lotes(caminho: Path, tamanho_lote: int):
    lote = []
    with open(caminho, "r", encoding="utf-8") as f:
        for linha in f:
            dados = json.loads(linha)
            lote.append(
                Document(
                    page_content=dados["page_content"],
                    metadata=dados["metadata"],
                )
            )
            if len(lote) == tamanho_lote:
                yield lote
                lote = []
        if lote:
            yield lote


def popular_elasticsearch(cliente: Elasticsearch) -> bool:
    secao("FASE 4 — Ingestão no Elasticsearch")

    total_no_arquivo = contar_chunks_no_arquivo(ARQUIVO_CHUNKS)
    info(f"Total de chunks no arquivo: {total_no_arquivo}")

    indice_existe = cliente.indices.exists(index=ELASTICSEARCH_INDEX)
    total_no_indice = 0
    if indice_existe:
        total_no_indice = cliente.count(index=ELASTICSEARCH_INDEX).get("count", 0)
        info(f"Documentos atualmente no índice '{ELASTICSEARCH_INDEX}': {total_no_indice}")

    minimo_aceitavel = total_no_arquivo * (1 - TOLERANCIA_CHUNKS_FALTANTES)
    if total_no_indice >= minimo_aceitavel:
        ok(f"Índice '{ELASTICSEARCH_INDEX}' já está populado. Pulando ingestão.")
        return True

    if indice_existe and total_no_indice > 0:
        aviso(
            f"Índice está parcialmente populado ({total_no_indice}/{total_no_arquivo}). "
            "Apagando para reingerir do zero (garantia de consistência)."
        )
        cliente.indices.delete(index=ELASTICSEARCH_INDEX)

    info("Iniciando ingestão. Pode demorar 10-20 minutos.")

    banco_lexical = ElasticsearchStore(
        es_url=ELASTICSEARCH_URL,
        index_name=ELASTICSEARCH_INDEX,
        strategy=ElasticsearchStore.BM25RetrievalStrategy(),
    )

    total_lotes = math.ceil(total_no_arquivo / TAMANHO_LOTE_INGESTAO)
    chunks_inseridos = 0

    try:
        for lote in tqdm(
            gerador_de_lotes(ARQUIVO_CHUNKS, TAMANHO_LOTE_INGESTAO),
            total=total_lotes,
            desc="Indexando no Elasticsearch",
        ):
            banco_lexical.add_documents(lote)
            chunks_inseridos += len(lote)

    except KeyboardInterrupt:
        print()
        aviso(
            f"Ingestão interrompida pelo usuário. "
            f"{chunks_inseridos}/{total_no_arquivo} chunks indexados parcialmente. "
            "Rode novamente para reiniciar do zero."
        )
        return False
    except Exception as e:
        print()
        erro(f"Erro durante a ingestão: {e}")
        return False

    ok(f"Ingestão concluída. {chunks_inseridos} chunks indexados em '{ELASTICSEARCH_INDEX}'.")
    return True


def main() -> int:
    print()
    print("╔" + "═" * 58 + "╗")
    print("║" + "  PREPARAÇÃO DE DADOS — RAG ANEEL".ljust(58) + "║")
    print("╚" + "═" * 58 + "╝")

    if not preparar_banco_chroma():
        erro("Falha na preparação do banco vetorial.")
        return 1

    if not preparar_chunks():
        erro("Falha na preparação dos chunks.")
        return 1

    cliente = aguardar_elasticsearch()
    if cliente is None:
        return 1

    if not popular_elasticsearch(cliente):
        erro("Falha na ingestão do Elasticsearch.")
        return 1

    secao("TUDO PRONTO!")
    ok("API disponível em: http://localhost:8000")
    ok("Documentação interativa: http://localhost:8000/docs")
    ok("Verificação de saúde: http://localhost:8000/health")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())