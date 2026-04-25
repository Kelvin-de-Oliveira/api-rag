import json
import math
from tqdm import tqdm
from langchain_core.documents import Document
from langchain_elasticsearch import ElasticsearchStore

# Configurações
TAMANHO_LOTE = 500  # Quantidade de documentos a serem enviados por lote para o Elasticsearch
DOCKER_ELASTICSEARCH_URL = "http://localhost:9200"
ELASTICSEARCH_INDEX_NAME = "aneel_lexical"
ARQUIVO_JSONL = "chunks\\chunks.jsonl"

def gerador_de_lotes_es(caminho_arquivo: str, tamanho_lote: int) -> iter:
    """Lê o JSONL de forma preguiçosa para não sobrecarregar a RAM do VS Code."""
    lote_atual = []
    with open(caminho_arquivo, 'r', encoding='utf-8') as f:
        for linha in f:
            dados = json.loads(linha)
            doc = Document(page_content=dados["page_content"], metadata=dados["metadata"])
            lote_atual.append(doc)
            
            if len(lote_atual) == tamanho_lote:
                yield lote_atual
                lote_atual = []
        if lote_atual:
            yield lote_atual

def inserir_elasticsearch(arquivo_jsonl: str = ARQUIVO_JSONL, nome_indice: str = ELASTICSEARCH_INDEX_NAME, url_es: str = DOCKER_ELASTICSEARCH_URL, tamanho_lote: int = TAMANHO_LOTE) -> None:
    """Função principal para popular o Elasticsearch com os chunks do JSONL."""
    print("🚀 Iniciando ingestão no Elasticsearch (Busca Lexical/BM25)...")
    
    # Conecta ao ES e define a estratégia EXCLUSIVA de BM25
    banco_lexical = ElasticsearchStore(
        es_url=url_es,
        index_name=nome_indice,
        strategy=ElasticsearchStore.BM25RetrievalStrategy()
    )
    
    # Contamos o total de linhas apenas para a barra de progresso
    with open(arquivo_jsonl, 'r', encoding='utf-8') as f:
        total_chunks = sum(1 for _ in f)
        
    total_lotes = math.ceil(total_chunks / tamanho_lote) if total_chunks else 0
    lotes = gerador_de_lotes_es(arquivo_jsonl, tamanho_lote)
    
    chunks_inseridos = 0
    for lote in tqdm(lotes, total=total_lotes, desc="Enviando para o Elasticsearch"):
        banco_lexical.add_documents(lote)
        chunks_inseridos += len(lote)
        
    print(f"\n✅ Concluído! {chunks_inseridos} chunks indexados no Elasticsearch.")

if __name__ == "__main__":
    inserir_elasticsearch()