import os
import json
import time
import math
from tqdm import tqdm
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

# ==========================================
# 1. CONFIGURAÇÕES
# ==========================================

load_dotenv() # Carrega a chave de API do Google do arquivo .env, se existir
TAMANHO_LOTE = 100 #Tamanho do lote enviado
TOTAL_CHUNKS_DIRETORIO = "chunks\\total_chunks_gerados.txt" #tem apenas um número inteiro
ARQUIVO_JSONL = "chunks\\chunks.jsonl"
DIRETORIO_CHROMA = "banco_chroma"
ARQUIVO_CHECKPOINT = "embedding_checkpoint.txt"
MODEL_EMBEDDING = "models/gemini-embedding-001"

try:
    if os.path.exists(TOTAL_CHUNKS_DIRETORIO):
            with open(TOTAL_CHUNKS_DIRETORIO, 'r', encoding='utf-8') as f:
                TOTAL_CHUNKS_ESPERADOS = int(f.read().strip())
except Exception as e:
    print(f"⚠️ Erro ao ler o total de chunks esperados: {e}")
    TOTAL_CHUNKS_ESPERADOS = 297858 # Valor padrão caso haja um problema

# ==========================================
# 2. FUNÇÕES DE CHECKPOINT
# ==========================================
def carregar_linha_atual(arquivo_checkpoint:str) -> int:
    """Carrega o número da última linha processada a partir do arquivo de checkpoint. Retorna 0 se o arquivo não existir ou estiver vazio."""
    if os.path.exists(arquivo_checkpoint):
        with open(arquivo_checkpoint, 'r') as f:
            return int(f.read().strip())
    return 0

def salvar_linha_atual(numero_linha: int, arquivo_checkpoint: str) -> None:
    """Salva o número da última linha processada no arquivo de checkpoint. Sobrescreve o arquivo a cada chamada para garantir que o progresso seja atualizado."""
    with open(arquivo_checkpoint, 'w') as f:
        f.write(str(numero_linha))

# ==========================================
# 3. O LEITOR PREGUIÇOSO (Gerador de Lotes)
# ==========================================

def gerador_de_lotes(caminho_arquivo: str, linha_inicio: int, tamanho_lote: int)-> iter:
    """Gerador que lê o arquivo JSONL a partir de uma linha específica e produz lotes de documentos para vetorização. Isso permite que o processo de vetorização seja eficiente em termos de memória, mesmo para arquivos muito grandes."""
    lote_atual = []
    
    with open(caminho_arquivo, 'r', encoding='utf-8') as f:
        # Pula as linhas já vetorizadas
        for _ in range(linha_inicio):
            next(f)
            
        for linha in f:
            dados = json.loads(linha)
            doc = Document(
                page_content=dados["page_content"], 
                metadata=dados["metadata"]
            )
            lote_atual.append(doc)
            
            if len(lote_atual) == tamanho_lote:
                yield lote_atual
                lote_atual = []
        
        if lote_atual:
            yield lote_atual

# ==========================================
# 4. O MOTOR DE VETORIZAÇÃO BLINDADO
# ==========================================
def processar_embeddings(arquivo_jsonl: str = ARQUIVO_JSONL, diretorio_chroma: str = DIRETORIO_CHROMA, arquivo_checkpoint: str = ARQUIVO_CHECKPOINT) -> None:
    """Função principal que orquestra o processo de vetorização, utilizando o Google Generative AI Embeddings e o ChromaDB. O processo é robusto, com tratamento de erros e um sistema de retry exponencial para lidar com limites de API, garantindo que o progresso seja salvo a cada lote processado."""
    print("🚀 A iniciar Vetorização em Nuvem (Embeddings)...")
    
    # Task_type configurado para melhorar a semântica de banco de conhecimento
    embeddings_google = GoogleGenerativeAIEmbeddings(
        model=MODEL_EMBEDDING,
        task_type="RETRIEVAL_DOCUMENT" 
    )
    
    banco_vetorial = Chroma(
        persist_directory=diretorio_chroma, 
        embedding_function=embeddings_google
    )
    
    linha_inicio = carregar_linha_atual(arquivo_checkpoint)
    
    if linha_inicio >= TOTAL_CHUNKS_ESPERADOS:
        print("🎉 O Banco de Dados Vetorial já está 100% completo!")
        return

    print(f"📍 A retomar a partir do chunk nº {linha_inicio}")
    
    lotes = gerador_de_lotes(arquivo_jsonl, linha_inicio, TAMANHO_LOTE)
    
    # math.ceil garante que o último lote incompleto seja contabilizado na barra
    total_lotes = math.ceil((TOTAL_CHUNKS_ESPERADOS - linha_inicio) / TAMANHO_LOTE)
    
    linha_atual = linha_inicio
    
    try:
        for lote in tqdm(lotes, total=total_lotes, desc="A vetorizar no Google"):
            
            sucesso_no_lote = False
            tentativas = 0
            
            # Limite aumentado para 8 tentativas (até ~4 minutos de paciência)
            while not sucesso_no_lote and tentativas < 8:
                try:
                    banco_vetorial.add_documents(lote)
                    sucesso_no_lote = True
                    
                except Exception as e:
                    tentativas += 1
                    tempo_espera = 2 ** tentativas # 2, 4, 8, 16, 32, 64... segundos
                    print(f"\n⚠️ Limite atingido/Erro na API: \n{str(e)[:100]}")
                    print(f"🔄 A aguardar {tempo_espera}s para a cota renovar (Tentativa {tentativas}/8)...")
                    time.sleep(tempo_espera)
            
            if not sucesso_no_lote:
                print(f"\n❌ Falha irreversível após 8 tentativas. Abortando para segurança.")
                break
                
            linha_atual += len(lote)
            salvar_linha_atual(linha_atual, arquivo_checkpoint)

    except KeyboardInterrupt:
        print("\n\n🛑 PROCESSO INTERROMPIDO PELO UTILIZADOR (Ctrl+C).")
        print(f"Progresso salvo no disco! O banco tem {linha_atual} chunks.")
    
    print("\n" + "="*50)
    print(f"✅ SESSÃO DE VETORIZAÇÃO ENCERRADA!")
    print(f"📊 Total de chunks injetados no ChromaDB: {linha_atual} / {TOTAL_CHUNKS_ESPERADOS}")
    print("="*50)

if __name__ == "__main__":
    processar_embeddings()