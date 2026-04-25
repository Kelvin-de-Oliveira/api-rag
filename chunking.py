import os
import json
from tqdm import tqdm
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

# ==========================================
# 1. CONFIGURAÇÕES
# ==========================================

ARQUIVO_SAIDA_JSONL = "chunks\\chunks.jsonl"
ARQUIVO_CHECKPOINTS = "chunks\\checkpoints_chunking.txt"
ARQUIVO_ERROS = "chunks\\erros_chunking.txt"
TOTAL_CHUNKS_GERADOS = "chunks\\total_chunks_gerados.txt"
PASTA_PARSEDS = "json_parsed"

# O Splitter Estrutural com os nomes
headers_para_fatiar = [
    ("#", "Titulo_Principal"),
    ("##", "Secao"),
    ("###", "Subsecao"),
]
markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_para_fatiar)

# O Splitter de Segurança para garantir que nenhum chunk ultrapasse o limite 
recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024,
    chunk_overlap=154,
    separators=["\n\n", "\n", ".", " ", ""]
)

# ==========================================
# 2. FUNÇÕES DE APOIO
# ==========================================

def verificar_existencia_arquivo(caminho_arquivo: str = ARQUIVO_SAIDA_JSONL) -> bool:
    """Verifica se o arquivo de saída JSONL existe e se o diretório está pronto. Cria o diretório se necessário."""
    diretorio = os.path.dirname(caminho_arquivo)
    if not os.path.exists(diretorio):
        os.makedirs(diretorio)
    return os.path.exists(caminho_arquivo)

def carregar_processados(arquivos_checkpoints: str = ARQUIVO_CHECKPOINTS) -> set:
    """Carrega o conjunto de arquivos já processados a partir do arquivo de checkpoints. Retorna um set para acesso rápido."""
    try:
        if not os.path.exists(arquivos_checkpoints):
            return set()
        with open(arquivos_checkpoints, 'r', encoding='utf-8') as f:
            return set(linha.strip() for linha in f)
    except Exception:
        return set()

def registrar_processado(nome_arquivo: str, arquivos_checkpoints: str = ARQUIVO_CHECKPOINTS)-> None:
    """Registra um arquivo como processado, adicionando seu nome ao arquivo de checkpoints. Usa append para garantir que o histórico seja mantido."""
    try:
        with open(arquivos_checkpoints, 'a', encoding='utf-8') as f:
            f.write(f"{nome_arquivo}\n")
    except Exception:
        pass

def registrar_erro(nome_arquivo: str, erro: Exception, arquivo_erros: str = ARQUIVO_ERROS)-> None:
    """Registra um erro ocorrido durante o processamento de um arquivo, salvando o nome do arquivo e a mensagem de erro em um arquivo de log. Usa append para garantir que o histórico seja mantido."""
    try:
        with open(arquivo_erros, 'a', encoding='utf-8') as f:
            f.write(f"ERRO: {nome_arquivo} | Motivo: {str(erro)}\n")
    except Exception:
        pass

def atualizar_total_chunks(qtd_chunks: int, arquivo_total: str = TOTAL_CHUNKS_GERADOS) -> None:
    """Atualiza o total de chunks gerados, somando a quantidade atual ao valor existente no arquivo."""
    try:
        if os.path.exists(arquivo_total):
            with open(arquivo_total, 'r', encoding='utf-8') as f:
                total_atual = int(f.read().strip() or "0")
                total_atual += qtd_chunks
            with open(arquivo_total, 'w', encoding='utf-8') as f:
                f.write(str(total_atual))
        else:
            with open(arquivo_total, 'w', encoding='utf-8') as f:
                f.write(str(qtd_chunks))
    except Exception as e:
        print(e)
        pass
    
# ==========================================
# 3. Processando JSONs
# ==========================================

def processar_em_massa(pasta_origem: str = PASTA_PARSEDS, arquivo_saida_jsonl: str = ARQUIVO_SAIDA_JSONL)-> None:
    """Processa em massa os arquivos JSON, fatiando o conteúdo Markdown e salvando em um arquivo JSONL compatível com LangChain. O processo é robusto, com checkpointing e registro de erros para garantir que nada seja perdido mesmo em caso de falhas."""
    print("Preparando a Linha de Montagem (Leitura de JSONs)...")
    total_chunks_gerados = 0
    try:
        verificar_existencia_arquivo()
        # Agora buscamos por arquivos .json
        todos_arquivos = [f for f in os.listdir(pasta_origem) if f.endswith('.json')]
    except Exception as e:
        print(f"❌ Erro fatal: Não consegui acessar a pasta '{pasta_origem}'.\nErro: {e}")
        return

    arquivos_ja_feitos = carregar_processados()
    arquivos_pendentes = [f for f in todos_arquivos if f not in arquivos_ja_feitos]
    
    print(f"📁 Total: {len(todos_arquivos)}\n")
    
    if not arquivos_pendentes:
        print("🎉 Todos os arquivos já foram processados!")
        return

    try:
        with open(arquivo_saida_jsonl, 'a', encoding='utf-8') as arquivo_jsonl:
            # 1. Variáveis para acompanhar o lote atual
            sucessos_lote = 0
            erros_lote = 0
            barra_progresso = tqdm(arquivos_pendentes, desc="Realizando Chunking em Massa...")
            for nome_arquivo in barra_progresso:
                caminho_completo = os.path.join(pasta_origem, nome_arquivo)
                
                try:
                    linhas_do_arquivo = []
                    total_chunks_arquivo = 0
                    # 1. Carrega o JSON Original
                    with open(caminho_completo, 'r', encoding='utf-8') as f:
                        json_data = json.load(f)
                    
                    # 2. Resgata os Metadados Globais (Ficha Catalográfica)
                    metadados_base = {
                        "titulo_global": json_data.get('titulo', 'Sem título'),
                        "ementa_global": json_data.get('ementa', 'Sem ementa'),
                        "id_processo": json_data.get('id', nome_arquivo.replace('.json', '')),
                        "data_publicacao": json_data.get('data_publicacao', ''),
                        "assunto": json_data.get('assunto', ''),
                        "autor": json_data.get('autor', '')
                    }
                    
                    # 3. Itera sobre os anexos internos do JSON
                    lista_documentos = json_data.get('documentos', [])
                    
                    for indice, doc in enumerate(lista_documentos):
                        conteudo_md = doc.get('texto_extraido_md') or doc.get('texto_extraido') or ""

                        # Adicionamos str() por segurança extrema
                        if not str(conteudo_md).strip():
                            continue
                            
                        # Metadados específicos deste anexo
                        metadados_especificos = metadados_base.copy()
                        # A chave DEVE ser "url" para que o nosso Front-end ache o link depois
                        metadados_especificos["url"] = doc.get('url', 'Sem URL') 
                        metadados_especificos["indice_documento"] = indice + 1
                        
                        # 4. Fatiamento em 2 Etapas
                        chunks_md = markdown_splitter.split_text(conteudo_md)
                        
                        for chunk in chunks_md:
                            # Funde o cabeçalho Markdown com a nossa ficha catalográfica
                            chunk.metadata.update(metadados_especificos)
                            sub_chunks = recursive_splitter.split_documents([chunk])
                            
                            # 5. Salva na linha de montagem (Formato compatível com LangChain)
                            for sub_chunk in sub_chunks:
                                linha = {
                                    "page_content": sub_chunk.page_content,
                                    "metadata": sub_chunk.metadata
                                }
                                linhas_do_arquivo.append(json.dumps(linha, ensure_ascii=False) + "\n")
                            
                            # Incrementamos os contadores assim que as linhas são escritas
                            qtd_sub_chunks = len(sub_chunks)
                            total_chunks_arquivo += qtd_sub_chunks

                    # Escrita transacional por arquivo: só persiste após processamento completo sem erro
                    arquivo_jsonl.writelines(linhas_do_arquivo)
                    total_chunks_gerados += total_chunks_arquivo
                    
                    # 3. Registra o sucesso e atualiza a interface
                    registrar_processado(nome_arquivo)
                    sucessos_lote += 1
                    barra_progresso.set_postfix({"Sucesso": sucessos_lote, "Erro": erros_lote, "Chunks Gerados": total_chunks_gerados})
                    
                except Exception as e:
                    registrar_erro(nome_arquivo, e)
                    # Registra o erro e atualiza a interface
                    erros_lote += 1
                    barra_progresso.set_postfix({"Sucesso": sucessos_lote, "Erro": erros_lote, "Chunks Gerados": total_chunks_gerados})
                    continue

        # Print final do relatório completo
        print("\n" + "="*50)
        print("\n✅ Processamento concluído com sucesso!")
        print("="*50)
        print(f"📊 Relatório da Sessão:")
        print(f"   - Ficheiros Processados: {sucessos_lote}")
        print(f"   - Ficheiros com Erro: {erros_lote}")
        print(f"   - Total de Chunks Criados: {total_chunks_gerados}")
        print("="*50)

    except KeyboardInterrupt:
        print("\n\n🛑 PROCESSO INTERROMPIDO PELO USUÁRIO (Ctrl+C). Trabalho salvo.")
    except Exception as e:
        print(f"\n\n❌ ERRO CRÍTICO:\n{e}")
    finally:
        atualizar_total_chunks(total_chunks_gerados)
        print(total_chunks_gerados)

if __name__ == "__main__":
    processar_em_massa()