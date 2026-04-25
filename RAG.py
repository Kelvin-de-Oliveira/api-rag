from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_elasticsearch import ElasticsearchStore
from langchain_core.prompts import PromptTemplate
from langchain_classic.retrievers import EnsembleRetriever 
from langchain_core.documents import Document
from dotenv import load_dotenv
import re

# ==========================================
# CONFIGURAÇÕES
# ==========================================

load_dotenv() # Carrega a chave de API do Google do arquivo .env, se existir
DIRETORIO_CHROMA = "banco_chroma"
MODEL_EMBEDDING = "models/gemini-embedding-001"
MODEL_GENERATIVE = "gemini-2.5-flash"
DOCKER_ELASTICSEARCH_URL = "http://localhost:9200"
ELASTICSEARCH_INDEX_NAME = "aneel_lexical"

def consultar_assistente_aneel(pergunta_usuario: str) -> tuple[str, list[Document]]:
    """Função principal que orquestra a consulta ao assistente especializado da ANEEL."""

    # ---------------------------------------------------------
    # FASE 1: BUSCA (Retriver Híbrido - Vetorial + Lexical)
    # ---------------------------------------------------------
    
    # 1. Recuperador Vetorial (ChromaDB + Gemini)
    caminho_banco_vetorial = DIRETORIO_CHROMA
    embeddings_google = GoogleGenerativeAIEmbeddings(model=MODEL_EMBEDDING)
    banco_vetorial = Chroma(persist_directory=caminho_banco_vetorial, embedding_function=embeddings_google)
    
    retriever_vetorial = banco_vetorial.as_retriever(
        search_type="mmr", 
        search_kwargs={"k": 6, "fetch_k": 100}
    )

    # 2. Recuperador Lexical (Elasticsearch via Docker)
    banco_lexical = ElasticsearchStore(
        es_url=DOCKER_ELASTICSEARCH_URL,
        index_name=ELASTICSEARCH_INDEX_NAME,
        strategy=ElasticsearchStore.BM25RetrievalStrategy()
    )
    retriever_palavra_chave = banco_lexical.as_retriever(search_kwargs={"k": 6})
    
    # 3. Ensemble - Mantendo a prioridade na palavra exata
    retriever_hibrido = EnsembleRetriever(
        retrievers=[retriever_palavra_chave, retriever_vetorial],
        weights=[0.6, 0.4] 
    )
    
    print("🔎 Realizando busca nos documentos...")
    documentos_recuperados = retriever_hibrido.invoke(pergunta_usuario)
    
    # ---------------------------------------------------------
    # FASE 2: AUGMENTATION (Construção do Prompt)
    # ---------------------------------------------------------

    textos_extraidos = []
    for i, doc in enumerate(documentos_recuperados):
        id_doc = doc.metadata.get('id_processo', 'Documento sem ID') 
        texto_formatado = f"--- Documento [{i+1}] ---\nID: {id_doc}\nTexto: {doc.page_content}"
        textos_extraidos.append(texto_formatado)
        
    contexto_injetado = "\n\n".join(textos_extraidos)

    template = """Você é um assistente técnico especializado na análise de Despachos e Documentos da ANEEL.
    Sua tarefa é responder à pergunta do usuário utilizando EXCLUSIVAMENTE os trechos de documentos fornecidos abaixo.
    
    Regras estritas:
    1. Se a resposta não estiver contida nos trechos abaixo, responda EXATAMENTE: "Desculpe, mas não encontrei essa informação nos documentos analisados."
    2. Não invente valores, datas ou dados que não estejam no contexto.
    3. Seja direto, claro e profissional.
    
    REGRA DE CITAÇÃO OBRIGATÓRIA:
    Sempre que utilizar uma informação de um documento, VOCÊ DEVE citar a fonte no final da frase correspondente, usando o número do documento entre colchetes.
    Exemplo: A potência instalada da usina é de 50kW [1].
    
    DOCUMENTOS RECUPERADOS (Contexto):
    {contexto}
    
    PERGUNTA DO USUÁRIO: 
    {pergunta}
    
    RESPOSTA:"""
    
    prompt = PromptTemplate(template=template, input_variables=["contexto", "pergunta"])
    prompt_final = prompt.format(contexto=contexto_injetado, pergunta=pergunta_usuario)

    # ---------------------------------------------------------
    # FASE 3: GENERATION (Geração via LLM)
    # ---------------------------------------------------------

    print("🧠 Consultando os documentos e gerando a resposta...")
    
    llm = ChatGoogleGenerativeAI(model=MODEL_GENERATIVE, temperature=0.2)
    resposta = llm.invoke(prompt_final)
    resposta_texto = resposta.content
    
    # ---------------------------------------------------------
    # FASE 4: FILTRAGEM E AGRUPAMENTO DE FONTES
    # ---------------------------------------------------------

    conteudos_entre_colchetes = re.findall(r'\[(.*?)\]', resposta_texto)
    
    citacoes = set()
    for conteudo in conteudos_entre_colchetes:
        numeros_encontrados = re.findall(r'\d+', conteudo)
        citacoes.update(numeros_encontrados)
    
    documentos_agrupados = {}
    
    if citacoes:
        for num_str in citacoes:
            indice_array = int(num_str) - 1 
            if 0 <= indice_array < len(documentos_recuperados):
                doc = documentos_recuperados[indice_array]

                id_doc = doc.metadata.get('id_processo', 'Documento sem ID')
                url_doc = doc.metadata.get('url', 'Link não disponível')
                
                if id_doc not in documentos_agrupados:
                    documentos_agrupados[id_doc] = {
                        "indices": [],
                        "url": url_doc,
                        "documento_base": doc
                    }
                documentos_agrupados[id_doc]["indices"].append(int(num_str))

    documentos_utilizados_final = []
    referencias_formatadas = []

    for id_doc, info in documentos_agrupados.items():
        indices_ordenados = sorted(info["indices"])
        tags_citacao = " ".join([f"[{i}]" for i in indices_ordenados])
        
        ref_str = f"{tags_citacao} 📄 Documento: {id_doc} | Link: {info['url']}"
        referencias_formatadas.append(ref_str)
        documentos_utilizados_final.append(info["documento_base"])

    if referencias_formatadas:
        resposta_texto += "\n\n**Fontes Consultadas:**\n" + "\n".join(referencias_formatadas)
    
    return resposta_texto, documentos_utilizados_final

#Código de teste
if __name__ == "__main__":
    pergunta = "Qual é a principal decisão tomada pelo Despacho Nº 244, de 28 de janeiro de 2016, em relação à unidade geradora UG2 da CGH Wasser Kraft?"
    
    resposta_texto, fontes_utilizadas = consultar_assistente_aneel(pergunta)
    
    print("\n" + "="*50)
    print("🤖 RESPOSTA DO ASSISTENTE:")
    print("="*50)
    print(resposta_texto)
    for i, doc in enumerate(fontes_utilizadas):
        print(f"\n--- Documento Utilizado [{i+1}] ---")
        print(f"ID: {doc.metadata.get('id_processo', 'Documento sem ID')}")
        print(f"URL: {doc.metadata.get('url', 'Link não disponível')}")
        print(f"Conteúdo: {doc.page_content[:500]}...")