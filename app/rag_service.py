import re
import time
import logging
from typing import Any

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_elasticsearch import ElasticsearchStore
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_classic.retrievers import EnsembleRetriever

from app import config

logger = logging.getLogger(__name__)

# Template do prompt — extraído como constante de módulo para facilitar ajustes
PROMPT_TEMPLATE = """Você é um assistente técnico especializado na análise de Despachos e Documentos da ANEEL.
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


class RAGService:
    """
    Serviço de RAG com componentes carregados uma única vez.
    """

    def __init__(self):
        logger.info("Inicializando RAGService...")
        inicio = time.time()

        #Modelo de embeddings 
        self._embeddings = GoogleGenerativeAIEmbeddings(model=config.MODEL_EMBEDDING)

        #Banco vetorial 
        self._banco_vetorial = Chroma(
            persist_directory=config.DIRETORIO_CHROMA,
            embedding_function=self._embeddings,
        )

        #Banco lexical 
        self._banco_lexical = ElasticsearchStore(
            es_url=config.ELASTICSEARCH_URL,
            index_name=config.ELASTICSEARCH_INDEX_NAME,
            strategy=ElasticsearchStore.BM25RetrievalStrategy(),
        )

        #Retrievers individuais
        retriever_vetorial = self._banco_vetorial.as_retriever(
            search_type="mmr",
            search_kwargs={"k": config.K_VETORIAL, "fetch_k": config.FETCH_K_VETORIAL},
        )
        retriever_lexical = self._banco_lexical.as_retriever(
            search_kwargs={"k": config.K_LEXICAL}
        )

        #Retriever híbrido 
        self._retriever_hibrido = EnsembleRetriever(
            retrievers=[retriever_lexical, retriever_vetorial],
            weights=[config.PESO_LEXICAL, config.PESO_VETORIAL],
        )

        #LLM para geração da resposta final
        self._llm = ChatGoogleGenerativeAI(
            model=config.MODEL_GENERATIVE,
            temperature=config.TEMPERATURA_LLM,
        )

        #Template de prompt 
        self._prompt = PromptTemplate(
            template=PROMPT_TEMPLATE,
            input_variables=["contexto", "pergunta"],
        )

        duracao = time.time() - inicio
        logger.info(f"RAGService inicializado em {duracao:.2f}s.")

    #Método publico:
    def consultar(self, pergunta: str) -> dict[str, Any]:
        """
        Executa o pipeline completo de RAG para uma pergunta.

        Retorna um dicionário com:
            - resposta (str): texto gerado pelo LLM, com citações [1], [2]...
            - fontes (list[dict]): documentos efetivamente citados na resposta
            - tempo_processamento_s (float): duração total em segundos
        """
        if not pergunta or not pergunta.strip():
            raise ValueError("A pergunta não pode ser vazia.")

        inicio = time.time()

        #Busca (retrieval híbrido)
        documentos_recuperados = self._buscar(pergunta)

        #Augmentation (montagem do prompt)
        prompt_final = self._montar_prompt(pergunta, documentos_recuperados)

        #Generation (resposta do LLM)
        resposta_texto = self._gerar(prompt_final)

        #Extração das fontes efetivamente citadas
        fontes = self._extrair_fontes_citadas(resposta_texto, documentos_recuperados)

        duracao = time.time() - inicio

        return {
            "resposta": resposta_texto,
            "fontes": fontes,
            "tempo_processamento_s": round(duracao, 2),
        }

    #Métodos privados (fases do pipeline)
    def _buscar(self, pergunta: str) -> list[Document]:
        """Busca documentos relevantes usando o retriever híbrido."""
        logger.info(f"Buscando documentos para: {pergunta[:80]}...")
        return self._retriever_hibrido.invoke(pergunta)

    def _montar_prompt(self, pergunta: str, documentos: list[Document]) -> str:
        """Formata os documentos recuperados e injeta no template de prompt."""
        textos_extraidos = []
        for i, doc in enumerate(documentos):
            id_doc = doc.metadata.get("id_processo", "Documento sem ID")
            texto_formatado = (
                f"--- Documento [{i + 1}] ---\n"
                f"ID: {id_doc}\n"
                f"Texto: {doc.page_content}"
            )
            textos_extraidos.append(texto_formatado)

        contexto_injetado = "\n\n".join(textos_extraidos)
        return self._prompt.format(contexto=contexto_injetado, pergunta=pergunta)

    def _gerar(self, prompt_final: str) -> str:
        """Chama o LLM e devolve apenas o texto da resposta."""
        logger.info("Gerando resposta via LLM...")
        resposta = self._llm.invoke(prompt_final)
        return resposta.content

    def _extrair_fontes_citadas(self, resposta_texto: str, documentos: list[Document]) -> list[dict[str, Any]]:
        """
        Parseia as citações [n] presentes no texto da resposta e devolve
        apenas os documentos efetivamente referenciados, agrupados por
        id_processo para evitar duplicatas.
        """
        conteudos_entre_colchetes = re.findall(r"\[(.*?)\]", resposta_texto)
        citacoes: set[int] = set()
        for conteudo in conteudos_entre_colchetes:
            for num_str in re.findall(r"\d+", conteudo):
                citacoes.add(int(num_str))

        if not citacoes:
            return []

        documentos_agrupados: dict[str, dict[str, Any]] = {}
        for num in citacoes:
            indice_array = num - 1
            if not (0 <= indice_array < len(documentos)):
                continue

            doc = documentos[indice_array]
            id_doc = doc.metadata.get("id_processo", "Documento sem ID")

            if id_doc not in documentos_agrupados:
                documentos_agrupados[id_doc] = {
                    "id_processo": id_doc,
                    "titulo": doc.metadata.get("titulo_global", ""),
                    "url": doc.metadata.get("url", ""),
                    "data_publicacao": doc.metadata.get("data_publicacao", ""),
                    "trecho": doc.page_content,
                    "indices_citacao": [],
                }
            documentos_agrupados[id_doc]["indices_citacao"].append(num)

        fontes = []
        for info in documentos_agrupados.values():
            info["indices_citacao"] = sorted(info["indices_citacao"])
            fontes.append(info)

        return fontes