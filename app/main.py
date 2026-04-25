import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from app.rag_service import RAGService
from app.schemas import (ComponenteSaude, ErroResponse, HealthResponse, PerguntaRequest, RespostaResponse,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)



@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=" * 60)
    logger.info("Iniciando API RAG ANEEL...")
    logger.info("=" * 60)

    try:
        app.state.rag_service = RAGService()
        logger.info("RAGService pronto. API aceitando requisições.")
    except Exception as e:
        logger.exception("Falha ao inicializar RAGService.")
        raise

    yield

    logger.info("Encerrando API RAG ANEEL.")


app = FastAPI(
    title="API RAG ANEEL",
    description=(
        "API de consulta a documentos da ANEEL utilizando RAG "
        "(Retrieval-Augmented Generation) com busca híbrida "
        "(vetorial via ChromaDB + lexical via Elasticsearch) e "
        "geração via Google Gemini."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

#CORS liberado por que ainda não sei qual o dominio do front, by:kelvin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



#ROTAS

@app.get("/", include_in_schema=False)
async def raiz():
    """Redireciona a raiz para a documentação Swagger."""
    return RedirectResponse(url="/docs")


@app.post(
    "/perguntar",
    response_model=RespostaResponse,
    responses={
        400: {"model": ErroResponse, "description": "Pergunta inválida"},
        500: {"model": ErroResponse, "description": "Erro interno"},
        503: {"model": ErroResponse, "description": "Serviço indisponível"},
    },
    summary="Consulta o assistente RAG",
    description=(
        "Recebe uma pergunta em linguagem natural e devolve uma resposta gerada "
        "pelo LLM com base nos documentos recuperados, junto da lista de fontes "
        "efetivamente citadas."
    ),
)
async def perguntar(payload: PerguntaRequest, request: Request) -> RespostaResponse:
    rag_service: RAGService = request.app.state.rag_service

    try:
        resultado = rag_service.consultar(payload.pergunta)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.exception("Erro inesperado ao consultar RAG.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro ao processar a pergunta: {e}",
        )

    return RespostaResponse(**resultado)


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Checagem de saúde dos componentes",
    description=(
        "Verifica o estado de cada componente do sistema (ChromaDB, "
        "Elasticsearch, Gemini). Útil para diagnosticar problemas durante "
        "a configuração inicial ou em caso de falhas."
    ),
)
async def health(request: Request) -> HealthResponse:
    rag_service: RAGService | None = getattr(request.app.state, "rag_service", None)
    componentes: list[ComponenteSaude] = []

    if rag_service is None:
        componentes.append(
            ComponenteSaude(
                nome="rag_service",
                status="erro",
                detalhe="RAGService não foi inicializado.",
            )
        )
        return HealthResponse(status="degraded", componentes=componentes)

    try:
        total_chroma = rag_service._banco_vetorial._collection.count()
        if total_chroma > 0:
            componentes.append(
                ComponenteSaude(
                    nome="chromadb",
                    status="ok",
                    detalhe=f"{total_chroma} documentos indexados.",
                )
            )
        else:
            componentes.append(
                ComponenteSaude(
                    nome="chromadb",
                    status="erro",
                    detalhe="ChromaDB acessível, mas vazio. Rode o setup de dados.",
                )
            )
    except Exception as e:
        componentes.append(
            ComponenteSaude(nome="chromadb", status="erro", detalhe=str(e))
        )

    try:
        from app import config as _cfg
 
        es_client = rag_service._banco_lexical.client
        nome_indice = _cfg.ELASTICSEARCH_INDEX_NAME
 
        if not es_client.ping():
            componentes.append(
                ComponenteSaude(
                    nome="elasticsearch",
                    status="erro",
                    detalhe="Sem resposta ao ping.",
                )
            )
        elif not es_client.indices.exists(index=nome_indice):
            componentes.append(
                ComponenteSaude(
                    nome="elasticsearch",
                    status="erro",
                    detalhe=f"Índice '{nome_indice}' não existe. Rode a ingestão.",
                )
            )
        else:
            count = es_client.count(index=nome_indice).get("count", 0)
            if count > 0:
                componentes.append(
                    ComponenteSaude(
                        nome="elasticsearch",
                        status="ok",
                        detalhe=f"{count} documentos indexados em '{nome_indice}'.",
                    )
                )
            else:
                componentes.append(
                    ComponenteSaude(
                        nome="elasticsearch",
                        status="erro",
                        detalhe=f"Índice '{nome_indice}' existe, mas está vazio.",
                    )
                )
    except Exception as e:
        componentes.append(
            ComponenteSaude(nome="elasticsearch", status="erro", detalhe=str(e))
        )

    #sem fazer chamada real para não desperdiçar quota atoa
    from app import config

    if config.GEMINI_API_KEY:
        componentes.append(
            ComponenteSaude(
                nome="gemini",
                status="ok",
                detalhe="Chave de API configurada.",
            )
        )
    else:
        componentes.append(
            ComponenteSaude(
                nome="gemini",
                status="erro",
                detalhe="GEMINI_API_KEY não configurada.",
            )
        )

    status_geral = (
        "ok" if all(c.status == "ok" for c in componentes) else "degraded"
    )
    return HealthResponse(status=status_geral, componentes=componentes)