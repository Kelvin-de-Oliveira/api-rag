from pydantic import BaseModel, Field


class PerguntaRequest(BaseModel):
    """Corpo do POST /perguntar."""

    pergunta: str = Field(
        ...,
        min_length=3,
        max_length=2000,
        description="Pergunta em linguagem natural sobre os documentos da ANEEL.",
        examples=[
            "Qual é a principal decisão tomada pelo Despacho Nº 244, "
            "de 28 de janeiro de 2016, em relação à unidade geradora UG2 "
            "da CGH Wasser Kraft?"
        ],
    )



class DocumentoFonte(BaseModel):
    """Um documento da ANEEL efetivamente citado pelo LLM na resposta."""

    id_processo: str = Field(
        ...,
        description="Identificador do processo/despacho na ANEEL.",
    )
    titulo: str = Field(
        default="",
        description="Título global do documento (quando disponível).",
    )
    url: str = Field(
        default="",
        description="Link para o documento original na ANEEL.",
    )
    data_publicacao: str = Field(
        default="",
        description="Data de publicação (quando disponível).",
    )
    trecho: str = Field(
        ...,
        description="Trecho do documento utilizado pelo LLM como contexto.",
    )
    indices_citacao: list[int] = Field(
        default_factory=list,
        description=(
            "Números das citações no texto da resposta que apontam para este "
            "documento. Ex.: [1, 3] significa que esta fonte aparece como [1] e [3]."
        ),
    )


class RespostaResponse(BaseModel):
    """Corpo da resposta do POST /perguntar."""

    resposta: str = Field(
        ...,
        description=(
            "Resposta gerada pelo LLM, com citações inline no formato [n] "
            "que referenciam itens em `fontes`."
        ),
    )
    fontes: list[DocumentoFonte] = Field(
        default_factory=list,
        description="Documentos efetivamente citados na resposta, sem duplicatas.",
    )
    tempo_processamento_s: float = Field(
        ...,
        description="Tempo total de processamento da pergunta, em segundos.",
    )


class ComponenteSaude(BaseModel):
    """Estado de um componente individual do sistema."""

    nome: str
    status: str = Field(..., description="'ok' ou 'erro'")
    detalhe: str = Field(default="")


class HealthResponse(BaseModel):
    """Corpo da resposta do GET /health."""

    status: str = Field(..., description="'ok' se todos os componentes OK, senão 'degraded'")
    componentes: list[ComponenteSaude]


class ErroResponse(BaseModel):
    """Formato padronizado para respostas de erro."""

    detail: str