# API RAG ANEEL

API REST que responde perguntas sobre Despachos e documentos da ANEEL utilizando RAG (Retrieval-Augmented Generation) com busca híbrida (vetorial + lexical) e geração via 
Google Gemini.


## Scripts originais (geração dos dados)

Os scripts que geram os dados usados por esta API (chunking dos documentos, geração de embeddings com Gemini, ingestão inicial no ChromaDB e no Elasticsearch) estão em um repositório separado:

🔗 **Repositório do pipeline de ingestão:** <[Desafio_RAG_NLP](https://github.com/joaopauloCand/Desafio_RAG_NLP.git)>

🔗 **Todos os dados utilizados aqui podem ser encontrados em:** ([dataset_huggingface](https://huggingface.co/datasets/joaopauloCand/Embeddings_RAG_ANEEL/tree/main))

Esses scripts foram executados **uma única vez** para gerar os artefatos disponibilizados no Hugging Face e baixados pelo `prepare_data.py`. Não são necessários para rodar esta API. O repositório foi preservado para fins de documentação e reprodutibilidade do pipeline completo.


## Pré-requisitos

- **Docker Desktop** com Docker Compose ([instalar](https://www.docker.com/products/docker-desktop/))
- **Python 3.10+** instalado no host (para rodar o script de preparação de dados)
- **Chave de API do Google Gemini** ([obter aqui](https://aistudio.google.com/app/apikey))
- ~10 GB livres em disco (banco vetorial + chunks + Elasticsearch)

## Como rodar

### 1. Clone o repositório

```bash
git clone https://github.com/kelvin-de-oliveira/api-rag
cd api-rag
```

### 2. Configure a chave da API do Gemini 

No Command Prompt (CMD):
```bash
cp .env.example .env
copy .env.example .env.
```
No PowerShell:
```bash
copy .env.example .env.
```
Abra o arquivo `.env` e substitua o valor de `GEMINI_API_KEY` pela sua chave.

### 3. Suba a infraestrutura (API + Elasticsearch)

```bash
docker-compose up -d
```

Aguarde cerca de 30 a 60 segundos até os containers ficarem `healthy`. Você pode acompanhar com:

```bash
docker-compose ps
```

### 4. Prepare os dados

Em um terminal fora do Docker, instale apenas as dependências mínimas e execute o script:

```bash
pip install -r requirements_setup.txt
python prepare_data.py
```
**Obs.:** Se voce estiver usando ambientes virtuais com uv, lembre-se de colocar uv antes de `pip install -r requirements_setup.txt`, se não as dependência serão instaladas no seu ambiente "geral". 

Esse script é **idempotente**: pode ser executado quantas vezes quiser. Na primeira execução, ele:
- Baixa o banco vetorial (~5 GB) do Hugging Face
- Baixa os chunks do Hugging Face
- Popula o índice no Elasticsearch (cerca de 2 a 5 minutos)

Em execuções seguintes, detecta o que já está pronto e pula automaticamente.

### 5. Pronto

Acesse:

- **Swagger interativo:** http://localhost:8000/docs
- **Healthcheck:** http://localhost:8000/health

## Como usar

### Endpoint principal

`POST /perguntar`

Exemplo com `curl` no terminal Windows:

```bash
curl -X POST http://localhost:8000/perguntar ^
     -H "Content-Type: application/json" ^
     -d "{\"pergunta\": \"Qual a decisão do Despacho 244 sobre a UG2 da CGH Wasser Kraft?\"}"

```

Resposta:

```json
{
  "resposta": "O Despacho Nº 244, de 28 de janeiro de 2016, resolve liberar a unidade geradora UG2, de 400 kW de capacidade instalada, da CGH Wasser Kraft, para início da operação em teste a partir do dia 29 de janeiro de 2016 [1].",
  "fontes": [
    {
      "id_processo": "2016-01-29_dsp_despacho_244_2016",
      "titulo": "DSP - DESPACHO 244/2016",
      "url": "http://www2.aneel.gov.br/cedoc/dsp2016244ti.pdf",
      "data_publicacao": "2016-01-29",
      "trecho": "...",
      "indices_citacao": [1]
    }
  ],
  "tempo_processamento_s": 7.96
}
```

### Interface local com Streamlit (extra opcional)

Se voce quiser um chatbot local para conversar com o RAG de forma mais pratica, use o modulo opcional em `chatRAG/`.

```bash
pip install -r chatRAG/requirements.txt
python -m streamlit run chatRAG/chatRAG.py
```

Depois abra no navegador o endereco mostrado pelo Streamlit (normalmente http://localhost:8501).

No painel lateral da interface, mantenha a URL base da API como `http://localhost:8000` (ou ajuste para o host/porta em uso).

> Esse chatbot é apenas um extra para conveniência, não é necessário para o funcionamento da API.

### Outros endpoints

- `GET /health` — verifica o estado de cada componente (ChromaDB, Elasticsearch, Gemini)
- `GET /docs` — interface Swagger interativa

## Estrutura do projeto

```
rag-aneel/
├── chatRAG/                   # Chatbot Streamlit opcional para consumir a API
│   ├── chatRAG.py             # Interface local de conversa com o RAG
│   ├── requirements.txt       # Dependencias exclusivas do chatbot
│   └── README.md              # Instrucoes do modulo ChatRAG
├── app/                       # Código da API (FastAPI)
│   ├── main.py                # Rotas e lifespan
│   ├── rag_service.py         # Lógica do RAG
│   ├── schemas.py             # Modelos Pydantic
│   └── config.py              # Configurações via env vars
├── Dockerfile                 # Imagem da API
├── docker-compose.yml         # Orquestração API + Elasticsearch
├── prepare_data.py            # Script idempotente de preparação
├── requirements.txt           # Dependências da API (Docker)
├── requirements-setup.txt     # Dependências do prepare_data.py
└── .env.example               # Template das variáveis de ambiente
```

## Arquitetura
 
```
                  ┌──────────────────────────────────────┐
                  │             Cliente                  │
                  └───────────────────┬──────────────────┘
                                      │ HTTP (JSON)
                  ┌───────────────────▼──────────────────┐
                  │         API FastAPI (porta 8000)     │
                  │                                      │
                  │   ┌────────────────────────────┐     │
                  │   │  RAGService                │     │
                  │   │   ┌──────────────────────┐ │     │
                  │   │   │ EnsembleRetriever    │ │     │
                  │   │   │  (peso 60% / 40%)    │ │     │
                  │   │   └──┬────────────────┬──┘ │     │
                  │   └──────┼────────────────┼────┘     │
                  └──────────┼────────────────┼──────────┘
                             │                │
                  ┌──────────▼─────┐  ┌───────▼──────────┐
                  │  Elasticsearch │  │    ChromaDB      │
                  │  (BM25/Lexical)│  │  (Vetorial/MMR)  │
                  │   container    │  │   bind mount     │
                  └────────────────┘  └──────────────────┘
                                                 │
                                                 │ Embeddings
                                      ┌──────────▼────────┐
                                      │  Google Gemini    │
                                      │  (embed + gen)    │
                                      └───────────────────┘
```



## Comandos úteis

```bash
docker-compose logs -f api          # Acompanha logs da API
docker-compose logs -f elasticsearch # Acompanha logs do ES
docker-compose down                  # Para tudo (mantém dados)
docker-compose down -v               # Para tudo E apaga dados do ES
docker-compose restart api           # Reinicia só a API
```

## Solução de problemas

**A API retorna `degraded` no `/health`?** Significa que algum componente não está OK. Cheque o detalhe de cada componente na resposta, mas geralmente é o Elasticsearch sem o índice populado (rode `python prepare_data.py`).

**Erro ao iniciar a API: `GEMINI_API_KEY não configurada`?** O arquivo `.env` não foi criado ou está com a chave em branco. Confira o passo 2.

**O `docker-compose up` falha?** Confira se o Docker Desktop está rodando e se as portas 8000 e 9200 estão livres.

**O `prepare_data.py` falha no download?**
- Confira sua conexão com a internet
- Apague os arquivos parciais (`banco_chroma.zip`, `chunks.zip`) e rode novamente — o script retoma do zero

**A primeira requisição após subir o Docker demora 30-60s?**
Comportamento esperado. Na primeira chamada, o sistema operacional carrega
metadados do banco vetorial e estabelece conexões com o Elasticsearch e
o Gemini. As chamadas seguintes respondem em milissegundos (`/health`, `/docs`)
ou em 5-10 segundos (`/perguntar`).
 
**Pergunta retorna "não encontrei essa informação"?** A LLM foi instruída a **não inventar** respostas. Se os documentos recuperados não cobrem a pergunta, ele explicitamente avisa. Tente reformular com termos mais específicos (números de despacho, datas, nomes próprios) pois esses ativam melhor a busca lexical.