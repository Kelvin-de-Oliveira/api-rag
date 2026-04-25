# ChatRAG (extra opcional)

Este diretorio contem uma interface local em Streamlit para conversar com a API RAG de forma pratica.

## Requisitos

- API RAG rodando (por padrao em http://localhost:8000)
- Python 3.10+

## Como rodar

No diretorio raiz do projeto:

```bash
pip install -r chatRAG/requirements.txt
python -m streamlit run chatRAG/chatRAG.py
```

Abra no navegador o endereco exibido pelo Streamlit (normalmente http://localhost:8501).

No painel lateral do app voce pode ajustar a URL base da API e o timeout.

## Observacoes

- O ChatRAG nao e necessario para a API funcionar.
- Ele apenas consome o endpoint `POST /perguntar` da API principal.
