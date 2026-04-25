import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Caminhos 
BASE_DIR = Path(__file__).resolve().parent.parent
DIRETORIO_CHROMA = os.getenv("DIRETORIO_CHROMA", str(BASE_DIR / "banco_chroma"))

# Elasticsearch
ELASTICSEARCH_URL = os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")
ELASTICSEARCH_INDEX_NAME = os.getenv("ELASTICSEARCH_INDEX_NAME", "aneel_lexical")

# Modelos do Google Gemini
MODEL_EMBEDDING = os.getenv("MODEL_EMBEDDING", "models/gemini-embedding-001")
MODEL_GENERATIVE = os.getenv("MODEL_GENERATIVE", "gemini-2.5-flash")

# Parâmetros de busca 
K_VETORIAL = int(os.getenv("K_VETORIAL", "6"))
FETCH_K_VETORIAL = int(os.getenv("FETCH_K_VETORIAL", "100"))
K_LEXICAL = int(os.getenv("K_LEXICAL", "6"))
PESO_LEXICAL = float(os.getenv("PESO_LEXICAL", "0.6"))
PESO_VETORIAL = float(os.getenv("PESO_VETORIAL", "0.4"))

# LLM
TEMPERATURA_LLM = float(os.getenv("TEMPERATURA_LLM", "0.2"))

# Valida presença da chave da API 
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError(
        "Variável de ambiente GEMINI_API_KEY (ou GOOGLE_API_KEY) não configurada. "
        "Crie um arquivo .env na raiz do projeto com: GEMINI_API_KEY=sua_chave_aqui"
    )

# O langchain-google-genai espera GOOGLE_API_KEY!
os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY
os.environ.pop("GEMINI_API_KEY", None)