import os

# Qdrant configurations
QDRANT_PATH = os.path.join(os.path.dirname(__file__), "qdrant_db")
COLLECTION_NAME = "financial_reports"

# Embedding settings
# Provider: "huggingface" hoặc "cohere"
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "cohere")

# HuggingFace model (dùng khi EMBEDDING_PROVIDER = "huggingface")
EMBEDDING_MODEL_NAME = "keepitreal/vietnamese-sbert"

# Cohere embedding (dùng khi EMBEDDING_PROVIDER = "cohere")
COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")
COHERE_EMBEDDING_MODEL = "embed-multilingual-v3.0"  # 1024 chiều, hỗ trợ 100+ ngôn ngữ

# Text Splitting settings
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200

# LLM Settings
# We use Groq's fast models. llama-3.3-70b-versatile or mixtral-8x7b-32768
LLM_MODEL = "llama-3.3-70b-versatile"
LLM_TEMPERATURE = 0.1 # Keep it low for precise factual retrieval
LLM_MAX_TOKENS = 1024

# Retrieval Settings
RETRIEVER_K = 5

# SQL Server configurations
SQL_SERVER = os.getenv("SQL_SERVER", "localhost")
SQL_DATABASE = os.getenv("SQL_DATABASE", "Financial_Analyst_DB")
SQL_TRUSTED_CONNECTION = os.getenv("SQL_TRUSTED_CONNECTION", "yes") # "yes" cho Windows Auth, "no" cho SQL Server Auth
SQL_USERNAME = os.getenv("SQL_USERNAME", "")
SQL_PASSWORD = os.getenv("SQL_PASSWORD", "")
