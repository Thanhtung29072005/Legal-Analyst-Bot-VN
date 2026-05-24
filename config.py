import os

# Qdrant configurations
QDRANT_PATH = os.path.join(os.path.dirname(__file__), "qdrant_db")
COLLECTION_NAME = "financial_reports"

# Embedding settings
# Option 1: keepitreal/vietnamese-sbert (Good for Vietnamese)
# Option 2: sentence-transformers/paraphrase-multilingual-mpnet-base-v2 (Multilingual)
EMBEDDING_MODEL_NAME = "keepitreal/vietnamese-sbert"

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
