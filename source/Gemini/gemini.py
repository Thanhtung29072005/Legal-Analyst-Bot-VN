import os
import sys

# Ensure parent directory is in path to import config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import config
from langchain_groq import ChatGroq
from qdrant_client import QdrantClient

def get_llm():
    """Khởi tạo và trả về đối tượng ChatGroq."""
    return ChatGroq(
        model=config.LLM_MODEL,
        temperature=config.LLM_TEMPERATURE,
        max_tokens=config.LLM_MAX_TOKENS
    )

def get_embeddings():
    """Khởi tạo và trả về embeddings model dựa theo cấu hình (Cohere hoặc HuggingFace)."""
    if config.EMBEDDING_PROVIDER == "cohere":
        from langchain_cohere import CohereEmbeddings
        # pyrefly: ignore [missing-argument]
        return CohereEmbeddings(
            cohere_api_key=config.COHERE_API_KEY,
            model=config.COHERE_EMBEDDING_MODEL,
        )
    else:
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL_NAME
        )

def get_qdrant_client():
    """Khởi tạo và trả về client kết nối Qdrant."""
    return QdrantClient(path=config.QDRANT_PATH)
