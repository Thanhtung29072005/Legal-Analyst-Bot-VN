# Legal Advisory Chatbot System

## Description 

The **Legal Advisory Chatbot System** is designed to assist users by answering questions related to Vietnamese law. By leveraging advanced Retrieval-Augmented Generation (RAG) techniques, this system can analyze and retrieve relevant legal documents from an extensive collection of official Vietnamese legal texts. The core functionality of the chatbot includes providing users with accurate and up-to-date legal information, helping them understand complex legal terminology, and offering clear, actionable advice based on Vietnamese legal documents. This system is highly valuable for individuals, businesses, and legal professionals seeking quick access to legal knowledge.

## Demo

![image](https://github.com/user-attachments/assets/443b303d-f11b-44ef-9299-63f5f13a5936)

## Models & Technologies

**LLM Model for Reasoning & Answering:**
- **Llama 3.3 70B (via Groq API)**: Used for precise, high-quality, and context-aware responses based on retrieved laws.

**Embeddings & Search Models:**
- **Cohere Embedding (`embed-multilingual-v3.0`)**: 1024-dimensional multilingual model optimized for search and retrieval.
- **HuggingFace Sentence Transformer (`keepitreal/vietnamese-sbert`)**: Alternative local embedding model optimized for Vietnamese.
- **Cohere Rerank (`rerank-multilingual-v3.0`)**: Used to re-rank retrieved documents, ensuring the top results are the most relevant.

**Database & Storage:**
- **Qdrant Vector Database**: For efficient high-dimensional vector search.
- **SQL Server (MS SQL)**: For tracking chat sessions, history, and PDF upload summaries.

## Features

- **Vietnamese Law Knowledge Base**: The chatbot is built upon a large dataset of Vietnamese legal documents, ensuring that the responses are based on reliable and official sources.
- **Hybrid Search & Re-ranking**: Combines vector database retrieval with Cohere's state-of-the-art re-ranking engine to ensure high precision in retrieved laws.
- **Dynamic PDF Uploading & Indexing**: Users can upload new legal PDF documents, which are automatically indexed into Qdrant, summarized, and integrated into the conversation context.
- **Session History Management**: Full conversation history tracking stored in SQL Server, allowing users to switch between, delete, or resume previous chats.
- **Interactive UI**: A sleek, modern, and fully responsive user interface built using HTML, modern CSS, and vanilla JS.

## Data Ingestion & Processing

The core of the system relies on high-quality legal documents. The steps involved in data processing include:

1. **Document Ingestion**: Parsing PDF documents using Python-based extractors to extract raw legal text.
2. **Text Chunking**: Splitting documents into manageable segments (chunk size of `2000` characters with `300` overlap) to keep paragraphs and articles intact.
3. **Vector Embeddings**: Converting text chunks into high-dimensional vector representations using Cohere or HuggingFace embeddings.
4. **Vector Storage (Qdrant)**: Storing the vectors in Qdrant collections (`vietnamese_laws`) to enable rapid semantic similarity search.

---

## Setup & Running the Application

### 1. Prerequisites
Ensure you have the following installed:
* [Docker & Docker Compose](https://www.docker.com/)
* SQL Server instance (local or remote)

### 2. Configuration
Create a `.env` file in the root directory:
```env
GROQ_API_KEY=your_groq_api_key
COHERE_API_KEY=your_cohere_api_key
EMBEDDING_PROVIDER=cohere # 'cohere' or 'huggingface'
LLM_MODEL=llama-3.3-70b-versatile

# SQL Server Configuration
SQL_SERVER=host.docker.internal
SQL_DATABASE=Legal_Chatbot_DB
SQL_TRUSTED_CONNECTION=no
SQL_USERNAME=your_username
SQL_PASSWORD=your_password
```

### 3. Running with Docker Compose
To build and start the entire stack (FastAPI web app, mounts local directories for dynamic templates, static files, and databases):

```bash
docker compose up --build
```
The application will be accessible at: `http://localhost:5000`
