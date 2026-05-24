import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_groq import ChatGroq
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_classic.chains.history_aware_retriever import create_history_aware_retriever

import config

class FinancialRAG:
    def __init__(self):
        # Initialize Embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL_NAME
        )
        
        # Initialize LLM
        self.llm = ChatGroq(
            model=config.LLM_MODEL,
            temperature=config.LLM_TEMPERATURE,
            max_tokens=config.LLM_MAX_TOKENS
        )
        
        from qdrant_client import QdrantClient
        self.client = QdrantClient(path=config.QDRANT_PATH)
        
        # Will hold the Qdrant vector store
        self.vectorstore = None

    def load_and_index_pdf(self, file_path):
        """Loads a PDF, splits it, and stores it in Qdrant Local."""
        loader = PyPDFLoader(file_path)
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        splits = text_splitter.split_documents(docs)

        try:
            self.client.get_collection(config.COLLECTION_NAME)
        except Exception:
            from qdrant_client.models import VectorParams, Distance
            vector_size = len(self.embeddings.embed_query("test"))
            self.client.create_collection(
                collection_name=config.COLLECTION_NAME,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
            )

        self.vectorstore = QdrantVectorStore(
            client=self.client,
            collection_name=config.COLLECTION_NAME,
            embedding=self.embeddings
        )
        self.vectorstore.add_documents(splits)
        return len(splits)

    def load_existing_db(self):
        """Loads an existing Qdrant DB if it exists."""
        if os.path.exists(config.QDRANT_PATH):
            # Check if collection exists
            try:
                self.client.get_collection(config.COLLECTION_NAME)
                self.vectorstore = QdrantVectorStore(
                    client=self.client, 
                    collection_name=config.COLLECTION_NAME, 
                    embedding=self.embeddings
                )
                return True
            except Exception:
                pass
        return False

    def get_conversation_chain(self):
        """Builds a history-aware RAG chain."""
        if not self.vectorstore:
            raise ValueError("Vectorstore not initialized. Please load a PDF first.")

        retriever = self.vectorstore.as_retriever(search_kwargs={"k": config.RETRIEVER_K})

        # 1. Contextualize Question Prompt (deals with history)
        contextualize_q_system_prompt = """Bạn là trợ lý AI. Dựa vào lịch sử hội thoại và câu hỏi mới nhất của người dùng,
        hãy viết lại câu hỏi thành một câu hỏi độc lập có đầy đủ ý nghĩa. Không cần trả lời câu hỏi, chỉ cần viết lại nếu cần thiết, ngược lại giữ nguyên."""
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        history_aware_retriever = create_history_aware_retriever(
            self.llm, retriever, contextualize_q_prompt
        )

        # 2. Answer Question Prompt
        qa_system_prompt = """Bạn là một chuyên gia phân tích tài chính cao cấp (Financial Analyst).
        Nhiệm vụ của bạn là phân tích báo cáo tài chính và trả lời câu hỏi của người dùng dựa trên thông tin được cung cấp dưới đây.
        
        Quy tắc:
        - Chỉ sử dụng dữ liệu trong Context để trả lời. Không bịa đặt số liệu.
        - Nếu số liệu hoặc thông tin không có trong Context, hãy nói rõ: "Tôi không tìm thấy thông tin này trong tài liệu."
        - Trả lời rõ ràng, mạch lạc, có thể dùng bullet points để dễ đọc. Trích dẫn nếu cần.
        - Nếu câu hỏi yêu cầu so sánh hoặc tính toán đơn giản từ các số liệu có sẵn, hãy thực hiện cẩn thận.
        
        Context:
        {context}"""
        
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", qa_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        
        question_answer_chain = create_stuff_documents_chain(self.llm, qa_prompt)
        
        # 3. Full Retrieval Chain
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
        
        return rag_chain

    def ask(self, question, chat_history):
        """Asks a question with history and returns the answer and sources."""
        chain = self.get_conversation_chain()
        # chat_history format should be a list of BaseMessage (HumanMessage, AIMessage)
        response = chain.invoke({"input": question, "chat_history": chat_history})
        
        answer = response["answer"]
        sources = []
        for doc in response.get("context", []):
            sources.append(f"Page {doc.metadata.get('page', 'Unknown')}")
            
        # Deduplicate sources
        sources = list(set(sources))
        return answer, sources
