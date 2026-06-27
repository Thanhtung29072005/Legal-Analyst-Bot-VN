import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
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
        # Initialize Embeddings theo provider
        if config.EMBEDDING_PROVIDER == "cohere":
            from langchain_cohere import CohereEmbeddings
          
            self.embeddings = CohereEmbeddings(
                cohere_api_key=config.COHERE_API_KEY,
                model=config.COHERE_EMBEDDING_MODEL,
            )
        else:
            from langchain_huggingface import HuggingFaceEmbeddings
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

    def load_and_index_pdf(self, file_path, session_id=None):
        """Loads a PDF, splits it, and stores it in Qdrant Local."""
        loader = PyPDFLoader(file_path)
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        splits = text_splitter.split_documents(docs)

        # Gắn nguồn (file_name) và session_id vào metadata
        import os
        file_name = os.path.basename(file_path)
        for doc in splits:
            doc.metadata["source"] = file_name
            if session_id is not None:
                doc.metadata["session_id"] = session_id

        # Kiểm tra và khởi tạo collection nếu chưa có
        try:
            self.client.get_collection(config.COLLECTION_NAME)
        except Exception:
            from qdrant_client.models import VectorParams, Distance
            vector_size = len(self.embeddings.embed_query("test"))
            self.client.create_collection(
                collection_name=config.COLLECTION_NAME,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
            )

        # Xóa các vector cũ của cùng file này để tránh trùng lặp tài liệu khi nạp lại
        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            self.client.delete(
                collection_name=config.COLLECTION_NAME,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="metadata.source",
                            match=MatchValue(value=file_name)
                        )
                    ]
                )
            )
        except Exception:
            pass

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

    def get_indexed_documents(self):
        """Lấy danh sách các tài liệu luật (tên file PDF) đã được nạp vào Qdrant."""
        if not self.vectorstore:
            return []
        try:
            response, _ = self.client.scroll(
                collection_name=config.COLLECTION_NAME,
                limit=1000,
                with_payload=True,
                with_vectors=False
            )
            sources = set()
            for point in response:
                payload = point.payload
                metadata = payload.get("metadata", {})
                source = metadata.get("source") or payload.get("source")
                if source:
                    sources.add(source)
            return sorted(list(sources))
        except Exception:
            return []

    def get_conversation_chain(self, session_id=None):
        """Builds a history-aware RAG chain."""
        if not self.vectorstore:
            raise ValueError("Vectorstore not initialized. Please load a PDF first.")

        search_kwargs = {"k": config.RETRIEVER_K}
        
        # Để tư vấn luật toàn hệ thống, ta không áp dụng bộ lọc session_id vào retriever.
        # Tất cả các phiên chat đều tìm kiếm chung trên kho dữ liệu luật pháp của Qdrant.

        retriever = self.vectorstore.as_retriever(search_kwargs=search_kwargs)

        # 1. Contextualize Question Prompt (deals with history)
        contextualize_q_system_prompt = """Bạn là trợ lý AI chuyên về luật pháp. Dựa vào lịch sử hội thoại và câu hỏi mới nhất của người dùng,
        hãy viết lại câu hỏi thành một câu hỏi độc lập có đầy đủ ý nghĩa pháp lý để phục vụ việc tra cứu. Không cần giải thích hay trả lời câu hỏi, chỉ cần viết lại nếu cần thiết, ngược lại giữ nguyên câu hỏi."""
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
        qa_system_prompt = """Bạn là một chuyên gia tư vấn luật pháp Việt Nam chuyên nghiệp (Legal Advisor).
        Nhiệm vụ của bạn là giải đáp thắc mắc của người dùng dựa trên thông tin ngữ cảnh (Context) các văn bản luật được cung cấp dưới đây.
        
        Quy tắc:
        - Chỉ sử dụng dữ liệu trong Context để trả lời. Không tự bịa đặt điều luật, số hiệu văn bản pháp lý hoặc thông tin không có trong tài liệu.
        - Trích dẫn rõ ràng tên văn bản luật, số hiệu, điều, khoản, điểm (Ví dụ: "Theo Điều 5 Luật Đất đai 2024...") khi trả lời để câu trả lời có tính thuyết phục cao.
        - Nếu thông tin không có trong Context hoặc Context không đủ để trả lời, hãy nói rõ: "Tôi không tìm thấy thông tin pháp lý này trong cơ sở dữ liệu luật hiện tại của hệ thống."
        - Trả lời rõ ràng, mạch lạc bằng tiếng Việt, đúng thuật ngữ pháp lý. Có thể dùng các gạch đầu dòng để phân tách các quy định pháp luật cho người dùng dễ đọc.
        
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

    def ask(self, question, chat_history, session_id=None):
        """Asks a question with history and returns the answer and sources."""
        chain = self.get_conversation_chain(session_id)
        # chat_history format should be a list of BaseMessage (HumanMessage, AIMessage)
        response = chain.invoke({"input": question, "chat_history": chat_history})
        
        answer = response["answer"]
        sources = []
        for doc in response.get("context", []):
            doc_source = doc.metadata.get('source', 'Tài liệu luật')
            doc_page = doc.metadata.get('page', 0) + 1
            sources.append(f"{doc_source} (Trang {doc_page})")
            
        # Deduplicate sources
        sources = list(set(sources))
        return answer, sources

    def summarize_pdf(self, file_path):
        """Extracts text from PDF and gets a full summary from LLM using stuffing."""
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        
        # Để tránh vượt giới hạn Rate Limit (12,000 Tokens/phút) của Groq:
        # Nếu file dài hơn 10 trang, chọn lọc lấy 8 trang đầu (chứa tổng quan/phạm vi)
        # và 2 trang cuối để tóm tắt.
        num_pages = len(docs)
        if num_pages <= 10:
            selected_docs = docs
        else:
            selected_docs = docs[:8] + docs[-2:]
            
        # Combine selected page contents
        full_text = "\n".join([doc.page_content for doc in selected_docs])
        
        # Cắt tiếp nếu tổng số ký tự vẫn quá lớn (giới hạn an toàn khoảng 28,000 ký tự ~ 7,000 tokens)
        max_chars = 28000
        if len(full_text) > max_chars:
            full_text = full_text[:max_chars] + "\n\n[Tài liệu đã được lược bớt một số trang giữa để tránh vượt giới hạn API]"
            
        prompt = f"""Bạn là một chuyên gia tư vấn luật pháp Việt Nam chuyên nghiệp (Legal Advisor).
Nhiệm vụ của bạn là đọc toàn bộ văn bản pháp lý/luật/nghị định/thông tư dưới đây và viết một bản tóm tắt phân tích pháp lý toàn diện, rõ ràng bằng tiếng Việt.

Yêu cầu nội dung bản tóm tắt cần làm rõ:
1. **Thông tin chung về văn bản:** 
   - Tên đầy đủ của văn bản luật/nghị định/thông tư/quyết định.
   - Cơ quan ban hành, ngày ban hành và ngày có hiệu lực (nếu có trong tài liệu).
2. **Phạm vi điều chỉnh & Đối tượng áp dụng:**
   - Văn bản này điều chỉnh những vấn đề gì và áp dụng cho những ai.
3. **Các nội dung cốt lõi/Điểm mới nổi bật:**
   - Tóm tắt các chương, điều khoản quan trọng nhất hoặc những điểm mới nổi bật so với quy định cũ.
4. **Các lưu ý quan trọng hoặc chế tài liên quan:**
   - Những quy định cần đặc biệt tuân thủ hoặc các chế tài xử phạt/nghĩa vụ pháp lý đáng chú ý.

Văn bản tài liệu:
---
{full_text}
---

Hãy viết bản tóm tắt một cách tự nhiên, mạch lạc, trực tiếp đi vào các nội dung chính của văn bản, sử dụng đúng thuật ngữ pháp lý.

BẢN TÓM TẮT VĂN BẢN PHÁP LUẬT CHI TIẾT:"""

        response = self.llm.invoke(prompt)
        return response.content
