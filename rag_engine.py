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
import re
from langchain_core.documents import Document

def split_by_articles(docs):
    """Chia văn bản luật thành các Điều (Articles) thay vì chia theo dung lượng ký tự ngẫu nhiên."""
    full_text = ""
    page_map = []  # Lưu dải chỉ số ký tự thuộc trang nào [(start_char_idx, end_char_idx, page_num)]
    
    for doc in docs:
        start_idx = len(full_text)
        full_text += doc.page_content + "\n"
        end_idx = len(full_text)
        page_map.append((start_idx, end_idx, doc.metadata.get("page", 0)))
        
    # Phát hiện điểm bắt đầu của Điều luật (ví dụ: "Điều 5. ..." ở đầu dòng)
    article_regex = re.compile(r'^(?:Điều|ĐIỀU)\s+(\d+[a-z]?)\b', re.MULTILINE)
    matches = list(article_regex.finditer(full_text))
    
    if len(matches) < 3:
        # Nếu có ít hơn 3 tiêu chuẩn Điều luật, có thể là văn bản thường -> dùng mặc định
        return None
        
    chunks = []
    chapter_regex = re.compile(r'(?:Chương|CHƯƠNG)\s+([I|V|X|L|C\d]+)', re.IGNORECASE)
    
    for i in range(len(matches)):
        start = matches[i].start()
        end = matches[i+1].start() if i + 1 < len(matches) else len(full_text)
        
        article_num = matches[i].group(1)
        chunk_text = full_text[start:end].strip()
        
        # Tìm xem Điều này thuộc trang nào trong PDF gốc
        page_num = 0
        for p_start, p_end, p_val in page_map:
            if p_start <= start < p_end:
                page_num = p_val
                break
                
        # Tìm Chương gần nhất nằm trước Điều luật này
        active_chapter = "Chưa rõ"
        text_before = full_text[:start]
        chapters = list(chapter_regex.finditer(text_before))
        if chapters:
            active_chapter = f"Chương {chapters[-1].group(1)}"
            
        doc = Document(
            page_content=chunk_text,
            metadata={
                "article": f"Điều {article_num}",
                "chapter": active_chapter,
                "page": page_num
            }
        )
        chunks.append(doc)
        
    return chunks

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

    def extract_entities_from_query(self, query: str):
        """Dùng LLM để trích xuất Điều, Chương và tên Luật từ câu hỏi của người dùng."""
        prompt = f"""Bạn là trợ lý AI chuyên về luật pháp Việt Nam. Nhiệm vụ của bạn là phân tích câu hỏi của người dùng và trích xuất các thông tin cấu trúc sau (nếu có):
        1. Số Điều luật (Ví dụ: "Điều 5", "Điều 100", "Điều 10a")
        2. Số Chương (Ví dụ: "Chương I", "Chương III")
        3. Loại văn bản hoặc Tên luật (Ví dụ: "Luật Đất Đai", "Luật Hôn nhân", "Nghị định 15")
        
        Hãy trả về kết quả dưới định dạng JSON với các khóa: "article", "chapter", "law_name". Nếu không có thông tin nào, hãy để giá trị là null.
        Không giải thích gì thêm, chỉ trả về chuỗi JSON hợp lệ.
        
        Ví dụ:
        "Điều kiện cấp sổ đỏ theo Điều 100 Luật Đất Đai là gì?" -> {{"article": "Điều 100", "chapter": null, "law_name": "Luật Đất Đai"}}
        "Quy định tại Chương II Luật Hôn nhân" -> {{"article": null, "chapter": "Chương II", "law_name": "Luật Hôn nhân"}}
        "Độ tuổi đăng ký kết hôn là bao nhiêu?" -> {{"article": null, "chapter": null, "law_name": null}}
        
        Câu hỏi: "{query}"
        JSON:"""
        
        try:
            response = self.llm.invoke(prompt)
            import json
            content = response.content.strip()
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            data = json.loads(content.strip())
            return data
        except Exception:
            return {"article": None, "chapter": None, "law_name": None}

    def rewrite_question(self, question, chat_history):
        """Viết lại câu hỏi dựa trên lịch sử chat để làm câu hỏi độc lập."""
        if not chat_history:
            return question
            
        prompt = f"""Bạn là trợ lý AI chuyên về luật pháp. Dựa vào lịch sử hội thoại và câu hỏi mới nhất của người dùng,
        hãy viết lại câu hỏi thành một câu hỏi độc lập có đầy đủ ý nghĩa pháp lý để phục vụ việc tra cứu. Không cần giải thích hay trả lời câu hỏi, chỉ cần viết lại nếu cần thiết, ngược lại giữ nguyên câu hỏi.
        
        Lịch sử chat:
        {chat_history}
        
        Câu hỏi mới: {question}
        Câu hỏi viết lại:"""
        
        try:
            response = self.llm.invoke(prompt)
            return response.content.strip()
        except Exception:
            return question

    def load_and_index_pdf(self, file_path, session_id=None):
        """Loads a PDF, splits it, and stores it in Qdrant Local."""
        loader = PyPDFLoader(file_path)
        docs = loader.load()

        # Thử phân chia theo cấu trúc các Điều luật (Articles) trước
        splits = split_by_articles(docs)
        is_article_split = splits is not None
        
        if not is_article_split:
            # Fallback nếu không phải cấu trúc luật: chia theo độ dài ký tự
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

    def delete_document(self, file_name):
        """Xóa hoàn toàn một tài liệu khỏi cơ sở dữ liệu Qdrant và thư mục data/laws."""
        import os
        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            # Xóa khỏi Qdrant
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
            # Xóa file vật lý nếu có
            file_path = os.path.join("data", "laws", file_name)
            if os.path.exists(file_path):
                os.remove(file_path)
            return True
        except Exception:
            return False

    def get_indexed_documents(self):
        """Lấy danh sách các tài liệu luật (tên file PDF) đã được nạp vào Qdrant."""
        if not self.vectorstore:
            return []
        try:
            sources = set()
            next_page_offset = None
            while True:
                response, next_page_offset = self.client.scroll(
                    collection_name=config.COLLECTION_NAME,
                    limit=1000,
                    offset=next_page_offset,
                    with_payload=True,
                    with_vectors=False
                )
                for point in response:
                    payload = point.payload
                    metadata = payload.get("metadata", {})
                    source = metadata.get("source") or payload.get("source")
                    if source:
                        sources.add(source)
                if next_page_offset is None:
                    break
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
        - Linh hoạt về thuật ngữ: Nếu người dùng sử dụng từ ngữ đời thường hoặc thuật ngữ không chuẩn xác tuyệt đối nhưng có ý nghĩa tương đương hoặc liên quan mật thiết với nội dung trong Context (ví dụ: dùng "vô ý giết người" thay vì "vô ý làm chết người"), hãy chủ động giải đáp dựa trên các điều luật liên quan đó và hướng dẫn lại thuật ngữ pháp lý chính xác cho họ. Không trả lời cứng nhắc là "không tìm thấy" trong trường hợp này.
        - Chỉ trả lời: "Tôi không tìm thấy thông tin pháp lý này trong cơ sở dữ liệu luật hiện tại của hệ thống." khi nội dung câu hỏi hoàn toàn lệch hướng, không liên quan hoặc Context không chứa bất kỳ thông tin nào liên quan.
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

    def get_qa_chain(self):
        """Tạo chain trả lời câu hỏi dựa trên Context."""
        qa_system_prompt = """Bạn là một chuyên gia tư vấn luật pháp Việt Nam chuyên nghiệp (Legal Advisor).
        Nhiệm vụ của bạn là giải đáp thắc mắc của người dùng dựa trên thông tin ngữ cảnh (Context) các văn bản luật được cung cấp dưới đây.
        
        Quy tắc:
        - Chỉ sử dụng dữ liệu trong Context để trả lời. Không tự bịa đặt điều luật, số hiệu văn bản pháp lý hoặc thông tin không có trong tài liệu.
        - Trích dẫn rõ ràng tên văn bản luật, số hiệu, điều, khoản, điểm (Ví dụ: "Theo Điều 5 Luật Đất đai 2024...") khi trả lời để câu trả lời có tính thuyết phục cao.
        - Linh hoạt về thuật ngữ: Nếu người dùng sử dụng từ ngữ đời thường hoặc thuật ngữ không chuẩn xác tuyệt đối nhưng có ý nghĩa tương đương hoặc liên quan mật thiết với nội dung trong Context (ví dụ: dùng "vô ý giết người" thay vì "vô ý làm chết người"), hãy chủ động giải đáp dựa trên các điều luật liên quan đó và hướng dẫn lại thuật ngữ pháp lý chính xác cho họ. Không trả lời cứng nhắc là "không tìm thấy" trong trường hợp này.
        - Chỉ trả lời: "Tôi không tìm thấy thông tin pháp lý này trong cơ sở dữ liệu luật hiện tại của hệ thống." khi nội dung câu hỏi hoàn toàn lệch hướng, không liên quan hoặc Context không chứa bất kỳ thông tin nào liên quan.
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
        return create_stuff_documents_chain(self.llm, qa_prompt)

    def ask(self, question, chat_history, session_id=None):
        """Asks a question with history and returns the answer and sources."""
        # 1. Viết lại câu hỏi độc lập
        rewritten_question = self.rewrite_question(question, chat_history)
        
        # 2. Trích xuất thực thể từ câu hỏi viết lại
        entities = self.extract_entities_from_query(rewritten_question)
        
        # 3. Ánh xạ law_name sang tên file nguồn trong Qdrant
        source_filter = None
        if entities.get("law_name"):
            normalized_law_name = entities["law_name"].lower().replace(" ", "")
            indexed_docs = self.get_indexed_documents()
            for doc_name in indexed_docs:
                normalized_doc = doc_name.lower().replace(" ", "")
                if normalized_law_name in normalized_doc or normalized_doc in normalized_law_name:
                    source_filter = doc_name
                    break
        
        # 4. Tạo bộ lọc Qdrant Filter
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        must_conditions = []
        
        if source_filter:
            must_conditions.append(
                FieldCondition(
                    key="metadata.source",
                    match=MatchValue(value=source_filter)
                )
            )
            
        if entities.get("article"):
            must_conditions.append(
                FieldCondition(
                    key="metadata.article",
                    match=MatchValue(value=entities["article"])
                )
            )
            
        if entities.get("chapter"):
            must_conditions.append(
                FieldCondition(
                    key="metadata.chapter",
                    match=MatchValue(value=entities["chapter"])
                )
            )
            
        qdrant_filter = Filter(must=must_conditions) if must_conditions else None
        
        # 5. Truy vấn tương đồng trong Vectorstore
        if not self.vectorstore:
            raise ValueError("Vectorstore not initialized.")
            
        results = []
        try:
            results = self.vectorstore.similarity_search(
                query=rewritten_question,
                k=config.RETRIEVER_K,
                filter=qdrant_filter
            )
        except Exception:
            # Bỏ qua nếu có lỗi lọc
            pass
            
        # 6. Fallback nếu không có kết quả với bộ lọc cứng
        if not results and qdrant_filter:
            try:
                # Relax filter: chỉ lọc theo tên file luật
                if source_filter:
                    relaxed_filter = Filter(must=[
                        FieldCondition(
                            key="metadata.source",
                            match=MatchValue(value=source_filter)
                        )
                    ])
                    results = self.vectorstore.similarity_search(
                        query=rewritten_question,
                        k=config.RETRIEVER_K,
                        filter=relaxed_filter
                    )
                # Nếu vẫn không thấy hoặc không có source, tìm kiếm không bộ lọc
                if not results:
                    results = self.vectorstore.similarity_search(
                        query=rewritten_question,
                        k=config.RETRIEVER_K
                    )
            except Exception:
                pass
                
        # 7. Trả lời câu hỏi bằng chain
        qa_chain = self.get_qa_chain()
        response = qa_chain.invoke({
            "input": rewritten_question,
            "chat_history": chat_history,
            "context": results
        })
        
        # 8. Định dạng nguồn dẫn chiếu chi tiết
        sources = []
        for doc in results:
            doc_source = doc.metadata.get('source', 'Tài liệu luật')
            doc_page = doc.metadata.get('page', 0) + 1
            doc_article = doc.metadata.get('article')
            doc_chapter = doc.metadata.get('chapter')
            
            ref_str = f"{doc_source} (Trang {doc_page})"
            if doc_article:
                if doc_chapter and doc_chapter != "Chưa rõ":
                    ref_str = f"{doc_article} ({doc_chapter}) - {ref_str}"
                else:
                    ref_str = f"{doc_article} - {ref_str}"
            sources.append(ref_str)
            
        # Deduplicate sources
        sources = list(set(sources))
        return response, sources

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
