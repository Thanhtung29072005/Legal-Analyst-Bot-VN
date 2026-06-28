import os
import sys

# Ensure parent directory is in path to import config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from langchain_community.document_loaders import PyPDFLoader

def summarize_pdf(rag_engine, file_path):
    """Extracts text from PDF and gets a full summary from LLM using stuffing."""
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    
    num_pages = len(docs)
    if num_pages <= 10:
        selected_docs = docs
    else:
        selected_docs = docs[:8] + docs[-2:]
        
    full_text = "\n".join([doc.page_content for doc in selected_docs])
    
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

    response = rag_engine.llm.invoke(prompt)
    return response.content
