import os
import sys

# Ensure parent directory is in path to import config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from langchain_community.document_loaders import PyPDFLoader

def summarize_pdf(rag_engine, file_path):
    """Tóm tắt văn bản PDF sử dụng kỹ thuật Map-Reduce để xử lý tài liệu dài."""
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    
    # 1. Phân bổ các trang cần đọc (Tối đa 20 trang để đảm bảo thời gian xử lý và giới hạn API)
    if len(docs) <= 20:
        selected_docs = docs
    else:
        # Lấy 12 trang đầu, 4 trang giữa và 4 trang cuối để có độ phủ tốt nhất
        mid_start = len(docs) // 2 - 2
        selected_docs = docs[:12] + docs[mid_start:mid_start+4] + docs[-4:]
        
    # 2. Gom các trang vào các nhóm (4 trang mỗi nhóm => tối đa 5 nhóm)
    pages_per_chunk = 4
    chunks = []
    current_chunk = []
    for i, doc in enumerate(selected_docs):
        current_chunk.append(doc.page_content)
        if len(current_chunk) == pages_per_chunk or i == len(selected_docs) - 1:
            chunks.append("\n".join(current_chunk))
            current_chunk = []
            
    # 3. Map Phase: Tóm tắt từng phân đoạn
    partial_summaries = []
    print(f"[*] Tiến hành Map Phase: Tóm tắt {len(chunks)} phân đoạn tài liệu...")
    for idx, chunk_text in enumerate(chunks):
        map_prompt = f"""Bạn là trợ lý AI chuyên về luật pháp Việt Nam. Dưới đây là nội dung của một phần trong văn bản pháp luật (Phần {idx+1}/{len(chunks)}).
Hãy tóm tắt ngắn gọn các nội dung pháp lý chính trong phần văn bản này dưới dạng các ý gạch đầu dòng (bullet points). 
Tập trung vào thông tin chung, phạm vi điều chỉnh, các quy định cốt lõi và các chế tài/nghĩa vụ pháp lý đáng chú ý. Chỉ phản hồi bản tóm tắt trực tiếp, không giải thích hay dẫn nhập dài dòng.

Nội dung phần văn bản:
---
{chunk_text}
---

Tóm tắt ngắn gọn:"""
        try:
            response = rag_engine.llm.invoke(map_prompt)
            partial_summaries.append(response.content.strip())
        except Exception as e:
            print(f"[!] Lỗi khi tóm tắt phân đoạn {idx+1}: {e}")
            partial_summaries.append(f"[Lỗi tóm tắt phân đoạn {idx+1}]")
            
    # 4. Reduce Phase: Tổng hợp các phân đoạn thành bản tóm tắt hoàn chỉnh
    print("[*] Tiến hành Reduce Phase: Tổng hợp toàn diện...")
    combined_partial_summaries = "\n\n".join([f"### Tóm tắt Phân đoạn {idx+1}:\n{summary}" for idx, summary in enumerate(partial_summaries)])
    
    reduce_prompt = f"""Bạn là một chuyên gia tư vấn luật pháp Việt Nam chuyên nghiệp (Legal Advisor).
Dưới đây là các bản tóm tắt phân đoạn của một văn bản pháp lý/luật/nghị định/thông tư. 
Nhiệm vụ của bạn là tổng hợp các bản tóm tắt phân đoạn này thành một bản tóm tắt phân tích pháp lý toàn diện, rõ ràng bằng tiếng Việt.

Yêu cầu nội dung bản tóm tắt tổng hợp cuối cùng cần làm rõ các phần sau:
1. **Thông tin chung về văn bản:** 
   - Tên đầy đủ của văn bản luật/nghị định/thông tư/quyết định.
   - Cơ quan ban hành, ngày ban hành và ngày có hiệu lực (nếu có).
2. **Phạm vi điều chỉnh & Đối tượng áp dụng:**
   - Văn bản này điều chỉnh những vấn đề gì và áp dụng cho những ai.
3. **Các nội dung cốt lõi/Điểm mới nổi bật:**
   - Tóm tắt các chương, điều khoản quan trọng nhất hoặc những điểm mới nổi bật so với quy định cũ.
4. **Các lưu ý quan trọng hoặc chế tài liên quan:**
   - Những quy định cần đặc biệt tuân thủ hoặc các chế tài xử phạt/nghĩa vụ pháp lý đáng chú ý.

Danh sách tóm tắt phân đoạn:
---
{combined_partial_summaries}
---

Hãy viết bản tóm tắt tổng hợp một cách tự nhiên, mạch lạc, trực tiếp đi vào các nội dung chính của văn bản, sử dụng đúng thuật ngữ pháp lý.

BẢN TÓM TẮT VĂN BẢN PHÁP LUẬT CHI TIẾT:"""
    
    try:
        final_response = rag_engine.llm.invoke(reduce_prompt)
        return final_response.content
    except Exception as e:
        print(f"[!] Lỗi trong bước tổng hợp cuối cùng: {e}")
        return "⚠️ Không thể tổng hợp bản tóm tắt do lỗi API: " + str(e)
