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
