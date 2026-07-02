import re
from langchain_core.documents import Document

# Ngưỡng ký tự tối đa cho mỗi chunk (nếu Điều dài hơn thì chia nhỏ)
MAX_ARTICLE_CHUNK_SIZE = 1500


def extract_law_name_from_filename(file_name: str) -> str:
    """
    Trích xuất tên luật có thể đọc được từ tên file PDF.
    Ví dụ: "luat-dat-dai-2024.pdf" -> "Luật Đất Đai 2024"
             "luat_hon_nhan_gia_dinh.pdf" -> "Luật Hôn Nhân Gia Đình"
             "Luat Doanh Nghiep 2020.pdf" -> "Luat Doanh Nghiep 2020"
    """
    # Bỏ phần mở rộng file
    name = re.sub(r'\.pdf$', '', file_name, flags=re.IGNORECASE)
    # Thay dấu gạch ngang và gạch dưới bằng dấu cách
    name = name.replace('-', ' ').replace('_', ' ')
    # Viết hoa chữ cái đầu mỗi từ
    name = ' '.join(word.capitalize() for word in name.split())
    return name.strip()


def _split_long_article(article_num: str, chunk_text: str, base_metadata: dict,
                         max_size: int = MAX_ARTICLE_CHUNK_SIZE) -> list:
    """
    Nếu nội dung một Điều quá dài (> max_size ký tự), chia thành nhiều phần nhỏ
    nhưng vẫn giữ nguyên article metadata để không mất ngữ cảnh khi filter.
    """
    if len(chunk_text) <= max_size:
        doc = Document(
            page_content=chunk_text,
            metadata={**base_metadata}
        )
        return [doc]

    # Ưu tiên tách theo dòng, rồi theo câu để giữ tính toàn vẹn ngữ nghĩa
    lines = chunk_text.split('\n')
    sub_chunks = []
    current_part = []
    current_len = 0
    part_idx = 1

    for line in lines:
        line_len = len(line) + 1  # +1 cho ký tự newline
        if current_len + line_len > max_size and current_part:
            # Lưu phần hiện tại
            part_text = '\n'.join(current_part).strip()
            if part_text:
                doc = Document(
                    page_content=part_text,
                    metadata={
                        **base_metadata,
                        "article": base_metadata["article"],
                        "title": f"{base_metadata['article']} (phần {part_idx})"
                    }
                )
                sub_chunks.append(doc)
                part_idx += 1
            current_part = [line]
            current_len = line_len
        else:
            current_part.append(line)
            current_len += line_len

    # Phần cuối còn lại
    if current_part:
        part_text = '\n'.join(current_part).strip()
        if part_text:
            title_suffix = f" (phần {part_idx})" if part_idx > 1 else ""
            doc = Document(
                page_content=part_text,
                metadata={
                    **base_metadata,
                    "article": base_metadata["article"],
                    "title": f"{base_metadata['article']}{title_suffix}"
                }
            )
            sub_chunks.append(doc)

    return sub_chunks if sub_chunks else [Document(page_content=chunk_text, metadata=base_metadata)]


# pyrefly: ignore [bad-function-definition]
def split_by_articles(docs, law_name: str = None):
    """
    Chia văn bản luật thành các Điều (Articles) thay vì chia theo dung lượng ký tự ngẫu nhiên.

    Cải tiến so với phiên bản cũ:
    - Bổ sung metadata: law_name, title (tên đầy đủ của Điều)
    - Smart chunking: Nếu một Điều quá dài (> MAX_ARTICLE_CHUNK_SIZE) thì chia thành nhiều
      phần nhỏ nhưng vẫn giữ nguyên article metadata để không ảnh hưởng đến entity filter.
    """
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

    # Cố gắng trích xuất tiêu đề luật từ nội dung (thường ở đầu văn bản)
    extracted_doc_title = _extract_document_title(full_text)

    for i in range(len(matches)):
        start = matches[i].start()
        end = matches[i+1].start() if i + 1 < len(matches) else len(full_text)

        article_num = matches[i].group(1)
        article_label = f"Điều {article_num}"
        chunk_text = full_text[start:end].strip()

        # Trích xuất tiêu đề của Điều (thường là dòng đầu tiên sau "Điều X.")
        article_title = _extract_article_title(chunk_text, article_label)

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

        # Metadata đầy đủ
        base_metadata = {
            "article": article_label,
            "chapter": active_chapter,
            "page": page_num,
            "title": article_title,
            "law_name": law_name or extracted_doc_title or "Văn bản pháp luật",
            "source_law": law_name or extracted_doc_title or "Văn bản pháp luật",  # alias để tìm kiếm
        }

        # Smart chunking: chia Điều dài thành nhiều phần nhỏ
        sub_docs = _split_long_article(article_num, chunk_text, base_metadata)
        chunks.extend(sub_docs)

    return chunks


def _extract_document_title(full_text: str) -> str:
    """
    Cố gắng trích xuất tên luật từ nội dung văn bản.
    Thường xuất hiện dưới dạng "LUẬT ... " hoặc "Luật ... " ở phần đầu.
    """
    # Tìm pattern "LUẬT [Tên Luật]" ở phần đầu văn bản (1500 ký tự đầu)
    header_text = full_text[:1500]
    patterns = [
        r'(?:LUẬT|Luật)\s+([\w\s\-]+?)(?:\n|Số|Căn cứ|Điều|\d{4})',
        r'(?:NGHỊ ĐỊNH|Nghị định)\s+(?:SỐ\s+)?(\d+[\w/\-]+)',
        r'(?:THÔNG TƯ|Thông tư)\s+(?:SỐ\s+)?(\d+[\w/\-]+)',
    ]
    for pattern in patterns:
        m = re.search(pattern, header_text, re.IGNORECASE)
        if m:
            title = m.group(0).strip()
            # Làm sạch: bỏ ký tự thừa ở cuối
            title = re.sub(r'\s+(Số|Căn cứ|Điều|\d{4}).*$', '', title, flags=re.IGNORECASE).strip()
            if len(title) > 5:
                return title
    return ""


def _extract_article_title(chunk_text: str, article_label: str) -> str:
    """
    Trích xuất tiêu đề của một Điều luật từ nội dung chunk.
    Ví dụ: "Điều 5. Quyền và nghĩa vụ của..." -> "Điều 5. Quyền và nghĩa vụ của..."
    """
    # Lấy dòng đầu tiên của chunk (thường chứa tiêu đề Điều)
    first_line = chunk_text.split('\n')[0].strip()
    if len(first_line) > 5:
        # Giới hạn độ dài tiêu đề
        return first_line[:200]
    return article_label
