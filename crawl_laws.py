import os
import re
import urllib.parse
from urllib.parse import urlparse, parse_qs, unquote
import requests
from bs4 import BeautifulSoup

def clean_filename(text):
    # Chuyển tiếng Việt có dấu thành không dấu và loại bỏ các ký tự lạ để làm tên file sạch
    import unicodedata
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
    text = re.sub(r'[^a-zA-Z0-9\s-]', '', text).strip().lower()
    text = re.sub(r'[\s-]+', '-', text)
    return text

def extract_ddg_real_url(url):
    # Giải mã link redirect của DuckDuckGo để lấy URL trực tiếp
    if "uddg=" in url:
        parsed = urlparse(url)
        queries = parse_qs(parsed.query)
        if "uddg" in queries:
            return unquote(queries["uddg"][0])
    return url

def search_legal_pdf(keyword):
    # Tìm kiếm file PDF chứa từ khóa toàn cầu để tránh lỗi cú pháp của DuckDuckGo
    query = f'{keyword} filetype:pdf'
    url = "https://html.duckduckgo.com/html/"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36"
    }
    data = {"q": query}
    
    print(f"[*] Đang tìm kiếm file PDF cho từ khóa: '{keyword}'...")
    try:
        response = requests.get(url, params=data, headers=headers, timeout=15)
        response.raise_for_status()
    except Exception as e:
        print(f"[!] Lỗi khi kết nối với công cụ tìm kiếm: {e}")
        return None
        
    soup = BeautifulSoup(response.text, "html.parser")
    results = soup.select("#links .result")
    
    # Các tên miền uy tín về luật của Việt Nam
    trusted_domains = ["chinhphu.vn", "vbpl.vn", "moj.gov.vn", "luatvietnam.vn", "thuvienphapluat.vn", "congbao.chinhphu.vn", "edu.vn"]
    
    pdf_links = []
    external_links = []
    
    for res in results:
        a_tag = res.select_one(".result__a")
        if a_tag:
            href = a_tag.get("href", "")
            real_url = extract_ddg_real_url(href)
            # Chỉ lấy các link dẫn trực tiếp tới tệp PDF
            if real_url.lower().endswith(".pdf") or "pdf" in real_url.lower():
                title = a_tag.get_text(strip=True)
                item = {"title": title, "url": real_url}
                
                # Phân loại nguồn tin cậy
                if any(dom in real_url.lower() for dom in trusted_domains):
                    pdf_links.append(item)
                else:
                    external_links.append(item)
                    
    # Ưu tiên các nguồn uy tín, nếu không có thì lấy nguồn khác
    if pdf_links:
        print(f"[+] Tìm thấy {len(pdf_links)} tài liệu từ nguồn pháp luật chính thống.")
        return pdf_links
    elif external_links:
        print(f"[!] Không tìm thấy nguồn chính thống, sử dụng {len(external_links)} tài liệu từ nguồn tham khảo khác.")
        return external_links
    return []

def download_pdf(url, filename):
    laws_dir = os.path.join("data", "laws")
    os.makedirs(laws_dir, exist_ok=True)
    dest_path = os.path.join(laws_dir, filename)
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36"
    }
    
    print(f"[*] Đang kết nối tải file từ nguồn chính thống...")
    try:
        res = requests.get(url, headers=headers, stream=True, timeout=30)
        res.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in res.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print(f"[+] Tải xuống thành công: {filename} (Kích thước: {os.path.getsize(dest_path)} bytes)")
        return dest_path
    except Exception as e:
        print(f"[!] Lỗi khi tải xuống file PDF: {e}")
        return None

def main():
    print("=" * 60)
    print("       CÔNG CỤ TỰ ĐỘNG TÌM KIẾM VÀ CÀO VĂN BẢN PHÁP LUẬT")
    print("=" * 60)
    
    import sys
    if len(sys.argv) > 1:
        keyword = " ".join(sys.argv[1:]).strip()
        print(f"[+] Nhận từ khóa từ tham số dòng lệnh: {keyword}")
    else:
        keyword = input("Nhập tên văn bản luật muốn tải (Ví dụ: Luật Đất đai 2024): ").strip()
        
    if not keyword:
        print("[!] Từ khóa không được để trống!")
        return
        
    results = search_legal_pdf(keyword)
    
    if not results:
        print("[!] Không tìm thấy file PDF văn bản pháp luật chính thống nào khớp với từ khóa.")
        print("    Gợi ý: Thử thêm từ khóa cụ thể hơn như kèm năm ban hành (Ví dụ: 'Luật Nhà ở 2023').")
        return
        
    print(f"\n[+] Tìm thấy {len(results)} tài liệu tiềm năng:")
    for i, res in enumerate(results[:5], 1):
        print(f"  {i}. {res['title']}")
        print(f"     Link: {res['url']}")
        
    # Tự động tải xuống kết quả phù hợp nhất đầu tiên
    best_match = results[0]
    print(f"\n[*] Tiến hành tải tài liệu tốt nhất: {best_match['title']}")
    
    # Tạo tên file chuẩn hóa
    file_name = clean_filename(keyword) + ".pdf"
    
    downloaded_path = download_pdf(best_match['url'], file_name)
    
    if downloaded_path:
        print("\n[+] Quá trình cào văn bản hoàn tất!")
        print(f"    File đã lưu tại: {downloaded_path}")
        print("    Bạn có thể chạy script 'ingest_laws.py' hoặc nạp trực tiếp qua Streamlit Web.")
        
if __name__ == "__main__":
    main()
