import os
import glob
from dotenv import load_dotenv

# Load env variables first
load_dotenv()

import config

def main():
    laws_dir = os.path.join("data", "laws")
    os.makedirs(laws_dir, exist_ok=True)
    
    # Lấy danh sách file PDF trong thư mục data/laws
    pdf_files = glob.glob(os.path.join(laws_dir, "*.pdf"))
    
    if not pdf_files:
        print(f"[*] Thư mục '{laws_dir}' trống hoặc không tìm thấy file PDF nào.")
        print("[*] Vui lòng đặt các file PDF luật vào thư mục này rồi chạy lại script, hoặc tải lên trực tiếp từ giao diện Web.")
        return
        
    print(f"[*] Tìm thấy {len(pdf_files)} file PDF luật để nạp...")
    
    # Khởi tạo RAG Engine
    from source.Function.search_Qdrant import FinancialRAG
    rag_engine = FinancialRAG()
    
    for pdf_path in pdf_files:
        file_name = os.path.basename(pdf_path)
        print(f"\n[+] Đang xử lý: {file_name}...")
        try:
            num_chunks = rag_engine.load_and_index_pdf(pdf_path)
            print(f"[-] Thành công! Đã nạp và lập chỉ mục {num_chunks} chunks cho: {file_name}")
        except Exception as e:
            print(f"[!] Lỗi khi xử lý {file_name}: {str(e)}")
            
    print("\n[*] Quá trình nạp luật hoàn tất!")

if __name__ == "__main__":
    main()
