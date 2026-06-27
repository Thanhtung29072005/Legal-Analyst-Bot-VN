import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from rag_engine import FinancialRAG

# Load environment variables
load_dotenv()

# Setup Streamlit page config
st.set_page_config(
    page_title="Trợ lý Luật pháp Việt Nam AI",
    page_icon="⚖️",
    layout="wide"
)

# Custom styling to make the app look premium
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Outfit:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<style>
    /* Styling headers and fonts */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: #f8fafc;
    }
    
    .stApp {
        background-color: #f8fafc;
    }
    
    /* Brand Header */
    .brand-header {
        margin-bottom: 2rem;
        padding-bottom: 1.5rem;
        border-bottom: 1px solid #e2e8f0;
    }
    .brand-tag {
        font-family: 'Outfit', sans-serif;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.15em;
        color: #b45309; /* Justice Gold */
        margin-bottom: 0.5rem;
    }
    .main-title {
        font-family: 'Outfit', sans-serif !important;
        font-weight: 800 !important;
        font-size: 2.25rem !important;
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        margin: 0 0 0.5rem 0 !important;
        line-height: 1.2 !important;
    }
    .subtitle {
        font-family: 'Inter', sans-serif;
        font-size: 1rem;
        color: #64748b;
        margin: 0 !important;
    }
    
    /* Welcome Card Styling */
    .welcome-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-left: 4px solid #b45309; /* Gold accent */
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 2rem;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.02);
    }
    .welcome-badge {
        display: inline-block;
        background-color: #fef3c7;
        color: #b45309;
        font-size: 0.7rem;
        font-weight: 700;
        padding: 0.25rem 0.5rem;
        border-radius: 4px;
        margin-bottom: 0.75rem;
        letter-spacing: 0.05em;
    }
    .welcome-card h3 {
        font-family: 'Outfit', sans-serif;
        color: #1e3a8a;
        margin: 0 0 0.5rem 0;
        font-size: 1.25rem;
        font-weight: 700;
    }
    .welcome-card p {
        font-family: 'Inter', sans-serif;
        font-size: 0.9rem;
        line-height: 1.6;
        color: #475569;
        margin: 0;
    }
    
    /* Sidebar Brand styling */
    .sidebar-brand {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 1.5rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #e2e8f0;
    }
    .sidebar-logo {
        font-size: 1.5rem;
    }
    .sidebar-title {
        font-family: 'Outfit', sans-serif;
        font-size: 1.1rem;
        font-weight: 700;
        color: #1e3a8a; /* Navy */
        letter-spacing: 0.05em;
    }
    
    /* Buttons custom styling */
    div.stButton > button {
        border-radius: 8px !important;
        border: 1px solid #cbd5e1 !important;
        font-weight: 500 !important;
        font-family: 'Inter', sans-serif !important;
        transition: all 0.2s ease-in-out !important;
        background-color: #ffffff !important;
        color: #334155 !important;
    }
    
    div.stButton > button:hover {
        border-color: #1e3a8a !important;
        color: #1e3a8a !important;
        background-color: rgba(30, 58, 138, 0.04) !important;
        transform: translateY(-1px);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    
    /* Sidebar Customizations */
    [data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #e2e8f0 !important;
    }
    
    [data-testid="stSidebar"] button {
        text-align: left !important;
        justify-content: flex-start !important;
        background-color: transparent !important;
        border: 1px solid transparent !important;
        color: #475569 !important;
        font-weight: 500 !important;
        border-radius: 8px !important;
        padding: 0.5rem 0.75rem !important;
        transition: all 0.2s ease !important;
    }
    
    [data-testid="stSidebar"] button:hover {
        background-color: #f1f5f9 !important;
        color: #1e3a8a !important;
        border-color: #e2e8f0 !important;
    }
    
    /* Active session indicator styling inside sidebar */
    [class*="st-key-active-session"] button {
        background-color: #eff6ff !important;
        color: #1e3a8a !important;
        border-left: 4px solid #1e3a8a !important;
        font-weight: 600 !important;
        border-radius: 0 8px 8px 0 !important;
    }
    
    /* Sidebar delete button (primary kind) */
    [data-testid="stSidebar"] button[kind="primary"] {
        background-color: #fee2e2 !important;
        color: #991b1b !important;
        border: 1px solid #fca5a5 !important;
    }
    [data-testid="stSidebar"] button[kind="primary"]:hover {
        background-color: #fca5a5 !important;
        color: #991b1b !important;
    }
    
    
    /* File uploader custom borders */
    [data-testid="stFileUploader"] {
        border: 1px dashed #cbd5e1 !important;
        border-radius: 8px !important;
        background-color: #f8fafc !important;
        padding: 10px !important;
    }
    [data-testid="stFileUploader"] section {
        background-color: transparent !important;
        padding: 0 !important;
    }
    
    /* Chat message container custom styles */
    div[data-testid="stChatMessage"] {
        background-color: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 12px !important;
        padding: 1.25rem !important;
        margin-bottom: 1rem !important;
        box-shadow: 0 1px 3px 0 rgba(15, 23, 42, 0.03) !important;
        transition: border-color 0.2s ease !important;
    }
    div[data-testid="stChatMessage"]:hover {
        border-color: #cbd5e1 !important;
    }
    
    /* Style avatars */
    div[data-testid="stChatMessageAvatar"] {
        background-color: #f1f5f9 !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 8px !important;
    }
    
    /* Container styling for user messages */
    div[class*="st-key-chat_user"] div[data-testid="stChatMessage"] {
        background-color: #f8fafc !important;
        border-left: 4px solid #64748b !important;
    }
    
    /* Container styling for assistant messages */
    div[class*="st-key-chat_assistant"] div[data-testid="stChatMessage"] {
        background-color: #eff6ff !important;
        border-left: 4px solid #1e3a8a !important;
    }
    
    /* Suggestion cards styling */
    [class*="st-key-suggestions-container"] button {
        text-align: left !important;
        justify-content: flex-start !important;
        background-color: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 12px !important;
        padding: 1rem !important;
        height: auto !important;
        min-height: 80px !important;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.02) !important;
        transition: all 0.2s ease-in-out !important;
    }
    [class*="st-key-suggestions-container"] button:hover {
        border-color: #b45309 !important; /* Gold */
        background-color: #fffbeb !important; /* Warm gold tint */
        transform: translateY(-2px);
        box-shadow: 0 4px 6px -1px rgba(180, 83, 9, 0.05) !important;
    }
    
    /* Chat input box */
    [data-testid="stChatInput"] {
        background-color: transparent !important;
    }
    [data-testid="stChatInput"] textarea {
        border-radius: 8px !important;
        border: 1px solid #cbd5e1 !important;
        transition: all 0.2s ease !important;
        background-color: #ffffff !important;
    }
    [data-testid="stChatInput"] textarea:focus {
        border-color: #2563eb !important;
        box-shadow: 0 0 0 2px rgba(37, 99, 235, 0.1) !important;
    }
    
    /* Custom disclaimer footer */
    .legal-footer {
        text-align: center;
        padding: 20px;
        font-size: 0.8rem;
        color: #64748b;
        border-top: 1px solid #e2e8f0;
        margin-top: 40px;
        line-height: 1.5;
        font-family: 'Inter', sans-serif;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Database with fallback
use_db = False
db_error_msg = None
db = None

try:
    from database import SQLDatabase
    db = SQLDatabase()
    # Test connection
    conn = db.get_connection()
    conn.close()
    use_db = True
except Exception as e:
    use_db = False
    db_error_msg = str(e)

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

if "summary" not in st.session_state:
    st.session_state.summary = None

if "session_id" not in st.session_state:
    st.session_state.session_id = None
    
@st.cache_resource
def get_rag_engine():
    engine = FinancialRAG()
    engine.load_existing_db()
    return engine

rag_engine = get_rag_engine()



if "db_ready" not in st.session_state:
    st.session_state.db_ready = rag_engine.vectorstore is not None

# Sidebar
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <span class="sidebar-logo">⚖️</span>
        <span class="sidebar-title">TRỢ LÝ LUẬT AI</span>
    </div>
    """, unsafe_allow_html=True)

    st.write("---")

    # Nút đoạn chat mới
    if st.button("Đoạn Chat Mới", use_container_width=True, key="btn_new_chat"):
        st.session_state.session_id = None
        st.session_state.messages = []
        st.session_state.summary = None
        st.session_state.db_ready = False
        st.rerun()

    # Nút xoá lịch sử (chỉ hiện khi đang trong 1 phiên)
    if st.session_state.session_id is not None and use_db:
        if st.button(" Xoá Cuộc Hội Thoại Này", use_container_width=True, key="btn_delete_chat",
                     type="primary"):
            db.delete_session(st.session_state.session_id)
            st.session_state.session_id = None
            st.session_state.messages = []
            st.session_state.summary = None
            st.session_state.db_ready = False
            st.rerun()

    st.write("---")
    
    uploaded_file = st.file_uploader("📄 Nạp thêm Văn bản Luật (PDF)", type=["pdf"])
    
    if st.button("⚙️  Nạp vào Kho Luật Chung", use_container_width=True):
        if uploaded_file is not None:
            with st.status("Đang phân tích và nạp tài liệu luật...", expanded=True) as status:
                st.write("Đang lưu trữ tệp PDF vào thư mục luật...")
                # Tạo thư mục data/laws nếu chưa tồn tại
                laws_dir = os.path.join("data", "laws")
                os.makedirs(laws_dir, exist_ok=True)
                
                # Lưu tệp PDF thực tế vào thư mục data/laws/
                save_path = os.path.join(laws_dir, uploaded_file.name)
                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
                
                st.write("Đang phân tích điều luật và tạo vector index...")
                try:
                    if use_db and st.session_state.session_id is None:
                        st.session_state.session_id = db.create_session(uploaded_file.name, "")
                        
                    num_chunks = rag_engine.load_and_index_pdf(save_path, st.session_state.session_id)
                    if num_chunks == 0:
                        if os.path.exists(save_path):
                            os.remove(save_path)
                        raise ValueError("File PDF này không chứa ký tự văn bản có thể trích xuất (có thể là file PDF scan/dạng ảnh). Hệ thống RAG yêu cầu file PDF dạng văn bản số (digital text) có thể chọn/sao chép được.")
                        
                    st.session_state.db_ready = True
                    st.write("Đang khởi tạo bản tóm tắt văn bản luật...")
                    summary_text = rag_engine.summarize_pdf(save_path)
                    st.session_state.summary = summary_text
                    
                    if use_db:
                        db.update_session_pdf(st.session_state.session_id, uploaded_file.name, summary_text)
                            
                    status.update(label=f"Nạp luật thành công! Đã lưu file và lập chỉ mục {num_chunks} điều khoản.", state="complete", expanded=False)
                    st.rerun()
                except Exception as e:
                    status.update(label=f"Lỗi: {str(e)}", state="error", expanded=False)
        else:
            st.warning("Vui lòng tải lên một file PDF trước khi nạp.")

    st.write("---")

    # Danh sách các cuộc trò chuyện
    if use_db:
        sessions = db.get_all_sessions()
        if sessions:
            for s in sessions:
                label = s["title"]
                is_active = (st.session_state.session_id == s["id"])
                btn_label = f"{label}"
                
                if is_active:
                    with st.container(key="active-session"):
                        if st.button(f"📌 {btn_label}", use_container_width=True, key=f"sess_{s['id']}"):
                            pass
                else:
                    if st.button(f"💬 {btn_label}", use_container_width=True, key=f"sess_{s['id']}"):
                        st.session_state.session_id = s["id"]
                        st.session_state.messages = db.get_chat_history(s["id"])
                        summary_info = db.get_session_summary(s["id"])
                        if summary_info:
                            st.session_state.summary = summary_info["pdf_summary"]
                            st.session_state.db_ready = summary_info["pdf_summary"] is not None
                        else:
                            st.session_state.summary = None
                            st.session_state.db_ready = False
                        st.rerun()
        else:
            st.caption("Chưa có cuộc hội thoại nào.")
    else:
        st.warning("⚠️ SQL Server chưa kết nối")
        st.caption(f"Lỗi: {db_error_msg}")



# Main Chat Area
st.markdown("""
<div class="brand-header">
    <div class="brand-tag">HỆ THỐNG TRỢ LÝ SỐ PHÁP LUẬT VIỆT NAM</div>
    <h1 class="main-title">Trợ lý Tư vấn Luật pháp AI ⚖️</h1>
    <p class="subtitle">Tra cứu cơ sở dữ liệu pháp luật chính thống, trích dẫn chính xác điều khoản và nguồn tài liệu</p>
</div>
""", unsafe_allow_html=True)

# Kiểm tra RAG engine có dữ liệu không (dù session mới hay cũ)
has_vectorstore = rag_engine.vectorstore is not None

if not has_vectorstore:
    st.info("👋 Chào mừng bạn! Thư viện luật hiện tại đang trống. Vui lòng nạp thêm tài liệu luật bằng cách chọn file PDF ở thanh bên trái và bấm **'Nạp vào Kho Luật Chung'**, hoặc chạy script `ingest_laws.py` từ thư mục dự án.")
else:
    # Hiển thị Bản tóm tắt nhanh văn bản luật vừa nạp nếu có
    if st.session_state.get("summary"):
        with st.expander("📌 Bản tóm tắt nhanh văn bản pháp luật vừa nạp", expanded=False):
            st.markdown(st.session_state.summary)

    # Display chat messages from history on app rerun
    for idx, message in enumerate(st.session_state.messages):
        role = "user" if isinstance(message, HumanMessage) else "assistant"
        with st.container(key=f"chat_{role}_{idx}"):
            with st.chat_message(role):
                st.markdown(message.content)

    # Hiển thị lời chào và gợi ý câu hỏi nếu chưa có tin nhắn nào
    if len(st.session_state.messages) == 0:
        st.markdown("""
        <div class="welcome-card">
            <div class="welcome-badge">⚖️ KHỞI ĐẦU CHAT</div>
            <h3>Chào mừng đến với Trợ Lý Luật Pháp Việt Nam AI</h3>
            <p>
                Hệ thống hỗ trợ tra cứu các văn bản luật có trong thư mục luật dùng chung. 
                Khi trả lời, AI sẽ tự động tìm kiếm cơ sở dữ liệu luật pháp và trích dẫn rõ ràng tên văn bản, điều, khoản và số trang tương ứng.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("#### 💡 Gợi ý một số câu hỏi mẫu:")
        with st.container(key="suggestions-container"):
            cols = st.columns(2)
            suggestions = [
                ("Luật Đất Đai", "Điều kiện để hộ gia đình, cá nhân được cấp sổ đỏ (Giấy chứng nhận quyền sử dụng đất) mới nhất là gì?", "🏞️"),
                ("Luật Hôn Nhân và Gia Đình", "Khi nào ly hôn thì được chia tài sản chung?", "🏠"),
                ("Luật Hình sự", "Tội lừa đảo chiếm đoạt tài sản bị phạt bao nhiêu năm?", "💼"),
                ("Luật Doanh Nghiệp", "Quy trình thủ tục thành lập công ty TNHH có từ 2 thành viên trở lên gồm những bước nào?", "🏢")
            ]
            
            for i, (category, prompt_text, icon) in enumerate(suggestions):
                col = cols[i % 2]
                if col.button(f"{icon} **{category}**: {prompt_text[:60]}...", use_container_width=True, key=f"sugg_{i}"):
                    st.session_state.temp_prompt = prompt_text
                    st.rerun()

    # Nhận câu hỏi từ input chat hoặc gợi ý mẫu
    prompt_input = st.chat_input("Nhập câu hỏi pháp luật (Ví dụ: Điều kiện được cấp sổ đỏ là gì?)")
    
    prompt = None
    if "temp_prompt" in st.session_state and st.session_state.temp_prompt:
        prompt = st.session_state.temp_prompt
        del st.session_state.temp_prompt
    elif prompt_input:
        prompt = prompt_input

    # React to user input
    if prompt:
        # Display user message in chat message container
        with st.container(key="chat_user_input"):
            with st.chat_message("user"):
                st.markdown(prompt)
        
        # Lưu tin nhắn người dùng vào DB nếu dùng SQL Server
        if use_db:
            if st.session_state.session_id is None:
                st.session_state.session_id = db.create_session()
            db.save_message(st.session_state.session_id, "user", prompt)
            
        # Display assistant response in chat message container
        with st.container(key="chat_assistant_response"):
            with st.chat_message("assistant"):
                with st.spinner("Đang tra cứu cơ sở dữ liệu luật và lập luận câu trả lời..."):
                    try:
                        # Kiểm tra xem người dùng có muốn tóm tắt báo cáo không
                        is_summary_query = any(kw in prompt.lower() for kw in ["tóm tắt", "tom tat", "summary", "khái quát", "khai quat", "sơ lược", "so luoc"])
                        
                        if is_summary_query and st.session_state.get("summary"):
                            answer = st.session_state.summary
                            sources = []
                        else:
                            answer, sources = rag_engine.ask(prompt, st.session_state.messages, st.session_state.session_id)
                        
                        st.markdown(answer)
                        if sources:
                            st.caption(f"Nguồn: {', '.join(sources)}")
                            
                        # Add to history
                        st.session_state.messages.append(HumanMessage(content=prompt))
                        st.session_state.messages.append(AIMessage(content=answer))
                        
                        # Lưu phản hồi của bot vào DB nếu dùng SQL Server
                        if use_db:
                            db.save_message(st.session_state.session_id, "assistant", answer)
                    except Exception as e:
                        st.error(f"Đã xảy ra lỗi khi truy vấn: {str(e)}")

# Tuyên bố miễn trừ trách nhiệm pháp lý ở cuối trang
st.markdown("""
<div class="legal-footer">
    ⚠️ <b>Khuyến cáo pháp lý:</b> Thông tin cung cấp bởi Trợ lý AI chỉ mang tính chất tham khảo dựa trên các tài liệu luật hiện có trong hệ thống và không thay thế cho các ý kiến tư vấn pháp lý chuyên môn từ Luật sư hoặc Cơ quan tư pháp có thẩm quyền.
</div>
""", unsafe_allow_html=True)
