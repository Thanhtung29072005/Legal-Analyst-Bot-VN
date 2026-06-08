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
    page_title="Financial Analyst Bot",
    page_icon="📈",
    layout="wide"
)

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
    st.title("Phân tích Báo cáo Tài chính 📈")
    st.write("Tải lên bản cáo bạch hoặc báo cáo thường niên (PDF) để chuyên gia AI phân tích.")
    
    # Quản lý phiên hội thoại từ SQL Server
    if use_db:
        st.success("🔌 Đã kết nối SQL Server")
        sessions = db.get_all_sessions()
        options = ["Cuộc trò chuyện mới ➕"] + [f"ID {s['id']}: {s['pdf_name']}" for s in sessions]
        
        default_index = 0
        if st.session_state.session_id is not None:
            for i, s in enumerate(sessions):
                if s["id"] == st.session_state.session_id:
                    default_index = i + 1
                    break
                    
        selected_option = st.selectbox(
            "Chọn phiên hội thoại",
            options=options,
            index=default_index,
            key="session_select"
        )
        
        current_selected_id = None
        if selected_option != "Cuộc trò chuyện mới ➕":
            current_selected_id = int(selected_option.split(":")[0].replace("ID ", ""))
            
        if current_selected_id != st.session_state.session_id:
            st.session_state.session_id = current_selected_id
            if current_selected_id is None:
                st.session_state.messages = []
                st.session_state.summary = None
                st.session_state.db_ready = False
            else:
                st.session_state.messages = db.get_chat_history(current_selected_id)
                summary_info = db.get_session_summary(current_selected_id)
                if summary_info:
                    st.session_state.summary = summary_info["pdf_summary"]
                    st.session_state.db_ready = summary_info["pdf_summary"] is not None
                else:
                    st.session_state.summary = None
                    st.session_state.db_ready = False
            st.rerun()
            
        if st.session_state.session_id is not None:
            if st.button("🗑️ Xóa cuộc hội thoại này"):
                db.delete_session(st.session_state.session_id)
                st.session_state.session_id = None
                st.session_state.messages = []
                st.session_state.summary = None
                st.session_state.db_ready = False
                st.rerun()
    else:
        st.warning("⚠️ Lịch sử chat (SQL Server) chưa kết nối")
        st.caption(f"Lỗi: {db_error_msg}")
        
    st.write("---")
    uploaded_file = st.file_uploader("Upload PDF file", type=["pdf"])
    
    if st.button("Xử lý Tài liệu"):
        if uploaded_file is not None:
            with st.status("Đang xử lý tài liệu...", expanded=True) as status:
                st.write("Đang đọc file PDF...")
                # Save uploaded file to temp
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name
                
                st.write("Đang trích xuất và nhúng Vector (Vectorizing)...")
                try:
                    # Tạo session_id trước nếu chưa có để tag vector vào Qdrant
                    if use_db and st.session_state.session_id is None:
                        st.session_state.session_id = db.create_session(uploaded_file.name, "")
                        
                    num_chunks = rag_engine.load_and_index_pdf(tmp_path, st.session_state.session_id)
                    st.session_state.db_ready = True
                    st.write("Đang khởi tạo bản tóm tắt báo cáo tài chính...")
                    summary_text = rag_engine.summarize_pdf(tmp_path)
                    st.session_state.summary = summary_text
                    
                    # Cập nhật thông tin tóm tắt vào SQL Server
                    if use_db:
                        db.update_session_pdf(st.session_state.session_id, uploaded_file.name, summary_text)
                            
                    status.update(label=f"Xử lý thành công! Đã tạo {num_chunks} chunks và bản tóm tắt.", state="complete", expanded=False)
                    st.rerun()
                except Exception as e:
                    status.update(label=f"Lỗi: {str(e)}", state="error", expanded=False)
                finally:
                    os.remove(tmp_path)
        else:
            st.warning("Vui lòng tải lên một file PDF trước khi xử lý.")

    if st.button("Làm mới cuộc trò chuyện"):
        st.session_state.session_id = None
        st.session_state.messages = []
        st.session_state.summary = None
        st.session_state.db_ready = False
        st.rerun()

# Main Chat Area
st.title("Trợ lý Phân tích AI 🤖")

if not st.session_state.db_ready:
    st.info("👋 Chào mừng bạn! Vui lòng tải lên một Báo cáo Tài chính (PDF) ở thanh bên trái để bắt đầu.")
else:
    # Hiển thị Bản tóm tắt nhanh báo cáo tài chính nếu có
    if st.session_state.get("summary"):
        with st.expander("📌 Bản tóm tắt nhanh báo cáo tài chính", expanded=False):
            st.markdown(st.session_state.summary)

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        role = "user" if isinstance(message, HumanMessage) else "assistant"
        with st.chat_message(role):
            st.markdown(message.content)

    # React to user input
    if prompt := st.chat_input("Nhập câu hỏi về báo cáo tài chính (Ví dụ: Doanh thu năm nay là bao nhiêu?)"):
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Lưu tin nhắn người dùng vào DB nếu dùng SQL Server
        if use_db:
            if st.session_state.session_id is None:
                st.session_state.session_id = db.create_session()
            db.save_message(st.session_state.session_id, "user", prompt)
            
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            with st.spinner("Đang phân tích số liệu..."):
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
