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

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

if "summary" not in st.session_state:
    st.session_state.summary = None
    
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
                    num_chunks = rag_engine.load_and_index_pdf(tmp_path)
                    st.session_state.db_ready = True
                    st.write("Đang khởi tạo bản tóm tắt báo cáo tài chính...")
                    st.session_state.summary = rag_engine.summarize_pdf(tmp_path)
                    status.update(label=f"Xử lý thành công! Đã tạo {num_chunks} chunks và bản tóm tắt.", state="complete", expanded=False)
                except Exception as e:
                    status.update(label=f"Lỗi: {str(e)}", state="error", expanded=False)
                finally:
                    os.remove(tmp_path)
        else:
            st.warning("Vui lòng tải lên một file PDF trước khi xử lý.")

    if st.button("Làm mới cuộc trò chuyện"):
        st.session_state.messages = []
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
                        answer, sources = rag_engine.ask(prompt, st.session_state.messages)
                    
                    st.markdown(answer)
                    if sources:
                        st.caption(f"Nguồn: {', '.join(sources)}")
                        
                    # Add to history
                    st.session_state.messages.append(HumanMessage(content=prompt))
                    st.session_state.messages.append(AIMessage(content=answer))
                except Exception as e:
                    st.error(f"Đã xảy ra lỗi khi truy vấn: {str(e)}")
