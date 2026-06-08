import pyodbc
import config
from langchain_core.messages import HumanMessage, AIMessage

class SQLDatabase:
    def __init__(self):
        self.server = config.SQL_SERVER
        self.database = config.SQL_DATABASE
        self.trusted = config.SQL_TRUSTED_CONNECTION
        self.username = config.SQL_USERNAME
        self.password = config.SQL_PASSWORD

    def get_connection(self):
        """Tạo kết nối tới SQL Server dựa trên cấu hình."""
        if self.trusted.lower() == "yes":
            conn_str = (
                f"DRIVER={{ODBC Driver 17 for SQL Server}};"
                f"SERVER={self.server};"
                f"DATABASE={self.database};"
                f"Trusted_Connection=yes;"
            )
        else:
            conn_str = (
                f"DRIVER={{ODBC Driver 17 for SQL Server}};"
                f"SERVER={self.server};"
                f"DATABASE={self.database};"
                f"UID={self.username};"
                f"PWD={self.password};"
            )
        return pyodbc.connect(conn_str)

    def create_session(self, pdf_name=None, pdf_summary=None):
        """Tạo một phiên hội thoại mới và trả về session_id."""
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute(
                "INSERT INTO Chat_Sessions (pdf_name, pdf_summary) OUTPUT INSERTED.id VALUES (?, ?)",
                (pdf_name, pdf_summary)
            )
            session_id = cursor.fetchone()[0]
            conn.commit()
            return session_id
        finally:
            conn.close()

    def update_session_pdf(self, session_id, pdf_name, pdf_summary):
        """Cập nhật thông tin file PDF và bản tóm tắt cho phiên chat hiện tại."""
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute(
                "UPDATE Chat_Sessions SET pdf_name = ?, pdf_summary = ? WHERE id = ?",
                (pdf_name, pdf_summary, session_id)
            )
            conn.commit()
        finally:
            conn.close()

    def save_message(self, session_id, sender, message):
        """Lưu tin nhắn vào cơ sở dữ liệu."""
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute(
                "INSERT INTO Chat_Messages (session_id, sender, message) VALUES (?, ?, ?)",
                (session_id, sender, message)
            )
            conn.commit()
        finally:
            conn.close()

    def get_all_sessions(self):
        """Lấy tất cả các phiên chat, sắp xếp theo thời gian tạo mới nhất."""
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("""
                SELECT s.id, s.pdf_name, s.created_at,
                       (SELECT TOP 1 message FROM Chat_Messages 
                        WHERE session_id = s.id AND sender = 'user' 
                        ORDER BY id ASC) as first_message
                FROM Chat_Sessions s ORDER BY s.id DESC
            """)
            rows = cursor.fetchall()
            result = []
            for r in rows:
                first_msg = r[3]
                # Dùng câu hỏi đầu tiên làm tiêu đề; nếu chưa có thì dùng tên file PDF
                if first_msg and len(first_msg.strip()) > 0:
                    title = first_msg[:45] + "..." if len(first_msg) > 45 else first_msg
                else:
                    title = r[1] or "Cuộc hội thoại mới"
                result.append({"id": r[0], "pdf_name": r[1] or "", "title": title, "created_at": r[2]})
            return result
        except Exception:
            return []
        finally:
            conn.close()

    def get_session_summary(self, session_id):
        """Lấy thông tin PDF và bản tóm tắt của một phiên chat."""
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT pdf_name, pdf_summary FROM Chat_Sessions WHERE id = ?", (session_id,))
            row = cursor.fetchone()
            if row:
                return {"pdf_name": row[0], "pdf_summary": row[1]}
            return None
        finally:
            conn.close()

    def get_chat_history(self, session_id):
        """Tải toàn bộ lịch sử tin nhắn của một phiên chat và chuyển đổi sang đối tượng LangChain."""
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute(
                "SELECT sender, message FROM Chat_Messages WHERE session_id = ? ORDER BY id ASC",
                (session_id,)
            )
            rows = cursor.fetchall()
            messages = []
            for row in rows:
                sender, message = row[0], row[1]
                if sender == "user":
                    messages.append(HumanMessage(content=message))
                else:
                    messages.append(AIMessage(content=message))
            return messages
        finally:
            conn.close()

    def delete_session(self, session_id):
        """Xóa một phiên hội thoại và toàn bộ tin nhắn liên quan (do cascade delete)."""
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("DELETE FROM Chat_Sessions WHERE id = ?", (session_id,))
            conn.commit()
        finally:
            conn.close()
