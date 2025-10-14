import streamlit as st
import requests

# --- Cấu hình ---
# Lấy API key từ Streamlit Secrets một cách an toàn
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except (FileNotFoundError, KeyError):
    st.error("Lỗi: Không tìm thấy GEMINI_API_KEY. Vui lòng thêm vào mục Secrets trong Settings.")
    st.stop() # Dừng ứng dụng nếu không có key

GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-pro:generateContent"

# --- Hàm gọi Gemini API (ĐÃ ĐƠN GIẢN HÓA) ---
def call_gemini_api(user_prompt, history):
    """Gửi yêu cầu đến Gemini API và trả về phản hồi."""
    headers = {"Content-Type": "application/json"}
    
    # SỬA LẠI PROMPT: Hướng dẫn AI trở thành một chatbot trò chuyện thông thường
    system_prompt = """
Bạn là một trợ lý AI hữu ích, thông minh và thân thiện. 
Nhiệm vụ của bạn là trò chuyện với người dùng một cách tự nhiên và trả lời các câu hỏi của họ về nhiều chủ đề khác nhau. 
Hãy trả lời bằng tiếng Việt, giữ giọng điệu gần gũi và tích cực.
"""
    
    full_prompt = f"{system_prompt}\n--- LỊCH SỬ HỘI THOẠI ---\n{history}\n--- CÂU HỎI MỚI ---\nNgười dùng: {user_prompt}"

    data = {"contents": [{"parts": [{"text": full_prompt}]}]}
    
    try:
        response = requests.post(
            f"{GEMINI_API_URL}?key={GEMINI_API_KEY}",
            headers=headers,
            json=data,
            timeout=60
        )
        response.raise_for_status()
        result = response.json()
        # Thêm kiểm tra key 'candidates' để tránh lỗi
        if 'candidates' in result and result['candidates']:
            return result['candidates'][0]['content']['parts'][0]['text'].strip()
        else:
            return "Xin lỗi, tôi không nhận được phản hồi hợp lệ. Vui lòng thử lại."
    except requests.exceptions.HTTPError as err:
        return f"Lỗi HTTP từ API: {err}. Phản hồi: {response.text}"
    except Exception as e:
        return f"Đã xảy ra lỗi khi gọi Gemini API: {e}."

# --- Giao diện Streamlit (ĐÃ ĐƠN GIẢN HÓA) ---
st.set_page_config(page_title="Trò chuyện cùng AI", page_icon="💬")

st.title("💬 Chatbot AI Đa Năng")
st.caption("Trò chuyện về mọi chủ đề cùng Gemini")

# Sidebar
with st.sidebar:
    st.header("⚙️ Tùy chỉnh")
    if st.button("Xóa lịch sử trò chuyện", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Khởi tạo session state cho lịch sử chat
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Chào bạn! Bạn muốn trò chuyện về chủ đề gì?"}]

# Hiển thị lịch sử chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Xử lý input từ người dùng
if user_input := st.chat_input("Nhập câu hỏi của bạn..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    
    with st.chat_message("assistant"):
        with st.spinner("🧠 AI đang suy nghĩ..."):
            # Chỉ lấy 10 tin nhắn gần nhất làm ngữ cảnh
            history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages[-10:]])
            response = call_gemini_api(user_input, history)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})














