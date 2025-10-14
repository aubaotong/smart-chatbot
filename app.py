import streamlit as st
import requests
import urllib.request
import csv
from io import StringIO

# --- Cấu hình ---
# Lấy API key từ Streamlit Secrets một cách an toàn
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except (FileNotFoundError, KeyError):
    st.error("Lỗi: Không tìm thấy GEMINI_API_KEY. Vui lòng thêm vào mục Secrets trong Settings.")
    st.stop() # Dừng ứng dụng nếu không có key

GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1/models/gemini-pro:generateContent"

# --- Hàm tải dữ liệu Sheets ---
@st.cache_data(ttl=300) # Cache dữ liệu trong 5 phút
def load_advice_from_sheets(sheet_key):
    """Tải và phân tích dữ liệu CSV từ Google Sheets."""
    if not sheet_key:
        return "Vui lòng nhập Google Sheets Key."
    url = f"https://docs.google.com/spreadsheets/d/{sheet_key}/export?format=csv&gid=0"
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            csv_data = response.read().decode('utf-8')
        reader = csv.DictReader(StringIO(csv_data))
        advice_list = []
        for row in reader:
            question = row.get('Câu hỏi', '').strip()
            advice = row.get('Lời khuyên', '').strip()
            if question and advice: # Chỉ thêm nếu cả hai cột đều có dữ liệu
                advice_list.append(f"Câu hỏi: {question} | Lời khuyên: {advice}")
        
        if not advice_list:
            st.warning("Google Sheets không có dữ liệu hoặc sai định dạng cột ('Câu hỏi', 'Lời khuyên').")
            return None

        st.success(f"✔️ Đã tải thành công {len(advice_list)} lời khuyên từ Google Sheets.")
        return "\n".join(advice_list)
    except Exception as e:
        st.error(f"Lỗi khi tải dữ liệu từ Google Sheets: {e}")
        return None

# --- Hàm gọi Gemini API ---
def call_gemini_api(user_prompt, sheets_data, history):
    """Gửi yêu cầu đến Gemini API và trả về phản hồi."""
    headers = {"Content-Type": "application/json"}
    
    # Cấu trúc prompt rõ ràng cho AI
    system_prompt = f"""
Bạn là một trợ lý AI thông minh và thân thiện, chuyên đưa ra lời khuyên hữu ích bằng tiếng Việt.
Dựa vào dữ liệu tham khảo sau đây để trả lời câu hỏi của người dùng:
--- DỮ LIỆU THAM KHẢO ---
{sheets_data if sheets_data else "Không có dữ liệu tham khảo."}
--- KẾT THÚC DỮ LIỆU ---

QUY TẮC:
- Trả lời tự nhiên, ngắn gọn và đi thẳng vào vấn đề.
- Nếu câu hỏi của người dùng khớp với "Câu hỏi" trong dữ liệu, hãy dùng "Lời khuyên" tương ứng.
- Nếu không tìm thấy thông tin phù hợp, hãy đưa ra một lời khuyên chung dựa trên kiến thức của bạn.
- Nếu người dùng hỏi 'hướng dẫn', hãy giải thích ngắn gọn cách bạn hoạt động.
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
        response.raise_for_status() # Báo lỗi nếu status code là 4xx hoặc 5xx
        result = response.json()
        return result['candidates'][0]['content']['parts'][0]['text'].strip()
    except requests.exceptions.HTTPError as err:
        return f"Lỗi HTTP từ API: {err}. Phản hồi từ server: {response.text}"
    except Exception as e:
        return f"Đã xảy ra lỗi khi gọi Gemini API: {e}."

# --- Giao diện Streamlit ---
st.set_page_config(page_title="Chatbot Tư Vấn", page_icon="🤖")

st.title("🤖 Chatbot Tư Vấn Thông Minh")
st.caption("Cung cấp lời khuyên dựa trên dữ liệu từ Google Sheets")

# Sidebar cho cấu hình
with st.sidebar:
    st.header("⚙️ Cấu hình")
    sheet_key = st.text_input(
        "Google Sheets Key",
        value="1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms",
        help="Nhập key từ URL của Google Sheets."
    )
    if st.button("Tải lại dữ liệu", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    st.divider()
    if st.button("Xóa lịch sử trò chuyện", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Tải dữ liệu và xử lý lỗi
sheets_data = load_advice_from_sheets(sheet_key)

# Khởi tạo session state cho lịch sử chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Hiển thị lịch sử chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Xử lý input từ người dùng
if user_input := st.chat_input("Bạn cần lời khuyên về điều gì?"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    
    with st.chat_message("assistant"):
        with st.spinner("🧠 Đang suy nghĩ..."):
            if not sheets_data:
                st.error("Không thể tạo phản hồi vì không có dữ liệu từ Google Sheets.")
            else:
                history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages[-10:]])
                response = call_gemini_api(user_input, sheets_data, history)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
