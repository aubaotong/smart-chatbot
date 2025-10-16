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

GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1/models/gemini-2.5-flash:generateContent"

# ĐỔI TÊN HÀM và cách định dạng dữ liệu để AI dễ hiểu hơn
@st.cache_data(ttl=100)
def load_data_from_sheets(sheet_key):
    url = f"https://docs.google.com/spreadsheets/d/{sheet_key}/export?format=csv&gid=0"
    try:
        with urllib.request.urlopen(url) as response:
            csv_data = response.read().decode('utf-8')
        reader = csv.DictReader(StringIO(csv_data))
        data_list = []
        for row in reader:
            if 'Day' in row and 'poto' in row and 'Tình trạng lúa' in row and 'mức độ nhiễm' in row:
                # Định dạng lại dữ liệu cho rõ ràng, dễ đọc
                data_list.append(
                    f"- Thời gian: {row['Day']}, Tên file ảnh: {row['poto']}, Tên bệnh: {row['Tình trạng lúa']}, Mức độ nhiễm: {row['mức độ nhiễm']}"
                )
        st.success(f"Đã tải {len(data_list)} dòng dữ liệu từ Sheets.")
        return "\n".join(data_list)
    except Exception as e:
        st.error(f"Lỗi tải Sheets: {e}")
        return "Không có dữ liệu Sheets."

# --- THAY ĐỔI LỚN NHẤT: SỬA LẠI "BỘ NÃO" CỦA AI ---
# Hàm gọi Gemini API đã được cập nhật để nhận cả dữ liệu Sheets và câu hỏi của người dùng
def call_gemini_api(sheets_data, user_prompt, history=""):
    if not GEMINI_API_KEY or GEMINI_API_KEY == "YOUR_API_KEY_HERE":
        return "Vui lòng cấu hình API key trong Streamlit Secrets."
    
    headers = {
        "Content-Type": "application/json"
    }
    
    # Đây là "bộ não" của AI, ra lệnh cho nó cách hành xử
    system_prompt = f"""
Bạn là chatbot AI thông minh tên là CHTN, một chuyên gia về nông nghiệp lúa nước.
Nhiệm vụ của bạn là trả lời câu hỏi của người nông dân một cách chính xác dựa vào dữ liệu được cung cấp dưới đây.

---
**DỮ LIỆU TỪ GOOGLE SHEETS:**
{sheets_data}
---

Hãy phân tích dữ liệu trên và lịch sử hội thoại để trả lời câu hỏi của người dùng.
- Trả lời bằng tiếng Việt, với giọng văn tự nhiên, lễ phép và thân thiện.
- Giữ câu trả lời ngắn gọn, đi thẳng vào vấn đề.
- Nếu dữ liệu không có thông tin để trả lời, hãy nói rằng "Con không tìm thấy thông tin này trong dữ liệu ạ."

Lịch sử hội thoại:
{history}

Câu hỏi của người dùng: {user_prompt}
"""

    data = {
        "contents": [
            {
                "parts": [
                    {
                        "text": system_prompt
                    }
                ]
            }
        ]
    }
    try:
        response = requests.post(
            f"{GEMINI_API_URL}?key={GEMINI_API_KEY}",
            headers=headers,
            json=data,
            timeout=60 # Thêm timeout để tránh chờ quá lâu
        )
        response.raise_for_status()
        result = response.json()
        return result['candidates'][0]['content']['parts'][0]['text'].strip()
    except requests.exceptions.HTTPError as err:
        return f"Lỗi HTTP từ API: {err}"
    except Exception as e:
        return f"Lỗi khi gọi API: {e}. Hãy kiểm tra API Key và kết nối mạng."

# --- Giao diện ứng dụng Streamlit ---
st.title("🤖 Chatbot Nông Nghiệp CHTN")

# Sidebar cho cấu hình
with st.sidebar:
    st.header("Cấu hình")
    # Sử dụng sheet key demo của bạn
    sheet_key = st.text_input("Google Sheets Key", 
                            value="1JBoW6Wnv6satuZHlNXgJP0lzRXhSqgYRTrWeBJTKk60")
    if st.button("Tải lại dữ liệu Sheets"):
        st.cache_data.clear()
        st.success("Đã xóa cache, dữ liệu sẽ được tải lại.")
        st.rerun()

# Tải dữ liệu (sử dụng tên hàm mới)
sheets_data = load_data_from_sheets(sheet_key)

# Khởi tạo session state cho lịch sử chat
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Chào bác, con là AI CHTN. Bác cần con tra cứu thông tin gì về cánh đồng ạ?"}]

# Hiển thị lịch sử chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input chat
if user_input := st.chat_input("Bác hãy nhập câu hỏi vào đây..."):
    # Thêm tin nhắn của người dùng vào lịch sử
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Tạo và hiển thị phản hồi của AI
    with st.chat_message("assistant"):
        with st.spinner("Con đang tìm thông tin..."):
            history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages[-5:]])
            # SỬA LẠI CÁCH GỌI HÀM: Truyền cả sheets_data và user_input
            response = call_gemini_api(sheets_data, user_input, history)
            st.markdown(response)
    
    # Lưu phản hồi của AI vào lịch sử
    st.session_state.messages.append({"role": "assistant", "content": response})

# Nút xóa lịch sử chat
if st.sidebar.button("Xóa lịch sử chat"):
    st.session_state.messages = [{"role": "assistant", "content": "Chào bác, con là AI CHTN. Bác cần con tra cứu thông tin gì về cánh đồng ạ?"}]
    st.rerun()

