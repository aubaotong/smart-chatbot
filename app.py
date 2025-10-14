import streamlit as st
import requests
import urllib.request
import csv
from io import StringIO

# Config (dùng key trực tiếp để test local)
GEMINI_API_KEY = "AIzaSyAiRgyISAwWihXhLVuCVbZnOm9RzXKO6J4"  # Thay bằng key mới nếu cần
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent"

# Hàm tải dữ liệu Sheets (chạy một lần khi app load)
@st.cache_data(ttl=300)  # Cache 5 phút, tự cập nhật
def load_advice_from_sheets(sheet_key):
    url = f"https://docs.google.com/spreadsheets/d/{sheet_key}/export?format=csv&gid=0"
    try:
        with urllib.request.urlopen(url) as response:
            csv_data = response.read().decode('utf-8')
        reader = csv.DictReader(StringIO(csv_data))
        advice_list = []
        for row in reader:
            if 'Câu hỏi' in row and 'Lời khuyên' in row:
                advice_list.append(f"Câu hỏi: {row['Câu hỏi']} | Lời khuyên: {row['Lời khuyên']}")
        st.success(f"Đã tải {len(advice_list)} lời khuyên từ Sheets.")
        return "\n".join(advice_list)
    except Exception as e:
        st.error(f"Lỗi tải Sheets: {e}")
        return "Không có dữ liệu Sheets."

# Hàm gọi Gemini API
def call_gemini_api(prompt, history=""):
    if GEMINI_API_KEY == "your_key_here":
        return "Vui lòng cấu hình API key trong Streamlit Secrets."
    headers = {
        "Content-Type": "application/json"
    }
    data = {
        "contents": [
            {
                "parts": [
                    {
                        "text": f"""
Bạn là chatbot AI thông minh, chuyên lời khuyên. 
Dữ liệu từ Sheets: {prompt}
Trả lời tự nhiên, thân thiện bằng tiếng Việt. Nếu không khớp, đưa lời khuyên chung.
Giữ ngắn gọn. Hỗ trợ 'hướng dẫn' để giải thích.
Lịch sử hội thoại: {history}
Người dùng: {prompt}
"""
                    }
                ]
            }
        ]
    }
    try:
        response = requests.post(
            f"{GEMINI_API_URL}?key={GEMINI_API_KEY}",
            headers=headers,
            json=data
        )
        response.raise_for_status()
        result = response.json()
        return result['candidates'][0]['content']['parts'][0]['text'].strip()
    except Exception as e:
        return f"Lỗi API: {e}. Kiểm tra key tại https://cloud.google.com/generative-ai."

# Streamlit App chính
st.title("🤖 Smart Chatbot AI (Gemini-powered)")

# Sidebar cho config
with st.sidebar:
    st.header("Cấu hình")
    sheet_key = st.text_input("Google Sheets Key (Enter cho demo)", 
                              value="1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms")
    if st.button("Tải lại dữ liệu Sheets"):
        st.cache_data.clear()
        st.rerun()

# Tải dữ liệu
sheets_data = load_advice_from_sheets(sheet_key)

# Khởi tạo session state cho lịch sử chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Hiển thị lịch sử chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input chat
if prompt := st.chat_input("Nhập câu hỏi của bạn..."):
    # Thêm user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Tạo response
    with st.chat_message("assistant"):
        history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages[-5:]])  # Giới hạn lịch sử
        with st.spinner("Đang suy nghĩ..."):
            response = call_gemini_api(sheets_data, history)
        st.markdown(response)
    
    # Lưu response
    st.session_state.messages.append({"role": "assistant", "content": response})

# Nút clear chat
if st.button("Xóa lịch sử chat"):
    st.session_state.messages = []
    st.rerun()

# (Giữ nguyên các hàm load_advice_from_sheets, call_gemini_api, và giao diện Streamlit như trước)


