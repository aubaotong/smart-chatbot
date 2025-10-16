import streamlit as st
import requests
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
@st.cache_data(ttl=100)  # tự cập nhật sau mỗi 100 s
def load_advice_from_sheets(sheet_key):
    url = f"https://docs.google.com/spreadsheets/d/{sheet_key}/export?format=csv&gid=0"
    try:
        with urllib.request.urlopen(url) as response:
            csv_data = response.read().decode('utf-8')
        reader = csv.DictReader(StringIO(csv_data))
        advice_list = []
        for row in reader:
            if 'Day' in row and 'poto' in row and 'Tình trạng lúa' in row and 'mức độ nhiễm' in row:
                advice_list.append(f"phút giây giờ ngày tháng chủa dữ liệu tình trạng bệnh và ảnh đc chụp: {row['Day']} | vị chí lưu file ảnh mà mô hình AI nhận biết bệnh đã xử lí : {row['poto']} | tình trạng bệnh mà lúa trên cánh đồng của người nông dân đang mắc phải (tên bệnh): {row['Tình trạng lúa']} | mức độ nhiễm được tôi chia làm 3 cấp mới nhiễm là tình trạng của bệnh vừa mới xuất hiện cần tập chung trị bệnh từ giai đoạn này tiếp theo là giai vừa ở giai đoạn này cần điều trị gắp giai cuối là nặng cần nói chia buồn cùng nông dân : {row['mức độ nhiễm']} ")
        st.success(f"Đã tải {len(advice_list)} lời khuyên từ Sheets.")
        return "\n".join(advice_list)
    except Exception as e:
        st.error(f"Lỗi tải Sheets: {e}")
        return "Không có dữ liệu Sheets."

# Hàm gọi Gemini API
def call_gemini_api(prompt, history=""):
    if GEMINI_API_KEY == "GEMINI_API_KEY":
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
Bạn là chatbot AI thông minh tên là CHTN, chuyên về nông nghiệp lúa nước có nghiệm vụ tư vấn cho người nông dân về tình trạng cánh đồng dựa trên dữ liệu được cấp bạn cần phân tích các số liệu để đưa ra lời khuyên hợp lí và ngắn gọn.
Dữ liệu từ Sheets: {prompt}
Trả lời tự nhiên, lễ phép, thân thiện bằng tiếng Việt.
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
st.title(" Chatbot AI CHTN 👻 ")

# Sidebar cho config
with st.sidebar:
    st.header("Cấu hình")
    sheet_key = st.text_input("Google Sheets Key (Enter cho demo)", 
                            value="1JBoW6Wnv6satuZHlNXgJP0lzRXhSqgYRTrWeBJTKk60")
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
if prompt := st.chat_input("Chào bác con là AI CHTN con sẽ trả lời về tình trạng của cánh đồng của Bác"):
    # Thêm user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Tạo response
    with st.chat_message("assistant"):
        history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages[-100:]])  # Giới hạn lịch sử
        with st.spinner("Con đang suy nghĩ..."):
            response = call_gemini_api(sheets_data, history)
        st.markdown(response)
    
    # Lưu response
    st.session_state.messages.append({"role": "assistant", "content": response})

# Nút clear chat
if st.button("Xóa lịch sử chat"):
    st.session_state.messages = []
    st.rerun()












