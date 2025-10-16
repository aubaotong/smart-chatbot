import streamlit as st
import requests
import urllib.request
import csv
from io import StringIO
import pandas as pd # Sử dụng Pandas để phân tích dữ liệu hiệu quả hơn

# --- Cấu hình ---
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except (FileNotFoundError, KeyError):
    st.error("Lỗi: Không tìm thấy GEMINI_API_KEY. Vui lòng thêm vào mục Secrets trong Settings.")
    st.stop()

GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1/models/gemini-2.5-flash:generateContent"

# --- NÂNG CẤP HÀM TẢI DỮ LIỆU ---
# Giờ đây hàm sẽ trả về một DataFrame của Pandas để dễ dàng phân tích
@st.cache_data(ttl=120) # Tăng thời gian cache lên 2 phút
def load_and_prepare_data(sheet_key):
    """Tải dữ liệu từ Google Sheets và trả về dưới dạng DataFrame của Pandas."""
    if not sheet_key:
        return None
    url = f"https://docs.google.com/spreadsheets/d/{sheet_key}/export?format=csv&gid=0"
    try:
        # Sử dụng Pandas để đọc CSV trực tiếp từ URL, mạnh mẽ và ổn định hơn
        df = pd.read_csv(url)
        
        # Kiểm tra các cột cần thiết
        required_columns = ['Day', 'poto', 'Tình trạng lúa', 'mức độ nhiễm']
        if not all(col in df.columns for col in required_columns):
            st.error(f"Lỗi: File Sheets phải chứa đủ các cột: {','.join(required_columns)}")
            return None
            
        st.success(f"Đã tải và xử lý thành công {len(df)} dòng dữ liệu từ Sheets.")
        return df
    except Exception as e:
        st.error(f"Lỗi khi tải hoặc xử lý dữ liệu từ Sheets: {e}")
        return None

# --- NÂNG CẤP "BỘ NÃO" CỦA AI ---
def call_gemini_api(dataframe, user_prompt, history=""):
    """Hàm gọi API Gemini với nhiệm vụ phân tích dữ liệu tổng quan."""
    if dataframe is None or dataframe.empty:
        return "Con chưa có dữ liệu từ Google Sheets để phân tích ạ. Bác vui lòng kiểm tra lại Sheet Key."

    # 1. Tự động tạo một bản tóm tắt dữ liệu
    # Đếm số lần xuất hiện của mỗi bệnh
    disease_counts = dataframe['Tình trạng lúa'].value_counts().to_string()
    # Đếm số lần xuất hiện của mỗi mức độ nhiễm
    severity_counts = dataframe['mức độ nhiễm'].value_counts().to_string()
    # Ngày bắt đầu và kết thúc
    start_date = pd.to_datetime(dataframe['Day']).min().strftime('%Y-%m-%d')
    end_date = pd.to_datetime(dataframe['Day']).max().strftime('%Y-%m-%d')

    # 2. Tạo prompt hệ thống mới, thông minh hơn
    system_prompt = f"""
Bạn là CHTN, một trợ lý AI nông nghiệp chuyên sâu. Nhiệm vụ của bạn là phân tích dữ liệu về tình trạng lúa và cung cấp một báo cáo súc tích, dễ hiểu cho người nông dân.

---
**BÁO CÁO TỔNG QUAN TỰ ĐỘNG:**

Dữ liệu được ghi nhận từ ngày **{start_date}** đến ngày **{end_date}**.

**1. Thống kê các loại bệnh:**
{disease_counts}

**2. Thống kê mức độ nhiễm:**
{severity_counts}
---

**NHIỆM VỤ CỦA BẠN:**

Dựa vào **BÁO CÁO TỔNG QUAN** ở trên và lịch sử trò chuyện, hãy trả lời câu hỏi của người dùng.

* **Nếu người dùng hỏi chung chung** (ví dụ: "tình hình thế nào?", "phân tích cho tôi", "báo cáo đi"), hãy diễn giải lại **BÁO CÁO TỔNG QUAN** bằng lời văn tự nhiên, thân thiện. Đưa ra nhận định quan trọng nhất (ví dụ: "Bệnh đạo ôn đang xuất hiện nhiều nhất ạ") và đề xuất hành động nếu cần.
* **Nếu người dùng hỏi câu hỏi cụ thể**, hãy sử dụng báo cáo để trả lời chính xác câu hỏi đó.
* Luôn trả lời bằng tiếng Việt, giọng văn lễ phép, gần gũi.

Lịch sử hội thoại:
{history}

Câu hỏi của người dùng: "{user_prompt}"
"""

    headers = {"Content-Type": "application/json"}
    data = {"contents": [{"parts": [{"text": system_prompt}]}]}

    try:
        response = requests.post(
            f"{GEMINI_API_URL}?key={GEMINI_API_KEY}",
            headers=headers,
            json=data,
            timeout=90
        )
        response.raise_for_status()
        result = response.json()
        return result['candidates'][0]['content']['parts'][0]['text'].strip()
    except Exception as e:
        return f"Lỗi khi gọi API: {e}"

# --- Giao diện ứng dụng Streamlit ---
st.title("🤖 Chatbot Nông Nghiệp CHTN")

with st.sidebar:
    st.header("Cấu hình")
    sheet_key = st.text_input("Google Sheets Key", value="1JBoW6Wnv6satuZHlNXgJP0lzRXhSqgYRTrWeBJTKk60")
    if st.button("Tải lại dữ liệu Sheets"):
        st.cache_data.clear()
        st.rerun()

# Tải và chuẩn bị dữ liệu
df_data = load_and_prepare_data(sheet_key)

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Chào bác, con là AI CHTN. Bác hỏi con về tình hình cánh đồng hoặc yêu cầu con phân tích tổng quan nhé."}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_input := st.chat_input("Bác cần con giúp gì ạ?"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Con đang phân tích toàn bộ dữ liệu..."):
            history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages[-5:]])
            # Gọi hàm API đã được nâng cấp
            response = call_gemini_api(df_data, user_input, history)
            st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})

if st.sidebar.button("Xóa lịch sử chat"):
    st.session_state.messages = [{"role": "assistant", "content": "Chào bác, con là AI CHTN. Bác hỏi con về tình hình cánh đồng hoặc yêu cầu con phân tích tổng quan nhé."}]
    st.rerun()

