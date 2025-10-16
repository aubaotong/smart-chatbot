import streamlit as st
import requests
import pandas as pd
from io import StringIO
import urllib.request

# --- Cấu hình ---
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except (FileNotFoundError, KeyError):
    st.error("Lỗi: Không tìm thấy GEMINI_API_KEY. Vui lòng thêm vào mục Secrets.")
    st.stop()

# SỬ DỤNG MODEL MỚI NHẤT VÀ NHANH NHẤT CỦA GEMINI
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1/models/gemini-2.5-flash:generateContent"

# --- BƯỚC 1: TẢI VÀ CHUẨN BỊ DỮ LIỆU (Giữ nguyên) ---
@st.cache_data(ttl=300) # Tăng thời gian cache lên 5 phút
def load_data_from_sheets(sheet_key):
    """Tải dữ liệu từ Google Sheets và trả về dưới dạng DataFrame."""
    if not sheet_key:
        return None
    url = f"https://docs.google.com/spreadsheets/d/{sheet_key}/export?format=csv&gid=0"
    try:
        df = pd.read_csv(url)
        df.columns = df.columns.str.strip() # Làm sạch tên cột
        required_columns = ['Day', 'poto', 'Tình trạng lúa', 'mức độ nhiễm']
        if not all(col in df.columns for col in required_columns):
            st.error(f"Lỗi: File Sheets phải chứa các cột: {', '.join(required_columns)}")
            st.warning(f"Các cột tìm thấy: {list(df.columns)}")
            return None
        st.success(f"Đã tải và xử lý {len(df)} dòng dữ liệu từ Sheets.")
        return df
    except Exception as e:
        st.error(f"Lỗi tải dữ liệu từ Sheets: {e}")
        return None

# --- BƯỚC 2: TẠO HÀM PHÂN TÍCH DỮ LIỆU RIÊNG (Tối ưu hóa lớn nhất) ---
# Hàm này cũng được cache, nó chỉ chạy lại khi dữ liệu nguồn thay đổi.
@st.cache_data
def analyze_data_summary(df):
    """Phân tích DataFrame và tạo ra một bản tóm tắt dạng văn bản."""
    if df is None or df.empty:
        return "Không có dữ liệu để phân tích."

    disease_counts = df['Tình trạng lúa'].value_counts().to_string()
    severity_counts = df['mức độ nhiễm'].value_counts().to_string()
    
    valid_dates = pd.to_datetime(df['Day'], errors='coerce')
    start_date = valid_dates.min().strftime('%Y-%m-%d')
    end_date = valid_dates.max().strftime('%Y-%m-%d')

    # Trả về một chuỗi văn bản đã được định dạng sẵn
    summary_text = f"""
**BÁO CÁO TỔNG QUAN TỰ ĐỘNG:**
- Dữ liệu được ghi nhận từ ngày **{start_date}** đến ngày **{end_date}**.
- **Thống kê các loại bệnh:**\n{disease_counts}
- **Thống kê mức độ nhiễm:**\n{severity_counts}
"""
    return summary_text

# --- BƯỚC 3: HÀM GỌI API SIÊU NHẸ ---
# Hàm này giờ chỉ nhận văn bản, không cần xử lý dữ liệu nặng nữa
# THAY THẾ HÀM CŨ BẰNG HÀM NÀY ĐỂ AI TRỞ NÊN THÔNG MINH HƠN
def call_gemini_api(summary_report, user_prompt, history=""):
    """Hàm gọi API Gemini với quy trình xử lý đa ý định thông minh."""
    
    # "Bộ não" mới của AI với quy trình xử lý logic
    system_prompt = f"""
Bạn là CHTN, một trợ lý AI nông nghiệp thân thiện và thông minh.

**QUY TRÌNH XỬ LÝ YÊU CẦU (Rất quan trọng):**

1.  **Phân tích ý định của người dùng:** Đọc kỹ "Câu hỏi của người dùng" và xác định xem họ muốn làm gì trong các trường hợp sau.
2.  **Hành động theo ý định:**
    * **Nếu người dùng chỉ chào hỏi** (ví dụ: "hello", "chào em", "xin chào"): Hãy chào lại một cách thân thiện và hỏi xem bác cần giúp gì. **TUYỆT ĐỐI KHÔNG trình bày báo cáo dữ liệu.**
    * **Nếu người dùng hỏi một cách chung chung về tình hình** (ví dụ: "tình hình thế nào?", "phân tích dữ liệu", "báo cáo tình hình cho tôi"): Hãy diễn giải lại **BÁO CÁO TỔNG QUAN** dưới đây bằng lời văn tự nhiên, đưa ra nhận định quan trọng nhất (ví dụ: "Con thấy bệnh đạo ôn có vẻ đang xuất hiện nhiều nhất ạ").
    * **Nếu người dùng hỏi một câu hỏi cụ thể** (ví dụ: "có bao nhiêu ca bệnh đốm trung?", "ngày cuối cùng ghi nhận là khi nào?"): Hãy dựa vào **BÁO CÁO TỔNG QUAN** để tìm và trả lời chính xác câu hỏi đó.
    * **Nếu người dùng hỏi ngoài lề/trò chuyện** (ví dụ: "bạn là ai?", "bạn làm được gì?"): Hãy trả lời một cách tự nhiên, đúng vai trò là một trợ lý AI nông nghiệp mà không cần dùng đến báo cáo.

---
**BÁO CÁO TỔNG QUAN (Chỉ sử dụng khi cần thiết)**
{summary_report}
---

Lịch sử hội thoại gần đây:
{history}

Câu hỏi của người dùng: "{user_prompt}"
"""
    headers = {"Content-Type": "application/json"}
    data = {"contents": [{"parts": [{"text": system_prompt}]}]}
    try:
        response = requests.post(f"{GEMINI_API_URL}?key={GEMINI_API_KEY}", headers=headers, json=data, timeout=60)
        response.raise_for_status()
        result = response.json()
        return result['candidates'][0]['content']['parts'][0]['text'].strip()
    except Exception as e:
        return f"Lỗi gọi API: {e}"

# --- Giao diện ứng dụng Streamlit ---
st.title("🤖 Chatbot Nông Nghiệp CHTN (Tốc độ cao)")

with st.sidebar:
    st.header("Cấu hình")
    sheet_key = st.text_input("Google Sheets Key", value="1JBoW6Wnv6satuZHlNXgJP0lzRXhSqgYRTrWeBJTKk60")
    if st.button("Tải lại & Phân tích dữ liệu"):
        st.cache_data.clear() # Xóa toàn bộ cache
        st.rerun()

# --- LUỒNG XỬ LÝ ĐÃ TỐI ƯU ---
# 1. Tải dữ liệu (chạy khi cần)
df_data = load_data_from_sheets(sheet_key)
# 2. Phân tích dữ liệu (chỉ chạy 1 lần sau khi tải)
data_summary = analyze_data_summary(df_data)

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Chào bác, con đã phân tích xong dữ liệu. Bác cần con tư vấn gì ạ?"}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_input := st.chat_input("Bác cần con giúp gì ạ?"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Con đang nghĩ câu trả lời..."):
            history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages[-5:]])
            # 3. Gọi API siêu nhẹ với bản tóm tắt đã có sẵn
            response = call_gemini_api(data_summary, user_input, history)
            st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})

if st.sidebar.button("Xóa lịch sử chat"):
    st.session_state.messages = [{"role": "assistant", "content": "Chào bác, con đã phân tích xong dữ liệu. Bác cần con tư vấn gì ạ?"}]
    st.rerun()




