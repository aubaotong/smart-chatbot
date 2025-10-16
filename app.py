import streamlit as st
import requests
import pandas as pd
from io import StringIO
import urllib.request
from datetime import datetime

# --- Cấu hình ---
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except (FileNotFoundError, KeyError):
    st.error("Lỗi: Không tìm thấy GEMINI_API_KEY. Vui lòng thêm vào mục Secrets.")
    st.stop()

GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1/models/gemini-2.5-flash:generateContent"

# --- BƯỚC 1: TẢI VÀ CHUẨN BỊ DỮ LIỆU ---
@st.cache_data(ttl=300)
def load_data_from_sheets(sheet_key):
    """Tải dữ liệu từ Google Sheets và trả về dưới dạng DataFrame."""
    if not sheet_key:
        return None
    url = f"https://docs.google.com/spreadsheets/d/{sheet_key}/export?format=csv&gid=0"
    try:
        df = pd.read_csv(url)
        df.columns = df.columns.str.strip()
        required_columns = ['Day', 'poto', 'Tình trạng lúa', 'mức độ nhiễm']
        if not all(col in df.columns for col in required_columns):
            st.error(f"Lỗi: File Sheets phải chứa các cột: {', '.join(required_columns)}")
            return None
        # Xử lý cột 'Day' để chuẩn bị cho việc vẽ đồ thị
        df['Day'] = pd.to_datetime(df['Day'], errors='coerce').dt.date
        df.dropna(subset=['Day'], inplace=True) # Bỏ các dòng có ngày không hợp lệ
        st.success(f"Đã tải và xử lý {len(df)} dòng dữ liệu từ Sheets.")
        return df
    except Exception as e:
        st.error(f"Lỗi tải dữ liệu từ Sheets: {e}")
        return None

# --- BƯỚC 2: PHÂN TÍCH DỮ LIỆU CHO CHATBOT ---
@st.cache_data
def analyze_data_summary(df):
    """Phân tích DataFrame và tạo ra một bản tóm tắt dạng văn bản cho chatbot."""
    if df is None or df.empty:
        return "Không có dữ liệu để phân tích."
    disease_counts = df['Tình trạng lúa'].value_counts().to_string()
    severity_counts = df['mức độ nhiễm'].value_counts().to_string()
    start_date = df['Day'].min().strftime('%Y-%m-%d')
    end_date = df['Day'].max().strftime('%Y-%m-%d')
    summary_text = f"""
**BÁO CÁO TỔNG QUAN TỰ ĐỘNG:**
- Dữ liệu được ghi nhận từ ngày **{start_date}** đến ngày **{end_date}**.
- **Thống kê các loại bệnh:**\n{disease_counts}
- **Thống kê mức độ nhiễm:**\n{severity_counts}
"""
    return summary_text

# --- BƯỚC 3: HÀM GỌI API GEMINI ---
def call_gemini_api(summary_report, user_prompt, history=""):
    """Hàm gọi API Gemini với quy trình xử lý đa ý định thông minh."""
    system_prompt = f"""
Bạn là CHTN, một trợ lý AI nông nghiệp thân thiện và thông minh. Dựa vào báo cáo và lịch sử chat, hãy trả lời người dùng theo các quy tắc sau:
- Nếu người dùng chào hỏi, hãy chào lại thân thiện.
- Nếu người dùng hỏi chung về tình hình, hãy tóm tắt báo cáo.
- Nếu người dùng hỏi cụ thể, hãy tìm thông tin trong báo cáo để trả lời.
- Nếu người dùng trò chuyện, hãy trả lời tự nhiên theo vai trò.
---
**BÁO CÁO TỔNG QUAN (Chỉ sử dụng khi cần thiết)**
{summary_report}
---
Lịch sử hội thoại gần đây: {history}
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
st.title("🤖 Chatbot & Phân Tích Nông Nghiệp CHTN")

# --- LUỒNG XỬ LÝ CHÍNH ---
df_data = load_data_from_sheets("1JBoW6Wnv6satuZHlNXgJP0lzRXhSqgYRTrWeBJTKk60")
data_summary = analyze_data_summary(df_data)

# --- TÍNH NĂNG MỚI: BIỂU ĐỒ TÌNH TRẠNG LÚA ---
if df_data is not None and not df_data.empty:
    with st.expander("📊 Xem biểu đồ phân tích tình trạng lúa theo thời gian", expanded=False):
        st.subheader("Phân tích số ca bệnh theo ngày")

        # Lấy danh sách các loại bệnh duy nhất để người dùng chọn
        all_diseases = df_data['Tình trạng lúa'].unique()
        
        # Tạo widget để người dùng chọn loại bệnh muốn xem
        selected_diseases = st.multiselect(
            "Chọn tình trạng lúa để hiển thị:",
            options=all_diseases,
            default=[disease for disease in all_diseases if disease not in ['healthy', 'khỏe mạnh', 'không xác định']] # Mặc định chọn các bệnh
        )

        # Lấy ngày bắt đầu và kết thúc từ dữ liệu
        min_date = df_data['Date'].min()
        max_date = df_data['Date'].max()

        # Tạo widget slider để người dùng chọn khoảng thời gian
        date_range = st.slider(
            "Chọn khoảng thời gian:",
            min_value=min_date,
            max_value=max_date,
            value=(min_date, max_date), # Mặc định chọn toàn bộ
            format="DD/MM/YYYY"
        )
        
        start_date, end_date = date_range

        if selected_diseases:
            # Lọc dữ liệu dựa trên lựa chọn của người dùng
            filtered_df = df_data[
                (df_data['Tình trạng lúa'].isin(selected_diseases)) &
                (df_data['Date'] >= start_date) &
                (df_data['Date'] <= end_date)
            ]

            # Đếm số lượng ca bệnh mỗi ngày
            chart_data = filtered_df.groupby(['Date', 'Tình trạng lúa']).size().reset_index(name='Số ca')
            
            # Chuyển đổi dữ liệu để vẽ biểu đồ (mỗi bệnh một cột)
            chart_data_pivot = chart_data.pivot(index='Date', columns='Tình trạng lúa', values='Số ca').fillna(0)

            st.write(f"Biểu đồ số ca bệnh từ ngày {start_date.strftime('%d/%m/%Y')} đến {end_date.strftime('%d/%m/%Y')}")
            st.line_chart(chart_data_pivot)
        else:
            st.info("Vui lòng chọn ít nhất một tình trạng lúa để xem biểu đồ.")

# --- Giao diện Chatbot ---
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
            response = call_gemini_api(data_summary, user_input, history)
            st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})

# --- Các nút điều khiển trong Sidebar ---
with st.sidebar:
    st.header("Cấu hình")
    st.text_input("Google Sheets Key", value="1JBoW6Wnv6satuZHlNXgJP0lzRXhSqgYRTrWeBJTKk60", disabled=True)
    if st.button("Tải lại & Phân tích dữ liệu"):
        st.cache_data.clear()
        st.rerun()
    if st.button("Xóa lịch sử chat"):
        st.session_state.messages = [{"role": "assistant", "content": "Chào bác, con đã phân tích xong dữ liệu. Bác cần con tư vấn gì ạ?"}]
        st.rerun()

