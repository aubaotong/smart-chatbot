import streamlit as st
import requests
import pandas as pd
from io import StringIO
import urllib.request
from datetime import datetime

# --- Cấu hình ---
st.set_page_config(
    page_title="Chatbot & Dashboard Nông nghiệp",
    page_icon="🌾",
    layout="wide"
)

try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except (FileNotFoundError, KeyError):
    st.error("Lỗi: Không tìm thấy GEMINI_API_KEY. Vui lòng thêm vào mục Secrets.")
    st.stop()

GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-latest:generateContent"

# --- Tải và chuẩn bị dữ liệu ---
@st.cache_data(ttl=300)
def load_data_from_sheets(sheet_key):
    """Tải và làm sạch dữ liệu từ Google Sheets."""
    if not sheet_key:
        return pd.DataFrame()
    url = f"https://docs.google.com/spreadsheets/d/{sheet_key}/export?format=csv&gid=0"
    try:
        df = pd.read_csv(url)
        df.columns = df.columns.str.strip()
        required_columns = ['Date', 'Tình trạng lúa', 'mức độ nhiễm']
        if not all(col in df.columns for col in required_columns):
            st.error(f"Lỗi: File Sheets phải chứa các cột: {', '.join(required_columns)}")
            return pd.DataFrame()
        
        # Tạo cột 'Date' chỉ chứa ngày để lọc và vẽ biểu đồ
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce').dt.date
        df.dropna(subset=['Date'], inplace=True)
        st.success(f"Đã tải và xử lý {len(df)} dòng dữ liệu từ Sheets.")
        return df
    except Exception as e:
        st.error(f"Lỗi tải dữ liệu từ Sheets: {e}")
        return pd.DataFrame()

# --- Phân tích dữ liệu cho Chatbot ---
@st.cache_data
def analyze_data_summary(df):
    """Tạo bản tóm tắt dạng văn bản cho chatbot."""
    if df is None or df.empty:
        return "Không có dữ liệu để phân tích."
    disease_counts = df['Tình trạng lúa'].value_counts().to_string()
    severity_counts = df['mức độ nhiễm'].value_counts().to_string()
    start_date = df['Date'].min().strftime('%Y-%m-%d')
    end_date = df['Date'].max().strftime('%Y-%m-%d')
    summary_text = f"""
**BÁO CÁO TỔNG QUAN TỰ ĐỘNG:**
- Dữ liệu được ghi nhận từ ngày **{start_date}** đến ngày **{end_date}**.
- **Thống kê các loại bệnh:**\n{disease_counts}
- **Thống kê mức độ nhiễm:**\n{severity_counts}
"""
    return summary_text

# --- Hàm gọi API Gemini ---
def call_gemini_api(summary_report, user_prompt, history=""):
    """Hàm gọi API Gemini với quy trình xử lý đa ý định."""
    system_prompt = f"""
Bạn là CHTN, trợ lý AI nông nghiệp. Dựa vào báo cáo và lịch sử chat, hãy trả lời người dùng theo các quy tắc sau:
- Nếu người dùng chào hỏi, hãy chào lại thân thiện.
- Nếu người dùng hỏi chung về tình hình, hãy tóm tắt báo cáo.
- Nếu người dùng hỏi cụ thể, hãy tìm thông tin trong báo cáo để trả lời.
---
**BÁO CÁO TỔNG QUAN:**
{summary_report}
---
Lịch sử hội thoại: {history}
Câu hỏi: "{user_prompt}"
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

# --- Giao diện ứng dụng ---
st.title("🌾 Chatbot & Dashboard Phân tích Nông nghiệp CHTN")

# --- LUỒNG XỬ LÝ CHÍNH ---
SHEET_KEY = "1JBoWWnv6satuZHlNXgJP0lzRXhSqgYRTrWeBJTKk60"
df_data = load_data_from_sheets(SHEET_KEY)
data_summary_for_chatbot = analyze_data_summary(df_data)

# --- TÍNH NĂNG MỚI: BẢNG ĐIỀU KHIỂN (DASHBOARD) TƯƠNG TÁC ---
if df_data is not None and not df_data.empty:
    with st.expander("📊 **Mở Bảng điều khiển Phân tích Dữ liệu**", expanded=False):
        
        # Lấy danh sách các loại bệnh (loại trừ các trạng thái khỏe mạnh)
        all_diseases = sorted([
            d for d in df_data['Tình trạng lúa'].unique() 
            if d not in ['healthy', 'khỏe mạnh', 'không xác định']
        ])
        
        # Tạo 2 cột để đặt các bộ lọc
        col1, col2 = st.columns(2)

        with col1:
            # Widget: Chọn nhiều loại bệnh
            selected_diseases = st.multiselect(
                "Chọn các loại bệnh để xem:",
                options=all_diseases,
                default=all_diseases[:3] # Mặc định chọn 3 bệnh đầu tiên
            )
        
        with col2:
            # Widget: Thanh trượt chọn khoảng thời gian
            min_date = df_data['Date'].min()
            max_date = df_data['Date'].max()
            
            date_range = st.slider(
                "Chọn khoảng thời gian:",
                min_value=min_date,
                max_value=max_date,
                value=(min_date, max_date),
                format="DD/MM/YYYY"
            )
            start_date, end_date = date_range

        # Lọc dữ liệu dựa trên lựa chọn của người dùng
        if selected_diseases:
            filtered_df = df_data[
                (df_data['Tình trạng lúa'].isin(selected_diseases)) &
                (df_data['Date'] >= start_date) &
                (df_data['Date'] <= end_date)
            ]

            if not filtered_df.empty:
                # --- Hiển thị Biểu đồ ---
                st.subheader("Số ca bệnh ghi nhận theo thời gian")
                chart_data = filtered_df.groupby(['Date', 'Tình trạng lúa']).size().reset_index(name='Số ca')
                chart_pivot = chart_data.pivot_table(index='Date', columns='Tình trạng lúa', values='Số ca', aggfunc='sum').fillna(0)
                st.line_chart(chart_pivot)

                # --- Hiển thị Thẻ số liệu (Metrics) ---
                latest_date_in_range = filtered_df['Date'].max()
                st.subheader(f"Thống kê chi tiết trong ngày gần nhất ({latest_date_in_range.strftime('%d/%m/%Y')})")
                
                latest_day_data = filtered_df[filtered_df['Date'] == latest_date_in_range]
                
                previous_dates = filtered_df[filtered_df['Date'] < latest_date_in_range]['Date'].unique()
                previous_day_data = pd.DataFrame()
                previous_day = None
                if len(previous_dates) > 0:
                    previous_day = max(previous_dates)
                    previous_day_data = filtered_df[filtered_df['Date'] == previous_day]

                # Tạo các cột để hiển thị thẻ
                metric_cols = st.columns(len(selected_diseases))
                for i, disease in enumerate(selected_diseases):
                    with metric_cols[i]:
                        current_count = latest_day_data[latest_day_data['Tình trạng lúa'] == disease].shape[0]
                        delta = None
                        if not previous_day_data.empty:
                            previous_count = previous_day_data[previous_day_data['Tình trạng lúa'] == disease].shape[0]
                            delta = current_count - previous_count
                        
                        st.metric(
                            label=disease.capitalize(),
                            value=current_count,
                            delta=f"{delta} ca" if delta is not None else None,
                            help=f"Tổng số ca '{disease}'. So sánh với ngày {previous_day.strftime('%d/%m/%Y') if previous_day else 'trước đó'}."
                        )
            else:
                st.warning("Không có dữ liệu cho các lựa chọn trong khoảng thời gian này.")
        else:
            st.info("Vui lòng chọn ít nhất một loại bệnh để xem phân tích.")

# --- Giao diện Chatbot ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Chào bác, con đã phân tích xong dữ liệu. Bác có thể hỏi con hoặc mở bảng điều khiển ở trên để xem chi tiết."}]

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
            response = call_gemini_api(data_summary_for_chatbot, user_input, history)
            st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})

# --- Các nút điều khiển trong Sidebar ---
with st.sidebar:
    st.header("Cấu hình")
    st.text_input("Google Sheets Key", value=SHEET_KEY, disabled=True)
    if st.button("Tải lại & Phân tích dữ liệu"):
        st.cache_data.clear()
        st.rerun()
    if st.button("Xóa lịch sử chat"):
        st.session_state.messages = [{"role": "assistant", "content": "Chào bác, con đã phân tích xong dữ liệu. Bác có thể hỏi con hoặc mở bảng điều khiển ở trên để xem chi tiết."}]
        st.rerun()

