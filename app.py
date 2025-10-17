import streamlit as st
import requests
import pandas as pd
from io import StringIO
import urllib.request
import altair as alt # Sử dụng thư viện Altair để vẽ biểu đồ nâng cao

# --- Cấu hình ---
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except (FileNotFoundError, KeyError):
    st.error("Lỗi: Không tìm thấy GEMINI_API_KEY. Vui lòng thêm vào mục Secrets.")
    st.stop()

GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1/models/gemini-2.5-flash:generateContent"

# --- Tải và chuẩn bị dữ liệu ---
@st.cache_data(ttl=300)
def load_data_from_sheets(sheet_key):
    if not sheet_key:
        return None
    url = f"https://docs.google.com/spreadsheets/d/{sheet_key}/export?format=csv&gid=0"
    try:
        df = pd.read_csv(url)
        df.columns = df.columns.str.strip()
        required_columns = ['Date', 'Tình trạng lúa', 'mức độ nhiễm']
        if not all(col in df.columns for col in required_columns):
            st.error(f"Lỗi: File Sheets phải chứa các cột: {', '.join(required_columns)}")
            return None
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce').dt.date
        df.dropna(subset=['Date'], inplace=True)
        st.success(f"Đã tải và xử lý {len(df)} dòng dữ liệu từ Sheets.")
        return df.sort_values(by='Date') # Sắp xếp dữ liệu theo ngày
    except Exception as e:
        st.error(f"Lỗi tải dữ liệu từ Sheets: {e}")
        return None

# --- CẬP NHẬT LOGIC: TÍNH ĐIỂM THEO TỪNG HÀNG DỮ LIỆU ---
@st.cache_data
def calculate_disease_scores(df):
    if df is None or df.empty:
        return pd.DataFrame(), []

    # Lọc bỏ các trạng thái không phải là bệnh
    disease_names = [d for d in df['Tình trạng lúa'].unique() if d not in ['healthy', 'Khỏe mạnh', 'Không xác định']]
    
    # Khởi tạo điểm số
    scores = {name: 0 for name in disease_names}
    scores_over_time = []
    
    # Duyệt qua từng hàng trong DataFrame đã được sắp xếp theo ngày
    for index, row in df.iterrows():
        date = row['Date']
        tinh_trang = row['Tình trạng lúa']
        muc_do = row['mức độ nhiễm']
        
        # Logic 1: Giảm điểm nếu có báo cáo "không nhiễm bệnh"
        if muc_do == 'không nhiễm bệnh':
            for disease in scores:
                scores[disease] = max(0, scores[disease] - 1)
        
        # Logic 2: Tăng điểm dựa trên mức độ nhiễm của từng bệnh cụ thể trong hàng đó
        elif tinh_trang in disease_names:
            if muc_do == 'Mới nhiễm':
                scores[tinh_trang] += 3
            elif muc_do == 'Nhiễm vừa':
                scores[tinh_trang] += 4
            elif muc_do == 'Nhiễm nặng':
                scores[tinh_trang] += 9
        
        # Ghi lại điểm số tại thời điểm của hàng dữ liệu này
        # Thêm một cột 'Record_ID' để đảm bảo mỗi điểm dữ liệu là duy nhất trên biểu đồ
        current_scores = {'Record_ID': index, 'Date': date, **scores}
        scores_over_time.append(current_scores)

    scores_df = pd.DataFrame(scores_over_time)
    
    # Kiểm tra cảnh báo dựa trên điểm số cuối cùng
    warnings = []
    if not scores_df.empty:
        last_scores = scores_df.iloc[-1]
        for disease, score in last_scores.items():
            if disease not in ['Record_ID', 'Date'] and score > 5:
                warnings.append(f"Bệnh '{disease}' đã vượt ngưỡng cảnh báo với {score} điểm!")

    return scores_df, warnings

# --- Hàm gọi API Gemini (Không đổi) ---
def call_gemini_api(summary_report, user_prompt, history=""):
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

@st.cache_data
def analyze_data_summary(df):
    if df is None or df.empty: return "Không có dữ liệu để phân tích."
    disease_counts = df['Tình trạng lúa'].value_counts().to_string()
    severity_counts = df['mức độ nhiễm'].value_counts().to_string()
    start_date = pd.to_datetime(df['Date'], errors='coerce').min().strftime('%Y-%m-%d')
    end_date = pd.to_datetime(df['Date'], errors='coerce').max().strftime('%Y-%m-%d')
    summary_text = f"Dữ liệu từ {start_date} đến {end_date}.\nBệnh:\n{disease_counts}\nMức độ:\n{severity_counts}"
    return summary_text
    
# --- Giao diện ứng dụng Streamlit ---
st.title("🚨 Hệ thống Cảnh báo & Chatbot Nông nghiệp CHTN")

# --- LUỒNG XỬ LÝ CHÍNH ---
df_data = load_data_from_sheets("1JBoW6Wnv6satuZHlNXgJP0lzRXhSqgYRTrWeBJTKk60")
scores_df, warnings = calculate_disease_scores(df_data)
data_summary_for_chatbot = analyze_data_summary(df_data)

# --- HIỂN THỊ BIỂU ĐỒ NGUY HIỂM ---
if scores_df is not None and not scores_df.empty:
    with st.expander("📈 Xem biểu đồ điểm nguy hiểm của bệnh", expanded=True):
        
        disease_cols = [col for col in scores_df.columns if col not in ['Date', 'Record_ID']]
        scores_df[disease_cols] = scores_df[disease_cols].clip(upper=10)

        min_date = scores_df['Date'].min()
        max_date = scores_df['Date'].max()
        
        start_date, end_date = st.slider(
            "Chọn khoảng ngày bạn muốn xem:",
            min_value=min_date,
            max_value=max_date,
            value=(min_date, max_date),
            format="DD/MM/YYYY"
        )

        filtered_df = scores_df[(scores_df['Date'] >= start_date) & (scores_df['Date'] <= end_date)]

        if not filtered_df.empty:
            # Chuyển đổi dữ liệu, bỏ cột Record_ID khỏi melt
            scores_melted = filtered_df.melt(id_vars=['Record_ID', 'Date'], var_name='Tên bệnh', value_name='Điểm nguy hiểm')

            rule = alt.Chart(pd.DataFrame({'y': [5]})).mark_rule(color='red', strokeDash=[5,5]).encode(y='y')

            # --- CẬP NHẬT LOGIC: Trục X giờ sẽ là Record_ID để thể hiện từng điểm dữ liệu ---
            line_chart = alt.Chart(scores_melted).mark_line().encode(
                x=alt.X('Record_ID', title='Dòng dữ liệu (theo thời gian)'),
                y=alt.Y('Điểm nguy hiểm', scale=alt.Scale(domain=[0, 10])),
                color='Tên bệnh',
                tooltip=['Date', 'Tên bệnh', 'Điểm nguy hiểm']
            ).interactive()

            final_chart = (line_chart + rule).properties(
                title='Diễn biến điểm nguy hiểm của các loại bệnh theo từng cập nhật'
            )

            st.altair_chart(final_chart, use_container_width=True)
        else:
            st.warning("Không có dữ liệu để hiển thị trong khoảng ngày đã chọn.")

# --- Giao diện Chatbot ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Chào bác, con là AI CHTN. Con sẽ theo dõi và cảnh báo nếu có dịch bệnh nguy hiểm."}]
    if warnings:
        warning_text = "⚠️ **CẢNH BÁO KHẨN!**\n\n" + "\n".join(f"- {w}" for w in warnings)
        st.session_state.messages.append({"role": "assistant", "content": warning_text})


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
    st.text_input("Google Sheets Key", value="1JBoW6Wnv6satuZHlNXgJP0lzRXhSqgYRTrWeBJTKk60", disabled=True)
    if st.button("Tải lại & Phân tích dữ liệu"):
        st.cache_data.clear()
        st.rerun()
    if st.button("Xóa lịch sử chat"):
        st.session_state.messages = [{"role": "assistant", "content": "Chào bác, con là AI CHTN. Con sẽ theo dõi và cảnh báo nếu có dịch bệnh nguy hiểm."}]
        st.rerun()
