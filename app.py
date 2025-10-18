import streamlit as st
import requests
import pandas as pd
from io import StringIO, BytesIO
import urllib.request
import altair as alt
import re

# Thư viện cho tính năng giọng nói
from streamlit_mic_recorder import mic_recorder
import speech_recognition as sr
from pydub import AudioSegment
from gtts import gTTS

# --- Cấu hình ---
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except (FileNotFoundError, KeyError):
    st.error("Lỗi: Không tìm thấy GEMINI_API_KEY. Vui lòng thêm vào mục Secrets.")
    st.stop()

GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1/models/gemini-2.5-flash:generateContent"

# --- Hàm dọn dẹp văn bản ---
def clean_text_for_speech(text: str) -> str:
    text = re.sub(r'\*\*', '', text)
    text = re.sub(r'\*', '', text)
    text = re.sub(r'^\s*-\s*', '', text, flags=re.MULTILINE)
    text = text.replace(':', ',')
    return text

# --- Hàm chuyển văn bản thành audio ---
def text_to_speech(text: str, language: str = 'vi'):
    try:
        cleaned_text = clean_text_for_speech(text)
        tts = gTTS(text=cleaned_text, lang=language, slow=False)
        fp = BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        return fp
    except Exception as e:
        st.error(f"Lỗi khi tạo âm thanh: {e}")
        return None

# --- Tải và chuẩn bị dữ liệu (Không thay đổi) ---
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
        return df.sort_values(by='Date')
    except Exception as e:
        st.error(f"Lỗi tải dữ liệu từ Sheets: {e}")
        return None

# --- LOGIC TÍNH ĐIỂM (Không thay đổi) ---
@st.cache_data
def calculate_disease_scores(df):
    if df is None or df.empty:
        return pd.DataFrame(), []
    disease_names = [d for d in df['Tình trạng lúa'].unique() if d not in ['healthy', 'Khỏe mạnh', 'Không xác định']]
    scores = {name: 0 for name in disease_names}
    scores_over_time = []
    for index, row in df.iterrows():
        date = row['Date']
        tinh_trang = row['Tình trạng lúa']
        muc_do = row['mức độ nhiễm']
        if tinh_trang in ['healthy', 'Khỏe mạnh'] or muc_do == 'không nhiễm bệnh':
            for disease in scores:
                scores[disease] = max(0, scores[disease] - 1)
        elif tinh_trang in disease_names:
            if muc_do == 'Mới nhiễm':
                scores[tinh_trang] += 3
            elif muc_do == 'Nhiễm vừa':
                scores[tinh_trang] += 4
            elif muc_do == 'Nhiễm nặng':
                scores[tinh_trang] += 9
        for disease in scores:
             if scores[disease] > 10: scores[disease] = 10
             if scores[disease] < 0: scores[disease] = 0
        current_scores = {'Record_ID': index, 'Date': date, **scores}
        scores_over_time.append(current_scores)
    scores_df = pd.DataFrame(scores_over_time)
    warnings = []
    if not scores_df.empty:
        last_scores = scores_df.iloc[-1]
        for disease, score in last_scores.items():
            if disease not in ['Record_ID', 'Date'] and score > 5:
                warnings.append(f"Bệnh '{disease}' đã vượt ngưỡng cảnh báo với {score} điểm!")
    return scores_df, warnings

# --- HÀM PHÂN TÍCH & GỌI API (Không thay đổi) ---
def analyze_scores_for_chatbot(scores_df):
    if scores_df is None or scores_df.empty:
        return "Hiện không có dữ liệu điểm nguy hiểm để phân tích."
    latest_scores = scores_df.iloc[-1]
    latest_date = latest_scores['Date'].strftime('%d-%m-%Y')
    disease_cols = [col for col in scores_df.columns if col not in ['Date', 'Record_ID']]
    summary_text = f"Báo cáo phân tích diễn biến điểm nguy hiểm (tính đến ngày {latest_date}):\n\n"
    summary_text += "**1. Điểm số hiện tại:**\n"
    for disease in disease_cols:
        score = latest_scores[disease]
        summary_text += f"- {disease}: {score} điểm.\n"
    summary_text += "\n**2. Phân tích xu hướng gần đây:**\n"
    if len(scores_df) > 1:
        for disease in disease_cols:
            recent_scores = scores_df[disease].tail(5)
            if len(recent_scores) > 2:
                trend_start = recent_scores.iloc[0]
                trend_end = recent_scores.iloc[-1]
                if trend_end > trend_start:
                    trend = "có xu hướng **TĂNG**."
                elif trend_end < trend_start:
                    trend = "có xu hướng **GIẢM**."
                else:
                    trend = "đang **ỔN ĐỊNH**."
            else:
                trend = "chưa đủ dữ liệu để phân tích xu hướng."
            summary_text += f"- {disease}: {trend}\n"
    else:
        summary_text += "- Chưa đủ dữ liệu để phân tích xu hướng.\n"
    return summary_text

def call_gemini_api(summary_report, user_prompt, history=""):
    system_prompt = f"""
Bạn là CHTN, một trợ lý AI nông nghiệp chuyên phân tích biểu đồ và dữ liệu diễn biến. Dựa vào "Báo cáo phân tích diễn biến điểm nguy hiểm" dưới đây, hãy trả lời người dùng như một chuyên gia.
QUY TẮC:
- Trả lời ngắn gọn, tập trung vào thông tin quan trọng nhất.
- Khi được hỏi về tình hình chung, hãy tóm tắt báo cáo, tập trung vào các bệnh có điểm số cao và xu hướng TĂNG. Đưa ra nhận định tổng quan.
- Khi được hỏi cụ thể về một bệnh, hãy dựa vào điểm số và xu hướng của bệnh đó để trả lời chi tiết.
- Chủ động đưa ra lời khuyên dựa trên phân tích. Ví dụ: "Điểm bệnh đạo ôn đang có xu hướng tăng nhanh, bác nên ưu tiên thăm đồng và kiểm tra các dấu hiệu của bệnh này."
- Nếu người dùng chào hỏi, hãy chào lại thân thiện và tóm tắt ngắn gọn tình hình nổi bật nhất (ví dụ: bệnh nào đang có điểm cao nhất).
- Luôn trả lời dựa trên báo cáo được cung cấp.
---
**BÁO CÁO PHÂN TÍCH DIỄN BIẾN ĐIỂM NGUY HIỂM (Dữ liệu chính để phân tích)**
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
st.title("WED HỆ THỐNG GIÁM SÁT & CHUẨN ĐOÁN BỆNH Ở LÚA CHTN")

# --- LUỒNG XỬ LÝ CHÍNH ---
df_data = load_data_from_sheets("1JBoW6Wnv6satuZHlNXgJP0lzRXhSqgYRTrWeBJTKk60")
scores_df, warnings = calculate_disease_scores(df_data)
data_for_chatbot = analyze_scores_for_chatbot(scores_df)

# --- HIỂN THỊ BIỂU ĐỒ NGUY HIỂM (Không thay đổi) ---
if scores_df is not None and not scores_df.empty:
    with st.expander("Xem biểu đồ điểm nguy hiểm của bệnh", expanded=True):
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
            scores_melted = filtered_df.melt(id_vars=['Record_ID', 'Date'], var_name='Tên bệnh', value_name='Điểm nguy hiểm')
            rule = alt.Chart(pd.DataFrame({'y': [5]})).mark_rule(color='red', strokeDash=[5,5]).encode(y='y')
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
    st.session_state.messages = []
    initial_msg = "Chào bác, con là AI CHTN. Con sẽ theo dõi và cảnh báo nếu có dịch bệnh nguy hiểm."
    st.session_state.messages.append({"role": "assistant", "content": initial_msg})
    if warnings:
        warning_text = "⚠️ **CẢNH BÁO KHẨN!**\n\n" + "\n".join(f"- {w}" for w in warnings)
        st.session_state.messages.append({"role": "assistant", "content": warning_text})

# Hiển thị lịch sử chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and "audio" in message:
            st.audio(message["audio"], format='audio/mp3', start_time=0)

# --- THAY ĐỔI: Giao diện trò chuyện đơn giản hơn ---
st.write("---")
st.write("**Trò chuyện bằng giọng nói:**")

audio_data = mic_recorder(start_prompt=" Bấm để nói", stop_prompt=" Đang xử lý...", key='mic_recorder')

if audio_data:
    # --- THAY ĐỔI: Thêm kiểm tra chống lặp ---
    # Chỉ xử lý nếu audio mới được ghi lại (dựa vào ID duy nhất)
    if st.session_state.get('last_audio_id') != audio_data['id']:
        st.session_state.last_audio_id = audio_data['id'] # Lưu ID của audio vừa xử lý
        try:
            audio_bytes = BytesIO(audio_data['bytes'])
            audio_segment = AudioSegment.from_file(audio_bytes)
            r = sr.Recognizer()
            with BytesIO() as wav_file:
                audio_segment.export(wav_file, format="wav")
                wav_file.seek(0)
                with sr.AudioFile(wav_file) as source:
                    audio = r.record(source)
            transcribed_text = r.recognize_google(audio, language="vi-VN")
            
            st.session_state.messages.append({"role": "user", "content": transcribed_text})
            
            history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages[-5:]])
            response = call_gemini_api(data_for_chatbot, transcribed_text, history)

            audio_file = text_to_speech(response)
            
            if audio_file:
                st.session_state.messages.append({"role": "assistant", "content": response, "audio": audio_file})
            else:
                st.session_state.messages.append({"role": "assistant", "content": response})

            st.rerun()

        except sr.UnknownValueError:
            st.toast("Con không nghe rõ, bác thử lại nhé!", icon="🤔")
        except Exception as e:
            st.error(f"Đã có lỗi xảy ra khi xử lý giọng nói: {e}")

# Ô nhập văn bản luôn hiển thị
if user_input := st.chat_input("Hoặc nhập tin nhắn tại đây..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Con đang nghĩ câu trả lời..."):
            history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages[-5:]])
            response = call_gemini_api(data_for_chatbot, user_input, history)
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
        st.session_state.messages = []
        if 'last_audio_id' in st.session_state:
            del st.session_state['last_audio_id']
        st.rerun()
