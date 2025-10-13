import requests
import urllib.request
import csv
from io import StringIO
import json

# Thay bằng API key của bạn từ https://x.ai/api
GROK_API_KEY = "yxai-r4VDlb4Cj21mkjI99TFqoQPZWAx0lclmtIonR9x23ycQjTCx8evMsHm9LDb2kPL0AkM7gNqrHI1NH8LF"  # Ví dụ: "gsk_abc123..."
GROK_API_URL = "https://api.x.ai/v1/chat/completions"  # Endpoint Grok API (tương tự OpenAI format)

# Hàm tải dữ liệu từ Google Sheets (như trước)
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
        print(f"Đã tải {len(advice_list)} lời khuyên từ Sheets.")
        return "\n".join(advice_list)
    except Exception as e:
        print(f"Lỗi tải Sheets: {e}")
        return "Không có dữ liệu Sheets."

# Hàm gọi Grok API
def call_grok_api(prompt, history=""):
    headers = {
        "Authorization": f"Bearer {GROK_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "grok-beta",  # Hoặc model mới nhất từ docs
        "messages": [
            {"role": "system", "content": f"""
            Bạn là một chatbot AI thông minh, chuyên đưa lời khuyên hữu ích. 
            Dữ liệu kiến thức từ Sheets (cập nhật liên tục): {prompt}
            Hãy trả lời tự nhiên, thân thiện bằng tiếng Việt. Nếu không khớp dữ liệu, đưa lời khuyên chung thông minh.
            Giữ câu trả lời ngắn gọn, hữu ích. Hỗ trợ lệnh đặc biệt: 'hướng dẫn' để giải thích cách dùng.
            """},
            {"role": "user", "content": f"{history}\nNgười dùng: {prompt}"}
        ],
        "max_tokens": 300,  # Giới hạn độ dài response
        "temperature": 0.7  # Độ sáng tạo (0.0-1.0)
    }
    try:
        response = requests.post(GROK_API_URL, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        return result['choices'][0]['message']['content'].strip()
    except Exception as e:
        return f"Lỗi API: {e}. Kiểm tra key và quota tại https://x.ai/api."

# Chatbot chính
def run_smart_chatbot():
    print("=== SMART CHATBOT AI (Grok-powered) ===")
    sheet_key = input("Nhập key Google Sheets (Enter cho demo): ").strip()
    if not sheet_key:
        sheet_key = "1JBoW6Wnv6satuZHlNXgJP0lzRXhSqgYRTrWeBJTKk60"
    
    # Tải dữ liệu Sheets và nhúng vào prompt system
    sheets_data = load_advice_from_sheets(sheet_key)
    
    print("\nChat bắt đầu! Type 'quit' để thoát. Tôi sẽ nhớ ngữ cảnh chat.\n")
    print("--- Khung Chat ---")
    
    history = ""  # Lưu lịch sử để AI nhớ (có thể mở rộng)
    
    while True:
        user_input = input("\nBạn: ").strip()
        if user_input.lower() == 'quit':
            print("Bot: Tạm biệt! Hẹn gặp lại.")
            break
        
        # Xây dựng prompt với lịch sử
        full_prompt = f"{history}\n{user_input}"
        response = call_grok_api(sheets_data, full_prompt)  # Nhúng Sheets vào system prompt
        
        print(f"Bot: {response}")
        
        # Cập nhật lịch sử (giới hạn để tránh token dài)
        history += f"\nBạn: {user_input}\nBot: {response}\n"
        if len(history) > 1000:  # Cắt ngắn nếu quá dài
            history = history[-1000:]

if __name__ == "__main__":
    run_smart_chatbot()
