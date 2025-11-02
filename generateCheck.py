import requests

telegram_token = '7854981054:AAG7MxxKd2BM6gtFmkn3w1pVsBv9vvRHjg8'
telegram_chat_id = '6336989243'

def send_test_message():
    message = "✅ Bu test xabari"
    url = f"https://api.telegram.org/bot{telegram_token}/sendMessage"
    payload = {
        'chat_id': telegram_chat_id,
        'text': message
    }
    response = requests.post(url, data=payload)
    print("Telegram javobi:", response.text)

send_test_message()
