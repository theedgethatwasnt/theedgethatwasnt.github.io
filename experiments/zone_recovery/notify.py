"""Telegram notification helper for zone recovery experiments."""
import os
import requests


BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
CHAT_ID   = os.environ.get("TELEGRAM_CHAT_ID", "")


def send_telegram(message: str) -> bool:
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        resp = requests.post(url, json={"chat_id": CHAT_ID, "text": message}, timeout=10)
        return resp.ok
    except Exception as e:
        print(f"[Telegram] Failed: {e}")
        return False
