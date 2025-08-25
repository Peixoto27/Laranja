# -*- coding: utf-8 -*-
"""
notifier_telegram.py
Envio de mensagens para Telegram com suporte a HTML e inline keyboard.
Usa TELEGRAM_BOT_TOKEN e TELEGRAM_CHAT_ID do ambiente.
"""

import os
import json
import requests

BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
CHAT_ID   = os.environ.get("TELEGRAM_CHAT_ID")

API_URL = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage" if BOT_TOKEN else None

def _ensure_env():
    if not BOT_TOKEN or not CHAT_ID:
        raise RuntimeError("TELEGRAM_BOT_TOKEN ou TELEGRAM_CHAT_ID não definidos no ambiente.")

def _post(data: dict, timeout: int = 12):
    resp = requests.post(API_URL, json=data, timeout=timeout)
    try:
        js = resp.json()
    except Exception:
        js = {"ok": False, "text": resp.text}
    ok = (resp.status_code == 200) and bool(js.get("ok"))
    print(f"[TG] tentativa 1, status={resp.status_code}, resp={json.dumps(js)[:200]}")
    return ok, js

def notify_telegram(text: str, parse_mode: str = "HTML", reply_markup: dict | None = None):
    """Envio simples direto por texto (compat)."""
    _ensure_env()
    data = {"chat_id": CHAT_ID, "text": text, "parse_mode": parse_mode}
    if reply_markup:
        data["reply_markup"] = reply_markup
    ok, _ = _post(data)
    return ok

def send_signal_notification(payload: dict):
    """
    API preferida pelo main.py.
    Aceita: {"text": "...", "parse_mode": "HTML", "reply_markup": {...}}
    """
    _ensure_env()
    text = payload.get("text") or payload.get("message") or ""
    parse_mode = payload.get("parse_mode", "HTML")
    reply_markup = payload.get("reply_markup")
    data = {"chat_id": CHAT_ID, "text": text, "parse_mode": parse_mode}
    if reply_markup:
        data["reply_markup"] = reply_markup
    ok, _ = _post(data)
    return ok
