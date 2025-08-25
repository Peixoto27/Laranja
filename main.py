# -*- coding: utf-8 -*-
"""
Crypto Trading AI — API Flask + Alertas Telegram

Recursos:
- Perfis por ALERT_MODE (conservador/balanceado/agressivo)
- TP/SL por ATR (fallback %) + validação de coerência
- Anti-duplicação de alertas (lado+entry+tp+sl) + cooldown + 1 alerta/símbolo/ciclo
- Mensagens Telegram em HTML (pro | card | compact) com emojis e botões
- Endpoints: /, /api/predictions, /api/status, /api/force-update, /api/test-ai, /health
- Dashboard dark
"""
import os, sys, time, json, threading, requests
from math import isnan
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple, Set
from flask import Flask, jsonify, render_template_string, request
from flask_cors import CORS

# ----------------- Config por ENV -----------------
ALERT_MODE = os.environ.get("ALERT_MODE", "balanceado").lower()

# defaults (balanceado)
ALERT_CONF_MIN = 0.65
ALERT_COOLDOWN_MIN = 30
UPDATE_INTERVAL_SEC = 300
USE_ATR_LEVELS = True
ATR_PERIOD = 14
TP_ATR_MULT = 2.0
SL_ATR_MULT = 1.0
TP_PCT = 0.02
SL_PCT = 0.01

if ALERT_MODE == "conservador":
    ALERT_CONF_MIN = 0.75
    ALERT_COOLDOWN_MIN = 45
    TP_ATR_MULT = 1.5
    SL_ATR_MULT = 1.0
elif ALERT_MODE == "agressivo":
    ALERT_CONF_MIN = 0.55
    ALERT_COOLDOWN_MIN = 15
    TP_ATR_MULT = 2.5
    SL_ATR_MULT = 1.2

# Permite overrides finos por ENV (opcional)
ALERT_CONF_MIN = float(os.environ.get("ALERT_CONF_MIN", ALERT_CONF_MIN))
ALERT_COOLDOWN_MIN = int(os.environ.get("ALERT_COOLDOWN_MIN", ALERT_COOLDOWN_MIN))
UPDATE_INTERVAL_SEC = int(os.environ.get("UPDATE_INTERVAL_SECONDS", UPDATE_INTERVAL_SEC))
USE_ATR_LEVELS = os.environ.get("USE_ATR_LEVELS", str(USE_ATR_LEVELS)).lower() in {"1","true","yes","on"}
ATR_PERIOD = int(os.environ.get("ATR_PERIOD", ATR_PERIOD))
TP_ATR_MULT = float(os.environ.get("TP_ATR_MULT", TP_ATR_MULT))
SL_ATR_MULT = float(os.environ.get("SL_ATR_MULT", SL_ATR_MULT))
TP_PCT = float(os.environ.get("TP_PCT", TP_PCT))
SL_PCT = float(os.environ.get("SL_PCT", SL_PCT))

# Rótulos e integrações
TIMEFRAME      = os.environ.get("TIMEFRAME", "H1")
STRATEGY_LABEL = os.environ.get("STRATEGY_LABEL", "RSI+MACD+EMA+BB")
TG_TOKEN       = os.environ.get("TELEGRAM_BOT_TOKEN")
TG_CHAT        = os.environ.get("TELEGRAM_CHAT_ID")

# Templates / botões
ALERT_TEMPLATE = os.environ.get("ALERT_TEMPLATE", "pro").lower()  # pro | card | compact
APP_BASE_URL   = os.environ.get("APP_BASE_URL")  # ex: https://seu-projeto.up.railway.app

# ----------------- Imports do projeto -----------------
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from predict_enhanced import predict_signal, load_model_and_scaler
    from coingecko_client import fetch_bulk_prices, fetch_ohlc, SYMBOL_TO_ID
    from config import SYMBOLS
except ImportError as e:
    print(f"❌ Import error: {e}")

# ----------------- Utils: HTML escape para Telegram -----------------
def html_escape(s: str) -> str:
    if s is None:
        return ""
    return (str(s)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;"))

# ----------------- Integração Telegram -----------------
_tg_fn = None
try:
    from notifier_telegram import send_signal_notification as _tg_fn
except Exception:
    try:
        from notifier_telegram import notify_telegram as _tg_fn
    except Exception:
        try:
            from publisher import send_signal_notification as _tg_fn
        except Exception:
            _tg_fn = None

def _notify_telegram_fallback(text: str, parse_mode: str = "HTML", reply_markup: dict | None = None):
    if not (TG_TOKEN and TG_CHAT):
        return
    try:
        url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
        data = {"chat_id": TG_CHAT, "text": text, "parse_mode": parse_mode}
        if reply_markup:
            data["reply_markup"] = reply_markup
        requests.post(url, json=data, timeout=12)
    except Exception as e:
        print(f"⚠️ Telegram fallback erro: {e}")

def notify_telegram_message(text: str, payload: dict | None = None, parse_mode: str = "HTML", reply_markup: dict | None = None):
    try:
        if callable(_tg_fn):
            try:
                if payload is None:
                    payload = {"text": text, "parse_mode": parse_mode}
                else:
                    payload = {**payload, "text": text, "parse_mode": parse_mode}
                if reply_markup:
                    payload["reply_markup"] = reply_markup
                _tg_fn(payload)
            except TypeError:
                _tg_fn(text)
        else:
            _notify_telegram_fallback(text, parse_mode=parse_mode, reply_markup=reply_markup)
    except Exception as e:
        print(f"⚠️ erro ao enviar telegram: {e}")
        _notify_telegram_fallback(text, parse_mode=parse_mode, reply_markup=reply_markup)

# ----------------- App e estado -----------------
app = Flask(__name__)
CORS(app)

predictions_cache: List[Dict[str, Any]] = []
last_update: datetime | None = None
model = None
scaler = None
_last_alert_time: Dict[str, datetime] = {}    # {symbol: datetime}
_last_alert_sig:  Dict[str, Tuple] = {}       # {symbol: (side, entry, tp, sl)}  -> evita duplicar

# ----------------- Utils: médias, MACD, ATR -----------------
def ema(values: List[float], period: int) -> List[float]:
    if len(values) < period: return []
    k = 2 / (period + 1)
    out = []
    s = sum(values[:period]) / period
    out.append(s)
    for v in values[period:]:
        s = v * k + s * (1 - k)
        out.append(s)
    return out

def sma(values: List[float], period: int) -> List[float]:
    if len(values) < period: return []
    out = []
    s = sum(values[:period])
    out.append(s / period)
    for i in range(period, len(values)):
        s += values[i] - values[i - period]
        out.append(s / period)
    return out

def macd(values: List[float], fast=12, slow=26, signal=9) -> Tuple[List[float], List[float]]:
    if len(values) < slow + signal: return [], []
    ema_fast = ema(values, fast)
    ema_slow = ema(values, slow)
    offs = len(ema_fast) - len(ema_slow)
    if offs > 0: ema_fast = ema_fast[offs:]
    elif offs < 0: ema_slow = ema_slow[-offs:]
    line = [a - b for a, b in zip(ema_fast, ema_slow)]
    sig = ema(line, signal)
    offs2 = len(line) - len(sig)
    if offs2 > 0: line = line[offs2:]
    return line, sig

def _wilder_ema(prev: float, value: float, period: int) -> float:
    return (prev * (period - 1) + value) / period

def compute_atr(candles: List[List[float]], period: int = 14) -> float | None:
    n = len(candles)
    if n < period + 1: return None
    trs: List[float] = []
    for i in range(1, n):
        _, o, h, l, c = candles[i]
        _, po, ph, pl, pc = candles[i - 1]
        tr = max(h - l, abs(h - pc), abs(l - pc))
        trs.append(float(tr))
    if len(trs) < period: return None
    atr = sum(trs[:period]) / period
    for tr in trs[period:]:
        atr = _wilder_ema(atr, tr, period)
    if atr is None or isnan(atr): return None
    return float(atr)

def price_levels_by_atr(side: str, price: float, atr: float,
                        tp_mult: float, sl_mult: float) -> Tuple[float, float]:
    if side == "COMPRA":
        tp = price + tp_mult * atr
        sl = price - sl_mult * atr
    else:
        tp = price - tp_mult * atr
        sl = price + sl_mult * atr
    return tp, sl

def enforce_coherence(side: str, entry: float, tp: float, sl: float) -> Tuple[float, float]:
    """
    Garante consistência:
      - COMPRA: tp > entry e sl < entry
      - VENDA : tp < entry e sl > entry
    Se necessário, troca tp <-> sl e aplica pequenos ajustes pelo fallback %.
    """
    if side == "COMPRA":
        if tp <= entry and sl >= entry:
            tp, sl = sl, tp
        if tp <= entry: tp = entry * (1 + max(TP_PCT, 0.001))
        if sl >= entry: sl = entry * (1 - max(SL_PCT, 0.001))
    else:
        if tp >= entry and sl <= entry:
            tp, sl = sl, tp
        if tp >= entry: tp = entry * (1 - max(TP_PCT, 0.001))
        if sl <= entry: sl = entry * (1 + max(SL_PCT, 0.001))
    return tp, sl

# ----------------- Templates e botões de alerta -----------------
def conf_bar(conf: float, steps: int = 10) -> str:
    try:
        n = max(0, min(steps, round(conf * steps)))
        return ("█" * n) + ("░" * (steps - n))
    except Exception:
        return "—"

def fmtnum(x):
    try:
        return f"{float(x):,.4f}".replace(",", "X").replace(".", ",").replace("X", ".")
    except Exception:
        return str(x)

def build_inline_keyboard(symbol: str) -> dict | None:
    kb = []
    tv = f"https://www.tradingview.com/symbols/{symbol.replace('USDT','')}USDT/"
    kb.append([{"text":"📈 TradingView", "url": tv}])
    if APP_BASE_URL:
        kb_row = []
        kb_row.append({"text":"🤖 Testar IA", "url": f"{APP_BASE_URL}/api/test-ai?symbol={symbol}"})
        kb_row.append({"text":"🔄 Forçar Update", "url": f"{APP_BASE_URL}/api/force-update"})
        kb.append(kb_row)
    return {"inline_keyboard": kb} if kb else None

def build_msg_html(style: str, symbol: str, side: str, conf: float, content: dict,
                   strat_label: str, timeframe: str, use_atr: bool) -> tuple[str, dict | None]:
    side_emoji = "🟢" if side == "COMPRA" else "🔴"
    rr_txt = content.get("rr")
    rr_txt = f"{rr_txt:.2f}" if isinstance(rr_txt, (int, float)) else (rr_txt or "—")
    pct24 = content.get("price_change_24h") or 0.0
    atr_pct = content.get("atr_pct")
    atr_extra = f" ({atr_pct}%)" if atr_pct is not None else ""
    kb = build_inline_keyboard(symbol)

    if style == "compact":
        head = f'{side_emoji} <b>{html_escape(side)}</b> | <b>{html_escape(symbol)}</b>\n'
        table = (
            "<pre>"
            f"Entrada  {fmtnum(content['entry_price']).rjust(12)}\n"
            f"Alvo     {fmtnum(content['tp']).rjust(12)}\n"
            f"Stop     {fmtnum(content['sl']).rjust(12)}\n"
            "</pre>"
        )
        tail = (
            f"Conf: <code>{int(conf*100)}%</code> • R:R <code>{html_escape(rr_txt)}</code>\n"
            f"{'ATR' if use_atr else '%'} • {html_escape(timeframe)} • 24h: <code>{pct24:+.2f}%</code>"
        )
        return head + table + tail, kb

    if style == "card":
        bar = conf_bar(conf)
        head = f'{side_emoji} <b>{html_escape(side)}</b> | <b>{html_escape(symbol)}</b>\n'
        barline = f"Confiança: <code>{int(conf*100)}%</code>\n<code>{bar}</code>\n"
        grid = (
            "<pre>"
            f"Entrada   {fmtnum(content['entry_price']).rjust(12)}\n"
            f"Alvo      {fmtnum(content['tp']).rjust(12)}\n"
            f"Stop      {fmtnum(content['sl']).rjust(12)}\n"
            f"R:R       {str(rr_txt).rjust(12)}\n"
            "</pre>"
        )
        tech = (
            f"📊 Tendência: <b>{html_escape(content['trend'])}</b> • "
            f"MACD: <b>{html_escape(content['macd_trend'])}</b> • "
            f"Vol: <b>{html_escape(content['volatility'])}</b>{html_escape(atr_extra)}\n"
            f"🧠 Estratégia: <i>{html_escape(strat_label)}</i> {'(ATR)' if use_atr else '(%)'}\n"
            f"⏱️ {html_escape(timeframe)} • 24h: <code>{pct24:+.2f}%</code>"
        )
        return head + barline + grid + tech, kb

    # default: pro
    head = f'{side_emoji} <b>{html_escape(side)}</b> | <b>{html_escape(symbol)}</b>\n'
    lines = [
        f'🎯 <b>Entrada:</b> <code>{html_escape(fmtnum(content["entry_price"]))}</code>',
        f'🎯 <b>Alvo:</b>    <code>{html_escape(fmtnum(content["tp"]))}</code>',
        f'🛑 <b>Stop:</b>    <code>{html_escape(fmtnum(content["sl"]))}</code>',
        f'📈 <b>Confiança:</b> <code>{int(conf*100)}%</code> • <b>R:R</b> <code>{html_escape(rr_txt)}</code>',
        f'🧠 Estratégia: <i>{html_escape(strat_label)}</i> {"(ATR)" if use_atr else "(%)"}',
        f'📊 <b>Tendência:</b> {html_escape(content["trend"])} • <b>MACD:</b> {html_escape(content["macd_trend"])} • '
        f'<b>Vol:</b> {html_escape(content["volatility"])}{html_escape(atr_extra)}',
        f'⏱️ {html_escape(timeframe)} • 24h: <code>{pct24:+.2f}%</code>'
    ]
    return head + "\n".join(lines), kb

# ----------------- IA -----------------
def load_ai_model():
    global model, scaler
    try:
        model, scaler = load_model_and_scaler()
        if model is not None:
            print("✅ Modelo de IA carregado com sucesso")
            return True
        print("❌ Falha ao carregar modelo de IA")
        return False
    except Exception as e:
        print(f"❌ Erro ao carregar modelo: {e}")
        return False

# ----------------- Predição + Alertas -----------------
def collect_and_predict():
    """Coleta dados, gera predições e dispara alertas conforme regra."""
    global predictions_cache, last_update
    try:
        print("🔄 Coletando dados e fazendo predições...")
        bulk_data = fetch_bulk_prices(SYMBOLS)  # preços, mcap, 24h
        predictions: List[Dict[str, Any]] = []
        sent_this_cycle: Set[str] = set()  # garante 1 alerta por símbolo por ciclo

        for symbol in SYMBOLS:
            try:
                coin_id = SYMBOL_TO_ID.get(symbol, symbol.replace("USDT", "").lower())
                ohlc_raw = fetch_ohlc(coin_id, days=30)  # [[ts, o, h, l, c], ...]
                if not ohlc_raw or len(ohlc_raw) < 60:
                    continue

                # normaliza para predict_signal + prepara closes para métricas
                candles_norm = []
                closes = []
                for ts, o, h, l, c in ohlc_raw:
                    candles_norm.append({
                        "timestamp": int(ts/1000),
                        "open": float(o), "high": float(h),
                        "low": float(l), "close": float(c)
                    })
                    closes.append(float(c))

                result, error = predict_signal(symbol, candles_norm)
                if not result:
                    continue

                cur = bulk_data.get(symbol, {})
                price_now = float(cur.get("usd", 0.0))
                pct24 = float(cur.get("usd_24h_change", 0.0))
                result.update({
                    "current_price": price_now,
                    "price_change_24h": pct24,
                    "market_cap": cur.get("usd_market_cap", 0.0)
                })
                predictions.append(result)

                # ===== Regra de ALERTA + payload completo =====
                conf = float(result.get("confidence") or 0.0)
                side = (result.get("signal") or "").upper()
                prob_buy = float(result.get("probability_buy") or 0.0)
                prob_sell = float(result.get("probability_sell") or 0.0)

                trigger = (
                    conf >= ALERT_CONF_MIN and
                    ((side == "COMPRA" and prob_buy >= ALERT_CONF_MIN) or
                     (side == "VENDA"  and prob_sell >= ALERT_CONF_MIN))
                )

                if trigger and symbol not in sent_this_cycle:
                    now = datetime.utcnow()
                    last_t = _last_alert_time.get(symbol)

                    entry = price_now
                    # ——— níveis por ATR (ou % fallback) ———
                    tp, sl = None, None
                    atr = None
                    if USE_ATR_LEVELS:
                        atr = compute_atr(ohlc_raw, ATR_PERIOD)
                        if atr:
                            tp, sl = price_levels_by_atr(side, entry, atr, TP_ATR_MULT, SL_ATR_MULT)
                    if tp is None or sl is None:
                        if side == "COMPRA":
                            tp = entry * (1 + TP_PCT)
                            sl = entry * (1 - SL_PCT)
                        else:
                            tp = entry * (1 - TP_PCT)
                            sl = entry * (1 + SL_PCT)

                    # --------- Validação de coerência ---------
                    tp, sl = enforce_coherence(side, entry, tp, sl)

                    # --------- Métrica R:R ---------
                    if side == "COMPRA":
                        rr = (tp - entry) / max(entry - sl, 1e-12)
                    else:
                        rr = (entry - tp) / max(sl - entry, 1e-12)

                    # --------- Anti-duplicação ---------
                    sig = (side, round(entry,6), round(tp,6), round(sl,6))
                    last_sig = _last_alert_sig.get(symbol)
                    cooldown_ok = (not last_t) or ((now - last_t) >= timedelta(minutes=ALERT_COOLDOWN_MIN))
                    is_dup = (last_sig == sig)
                    if cooldown_ok and not is_dup:
                        _last_alert_sig[symbol] = sig
                        sent_this_cycle.add(symbol)

                        # --------- Comentário técnico ---------
                        trend_txt = "Neutra"
                        vol_txt = "—"
                        atr_pct = None
                        if atr and entry > 0:
                            atr_pct = (atr / entry) * 100.0
                            if atr_pct < 1.5: vol_txt = "Baixa"
                            elif atr_pct < 3.0: vol_txt = "Média"
                            else: vol_txt = "Alta"

                        def _sma(vals, n):
                            return sma(vals, n)[-1] if len(vals) >= n else None

                        sma20_last = _sma(closes, 20)
                        sma50_last = _sma(closes, 50)
                        macd_line, macd_sig = macd(closes, 12, 26, 9)
                        if sma20_last is not None and sma50_last is not None:
                            if sma20_last > sma50_last: trend_txt = "Alta"
                            elif sma20_last < sma50_last: trend_txt = "Baixa"
                        macd_txt = "—"
                        if macd_line and macd_sig:
                            macd_txt = "Alta" if macd_line[-1] > macd_sig[-1] else "Baixa"

                        content = {
                            "symbol": symbol,
                            "side": side,
                            "strategy": STRATEGY_LABEL,
                            "timeframe": TIMEFRAME,
                            "entry": round(entry, 6),
                            "entry_price": round(entry, 6),
                            "target_price": round(tp, 6),
                            "tp": round(tp, 6),
                            "stop_loss": round(sl, 6),
                            "sl": round(sl, 6),
                            "current_price": round(price_now, 6),
                            "confidence": conf,
                            "probability_buy": prob_buy,
                            "probability_sell": prob_sell,
                            "price_change_24h": pct24,
                            "rr": round(rr, 2),
                            "atr": round(atr, 6) if atr else None,
                            "atr_pct": round(atr_pct, 2) if atr_pct else None,
                            "trend": trend_txt,
                            "macd_trend": macd_txt,
                            "volatility": vol_txt,
                            "timestamp": now.isoformat() + "Z"
                        }

                        # ----- Mensagem estilizada + botões -----
                        txt, kb = build_msg_html(ALERT_TEMPLATE, symbol, side, conf, content, STRATEGY_LABEL, TIMEFRAME, USE_ATR_LEVELS)
                        notify_telegram_message(txt, payload=content, parse_mode="HTML", reply_markup=kb)
                        _last_alert_time[symbol] = now

            except Exception as e:
                print(f"❌ Erro ao processar {symbol}: {e}")
                continue

        predictions_cache = predictions
        last_update = datetime.utcnow()
        print(f"✅ Predições atualizadas: {len(predictions)} símbolos")

    except Exception as e:
        print(f"❌ Erro na coleta/predição: {e}")

def background_updater():
    while True:
        try:
            collect_and_predict()
        except Exception as e:
            print(f"❌ Erro no updater: {e}")
        time.sleep(UPDATE_INTERVAL_SEC)

# ----------------- Views -----------------
@app.route("/")
def index():
    html = r"""
<!doctype html>
<html lang="pt-br">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Crypto Trading AI — Dashboard</title>
<style>
  :root{--bg:#0b0b0b;--panel:#111214;--ring:#2a2c31;--text:#e8e8e8;--sub:#a9a9a9;--accent:#6ee7ff;--good:#22c55e;--bad:#ef4444}
  *{box-sizing:border-box} body{margin:0;background:#0b0b0b;color:var(--text);font:14px/1.45 ui-sans-serif,system-ui,Segoe UI,Roboto,Arial}
  .wrap{max-width:1200px;margin:0 auto;padding:20px}
  header{display:flex;justify-content:space-between;gap:12px;align-items:center}
  .brand{display:flex;gap:10px;align-items:center}
  .logo{width:36px;height:36px;border-radius:12px;background:linear-gradient(135deg,var(--accent),#9bffd6)}
  h1{margin:0;font-size:18px} .sub{color:var(--sub);font-size:12px}
  .bar{display:flex;gap:10px;flex-wrap:wrap}
  .input,.select,.btn{background:var(--panel);border:1px solid var(--ring);color:var(--text);padding:10px 12px;border-radius:10px}
  .btn{cursor:pointer}
  .status{display:flex;gap:10px;flex-wrap:wrap;margin:14px 0}
  .badge{background:#16181b;border:1px solid var(--ring);padding:6px 10px;border-radius:999px;font-size:12px;color:#d7d7d7}
  .grid{display:grid;grid-template-columns:repeat(12,1fr);gap:14px}
  @media (max-width:1100px){.grid{grid-template-columns:repeat(8,1fr)}}
  @media (max_width:780px){.grid{grid-template-columns:repeat(4,1fr)}}
  .card{grid-column:span 4;background:linear-gradient(180deg,#0f1012,#0b0c0e);border:1px solid var(--ring);border-radius:14px;padding:14px}
  .sym{font-weight:600} .price{font-size:20px;margin:6px 0} .buy{color:var(--good)} .sell{color:var(--bad)}
  .muted{color:#9aa0a6;font-size:12px;margin-top:8px}
  .meter{--p:0;height:8px;border-radius:999px;background:#1a1b1e;border:1px solid var(--ring);position:relative;overflow:hidden;margin-top:8px}
  .meter::after{content:"";position:absolute;inset:0;transform:scaleX(calc(var(--p)));transform-origin:left;background:linear-gradient(90deg,#1ad1ff,#3cf8b2)}
</style>
</head>
<body>
  <div class="wrap">
    <header>
      <div class="brand"><div class="logo"></div><div><h1>Crypto Trading AI</h1><div class="sub">IA + Técnico • Alertas Telegram</div></div></div>
      <div class="bar">
        <input id="q" class="input" placeholder="Buscar símbolo (ex: BTCUSDT)" />
        <select id="minConf" class="select">
          <option value="0">Conf ≥ 0%</option>
          <option value="0.5">Conf ≥ 50%</option>
          <option value="0.6" selected>Conf ≥ 60%</option>
          <option value="0.7">Conf ≥ 70%</option>
          <option value="0.8">Conf ≥ 80%</option>
        </select>
        <button class="btn" id="refresh">Atualizar</button>
      </div>
    </header>

    <section class="status">
      <span class="badge" id="badge-api">API online</span>
      <span class="badge" id="badge-ia">IA: carregando…</span>
      <span class="badge" id="badge-count">0 sinais</span>
      <span class="badge" id="badge-time">—</span>
    </section>

    <section id="grid" class="grid"></section>
    <div id="empty" class="badge" style="display:none">Sem resultados</div>
  </div>

<script>
const grid = document.getElementById('grid');
const q = document.getElementById('q');
const minConf = document.getElementById('minConf');
const btnRefresh = document.getElementById('refresh');
const badgeCount = document.getElementById('badge-count');
const badgeTime = document.getElementById('badge-time');
const badgeIA = document.getElementById('badge-ia');

let cache=[], lastUpdate=null;

function fmt(n,d=2){ if(n==null) return "—"; try{ return Number(n).toFixed(d);}catch(_){return String(n)} }

function render(){
  const term=(q.value||"").trim().toUpperCase();
  const min=parseFloat(minConf.value||"0");
  const data=cache.filter(x=>(!term || (x.symbol||"").toUpperCase().includes(term)))
                  .filter(x=> (x.confidence??0) >= min);
  grid.innerHTML="";
  data.forEach(s=>{
    const side=(s.signal||"").toUpperCase();
    const pct24=s.price_change_24h??0;
    const card=document.createElement('div');
    card.className='card';
    card.innerHTML=`
      <div class="sym">${s.symbol}</div>
      <div class="price">$${fmt(s.current_price, s.current_price>100?2:4)}</div>
      <div class="${side==='COMPRA'?'buy':'sell'}">${side} • Conf ${( (s.confidence??0)*100 ).toFixed(0)}%</div>
      <div>24h: ${pct24>=0?'▲':'▼'} ${fmt(pct24,2)}%</div>
      <div class="meter" style="--p:${(s.confidence??0)}"></div>
      <div class="muted">${lastUpdate? new Date(lastUpdate).toLocaleString(): '—'}</div>
    `;
    grid.appendChild(card);
  });
  badgeCount.textContent=`${data.length} sinais`;
  badgeTime.textContent= lastUpdate? new Date(lastUpdate).toLocaleTimeString() : '—';
}

async function pullStatus(){
  try{ const r=await fetch('/api/status'); const js=await r.json(); badgeIA.textContent = js.model_loaded? 'IA: ok' : 'IA: off'; }catch(e){ badgeIA.textContent='IA: ?';}
}
async function pull(){
  try{ const r=await fetch('/api/predictions'); const js=await r.json(); if(js.success){ cache=js.predictions||[]; lastUpdate=js.last_update; render(); } }catch(e){ }
}
btnRefresh.addEventListener('click', pull); q.addEventListener('input', render); minConf.addEventListener('change', render);
pullStatus(); pull(); setInterval(()=>{ pullStatus(); pull(); }, 60000);
</script>
</body>
</html>
"""
    return render_template_string(html)

# ----------------- API: dados -----------------
@app.route("/api/predictions")
def get_predictions():
    global predictions_cache, last_update
    if not predictions_cache or not last_update or (datetime.utcnow() - last_update).seconds > UPDATE_INTERVAL_SEC:
        collect_and_predict()
    return jsonify({
        "success": True,
        "predictions": predictions_cache,
        "last_update": last_update.isoformat() if last_update else None,
        "total_signals": len(predictions_cache)
    })

@app.route("/api/status")
def get_status():
    return jsonify({
        "status": "online",
        "model_loaded": model is not None,
        "last_update": last_update.isoformat() if last_update else None,
        "cached_predictions": len(predictions_cache),
        "alert_conf_min": ALERT_CONF_MIN,
        "alert_cooldown_min": ALERT_COOLDOWN_MIN,
        "use_atr_levels": USE_ATR_LEVELS,
        "atr_period": ATR_PERIOD,
        "tp_atr_mult": TP_ATR_MULT,
        "sl_atr_mult": SL_ATR_MULT,
        "alert_mode": ALERT_MODE,
        "alert_template": ALERT_TEMPLATE
    })

@app.route("/api/force-update")
def force_update():
    try:
        collect_and_predict()
        return jsonify({"success": True, "message": "Predições atualizadas", "count": len(predictions_cache)})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/test-ai")
def test_ai():
    """
    Diagnóstico: roda a IA ao vivo num símbolo (padrão BTCUSDT), sem cache e sem Telegram.
    Params: symbol (ex.: DOTUSDT), days (ex.: 14)
    """
    try:
        symbol = (request.args.get("symbol") or "BTCUSDT").upper().strip()
        days = int(request.args.get("days") or 30)

        coin_id = SYMBOL_TO_ID.get(symbol, symbol.replace("USDT", "").lower())
        ohlc_raw = fetch_ohlc(coin_id, days=days)
        if not ohlc_raw or len(ohlc_raw) < 20:
            return jsonify({"success": False, "error": f"OHLC insuficiente para {symbol} (days={days})", "symbol": symbol}), 400

        candles = []
        for ts, o, h, l, c in ohlc_raw:
            candles.append({
                "timestamp": int(ts/1000),
                "open": float(o), "high": float(h),
                "low": float(l), "close": float(c)
            })

        result, error = predict_signal(symbol, candles)
        price_ctx = fetch_bulk_prices([symbol]).get(symbol, {})
        now_iso = datetime.utcnow().isoformat() + "Z"

        return jsonify({
            "success": True if result else False,
            "model_loaded": (model is not None),
            "symbol": symbol,
            "days": days,
            "timestamp": now_iso,
            "prediction": result,
            "error": error,
            "context": {
                "current_price": price_ctx.get("usd"),
                "price_change_24h": price_ctx.get("usd_24hr_change") or price_ctx.get("usd_24h_change"),
                "market_cap": price_ctx.get("usd_market_cap")
            }
        }), (200 if result else 500)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/health")
def health():
    return jsonify(ok=True, model_loaded=(model is not None)), 200

# ----------------- Boot -----------------
if __name__ == "__main__":
    print("🚀 Iniciando Crypto Trading API...")
    load_ai_model()
    collect_and_predict()
    threading.Thread(target=background_updater, daemon=True).start()
    print("✅ API iniciada com sucesso!")
    port = int(os.environ.get("PORT", 5000))
    print(f"🌐 Acesse: http://0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
