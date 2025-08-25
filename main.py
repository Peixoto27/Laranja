# -*- coding: utf-8 -*-
"""
API Flask para Sistema de Trading de Criptomoedas com IA + Alertas Telegram
- TP/SL por ATR (configurável) com fallback por porcentagem
"""
import os, sys, time, json, threading, requests
from math import isnan
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple
from flask import Flask, jsonify, render_template_string
from flask_cors import CORS

# ----------------- Config por ENV -----------------
ALERT_CONF_MIN        = float(os.environ.get("ALERT_CONF_MIN", "0.60"))       # confiança mínima p/ alertar (0–1)
ALERT_COOLDOWN_MIN    = int(os.environ.get("ALERT_COOLDOWN_MIN", "30"))       # silêncio por símbolo (min)
UPDATE_INTERVAL_SEC   = int(os.environ.get("UPDATE_INTERVAL_SECONDS", "300")) # ciclo (s)

# Fallback por % (caso ATR indisponível)
TP_PCT                = float(os.environ.get("TP_PCT", "0.020"))              # 2%
SL_PCT                = float(os.environ.get("SL_PCT", "0.010"))              # 1%

# Níveis por ATR
USE_ATR_LEVELS        = os.environ.get("USE_ATR_LEVELS", "true").lower() in {"1","true","yes","on"}
ATR_PERIOD            = int(os.environ.get("ATR_PERIOD", "14"))
TP_ATR_MULT           = float(os.environ.get("TP_ATR_MULT", "2.0"))
SL_ATR_MULT           = float(os.environ.get("SL_ATR_MULT", "1.0"))

# Rótulos
TIMEFRAME             = os.environ.get("TIMEFRAME", "H1")
STRATEGY_LABEL        = os.environ.get("STRATEGY_LABEL", "RSI+MACD+EMA+BB")

# Telegram (fallback HTTP se módulo local não for encontrado)
TG_TOKEN              = os.environ.get("TELEGRAM_BOT_TOKEN")
TG_CHAT               = os.environ.get("TELEGRAM_CHAT_ID")

# ----------------- Imports do projeto -----------------
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from predict_enhanced import predict_signal, load_model_and_scaler
    from coingecko_client import fetch_bulk_prices, fetch_ohlc, SYMBOL_TO_ID
    from config import SYMBOLS
except ImportError as e:
    print(f"❌ Import error: {e}")

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

def _notify_telegram_fallback(text: str):
    if not (TG_TOKEN and TG_CHAT):
        return
    try:
        url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
        data = {"chat_id": TG_CHAT, "text": text, "parse_mode": "Markdown"}
        requests.post(url, json=data, timeout=12)
    except Exception as e:
        print(f"⚠️ Telegram fallback erro: {e}")

def notify_telegram_message(text: str, payload: dict | None = None):
    try:
        if callable(_tg_fn):
            try:
                _tg_fn(payload or {"text": text})
            except TypeError:
                _tg_fn(text)
        else:
            _notify_telegram_fallback(text)
    except Exception as e:
        print(f"⚠️ erro ao enviar telegram: {e}")
        _notify_telegram_fallback(text)

# ----------------- App e estado -----------------
app = Flask(__name__)
CORS(app)

predictions_cache: List[Dict[str, Any]] = []
last_update: datetime | None = None
model = None
scaler = None
_last_alert_time: Dict[str, datetime] = {}  # {symbol: datetime}

# ----------------- Utils: ATR -----------------
def _wilder_ema(prev: float, value: float, period: int) -> float:
    """Wilder EMA para ATR."""
    return (prev * (period - 1) + value) / period

def compute_atr(candles: List[List[float]], period: int = 14) -> float | None:
    """
    candles: [[ts, open, high, low, close], ...] (como o CoinGecko retorna)
    retorna ATR (float) ou None se não puder calcular
    """
    n = len(candles)
    if n < period + 1:
        return None
    trs: List[float] = []
    for i in range(1, n):
        _, o, h, l, c = candles[i]
        _, po, ph, pl, pc = candles[i - 1]
        tr = max(h - l, abs(h - pc), abs(l - pc))
        trs.append(float(tr))
    if len(trs) < period:
        return None
    # média inicial
    atr = sum(trs[:period]) / period
    # suavização de Wilder
    for tr in trs[period:]:
        atr = _wilder_ema(atr, tr, period)
    if atr is None or isnan(atr):
        return None
    return float(atr)

def price_levels_by_atr(side: str, price: float, atr: float,
                        tp_mult: float, sl_mult: float) -> Tuple[float, float]:
    """
    Retorna (tp, sl) a partir do preço e ATR.
    """
    if side == "COMPRA":
        tp = price + tp_mult * atr
        sl = price - sl_mult * atr
    else:  # VENDA
        tp = price - tp_mult * atr
        sl = price + sl_mult * atr
    return tp, sl

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

        for symbol in SYMBOLS:
            try:
                coin_id = SYMBOL_TO_ID.get(symbol, symbol.replace("USDT", "").lower())
                ohlc_raw = fetch_ohlc(coin_id, days=30)  # [[ts, o, h, l, c], ...]
                if not ohlc_raw or len(ohlc_raw) < 60:
                    continue

                # normaliza para predict_signal
                candles_norm = []
                for ts, o, h, l, c in ohlc_raw:
                    candles_norm.append({
                        "timestamp": int(ts/1000),
                        "open": float(o), "high": float(h),
                        "low": float(l), "close": float(c)
                    })

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

                if trigger:
                    now = datetime.utcnow()
                    last = _last_alert_time.get(symbol)
                    if (not last) or ((now - last) >= timedelta(minutes=ALERT_COOLDOWN_MIN)):
                        entry = price_now

                        # ——— níveis por ATR (ou % fallback) ———
                        tp, sl, rr = None, None, None
                        if USE_ATR_LEVELS:
                            atr = compute_atr(ohlc_raw, ATR_PERIOD)
                            if atr:
                                tp, sl = price_levels_by_atr(side, entry, atr, TP_ATR_MULT, SL_ATR_MULT)
                        if tp is None or sl is None:
                            # fallback por porcentagem
                            if side == "COMPRA":
                                tp = entry * (1 + TP_PCT)
                                sl = entry * (1 - SL_PCT)
                            else:
                                tp = entry * (1 - TP_PCT)
                                sl = entry * (1 + SL_PCT)

                        if side == "COMPRA":
                            rr = (tp - entry) / max(entry - sl, 1e-12)
                        else:
                            rr = (entry - tp) / max(sl - entry, 1e-12)

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
                            "rr": round(rr, 2) if rr == rr else None,  # evita NaN
                            "timestamp": now.isoformat() + "Z"
                        }

                        txt = (
                            f"📢 Novo sinal para {symbol}\n"
                            f"🎯 Entrada: {content['entry_price']}\n"
                            f"🎯 Alvo:   {content['tp']}\n"
                            f"🛑 Stop:   {content['sl']}\n"
                            f"📈 Confiança: {conf*100:.0f}%\n"
                            f"🧠 Estratégia: {STRATEGY_LABEL} "
                            f"{'(ATR)' if USE_ATR_LEVELS else '(%)'}\n"
                            f"⏱️ {TIMEFRAME} • RR: {content['rr'] if content['rr'] is not None else '—'}"
                        )
                        notify_telegram_message(txt, payload=content)
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
  @media (max-width:780px){.grid{grid-template-columns:repeat(4,1fr)}}
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
        "sl_atr_mult": SL_ATR_MULT
    })

@app.route("/api/force-update")
def force_update():
    try:
        collect_and_predict()
        return jsonify({"success": True, "message": "Predições atualizadas", "count": len(predictions_cache)})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/health")
def health():
    return jsonify(ok=True, model_loaded=(model is not None)), 200

if __name__ == "__main__":
    print("🚀 Iniciando Crypto Trading API...")
    load_ai_model()
    collect_and_predict()
    threading.Thread(target=background_updater, daemon=True).start()
    print("✅ API iniciada com sucesso!")
    port = int(os.environ.get("PORT", 5000))
    print(f"🌐 Acesse: http://0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
