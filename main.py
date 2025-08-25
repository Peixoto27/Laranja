# -*- coding: utf-8 -*-
"""
API Flask para Sistema de Trading de Criptomoedas com IA + Alertas Telegram
"""
import os
import sys
import time
import json
import threading
import requests
from datetime import datetime, timedelta
from flask import Flask, jsonify, render_template_string
from flask_cors import CORS

# ---------- Config de alertas via ENV ----------
ALERT_CONF_MIN = float(os.environ.get("ALERT_CONF_MIN", "0.60"))   # limiar de confiança (0-1)
ALERT_COOLDOWN_MIN = int(os.environ.get("ALERT_COOLDOWN_MIN", "30"))  # min por símbolo
UPDATE_INTERVAL_SECONDS = int(os.environ.get("UPDATE_INTERVAL_SECONDS", "300"))  # 5min
TG_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TG_CHAT = os.environ.get("TELEGRAM_CHAT_ID")

# ---------- Imports do projeto ----------
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from predict_enhanced import predict_signal, load_model_and_scaler
    from coingecko_client import fetch_bulk_prices, fetch_ohlc, SYMBOL_TO_ID
    from config import SYMBOLS
except ImportError as e:
    print(f"❌ Import error: {e}")

# ---------- Tentativa de integrar com seus módulos de Telegram ----------
# tenta vários nomes de função para compatibilidade com seu repo
_tg_fn = None
try:
    from notifier_telegram import send_signal_notification as _tg_fn  # nome usado em alguns módulos
except Exception:
    try:
        from notifier_telegram import notify_telegram as _tg_fn
    except Exception:
        try:
            from publisher import send_signal_notification as _tg_fn
        except Exception:
            _tg_fn = None

def _notify_telegram_fallback(text: str):
    """Fallback direto via HTTP caso nenhum módulo de notificação esteja disponível."""
    if not (TG_TOKEN and TG_CHAT):
        return
    try:
        url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
        data = {"chat_id": TG_CHAT, "text": text, "parse_mode": "Markdown"}
        requests.post(url, data=data, timeout=10)
    except Exception as e:
        print(f"⚠️ Telegram fallback erro: {e}")

def notify_telegram_message(text: str, payload: dict | None = None):
    """Função única de envio: usa módulo do projeto se existir; senão, fallback HTTP."""
    try:
        if callable(_tg_fn):
            # muitos projetos esperam dict; se for string, enviamos como 'text'
            try:
                _tg_fn(payload or {"text": text})
            except TypeError:
                _tg_fn(text)  # se a assinatura for (str)
        else:
            _notify_telegram_fallback(text)
    except Exception as e:
        print(f"⚠️ erro ao enviar telegram: {e}")
        _notify_telegram_fallback(text)

# ---------- App ----------
app = Flask(__name__)
CORS(app)

# ---------- Estado ----------
predictions_cache = []
last_update = None
model = None
scaler = None
_last_alert_time = {}  # {symbol: datetime}

# ---------- IA ----------
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

# ---------- Predição + Alertas ----------
def collect_and_predict():
    """Coleta dados, gera predições e dispara alertas conforme regra."""
    global predictions_cache, last_update
    try:
        print("🔄 Coletando dados e fazendo predições...")
        bulk_data = fetch_bulk_prices(SYMBOLS)  # preços, mcap, 24h
        predictions = []

        for symbol in SYMBOLS:
            try:
                coin_id = SYMBOL_TO_ID.get(symbol, symbol.replace("USDT", "").lower())
                ohlc_data = fetch_ohlc(coin_id, days=30)
                if not ohlc_data or len(ohlc_data) < 60:
                    continue

                # normaliza velas no formato esperado por predict_signal
                candles = []
                for ts, o, h, l, c in ohlc_data:
                    candles.append({
                        "timestamp": int(ts / 1000),
                        "open": float(o),
                        "high": float(h),
                        "low": float(l),
                        "close": float(c)
                    })

                result, error = predict_signal(symbol, candles)
                if not result:
                    # print(f"⚠️ {symbol}: {error}")
                    continue

                # agrega dados atuais
                cur = bulk_data.get(symbol, {})
                result.update({
                    "current_price": cur.get("usd", 0.0),
                    "price_change_24h": cur.get("usd_24h_change", 0.0),
                    "market_cap": cur.get("usd_market_cap", 0.0)
                })
                predictions.append(result)

                # ===== Regras de ALERTA =====
                conf = float(result.get("confidence") or 0.0)
                side = (result.get("signal") or "").upper()
                prob_buy = float(result.get("probability_buy") or 0.0)
                prob_sell = float(result.get("probability_sell") or 0.0)

                # exemplo de regra base (pode ajustar):
                trigger = (
                    conf >= ALERT_CONF_MIN and
                    ((side == "COMPRA" and prob_buy >= ALERT_CONF_MIN) or
                     (side == "VENDA"  and prob_sell >= ALERT_CONF_MIN))
                )

                if trigger:
                    now = datetime.utcnow()
                    last = _last_alert_time.get(symbol)
                    if (not last) or ((now - last) >= timedelta(minutes=ALERT_COOLDOWN_MIN)):
                        price = result.get("current_price")
                        pct24 = result.get("price_change_24h")
                        msg = (
                            f"🚨 *Sinal Forte Detectado*\n"
                            f"• Par: *{symbol}*\n"
                            f"• Ação: *{side}*\n"
                            f"• Confiança: *{conf*100:.0f}%*\n"
                            f"• Prob Buy/Sell: {prob_buy*100:.0f}% / {prob_sell*100:.0f}%\n"
                            f"• Preço: ${price:,.4f}\n"
                            f"• 24h: {pct24:+.2f}%\n"
                            f"• Hora: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC"
                        )
                        notify_telegram_message(msg, payload={"symbol": symbol, "signal": side, "confidence": conf})
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
    """Atualizador periódico."""
    while True:
        try:
            collect_and_predict()
        except Exception as e:
            print(f"❌ Erro no updater: {e}")
        time.sleep(UPDATE_INTERVAL_SECONDS)

# ---------- Views ----------
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
    # garante cache fresco
    if not predictions_cache or not last_update or (datetime.utcnow() - last_update).seconds > UPDATE_INTERVAL_SECONDS:
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
        "alert_cooldown_min": ALERT_COOLDOWN_MIN
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

# ---------- Boot ----------
if __name__ == "__main__":
    print("🚀 Iniciando Crypto Trading API...")
    load_ai_model()
    collect_and_predict()
    threading.Thread(target=background_updater, daemon=True).start()
    print("✅ API iniciada com sucesso!")
    port = int(os.environ.get("PORT", 5000))
    print(f"🌐 Acesse: http://0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
