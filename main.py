# -*- coding: utf-8 -*-
"""
API Flask para Sistema de Trading de Criptomoedas com IA
"""
import os
import sys
from flask import Flask, jsonify, render_template_string
from flask_cors import CORS
import json
import numpy as np
from datetime import datetime
import threading
import time

# Adicionar diretório atual ao path para imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Imports do projeto
try:
    from predict_enhanced import predict_signal, load_model_and_scaler
    from coingecko_client import fetch_bulk_prices, fetch_ohlc, SYMBOL_TO_ID
    from config import SYMBOLS
except ImportError as e:
    print(f"Erro ao importar módulos: {e}")

app = Flask(__name__)
CORS(app)  # Permitir CORS para todas as rotas

# Variáveis globais
predictions_cache = {}
last_update = None
model = None
scaler = None


def load_ai_model():
    """Carrega o modelo de IA"""
    global model, scaler
    try:
        model, scaler = load_model_and_scaler()
        if model is not None:
            print("✅ Modelo de IA carregado com sucesso")
            return True
        else:
            print("❌ Falha ao carregar modelo de IA")
            return False
    except Exception as e:
        print(f"❌ Erro ao carregar modelo: {e}")
        return False


def collect_and_predict():
    """Coleta dados e faz predições"""
    global predictions_cache, last_update

    try:
        print("🔄 Coletando dados e fazendo predições...")
        bulk_data = fetch_bulk_prices(SYMBOLS)
        predictions = []

        for symbol in SYMBOLS:
            try:
                coin_id = SYMBOL_TO_ID.get(symbol, symbol.replace("USDT", "").lower())
                ohlc_data = fetch_ohlc(coin_id, days=30)
                if not ohlc_data or len(ohlc_data) < 60:
                    continue

                candles = []
                for ts, o, h, l, c in ohlc_data:
                    candles.append({
                        "timestamp": int(ts/1000),
                        "open": float(o),
                        "high": float(h),
                        "low": float(l),
                        "close": float(c)
                    })

                result, error = predict_signal(symbol, candles)
                if result:
                    current_data = bulk_data.get(symbol, {})
                    result.update({
                        'current_price': current_data.get('usd', 0),
                        'price_change_24h': current_data.get('usd_24h_change', 0),
                        'market_cap': current_data.get('usd_market_cap', 0)
                    })
                    predictions.append(result)
            except Exception as e:
                print(f"❌ Erro ao processar {symbol}: {e}")
                continue

        predictions_cache = predictions
        last_update = datetime.now()
        print(f"✅ Predições atualizadas: {len(predictions)} símbolos")

    except Exception as e:
        print(f"❌ Erro na coleta de dados: {e}")


def background_updater():
    """Atualiza predições em background"""
    while True:
        try:
            collect_and_predict()
            time.sleep(300)  # 5 minutos
        except Exception as e:
            print(f"❌ Erro no background updater: {e}")
            time.sleep(60)


@app.route('/')
def index():
    html = r"""
    <!doctype html>
    <html lang="pt-br">
    <head>
      <meta charset="utf-8" />
      <meta name="viewport" content="width=device-width, initial-scale=1" />
      <title>Crypto Trading AI — Dashboard</title>
      <style>
        body { margin:0; font-family:sans-serif; background:#0b0b0b; color:#e8e8e8; }
        .wrap { max-width:1200px; margin:0 auto; padding:20px; }
        h1 { color:#6ee7ff; }
        .grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(300px,1fr)); gap:15px; }
        .card { background:#111214; border:1px solid #2a2c31; border-radius:12px; padding:16px; }
        .sym { font-weight:bold; font-size:18px; }
        .price { font-size:20px; margin:8px 0; }
        .buy { color:#22c55e; } .sell { color:#ef4444; }
        .confidence { font-size:14px; margin-top:6px; }
        .muted { color:#9aa0a6; font-size:12px; margin-top:10px; }
      </style>
    </head>
    <body>
      <div class="wrap">
        <h1>🚀 Crypto Trading AI</h1>
        <p>Dashboard em tema Black — sinais em tempo real</p>
        <div id="signals" class="grid"></div>
      </div>
    <script>
      async function loadPredictions(){
        const res = await fetch('/api/predictions');
        const js = await res.json();
        const div = document.getElementById('signals');
        div.innerHTML = '';
        (js.predictions||[]).forEach(p=>{
          const el = document.createElement('div');
          el.className='card';
          el.innerHTML = `
            <div class="sym">${p.symbol}</div>
            <div class="price">$${(p.current_price||0).toFixed(4)}</div>
            <div class="${p.signal==='COMPRA'?'buy':'sell'}">${p.signal} — Confiança ${(p.confidence*100).toFixed(1)}%</div>
            <div class="confidence">Prob Compra: ${(p.probability_buy*100).toFixed(1)}% | Prob Venda: ${(p.probability_sell*100).toFixed(1)}%</div>
            <div class="muted">${new Date(p.timestamp).toLocaleString()}</div>
          `;
          div.appendChild(el);
        });
      }
      loadPredictions();
      setInterval(loadPredictions,60000);
    </script>
    </body>
    </html>
    """
    return render_template_string(html)


@app.route('/api/predictions')
def get_predictions():
    global predictions_cache, last_update
    if not predictions_cache or not last_update or (datetime.now() - last_update).seconds > 300:
        collect_and_predict()
    return jsonify({
        'success': True,
        'predictions': predictions_cache,
        'last_update': last_update.isoformat() if last_update else None,
        'total_signals': len(predictions_cache)
    })


@app.route('/api/status')
def get_status():
    return jsonify({
        'status': 'online',
        'model_loaded': model is not None,
        'last_update': last_update.isoformat() if last_update else None,
        'cached_predictions': len(predictions_cache)
    })


@app.route('/api/force-update')
def force_update():
    try:
        collect_and_predict()
        return jsonify({
            'success': True,
            'message': 'Predições atualizadas',
            'predictions_count': len(predictions_cache)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


if __name__ == '__main__':
    print("🚀 Iniciando Crypto Trading API...")
    load_ai_model()
    collect_and_predict()
    updater_thread = threading.Thread(target=background_updater, daemon=True)
    updater_thread.start()
    print("✅ API iniciada com sucesso!")

    # 🚨 Porta dinâmica para Railway
    port = int(os.environ.get("PORT", 5000))
    print(f"🌐 Acesse: http://0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
