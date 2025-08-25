# -*- coding: utf-8 -*-
"""
coingecko_client.py
Coleta preços e OHLC do Coingecko com backoff e limites.
"""

import time
import math
import requests
from typing import Dict, List, Tuple

BASE = "https://api.coingecko.com/api/v3"

# Mapeamento rápido de symbols USDT -> id no Coingecko
SYMBOL_TO_ID: Dict[str, str] = {
    "BTCUSDT": "bitcoin",
    "ETHUSDT": "ethereum",
    "BNBUSDT": "binancecoin",
    "XRPUSDT": "ripple",
    "DOGEUSDT": "dogecoin",
    "SOLUSDT": "solana",
    "MATICUSDT": "matic-network",
    "DOTUSDT": "polkadot",
    "LTCUSDT": "litecoin",
    "LINKUSDT": "chainlink",
    "ADAUSDT":  "cardano",
    "TRXUSDT":  "tron",
    "SHIBUSDT": "shiba-inu",
    "AVAXUSDT": "avalanche-2",
    "TONUSDT":  "toncoin",
    "NEARUSDT": "near",
    "APTUSDT":  "aptos",
    "OPUSDT":   "optimism",
    "ARBUSDT":  "arbitrum",
    "ATOMUSDT": "cosmos",
    "SUIUSDT":  "sui",
    "PEPEUSDT": "pepe",
    "FTMUSDT":  "fantom",
    "ICPUSDT":  "internet-computer",
    "ETCUSDT":  "ethereum-classic",
}

def _get_json(url: str, params: dict | None = None, retries: int = 6, base_delay: float = 0.8):
    """GET com backoff para lidar com 429/5xx."""
    for i in range(1, retries + 1):
        try:
            r = requests.get(url, params=params, timeout=20)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 429:
                # respeita "Retry-After" se vier
                ra = r.headers.get("Retry-After")
                wait = float(ra) if ra else base_delay * (1.6 ** (i - 1))
                print(f"⚠️ 429 {url} — aguardando {wait:.1f}s (tentativa {i}/{retries})")
                time.sleep(wait)
                continue
            # outros 5xx: backoff
            if 500 <= r.status_code < 600:
                wait = base_delay * (1.6 ** (i - 1))
                print(f"⚠️ {r.status_code} {url} — aguardando {wait:.1f}s (tentativa {i}/{retries})")
                time.sleep(wait)
                continue
            r.raise_for_status()
        except requests.RequestException as e:
            wait = base_delay * (1.6 ** (i - 1))
            print(f"⚠️ erro de rede {e} — aguardando {wait:.1f}s (tentativa {i}/{retries})")
            time.sleep(wait)
    raise RuntimeError(f"Falha ao obter {url} depois de {retries} tentativas")

def chunked(lst: List[str], n: int) -> List[List[str]]:
    return [lst[i:i+n] for i in range(0, len(lst), n)]

def fetch_bulk_prices(symbols: List[str]) -> Dict[str, Dict]:
    """
    Retorna dict:
      { "BTCUSDT": {"usd": 110000.0, "usd_24h_change": -2.1, "usd_market_cap": ...}, ... }
    Faz chamadas em lotes (até ~120 ids por request).
    """
    out: Dict[str, Dict] = {}
    ids = []
    sym_by_id: Dict[str, str] = {}
    for sym in symbols:
        cid = SYMBOL_TO_ID.get(sym, sym.replace("USDT", "").lower())
        sym_by_id[cid] = sym
        ids.append(cid)

    # Coingecko suporta muitos ids; usamos lotes de 100 para sobrar margem.
    for group in chunked(ids, 100):
        params = {
            "ids": ",".join(group),
            "vs_currencies": "usd",
            "include_24hr_change": "true",
            "include_market_cap": "true",
        }
        url = f"{BASE}/simple/price"
        js = _get_json(url, params=params)
        for cid, data in (js or {}).items():
            sym = sym_by_id.get(cid)
            if not sym:
                continue
            out[sym] = {
                "usd": float(data.get("usd", 0.0) or 0.0),
                "usd_24h_change": float(data.get("usd_24h_change", 0.0) or 0.0),
                "usd_market_cap": float(data.get("usd_market_cap", 0.0) or 0.0),
            }
    return out

def fetch_ohlc(coin_id: str, days: int = 30, vs: str = "usd") -> List[List[float]]:
    """
    OHLC no formato [[ts, open, high, low, close], ...]
    docs: /coins/{id}/ohlc?vs_currency=usd&days=30
    """
    url = f"{BASE}/coins/{coin_id}/ohlc"
    js = _get_json(url, params={"vs_currency": vs, "days": str(days)})
    # a API já retorna [timestamp, o, h, l, c]
    return [[float(a), float(b), float(c), float(d), float(e)] for a, b, c, d, e in (js or [])]
