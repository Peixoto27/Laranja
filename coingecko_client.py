# -*- coding: utf-8 -*-
"""
Cliente CoinGecko com retry/backoff e chamadas em lote para evitar 429.
- fetch_bulk_prices: preços/24h em batches (default 5) com retry e jitter
- fetch_ohlc: OHLC com retry e respeito a Retry-After

Leitura de configs opcionais de config.py:
  API_DELAY_BULK, API_DELAY_OHLC, MAX_RETRIES, BACKOFF_BASE
"""

from __future__ import annotations
import time
import random
import requests
from typing import List, Dict, Any

# ---- Config (com defaults) ----
try:
    from config import API_DELAY_BULK, API_DELAY_OHLC, MAX_RETRIES, BACKOFF_BASE
except Exception:
    API_DELAY_BULK = 1.5
    API_DELAY_OHLC = 1.5
    MAX_RETRIES = 5
    BACKOFF_BASE = 1.8

# ---- Mapa TICKER -> CoinGecko ID ----
SYMBOL_TO_ID: Dict[str, str] = {
    "BTCUSDT": "bitcoin",
    "ETHUSDT": "ethereum",
    "BNBUSDT": "binancecoin",
    "XRPUSDT": "ripple",
    "ADAUSDT": "cardano",
    "DOGEUSDT": "dogecoin",
    "SOLUSDT": "solana",
    "MATICUSDT": "matic-network",
    "DOTUSDT": "polkadot",
    "LTCUSDT": "litecoin",
    "LINKUSDT": "chainlink",
    "TRXUSDT": "tron",
    "FILUSDT": "filecoin",
    "ATOMUSDT": "cosmos",
    "AVAXUSDT": "avalanche-2",
    "XLMUSDT": "stellar",
    "APTUSDT": "aptos",
    "INJUSDT": "injective-protocol",
    "ARBUSDT": "arbitrum",
    "OPUSDT": "optimism",
    "SUIUSDT": "sui",
    "PEPEUSDT": "pepe",
    "SHIBUSDT": "shiba-inu",
    "ETCUSDT": "ethereum-classic",
    "NEARUSDT": "near",
    "AAVEUSDT": "aave",
    "UNIUSDT": "uniswap",
    "XTZUSDT": "tezos",
    "ICPUSDT": "internet-computer",
    "FTMUSDT": "fantom",
    "GRTUSDT": "the-graph",
    "EGLDUSDT": "multiversx",
    "CHZUSDT": "chiliz",
    "ALGOUSDT": "algorand",
    "SANDUSDT": "the-sandbox",
    "MANAUSDT": "decentraland",
    "IMXUSDT": "immutable-x",
    "RUNEUSDT": "thorchain",
    "RNDRUSDT": "render-token",
    "TIAUSDT": "celestia",
    "JUPUSDT": "jupiter-exchange-solana",
    "PYTHUSDT": "pyth-network",
}

API_BASE = "https://api.coingecko.com/api/v3"

session = requests.Session()
session.headers.update({"User-Agent": "CryptonSignals/1.0 (Railway)"})


def _to_cg_id(sym_or_id: str) -> str:
    if not sym_or_id:
        return ""
    s = sym_or_id.strip().upper()
    if s in SYMBOL_TO_ID:
        return SYMBOL_TO_ID[s]
    if s.endswith("USDT") or s.endswith("USDC"):
        s = s[:-4]
    return s.lower()


def fetch_bulk_prices(symbols: List[str], batch_size: int = 5) -> Dict[str, Any]:
    """
    Busca preços e variação 24h para uma lista de símbolos.
    Faz chamadas em lotes de 'batch_size' (default 5) com retry/backoff.
    Retorna dict usando TICKERS originais como chaves.
    """
    out: Dict[str, Any] = {}
    if not symbols:
        return out

    url = f"{API_BASE}/simple/price"

    for i in range(0, len(symbols), batch_size):
        batch = symbols[i:i + batch_size]
        ids = ",".join(_to_cg_id(s) for s in batch)
        params = {"ids": ids, "vs_currencies": "usd", "include_24hr_change": "true"}

        delay = API_DELAY_BULK
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                time.sleep(delay)
                resp = session.get(url, params=params, timeout=20)

                if resp.status_code == 200:
                    data = resp.json()
                    for s in batch:
                        cid = _to_cg_id(s)
                        if cid in data:
                            out[s] = data[cid]
                    time.sleep(API_DELAY_BULK + random.uniform(0.4, 1.2))
                    break

                if resp.status_code == 429:
                    retry_after = resp.headers.get("Retry-After")
                    wait = float(retry_after) if retry_after else delay
                    wait += random.uniform(0.8, 2.0)
                    print(f"⚠️ 429 PRICE ids={ids[:60]}... aguardando {round(wait,1)}s "
                          f"(tentativa {attempt}/{MAX_RETRIES})")
                    time.sleep(wait)
                    delay = min(delay * BACKOFF_BASE, 60.0)
                    continue

                if 500 <= resp.status_code < 600:
                    wait = delay + random.uniform(0.8, 2.0)
                    print(f"⚠️ {resp.status_code} PRICE ids={ids[:60]}... aguardando {round(wait,1)}s")
                    time.sleep(wait)
                    delay = min(delay * BACKOFF_BASE, 60.0)
                    continue

                resp.raise_for_status()

            except requests.RequestException as e:
                wait = delay + random.uniform(0.8, 2.0)
                print(f"⚠️ Erro PRICE ids={ids[:60]}...: {e} "
                      f"(tentativa {attempt}/{MAX_RETRIES}). Aguardando {round(wait,1)}s")
                time.sleep(wait)
                delay = min(delay * BACKOFF_BASE, 60.0)
        else:
            print(f"⛔ Falha PRICE ids={ids[:60]}... após {MAX_RETRIES} tentativas; seguindo.")

    return out


def fetch_ohlc(sym_or_id: str, days: int = 1) -> list:
    """
    Busca OHLC com retry/backoff. Aceita TICKER (ex.: BTCUSDT) ou ID (ex.: bitcoin).
    Retorna lista: [[ts, open, high, low, close], ...]
    """
    coin_id = _to_cg_id(sym_or_id)
    url = f"{API_BASE}/coins/{coin_id}/ohlc"
    params = {"vs_currency": "usd", "days": days}

    delay = API_DELAY_OHLC
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            time.sleep(delay)
            resp = session.get(url, params=params, timeout=25)

            if resp.status_code == 200:
                return resp.json()

            if resp.status_code == 429:
                retry_after = resp.headers.get("Retry-After")
                wait = float(retry_after) if retry_after else delay
                wait += random.uniform(0.8, 2.0)
                print(f"⚠️ 429 OHLC {coin_id}: aguardando {round(wait,1)}s "
                      f"(tentativa {attempt}/{MAX_RETRIES})")
                time.sleep(wait)
                delay = min(delay * BACKOFF_BASE, 60.0)
                continue

            if 500 <= resp.status_code < 600:
                wait = delay + random.uniform(0.8, 2.0)
                print(f"⚠️ {resp.status_code} OHLC {coin_id}: aguardando {round(wait,1)}s")
                time.sleep(wait)
                delay = min(delay * BACKOFF_BASE, 60.0)
                continue

            resp.raise_for_status()

        except requests.RequestException as e:
            wait = delay + random.uniform(0.8, 2.0)
            print(f"⚠️ Erro OHLC {coin_id}: {e} (tentativa {attempt}/{MAX_RETRIES}). "
                  f"Aguardando {round(wait,1)}s")
            time.sleep(wait)
            delay = min(delay * BACKOFF_BASE, 60.0)

    return []
