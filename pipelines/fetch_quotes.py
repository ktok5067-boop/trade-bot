"""Котировки Мосбиржи через официальный MOEX ISS API (бесплатный, без ключа).

Документация: https://www.moex.com/a2196 (ISS API Reference)
"""
from __future__ import annotations

import logging
from datetime import date, timedelta

from . import db
from .http import get_json
from .settings import CONFIG, MSK
from zoneinfo import ZoneInfo

log = logging.getLogger("quotes")

BOARD = "TQBR"          # основной режим торгов акциями
SEC_BASE = ("https://iss.moex.com/iss/engines/stock/markets/shares"
            f"/boards/{BOARD}/securities")
IDX_URL = ("https://iss.moex.com/iss/engines/stock/markets/index"
           "/boards/SNDX/securities.json")


def watchlist() -> list[dict]:
    return CONFIG["watchlist"]


def tickers() -> list[str]:
    return [w["ticker"] for w in watchlist()]


def fetch_snapshot() -> list[dict]:
    """Текущий срез marketdata по watchlist. Сохраняет в БД, возвращает строки."""
    secs = ",".join(tickers())
    d = get_json(SEC_BASE + ".json", {
        "iss.meta": "off", "iss.only": "securities,marketdata", "securities": secs,
    })
    scols = d["securities"]["columns"]
    mcols = d["marketdata"]["columns"]
    smap = {r[0]: dict(zip(scols, r)) for r in d["securities"]["data"]}
    names = {w["ticker"]: w.get("name", w["ticker"]) for w in watchlist()}

    rows = []
    ts = db.now_iso()
    for r in d["marketdata"]["data"]:
        m = dict(zip(mcols, r))
        secid = m["SECID"]
        s = smap.get(secid, {})
        last = m.get("LAST") or m.get("CLOSEPRICE") or s.get("PREVPRICE")
        if last is None:
            continue
        rows.append({
            "secid": secid, "ts": ts,
            "shortname": s.get("SHORTNAME") or names.get(secid, secid),
            "last": last,
            "open": m.get("OPEN"), "high": m.get("HIGH"), "low": m.get("LOW"),
            "prev": s.get("PREVPRICE"),
            "chg_pct": m.get("LASTTOPREVPRICE"),
            "voltoday": m.get("VOLTODAY") or 0,
        })
    if rows:
        db.save_prices(rows)
        log.info("snapshot: %d тикеров", len(rows))
    return rows


def fetch_index() -> dict:
    """Индекс MOEX (IMOEX) — общий контекст рынка."""
    try:
        d = get_json(IDX_URL, {
            "iss.meta": "off", "iss.only": "marketdata",
            "securities": "IMOEX",
        })
        cols = d["marketdata"]["columns"]
        for r in d["marketdata"]["data"]:
            m = dict(zip(cols, r))
            if m["SECID"] == "IMOEX":
                return {"value": m.get("CURRENTVALUE"),
                        "chg_pct": m.get("LASTCHANGEPRC")}
    except Exception as e:
        log.warning("IMOEX: %s", e)
    return {}


def fetch_candles(secid: str, days: int = 380, interval: int = 24) -> int:
    """Дневные свечи (interval=24) за `days` дней (нужно для 52-недельных уровней)."""
    frm = (date.today() - timedelta(days=days)).isoformat()
    d = get_json(f"{SEC_BASE}/{secid}/candles.json", {
        "iss.meta": "off", "interval": interval, "from": frm,
    })
    cols = d["candles"]["columns"]
    rows = []
    for r in d["candles"]["data"]:
        m = dict(zip(cols, r))
        rows.append({"ts": m["begin"][:10], "open": m["open"], "close": m["close"],
                     "high": m["high"], "low": m["low"], "volume": m["volume"]})
    if rows:
        db.save_candles(secid, rows)
    return len(rows)


def update_all() -> tuple[list[dict], dict]:
    """Снапшот цен + индекс + свечи. Возвращает (prices, index)."""
    prices = fetch_snapshot()
    idx = fetch_index()
    for secid in tickers():
        try:
            n = fetch_candles(secid)
            if n:
                log.info("%s: %d дневных свечей", secid, n)
        except Exception as e:
            log.warning("%s candles: %s", secid, e)
    return prices, idx


if __name__ == "__main__":
    db.init_db()
    prices, idx = update_all()
    for p in prices:
        print(f"{p['secid']:<6} {p['shortname']:<12} last={p['last']} chg={p['chg_pct']}%")
    print("IMOEX:", idx)
