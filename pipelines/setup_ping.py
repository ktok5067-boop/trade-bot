"""Короткие пинги «посмотри на бумагу» — владелец торгует сам на бирже.

В отличие от ленты интрадей-событий, здесь НЕ сыплем по любому движению.
Пингуем только когда сетап реально сильный: вердикт «вход» либо балл >= порога.
Сообщение — ОДНА строка: тикер, вход, стоп, цель, риск. Без простыней.

Антиспам: один пинг на тикер в день (свинг-сетап живёт часами, повторять
каждый час незачем). Порог и включение — в config.yaml (intraday).
"""
from __future__ import annotations

import logging

from . import db, deliver, intraday
from .settings import CONFIG

log = logging.getLogger("setup_ping")

_SCORE_THRESHOLD = 2.0  # балл, начиная с которого сетап считается «сильным»


def _cfg():
    return CONFIG.get("intraday", {})


def _pinged_today(ticker: str) -> bool:
    return db.get_meta(f"setuppng_{ticker}") == db.now_msk_date()


def _mark_pinged(ticker: str):
    db.set_meta(f"setuppng_{ticker}", db.now_msk_date())


def _short_line(row: dict) -> str:
    v = row["verdict"]
    mark = "✅" if v["call"] == "enter" else "📈"
    risk = v.get("risk_pct")
    risk_s = f" риск {risk}%" if risk is not None else ""
    return (f"{mark} <b>{row['ticker']}</b>: вход {v['entry']:.4g} · "
            f"стоп {v['stop']:.4g} · цель {v['target']:.4g}{risk_s}")


def scan_and_ping():
    """Ранжирует watchlist и шлёт короткий пинг по САМЫМ сильным сетапам.

    Пингуем не всех «enter» подряд (их может набраться десяток — это спам),
    а только топ-N по баллу, которые ещё не пинговали сегодня. Так владелец
    видит 1-3 лучшие бумаги, а не простыню.
    """
    cfg = _cfg()
    if not cfg.get("setup_pings_enabled", True):
        return
    threshold = float(cfg.get("setup_ping_score", _SCORE_THRESHOLD))
    max_per_round = int(cfg.get("setup_ping_max", 3))
    from . import fetch_quotes
    try:
        result = intraday.rank_paper(fetch_quotes.tickers(), top_n=max_per_round * 3)
        scored = result[0] if isinstance(result, tuple) else result
    except Exception as e:
        log.warning("setup ping rank: %s", e)
        return

    lines = []
    for row in scored:
        if len(lines) >= max_per_round:
            break
        v = row["verdict"]
        if v["call"] == "avoid":
            continue
        strong = (v["call"] == "enter") or (v["score"] >= threshold)
        if not strong:
            continue
        if _pinged_today(row["ticker"]):
            continue
        lines.append(row)

    if not lines:
        return

    # Отправляем ОДНИМ сообщением все свежие пинги (не спам по одной).
    if len(lines) == 1:
        text = "📢 <b>Стоит глянуть</b>\n\n" + _short_line(lines[0])
    else:
        text = "📢 <b>Стоит глянуть</b>\n\n" + "\n".join(_short_line(r) for r in lines)

    try:
        deliver.send_text(text)
    except Exception as e:
        log.warning("setup ping send: %s", e)
        return
    for row in lines:
        _mark_pinged(row["ticker"])
    log.info("setup ping: %s", ", ".join(r["ticker"] for r in lines))
