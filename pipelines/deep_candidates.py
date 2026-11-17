"""Кандидаты на глубокий разбор TradingAgents.

Раз в день (после вечернего итога) бот отбирает несколько бумаг с лучшим
сетапом и присылает их списком: «прогони по ним глубокий разбор».

Идея: бот — это широкий недорогой обход всех 33 бумаг, а TradingAgents —
дорогой глубокий разбор единиц. Бот подсказывает, на какие бумаги потратить
время/токены аналитиков.
"""
from __future__ import annotations

import logging

from . import db, intraday
from .settings import CONFIG

log = logging.getLogger("deep_candidates")


def _sent_today() -> bool:
    return db.get_meta("last_deep_candidates_date") == db.now_msk_date()


def build_candidates(top_n: int = 3) -> list[dict]:
    """Топ-N бумаг watchlist по баллу сетапа (без явных «не входить»)."""
    from . import fetch_quotes
    result = intraday.rank_paper(fetch_quotes.tickers(), top_n=top_n * 2)
    scored = result[0] if isinstance(result, tuple) else result  # (scored, idx)
    picked = []
    for row in scored:
        v = row["verdict"]
        if v["call"] == "avoid":
            continue
        picked.append(row)
        if len(picked) >= top_n:
            break
    return picked


def _line(row: dict) -> str:
    v = row["verdict"]
    emoji = {"enter": "✅", "wait": "⏸", "avoid": "⛔"}.get(v["call"], "•")
    return (
        f"{emoji} <b>{row['ticker']}</b> — балл {v['score']}, {v['call']}\n"
        f"    вход {v['entry']:.4g} · стоп {v['stop']:.4g} · цель {v['target']:.4g} ₽\n"
        f"    риск {v['risk_pct']}% · горизонт {v.get('horizon', 'swing')}"
        f" · отмена ниже {v.get('invalidate', v['stop']):.4g}"
    )


def send_candidates(top_n: int = 3):
    """Отправить список кандидатов на глубокий разбор (один раз в день)."""
    if _sent_today():
        return
    cfg = CONFIG.get("intraday", {})
    if not cfg.get("deep_candidates_enabled", True):
        return
    try:
        rows = build_candidates(top_n)
    except Exception as e:
        log.warning("deep candidates: %s", e)
        return
    if not rows:
        return

    tickers = ", ".join(r["ticker"] for r in rows)
    body = "\n\n".join(_line(r) for r in rows)
    text = (
        "Кандидаты на глубокий разбор (бот прошёл весь список)\n"
        "=====================================================\n\n"
        f"{body}\n\n"
        "Запустить разбор (15-30 мин на бумагу):\n"
        + "\n".join(f"  deep_analysis.bat {r['ticker']}" for r in rows)
        + f"\n\nБот показал ЧТО смотреть: {tickers}."
    )
    # Владелец просил не спамить в чат — пишем в файл, в Telegram не шлём.
    # Файл — в data/, рядом с persist-базой; переживает рестарты локально.
    from .settings import DATA_DIR
    out = DATA_DIR / "deep_candidates.txt"
    try:
        out.write_text(text, encoding="utf-8")
    except Exception as e:
        log.warning("deep candidates write: %s", e)
        return
    db.set_meta("last_deep_candidates_date", db.now_msk_date())
    log.info("кандидаты на разбор записаны в %s: %s", out, tickers)
