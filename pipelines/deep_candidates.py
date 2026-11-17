"""Кандидаты на глубокий разбор TradingAgents.

Раз в день (после вечернего итога) бот отбирает несколько бумаг с лучшим
сетапом и присылает их списком: «прогони по ним глубокий разбор».

Идея: бот — это широкий недорогой обход всех 33 бумаг, а TradingAgents —
дорогой глубокий разбор единиц. Бот подсказывает, на какие бумаги потратить
время/токены аналитиков.
"""
from __future__ import annotations

import logging

from . import db, deliver, intraday
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
        "🔎 <b>Кандидаты на глубокий разбор</b>\n"
        "Бот прошёл весь список; эти бумаги сочёл самыми интересными.\n\n"
        f"{body}\n\n"
        "<b>Как получить разбор:</b>\n"
        "Запустите на компьютере:\n"
        + "\n".join(f"<code>deep_analysis.bat {r['ticker']}</code>" for r in rows)
        + f"\n\n💡 Аналитики работают <b>15-30 минут</b> на бумагу и объяснят "
        f"<b>почему</b>: спор быка с медведем, уровни, риски.\n"
        f"Бот показал <b>что</b> смотреть — это набор: {tickers}."
    )
    try:
        deliver.send_text(text)
    except Exception as e:
        log.warning("deep candidates send: %s", e)
        return
    db.set_meta("last_deep_candidates_date", db.now_msk_date())
    log.info("кандидаты на разбор отправлены: %s", tickers)
