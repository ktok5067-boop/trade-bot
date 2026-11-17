"""Дивидендный календарь: известные даты отсечек → расчёт доходности → алерты.

Данные берутся из data/dividends.json (владелец заполняет за 2 минуты с dohod.ru —
надёжных бесплатных API по дивидендам МОЕХ нет; ISS дивиденды больше не отдаёт).

Формат файла:
[
  {"ticker": "SBER", "amount": 33.3, "regClose": "2026-07-18", "note": "промежуточные"}
]

Логика: для каждой записи с датой отсечки в ближайшие days дней считается
дивидендная доходность по живой цене; если чистый шанс положительный
(доходность > порога), алерт в Telegram с дедупликацией по тикер+дата.
Команда /dividends в боте показывает таблицу грядущих отсечек.
"""
from __future__ import annotations

import json
import logging
from datetime import date, datetime, timedelta

from . import db, fetch_quotes
from .settings import DATA_DIR

log = logging.getLogger("dividends")

DIV_FILE = DATA_DIR / "dividends.json"
EXAMPLE_FILE = DATA_DIR / "dividends.example.json"

EXAMPLE = [
    {"ticker": "SBER", "amount": 33.30, "regClose": "2027-07-18", "note": "пример: заполняется с dohod.ru"},
]


def load_calendar() -> list[dict]:
    """Читает календарь; пустой файл создаётся с примером."""
    if not DIV_FILE.exists():
        try:
            DIV_FILE.write_text(json.dumps(EXAMPLE, ensure_ascii=False, indent=1),
                                encoding="utf-8")
            EXAMPLE_FILE.write_text(json.dumps(EXAMPLE, ensure_ascii=False, indent=1),
                                    encoding="utf-8")
            log.info("dividends: создан пустой календарь %s", DIV_FILE)
        except Exception as e:
            log.warning("dividends: не смог создать файл: %s", e)
        return []
    try:
        data = json.loads(DIV_FILE.read_text(encoding="utf-8"))
        return [d for d in data if isinstance(d, dict) and d.get("ticker")]
    except Exception as e:
        log.warning("dividends: битый %s: %s", DIV_FILE, e)
        return []


def upcoming(G=None, days: int = 45, min_yield: float = 0.0) -> list[dict]:
    """Грядущие отсечки по watchlist с расчётом доходности по живой цене.

    Возвращает список: {ticker, amount, regClose, daysLeft, price, yieldPct, note}.
    """
    entries = load_calendar()
    if not entries:
        return []
    today = date.today()
    horizon = today + timedelta(days=days)
    prices = {t: (db.quote_stats(t) or {}).get("last") for t in fetch_quotes.tickers()}
    out = []
    for e in entries:
        try:
            rc = datetime.strptime(str(e["regClose"])[:10], "%Y-%m-%d").date()
        except (KeyError, ValueError):
            continue
        if not (today <= rc <= horizon):
            continue
        ticker = str(e["ticker"]).upper()
        price = prices.get(ticker)
        amount = float(e.get("amount") or 0)
        if not price or price <= 0 or amount <= 0:
            continue
        out.append({
            "ticker": ticker,
            "amount": amount,
            "regClose": rc.isoformat(),
            "daysLeft": (rc - today).days,
            "price": price,
            "yieldPct": round(amount / price * 100, 2),
            "note": str(e.get("note") or "")[:60],
        })
    out.sort(key=lambda x: x["yieldPct"], reverse=True)
    if min_yield > 0:
        out = [x for x in out if x["yieldPct"] >= min_yield]
    return out


def _alert_key(e: dict) -> str:
    return f"div_alert_{e['ticker']}_{e['regClose']}"


def check_and_alert() -> int:
    """Раз в сутки: алерты по отсечкам в ближайшие 14 дней с доходностью >= 4%.
    Возвращает число отправленных алертов. Дедуп — meta по тикер+дата."""
    from .settings import CONFIG
    cfg = (CONFIG.get("dividends") or {})
    if not cfg.get("alerts", True):
        return 0
    days = int(cfg.get("alert_days", 14))
    min_yield = float(cfg.get("min_yield", 4.0))
    sent = 0
    for e in upcoming(days=days, min_yield=min_yield):
        key = _alert_key(e)
        if db.get_meta(key):
            continue
        db.set_meta(key, db.now_iso())
        from . import deliver
        deliver.send_text(
            f"💰 <b>Дивидендная отсечка близко</b>\n"
            f"<b>{e['ticker']}</b>: {e['amount']:.2f} ₽ "
            f"(доходность ~{e['yieldPct']:.1f}%)\n"
            f"Закрытие реестра: <b>{e['regClose']}</b> (через {e['daysLeft']} дн.)\n"
            f"Цена сейчас: {e['price']:.2f} ₽\n"
            f"<i>После отсечки цена обычно падает примерно на размер дивиденда — "
            f"стратегия работает только если падение меньше дивиденда. "
            f"Не ИИР.</i>")
        sent += 1
        log.info("dividends: алерт %s %s (%.1f%%)", e["ticker"], e["regClose"], e["yieldPct"])
    return sent


def format_table() -> str:
    """Таблица для команды /dividends."""
    rows = upcoming(days=60)
    if not rows:
        if not load_calendar():
            return ("📅 Дивидендный календарь пуст.\n"
                    "Заполни <b>data/dividends.json</b> (2 минуты с dohod.ru → раздел «Дивиденды»):\n"
                    "<code>[{\"ticker\": \"SBER\", \"amount\": 33.3, \"regClose\": \"2027-07-18\"}]</code>\n"
                    "Бот сам посчитает доходность и напомнит об отсечке.")
        return "📅 Грядущих отсечек в ближайшие 60 дней нет."
    lines = ["📅 <b>Грядущие дивидендные отсечки</b>"]
    for e in rows:
        lines.append(f"• <b>{e['ticker']}</b>: {e['amount']:.2f} ₽ (~{e['yieldPct']:.1f}%) — "
                     f"реестр {e['regClose']} (через {e['daysLeft']} дн.)")
    lines.append("\n<i>Доходность по живой цене. После отсечки цена падает ~на дивиденд. Не ИИР.</i>")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    db.init_db()
    print(json.dumps(upcoming(days=60), ensure_ascii=False, indent=1))
    if "--alert" in sys.argv:
        print("alerts sent:", check_and_alert())
