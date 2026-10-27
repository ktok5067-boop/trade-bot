"""Главный вход: планировщик задач + запуск Telegram-бота.

Цикл каждые 20 секунд проверяет, что пора делать:
- сбор новостей (RSS + Exa) каждые news_interval_min
- снапшот котировок каждые quotes_interval_min в торговые дни/часы
- срочные алерты по важным новостям (impact >= порога, с антиспамом)
- брифинг / итоги дня по расписанию в локальной зоне владельца (schedule.tz)
- пинг Healthchecks.io
"""
from __future__ import annotations

import logging
import time
import urllib.request
import zoneinfo
from datetime import datetime, timedelta, timezone

from . import analyze, bot, db, deliver, fetch_quotes, fetch_news, llm
from .settings import (CONFIG, HEALTHCHECK_URL, LOCAL_TZ, MSK,
                       TELEGRAM_BOT_TOKEN, setup_logging)

log = logging.getLogger("scheduler")
MSK_TZ = zoneinfo.ZoneInfo(MSK)
LOCAL = zoneinfo.ZoneInfo(LOCAL_TZ)


def now_local() -> datetime:
    """Время в зоне владельца (расписания брифингов)."""
    return datetime.now(LOCAL)


def minutes_since(meta_key: str, default_min: float = 1e9) -> float:
    last = db.get_meta(meta_key)
    if not last:
        return default_min
    dt = datetime.fromisoformat(last)  # хранится с явным смещением
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=LOCAL)
    return (datetime.now(timezone.utc) - dt.astimezone(timezone.utc)).total_seconds() / 60


def touch(meta_key: str):
    db.set_meta(meta_key, now_local().isoformat(timespec="seconds"))


def is_session(now_local_dt: datetime) -> bool:
    """Торговое окно MOEX задаётся по МСК независимо от зоны владельца."""
    cfg = CONFIG["schedule"]
    msk = now_local_dt.astimezone(MSK_TZ)
    start = datetime.strptime(cfg["session_start"], "%H:%M").time()
    end = datetime.strptime(cfg["session_end"], "%H:%M").time()
    return msk.weekday() < 6 and start <= msk.time() <= end


def _parse_hhmm(s: str):
    h, m = s.split(":")
    return int(h), int(m)


def process_alerts():
    cfg = CONFIG["alerts"]
    fresh = db.high_impact_unalerted(cfg["impact_threshold"])
    if not fresh:
        return
    # алертим только новости по нашим тикерам либо события максимальной важности
    fresh = [n for n in fresh if n["tickers"] or n["impact"] >= 4]
    if not fresh:
        return
    # холодный старт: после перезапуска бота (на HF база пустая) накопившиеся
    # новости помечаем без отправки, чтобы не заспамить повторами одних и тех же
    boot = db.get_meta("boot_id")
    if boot and db.get_meta("alerts_marked_boot") != boot:
        db.mark_news_alerted([n["id"] for n in fresh])
        db.set_meta("alerts_marked_boot", boot)
        log.info("холодный старт: %d накопившихся новостей помечены без алертов", len(fresh))
        return
    quota = cfg["max_per_hour"] - db.alerts_last_hour()
    for n in fresh[:max(0, quota)]:
        comment = ""
        if cfg.get("llm_comment"):
            try:
                comment = llm.chat(
                    [{"role": "system", "content":
                        "Одной фразой (до 30 слов): что эта новость значит для указанных "
                        "российских акций? Без рекомендаций покупать/продавать."},
                     {"role": "user", "content":
                        f"Новость: {n['title']} ({n['source']})\nТикеры: {n['tickers']}"}],
                    max_tokens=120, temperature=0.2)
            except Exception as e:
                log.warning("alert comment: %s", e)
        deliver.send_alert(n, comment)
        db.mark_news_alerted([n["id"]])
        log.info("алерт отправлен: %s", n["title"][:80])


def refresh_candles():
    for w in fetch_quotes.watchlist():
        try:
            fetch_quotes.fetch_candles(w["ticker"])
        except Exception as e:
            log.warning("candles %s: %s", w["ticker"], e)


def scheduled_report(kind: str):
    touch(f"last_{kind}_date")
    fetch_quotes.fetch_snapshot()
    refresh_candles()
    signals = analyze.run_analysis(kind)
    deliver.send_report(kind, signals)
    if kind == "evening":
        q = deliver.quality_summary()
        if q:
            deliver.send_text(q)
    log.info("отчёт %s отправлен (%d сигналов)", kind, len(signals))


def healthcheck_ping(ok: bool = True):
    if not HEALTHCHECK_URL:
        return
    try:
        urllib.request.urlopen(HEALTHCHECK_URL + ("" if ok else "/fail"), timeout=10)
    except Exception as e:
        log.warning("healthcheck: %s", e)


def run_cycle():
    cfg = CONFIG["schedule"]
    now = now_local()
    ran = False

    # --- новости (24/7) ---
    if minutes_since("last_news") >= cfg["news_interval_min"]:
        touch("last_news")
        fetch_news.run()
        process_alerts()
        healthcheck_ping()
        ran = True

    # --- котировки (в торговые дни/часы, окно задаётся по МСК) ---
    if is_session(now) and minutes_since("last_quotes") >= cfg["quotes_interval_min"]:
        touch("last_quotes")
        fetch_quotes.fetch_snapshot()
        ran = True

    # --- брифинг (время в зоне владельца) ---
    bh, bm = _parse_hhmm(cfg["brief_time"])
    brief_dt = now.replace(hour=bh, minute=bm, second=0, microsecond=0)
    if (brief_dt <= now <= brief_dt + timedelta(minutes=90)
            and db.get_meta("last_brief_date") != now.strftime("%Y-%m-%d")):
        scheduled_report("brief")
        ran = True

    # --- вечерний итог ---
    eh, em = _parse_hhmm(cfg["evening_time"])
    eve_dt = now.replace(hour=eh, minute=em, second=0, microsecond=0)
    if (eve_dt <= now <= eve_dt + timedelta(minutes=90)
            and db.get_meta("last_evening_date") != now.strftime("%Y-%m-%d")):
        scheduled_report("evening")
        ran = True

    return ran


def start_health_server(port: int):
    """Мини-HTTP для платформ, требующих открытый порт (Hugging Face Spaces: 7860)."""
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"trade-bot is running")

        def log_message(self, *args):
            pass

    server = HTTPServer(("0.0.0.0", port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True,
                     name="health-http").start()
    log.info("health-сервер слушает 0.0.0.0:%d", port)


def main():
    setup_logging()
    db.init_db()
    import uuid
    db.set_meta("boot_id", uuid.uuid4().hex[:12])  # метка текущего запуска
    log.info("=== trade-bot запускается, watchlist: %s ===", fetch_quotes.tickers())

    import os
    if os.environ.get("PORT"):  # Hugging Face Spaces и подобные платформы
        try:
            start_health_server(int(os.environ["PORT"]))
        except OSError as e:
            log.warning("health-сервер не поднялся (порт занят?): %s", e)

    if TELEGRAM_BOT_TOKEN:
        bot.start_polling()
    else:
        log.warning("Telegram-бот не запущен: нет TELEGRAM_BOT_TOKEN "
                    "(сбор данных и анализ всё равно работают)")

    while True:
        try:
            run_cycle()
        except KeyboardInterrupt:
            log.info("остановка")
            break
        except Exception:
            log.exception("cycle failed")
        time.sleep(20)


if __name__ == "__main__":
    main()
