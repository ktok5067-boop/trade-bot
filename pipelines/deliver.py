"""Доставка в Telegram: отчёты, алерты, отправка текста/голоса."""
from __future__ import annotations

import html
import logging
import zoneinfo
from datetime import datetime

from .http import session as requests_session
from . import db, llm
from .settings import LOCAL_TZ, MSK, REPORT_CHAT_ID, SHOW_DISCLAIMER, TELEGRAM_BOT_TOKEN, TZ_LABEL

log = logging.getLogger("deliver")

API = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"

DISCLAIMER = ("\n\n⚠️ <i>Не является индивидуальной инвестиционной рекомендацией. "
              "Сигналы генерируются автоматически и могут быть ошибочными.</i>") \
    if SHOW_DISCLAIMER else ""

CONF_RU = {"low": "низкая", "medium": "средняя", "high": "высокая"}
DIR_EMOJI = {"long": "📈 LONG", "short": "📉 SHORT", "flat": "➖ FLAT"}
KIND_TITLE = {"brief": "🌅 Утренний брифинг", "evening": "🌙 Итоги дня",
              "alert": "⚡ Срочный алерт"}


def tg(method: str, **params) -> dict:
    r = requests_session.post(f"{API}/{method}", json=params, timeout=30)
    d = r.json()
    if not d.get("ok"):
        raise RuntimeError(f"Telegram {method}: {d.get('description')}")
    return d


def _escape(s: str) -> str:
    return html.escape(str(s), quote=False)


def send_text(text: str, chat_id: str | int | None = None, try_html: bool = True):
    chat_id = chat_id or REPORT_CHAT_ID
    if not chat_id:
        log.warning("Не задан REPORT_CHAT_ID/TELEGRAM_ALLOWED_IDS — некому слать отчёт")
        return
    if try_html:
        try:
            tg("sendMessage", chat_id=chat_id, text=text, parse_mode="HTML",
               disable_web_page_preview=True)
            return
        except RuntimeError as e:
            log.warning("HTML-отправка не прошла (%s) — шлю как plain text", e)
    tg("sendMessage", chat_id=chat_id, text=text.replace("<", "["), disable_web_page_preview=True)


def render_signal(s: dict, prices: dict) -> str:
    p = prices.get(s["ticker"], {})
    price = f"{p['last']} ₽ ({p['chg_pct']:+.2f}%)" \
        if p.get("last") and p.get("chg_pct") is not None else f"{p.get('last', 'n/a')} ₽"
    # flat-сигналы компактно: одна строка, без развёрнутых блоков
    if s["direction"] == "flat":
        why = _escape(s["rationale"][:90])
        return (f"\n⚪ <b>{_escape(s['ticker'])}</b> — без сигнала: {why}")
    src = ""
    if s.get("sources"):
        links = " · ".join(f'<a href="{x["url"]}">{_escape(x.get("title", "источник")[:40])}</a>'
                           for x in s["sources"])
        src = f"\n• Источники: {links}"
    tech = f"\n• Техника: {_escape(s['tech_view'])}" if s.get("tech_view") else ""
    return (f"\n<b>{_escape(s['ticker'])}</b> — {DIR_EMOJI.get(s['direction'], s['direction'])} "
            f"(уверенность: {CONF_RU.get(s['confidence'], s['confidence'])}"
            f"{', горизонт: ' + _escape(s['horizon']) if s.get('horizon') else ''})\n"
            f"• Котировка: {price}\n"
            f"• {_escape(s['rationale'])}{tech}\n"
            f"• Риски: {_escape(s['risks'])}{src}")


TG_LIMIT = 4000  # лимит Telegram 4096, с запасом


def send_report(kind: str, signals: list[dict], chat_id: str | int | None = None):
    prices = db.latest_prices()
    title = KIND_TITLE.get(kind, kind)
    date_str = datetime.now(zoneinfo.ZoneInfo(LOCAL_TZ)).strftime("%d.%m.%Y %H:%M ") + TZ_LABEL
    header = f"<b>{title} — {_escape(date_str)}</b>"
    footer = f"{llm.usage_report()}{DISCLAIMER}"

    blocks = [render_signal(s, prices) for s in signals]
    messages: list[str] = []
    cur = header
    for b in blocks:
        if len(cur) + len(b) + 1 > TG_LIMIT:
            messages.append(cur)
            cur = ""
        cur += "\n" + b
    if len(cur) + len(footer) + 1 <= TG_LIMIT:
        cur += "\n" + footer
    else:
        messages.append(cur)
        cur = footer
    messages.append(cur)
    if len(messages) > 1:
        log.info("отчёт длинный — отправляю в %d сообщениях", len(messages))
    for m in messages:
        if m.strip():
            send_text(m, chat_id=chat_id)


def send_alert(news: dict, comment: str = ""):
    tickers = ", ".join(news["tickers"]) or "рынок в целом"
    text = (f"⚡ <b>Алерт (важность {news['impact']}/5)</b>\n"
            f"Затрагивает: <b>{_escape(tickers)}</b>\n"
            f"{_escape(news['title'])}\n"
            f"<a href=\"{news['url']}\">{_escape(news['source'])}</a>")
    if comment:
        text += f"\n\n💬 {_escape(comment)}"
    send_text(text)
    db.save_signal("alert", {"news_id": news.get("id"), "title": news["title"],
                             "tickers": news["tickers"], "comment": comment})


def quality_summary() -> str:
    """Hit-rate: сигналы 4+ часов назад против фактического движения цены."""
    from . import analyze
    rows = [s for s in db.signals_since(hours=24) if s["kind"] in ("brief", "evening")]
    checked, hit = 0, 0
    lines = []
    for row in rows:
        for s in row["payload"]:
            if s["direction"] not in ("long", "short"):
                continue
            created = row["created_at"]
            price_then = db.price_at(s["ticker"], created)
            price_now = (db.quote_stats(s["ticker"]) or {}).get("last") or \
                db.latest_prices().get(s["ticker"], {}).get("last")
            if not price_then or not price_now:
                continue
            chg = price_now / price_then - 1
            ok = (s["direction"] == "long" and chg > 0) or \
                 (s["direction"] == "short" and chg < 0)
            checked += 1
            hit += ok
            if len(lines) < 10:
                lines.append(f"{s['ticker']} {s['direction']}: {chg:+.2%} — {'✅' if ok else '❌'}")
    if not checked:
        return ""
    rate = hit / checked * 100
    return (f"\n\n📊 <b>Качество сигналов (24ч):</b> {hit}/{checked} верно ({rate:.0f}%)\n"
            + "\n".join(_escape(l) for l in lines))


def send_voice_ogg(ogg_bytes: bytes, chat_id: str | int):
    tg("sendVoice", chat_id=chat_id,
       voice=("reply.ogg", ogg_bytes, "audio/ogg"))
