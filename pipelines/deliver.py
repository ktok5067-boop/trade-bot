"""Доставка в Telegram: отчёты, алерты, отправка текста/голоса."""
from __future__ import annotations

import html
import logging
import zoneinfo
from datetime import datetime

import requests

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
    """POST к Telegram с ретраями на сетевые ошибки / 429 / 5xx.

    До 3 ретраев с паузами 2/5/10 сек. 429 уважает retry_after от Telegram.
    """
    import time
    delays = (2, 5, 10)
    last_err = None
    for attempt in range(1 + len(delays)):
        try:
            r = requests_session.post(f"{API}/{method}", json=params, timeout=30)
        except (requests.RequestException, OSError) as e:
            last_err = e
        else:
            if r.status_code == 429:
                retry_after = None
                try:
                    retry_after = int(r.json().get("parameters", {}).get("retry_after", 0))
                except Exception:
                    retry_after = None
                last_err = RuntimeError("Telegram 429 (rate limit)")
                if retry_after:
                    # уважаем retry_after, но не растягиваем сверх разумного
                    delays = (retry_after, retry_after, retry_after)
            elif r.status_code >= 500:
                last_err = RuntimeError(f"Telegram {method}: HTTP {r.status_code}")
            else:
                try:
                    d = r.json()
                except Exception as e:
                    last_err = e
                else:
                    if not d.get("ok"):
                        raise RuntimeError(f"Telegram {method}: {d.get('description')}")
                    return d
        # дошёл сюда — была ошибка, ретраим если остались попытки
        if attempt >= len(delays):
            break
        log.warning("Telegram %s: %s — retry %d in %ss",
                    method, last_err, attempt + 1, delays[attempt])
        time.sleep(delays[attempt])
    raise RuntimeError(f"Telegram {method} failed after retries: {last_err}")


def _escape(s: str) -> str:
    return html.escape(str(s), quote=False)


def send_text(text: str, chat_id: str | int | None = None, try_html: bool = True,
              reply_markup: dict | None = None):
    chat_id = chat_id or REPORT_CHAT_ID
    if not chat_id:
        log.warning("Не задан REPORT_CHAT_ID/TELEGRAM_ALLOWED_IDS — некому слать отчёт")
        return
    kwargs = {"chat_id": chat_id, "text": text, "disable_web_page_preview": True}
    if reply_markup:
        kwargs["reply_markup"] = reply_markup
    if try_html:
        try:
            tg("sendMessage", parse_mode="HTML", **kwargs)
            return
        except RuntimeError as e:
            log.warning("HTML-отправка не прошла (%s) — шлю как plain text", e)
    try:
        tg("sendMessage", text=text.replace("<", "["), **{k: v for k, v in kwargs.items() if k != "text"})
    except RuntimeError as e:
        log.warning("plain-отправка не прошла: %s", e)


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
    """Качество сигналов: только старше 4 часов, с учётом издержек.

    Цена входа берётся из payload (price_at_signal) — работает и после деплоя,
    когда эфемерный кеш котировок пуст.
    """
    from datetime import datetime, timezone
    from .settings import CONFIG
    fee = float((CONFIG.get("intraday") or {}).get("fee_pct", 1.0))
    rows = [s for s in db.signals_since(hours=24) if s["kind"] in ("brief", "evening")]
    checked, hit, in_plus = 0, 0, 0
    total_net = 0.0
    lines = []
    now_utc = datetime.now(timezone.utc)
    for row in rows:
        try:
            created = datetime.fromisoformat(str(row["created_at"]).replace("Z", "+00:00"))
            if created.tzinfo is None:
                created = created.replace(tzinfo=timezone.utc)
        except (ValueError, TypeError):
            continue
        if (now_utc - created).total_seconds() < 4 * 3600:
            continue  # сигналу меньше 4 часов — судить рано
        for s in row["payload"]:
            if s.get("direction") not in ("long", "short"):
                continue
            price_then = s.get("price_at_signal") or db.price_at(s["ticker"], row["created_at"])
            price_now = db.latest_prices().get(s["ticker"], {}).get("last") or \
                (db.quote_stats(s["ticker"]) or {}).get("last")
            if not price_then or not price_now:
                continue
            chg = (price_now / price_then - 1) * 100
            net = chg - fee if s["direction"] == "long" else -chg - fee
            ok = chg > 0 if s["direction"] == "long" else chg < 0
            checked += 1
            hit += ok
            in_plus += net > 0
            total_net += net
            if len(lines) < 10:
                lines.append(f"{s['ticker']} {s['direction']}: {chg:+.2f}% "
                             f"(после издержек {net:+.2f}%) — {'✅' if ok else '❌'}")
    if not checked:
        return ""
    rate = hit / checked * 100
    return (f"\n\n📊 <b>Качество сигналов (24ч, старше 4ч):</b> {hit}/{checked} верно ({rate:.0f}%)\n"
            f"После издержек {fee}% в плюсе: {in_plus}/{checked} · суммарно {total_net:+.2f}%\n"
            + "\n".join(_escape(l) for l in lines))


def paper_summary() -> str:
    """Сводка бумажного эксперимента: для вечернего отчёта и команды /paper."""
    from datetime import datetime, timezone
    from . import store
    try:
        stats = store.trade_stats(365, mode="paper")
        pos = store.active_positions(mode="paper")
        start = store.get_meta("paper_experiment_start", "")
    except Exception as e:
        log.warning("paper_summary: %s", e)
        return ""
    days = 0
    start_line = ""
    if start:
        try:
            dt = datetime.fromisoformat(start.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            days = (datetime.now(timezone.utc) - dt).days
            start_line = f"Старт: {dt.strftime('%d.%m.%Y')} · дней в эксперименте: {days}\n"
        except ValueError:
            pass
    wr = f"{stats['win_rate']}%" if stats["win_rate"] is not None else "—"
    avg = f"{stats['avg_pnl_pct']:+.2f}%" if stats["avg_pnl_pct"] is not None else "—"
    open_lines = "\n".join(
        f"• {p['ticker']}: вход {float(p['entry_price']):g} · стоп {float(p['stop']):g} · "
        f"цель {float(p['target']):g}" for p in pos) or "• открытых нет"
    note = ""
    if stats["closed"] < 30:
        note = ("\n<i>Сделок пока мало — винрейт без большой выборки не показатель, "
                "наблюдаем дальше.</i>")
    return (f"🧪 <b>Бумажная торговля</b> (вход только по вердикту «enter», "
            f"выход: цель/стоп/время ~4 торговых дня)\n"
            f"{start_line}\n"
            f"Открыто сейчас ({len(pos)}):\n{open_lines}\n\n"
            f"Закрыто: {stats['closed']} · в плюс: {stats['wins']} · в минус: {stats['losses']}\n"
            f"Винрейт: {wr} · средняя сделка: {avg}\n"
            f"Итог (с издержками ~{stats['avg_fee_pct'] or 0}%): <b>{stats['total_pnl_pct']:+.2f}%</b> · "
            f"макс. просадка: {stats['max_drawdown_pct']:.2f}%{note}")


def send_voice_ogg(ogg_bytes: bytes, chat_id: str | int):
    tg("sendVoice", chat_id=chat_id,
       voice=("reply.ogg", ogg_bytes, "audio/ogg"))
