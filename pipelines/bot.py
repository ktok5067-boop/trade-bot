"""Telegram-интерфейс: команды и голосовые сообщения (long polling, отдельный поток)."""
from __future__ import annotations

import logging
import re
import threading

from .http import session as requests_session
from . import analyze, db, deliver, fetch_quotes, llm
from .settings import (_ALLOWED_IDS, DEEPSEEK_API_KEY, EXA_API_KEY, GROQ_API_KEY,
                       GROQ_TTS_MODEL, GROQ_TTS_VOICE, TELEGRAM_BOT_TOKEN,
                       VOICE_REPLIES, setup_logging)

log = logging.getLogger("bot")

API = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"
HELP = """Команды:
/idea SPBE — вердикт: входить или нет (стоп/цель/риск)
/top — что лучше взять на интрадей (3-4 дня)
/pool — пул внимания · /pool add SBER · /pool del SBER
/quote SBER — котировка (по-русски тоже: /quote спб биржа)
/trade SPBE 178.3 — открыть сделку (записывает в трекер)
/close 1 182 — закрыть сделку №1 по цене 182 (посчитает P&L)
/stats — ваша статистика сделок (винрейт, итог)
/paper — бумажная торговля: бот сам входит/выходит и ведёт статистику
/backtest — качество прошлых сигналов с учётом комиссии
/signals — последние сигналы
/brief — анализ сейчас · /status — состояние бота
Голосовое сообщение или просто текст — задайте вопрос."""

# Кнопки-меню (ReplyKeyboardMarkup) — дублируют главные функции
MENU_BUTTONS = [
    ["🎯 Идея", "🏆 Топ 3-4 дня"],
    ["⚡ Пул", "📊 Сигналы"],
    ["📈 Котировка", "💼 Сделки"],
    ["🧪 Бумага", "📉 Бэктест"],
    ["ℹ️ Статус"],
]


def _menu_keyboard() -> dict:
    return {"keyboard": MENU_BUTTONS, "resize_keyboard": True, "one_time_keyboard": False}


def _button_to_command(text: str) -> str | None:
    m = {
        "🎯 Идея": "/idea",
        "🏆 Топ 3-4 дня": "/top",
        "⚡ Пул": "/pool",
        "📊 Сигналы": "/signals",
        "📈 Котировка": "/quote",
        "💼 Сделки": "/stats",
        "🧪 Бумага": "/paper",
        "📉 Бэктест": "/backtest",
        "ℹ️ Статус": "/status",
    }
    return m.get(text)


def _authorized(chat_id: int) -> bool:
    return not _ALLOWED_IDS or chat_id in _ALLOWED_IDS


def _stem(s: str) -> str:
    """Отрезаем окончание: 'спб биржа' -> 'спб бир', чтобы ловить 'спб бирже'."""
    s = s.lower().strip()
    return s[:-2] if len(s) >= 7 else s


def resolve_tickers(question: str) -> list[str]:
    """Какие тикеры упоминаются в вопросе: латиницей, русским именем или ключевым словом.

    Понимает и 'SPBE', и 'спб бирже', и 'сбербанк' в любом падеже.
    """
    q = question.lower()
    q_upper = question.upper()
    found = []
    for w in fetch_quotes.watchlist():
        t = w["ticker"]
        hit = bool(re.search(rf"\b{re.escape(t)}\b", q_upper))
        if not hit:
            name = _stem(w.get("name") or "")
            if name and name in q:
                hit = True
        if not hit:
            for kw in w.get("keywords", []):
                stem = _stem(kw)
                if stem and stem in q:
                    hit = True
                    break
        if hit:
            found.append(t)
    return found


# ---------------- Голос ----------------

def _transcribe_ogg(file_id: str) -> str:
    info = requests_session.get(f"{API}/getFile", params={"file_id": file_id}, timeout=30).json()
    path = info["result"]["file_path"]
    data = requests_session.get(f"{API}/file/bot{TELEGRAM_BOT_TOKEN}/{path}", timeout=60).content
    r = requests_session.post(
        "https://api.groq.com/openai/v1/audio/transcriptions",
        headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
        files={"file": ("voice.ogg", data, "audio/ogg")},
        data={"model": "whisper-large-v3", "language": "ru"},
        timeout=120,
    )
    r.raise_for_status()
    return r.json().get("text", "")


def _tts(text: str) -> bytes:
    r = requests_session.post(
        "https://api.groq.com/openai/v1/audio/speech",
        headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
        json={"model": GROQ_TTS_MODEL, "voice": GROQ_TTS_VOICE,
              "input": text[:1500], "response_format": "opus"},
        timeout=120)
    r.raise_for_status()
    return r.content


def _send_voice(chat_id: int, ogg: bytes):
    requests_session.post(f"{API}/sendVoice", data={"chat_id": chat_id},
                  files={"voice": ("reply.ogg", ogg, "audio/ogg")}, timeout=60)


# ---------------- Контекст для ответов ----------------

def _context(question: str = "") -> str:
    lines = []
    wl = fetch_quotes.watchlist()
    tickers = [w["ticker"] for w in wl]
    names = {w["ticker"]: w.get("name", "") for w in wl}

    # полный список отслеживания — чтобы модель не выдумывала "что отслеживается"
    lines.append(f"WATCHLIST — все отслеживаемые бумаги ({len(tickers)} шт.):")
    lines.append("; ".join(f"{t} ({names[t]})" for t in tickers))
    lines.append("")

    # о какой бумаге конкретно спрашивают — даём подробный блок со свежими данными
    asked = resolve_tickers(question)
    if asked:
        try:
            fetch_quotes.fetch_snapshot()  # свежие котировки, один запрос к MOEX
        except Exception:
            pass
        prices = db.latest_prices()
        for t in asked:
            p = prices.get(t)
            if p:
                lines.append(f"Котировка {t}: {p['last']} ₽ ({p['chg_pct']:+.2f}%) на {p['ts']} МСК")
            else:
                # запасной вариант: последняя цена закрытия из свечей
                candles = db.get_candles(t, limit=1)
                if candles:
                    lines.append(f"Котировка {t} (закрытие {candles[0]['ts']}): "
                                 f"{candles[0]['close']} ₽ — свежего среза нет, биржа закрыта")
                else:
                    lines.append(f"Котировка {t}: данных нет")
            s = db.quote_stats(t) or {}
            if s.get("chg_5d") is not None:
                lines.append(f"{t}: за 5 дней {s['chg_5d']:+.1f}%{s.get('volume', '')}")
            try:
                from . import indicators
                ind = indicators.compute(db.get_candles(t))
                if ind:
                    lines.append("Техника: " + indicators.to_prompt_line(t, ind))
            except Exception:
                pass
            tnews = [n for n in db.recent_news(hours=48, limit=500) if t in n["tickers"]][:5]
            if tnews:
                lines.append(f"Новости {t} за 48ч:")
                lines += [f"- [{n['impact']}] {n['title'][:110]}" for n in tnews]
            else:
                lines.append(f"Новостей по {t} за 48ч нет.")
        lines.append("")

    sigs = [s for row in db.signals_since(hours=24)
            if row["kind"] in ("brief", "evening") for s in row["payload"]]
    if sigs:
        lines.append("Последние сигналы:")
        lines += [f"- {s['ticker']}: {s['direction']} ({s['confidence']}) — {s['rationale'][:120]}"
                  for s in sigs[:10]]
    news = db.recent_news(hours=24, limit=15)
    if news:
        lines.append("Свежие новости:")
        lines += [f"- [{n['impact']}] {n['title'][:100]}" for n in news]
    return "\n".join(lines) or "Данных пока нет."


BOT_SYSTEM = ("Ты — личный торговый ассистент владельца этого бота в Telegram. "
              "Отвечай кратко (до 120 слов) по-русски, опираясь только на контекст данных.\n"
              "ВАЖНО: список отслеживаемых бумаг (WATCHLIST) указан в контексте полностью — "
              "никогда не выдумывай и не сокращай его. Если бумаги из WATCHLIST нет в данных "
              "(нет свежей котировки или сигнала), честно скажи: 'свежих данных по ней пока нет, "
              "брифинг обновит картину в 09:00/22:00 ВЛТ'. Если спрашивают про бумагу НЕ из "
              "WATCHLIST — скажи, что она не отслеживается, и владелец может добавить её в "
              "config.yaml сам (или попросить ассистента).")


def _answer(text: str) -> str:
    try:
        ctx = _context(text)
        return llm.chat([{"role": "system", "content": BOT_SYSTEM},
                         {"role": "user", "content": f"Контекст:\n{ctx}\n\nВопрос: {text}"}],
                        max_tokens=700, temperature=0.3)
    except Exception as e:
        return f"Не смог ответить: {e}"


# ---------------- Обработка апдейтов ----------------

def handle_update(msg: dict):
    chat_id = msg["chat"]["id"]
    if not _authorized(chat_id):
        log.warning("чужой chat_id=%s — игнор", chat_id)
        return

    if "voice" in msg or "audio" in msg:
        fid = msg.get("voice", msg.get("audio", {}))["file_id"]
        if not GROQ_API_KEY:
            deliver.send_text("Голос не настроен (нет GROQ_API_KEY). Напишите текстом.", chat_id)
            return
        try:
            text = _transcribe_ogg(fid)
            log.info("voice → %r", text)
            answer = _answer(text)
            deliver.send_text(f"🎙 Вы: {text}\n\n{answer}", chat_id)
            if VOICE_REPLIES:
                _send_voice(chat_id, _tts(answer))
        except Exception as e:
            log.exception("voice failed")
            deliver.send_text(f"Ошибка обработки голоса: {e}", chat_id)
        return

    text = (msg.get("text") or "").strip()
    if not text:
        return

    # Кнопки меню → команды
    btn = _button_to_command(text)
    if btn:
        text = btn
        if text == "/quote":
            text = "/quote"  # без аргумента — спросим в идее ниже

    if text.startswith("/start") or text.startswith("/help"):
        deliver.send_text(HELP, chat_id, reply_markup=_menu_keyboard())
    elif text.startswith("/status"):
        n = len(db.recent_news(hours=24, limit=1000))
        s = len(db.signals_since(hours=24))
        deliver.send_text(f"Состояние:\n• новостей за 24ч: {n}\n• сигналов за 24ч: {s}\n"
                          f"• {llm.usage_report()}\n"
                          f"• DeepSeek: {'✅' if DEEPSEEK_API_KEY else '❌ нет ключа'}\n"
                          f"• Groq (голос): {'✅' if GROQ_API_KEY else '❌ нет ключа'}\n"
                          f"• Exa: {'✅' if EXA_API_KEY else '➖'}", chat_id)
    elif text.startswith("/watchlist"):
        items = "\n".join(f"• {w['ticker']} — {w.get('name', '')}" for w in fetch_quotes.watchlist())
        deliver.send_text(f"Watchlist:\n{items}", chat_id)
    elif text.startswith("/quote"):
        parts = text.split(maxsplit=1)
        arg = (parts[1] if len(parts) > 1 else "").strip()
        known = {w["ticker"] for w in fetch_quotes.watchlist()}
        t = arg.upper()
        if t not in known and arg:  # попробуем распознать по-русски: "спб биржа", "сбер"…
            resolved = resolve_tickers(arg)
            if resolved:
                t = resolved[0]
        if t in known:
            try:
                fetch_quotes.fetch_snapshot()  # свежий срез, если биржа открыта
            except Exception:
                pass
            prices = db.latest_prices()
            if t in prices:
                p = prices[t]
                chg = f" ({p['chg_pct']:+.2f}%)" if p.get("chg_pct") is not None else ""
                wname = next((w.get("name", "") for w in fetch_quotes.watchlist()
                              if w["ticker"] == t), "")
                deliver.send_text(f"{t} ({wname}): {p['last']} ₽{chg} (на {p['ts']} МСК)",
                                  chat_id)
            else:
                candles = db.get_candles(t, limit=1)
                if candles:
                    deliver.send_text(f"{t}: биржа закрыта, последнее закрытие "
                                      f"({candles[0]['ts']}): {candles[0]['close']} ₽", chat_id)
                else:
                    deliver.send_text(f"{t}: котировок нет. Посмотрите /watchlist", chat_id)
        else:
            deliver.send_text(f"«{arg}» нет в watchlist. Посмотрите /watchlist", chat_id)

    elif text.startswith("/pool"):
        parts = text.split(maxsplit=2)
        sub = (parts[1].lower() if len(parts) > 1 else "")
        if sub in ("add", "del", "remove"):
            arg = (parts[2] if len(parts) > 2 else "").upper().strip()
            if not arg:
                deliver.send_text("Укажите тикер: /pool add SBER", chat_id)
                return
            pool = db.get_pool()
            if sub == "add":
                if arg in pool:
                    deliver.send_text(f"{arg} уже в пуле.", chat_id)
                    return
                pool.append(arg)
                db.set_pool(pool)
                deliver.send_text(f"✅ {arg} добавлен в пул: {', '.join(pool)}", chat_id)
            else:
                if arg not in pool:
                    deliver.send_text(f"{arg} нет в пуле.", chat_id)
                    return
                if len(pool) == 1:
                    deliver.send_text("Пул не может быть пустым — оставьте хотя бы одну бумагу.", chat_id)
                    return
                pool.remove(arg)
                db.set_pool(pool)
                deliver.send_text(f"➖ {arg} убран. Пул: {', '.join(pool)}", chat_id)
        else:
            pool = db.get_pool()
            deliver.send_text(
                "🎯 Пул внимания (интрадей-сканер следит за этими бумагами):\n"
                + "\n".join(f"• {t}" for t in pool)
                + "\n\nДобавить: /pool add SBER · убрать: /pool del SBER", chat_id)

    elif text.startswith("/idea"):
        parts = text.split(maxsplit=1)
        arg = (parts[1] if len(parts) > 1 else "").strip()
        pool = db.get_pool()
        t = arg.upper() if arg.upper() in pool else ""
        if not t and arg:
            resolved = resolve_tickers(arg)
            t = resolved[0] if resolved else ""
        if not t:
            t = pool[0] if pool else "SPBE"
        deliver.send_text(f"Считаю {t}…", chat_id)
        try:
            from . import indicators, intraday
            bars = intraday.fetch_bars(t)
            ind = intraday.analyze_bars(bars)
            if not ind:
                deliver.send_text(f"{t}: интрадей-данных недостаточно (биржа закрыта?).", chat_id)
                return
            daily_ind = None
            try:
                daily_ind = indicators.compute(db.get_candles(t))
            except Exception:
                pass
            idx = fetch_quotes.fetch_index()
            news = db.recent_news(hours=48, limit=60)
            v = intraday.verdict(t, ind, daily_ind, idx=idx, news=news)
            daily_line = indicators.to_prompt_line(t, daily_ind) if daily_ind else ""
            line = intraday.summary_line(t, ind)
            events = intraday.detect_events(t, ind)
            ev_lines = "\n".join(
                f"• {intraday.RUS.get(e['type'], e['type'])} (движение {e['move']:+.2f}%, объём x{e['vol']})"
                for e in events) or "• активных событий нет — рынок спокоен"
            body = (f"🎯 <b>{t}</b>\n{line}\n\n"
                    f"{intraday._verdict_line(v)}\n\n"
                    f"События:\n{ev_lines}")
            if daily_line:
                body += f"\n\nДневная: {daily_line}"
            deliver.send_text(body, chat_id)
        except Exception as e:
            log.exception("idea failed")
            deliver.send_text(f"Ошибка идеи: {e}", chat_id)

    elif text.startswith("/top"):
        deliver.send_text("Сканирую рынок на лучшие сетапы (3-4 дня)… это ~1 минута", chat_id)
        try:
            from . import intraday
            tickers = [w["ticker"] for w in fetch_quotes.watchlist()]
            top = intraday.screener(tickers, top_n=3)
            if not top:
                deliver.send_text("Не удалось получить данные ни по одной бумаге.", chat_id)
                return
            out = ["🏆 <b>Топ сетапов на интрадей (3-4 дня)</b>\n"]
            for i, s in enumerate(top, 1):
                v = s["verdict"]
                out.append(f"{i}. <b>{s['ticker']}</b> — {intraday._verdict_line(v).splitlines()[0]}")
                out.append(f"   {s['ticker']}: {s['ind']['last']:.4g} ₽ · тренд {s['daily_trend']} · "
                           f"балл {v['score']}")
                if v["entry"] and v["stop"] and v["target"]:
                    out.append(f"   вход {v['entry']:.4g} → цель {v['target']:.4g} · стоп {v['stop']:.4g} "
                               f"(риск {v['risk_pct']}%)")
            out.append("\nРазбор картины, не рекомендация.")
            deliver.send_text("\n".join(out), chat_id)
        except Exception as e:
            log.exception("top failed")
            deliver.send_text(f"Ошибка скринера: {e}", chat_id)

    elif text.startswith("/trade"):
        from . import store
        parts = text.split()
        # /trade SPBE 178.3
        if len(parts) < 3:
            deliver.send_text("Формат: /trade SPBE 178.3", chat_id)
            return
        t = parts[1].upper()
        try:
            price = float(parts[2].replace(",", "."))
        except ValueError:
            deliver.send_text("Цена не распознана.", chat_id)
            return
        tid = store.add_trade(t, price)
        deliver.send_text(f"📝 Сделка #{tid}: <b>{t}</b> вход {price:g} ₽ записана.\n"
                          f"Закрыть: /close {tid} ЦЕНА", chat_id)

    elif text.startswith("/close"):
        from . import store
        parts = text.split()
        if len(parts) < 3:
            deliver.send_text("Формат: /close 1 182", chat_id)
            return
        try:
            tid = int(parts[1])
            price = float(parts[2].replace(",", "."))
        except ValueError:
            deliver.send_text("Номер сделки или цена не распознаны.", chat_id)
            return
        r = store.close_trade(tid, price)
        if not r:
            deliver.send_text(f"Сделка #{tid} не найдена.", chat_id)
            return
        emoji = "🟢" if r["pnl_pct"] > 0 else ("🔴" if r["pnl_pct"] < 0 else "⚪")
        deliver.send_text(f"{emoji} Сделка #{r['id']} ({r['ticker']}) закрыта: "
                          f"{r['entry']:g} → {r['exit']:g} ₽\n"
                          f"Итог (с учётом комиссии): <b>{r['pnl_pct']:+.2f}%</b>", chat_id)

    elif text.startswith("/paper"):
        from . import store
        pos = store.active_positions(mode="paper")
        stats = store.trade_stats(30, mode="paper")
        open_lines = "\n".join(
            f"• {p['ticker']}: вход {float(p['entry_price']):g} · стоп {float(p['stop']):g} · "
            f"цель {float(p['target']):g}" for p in pos) or "• открытых нет"
        wr = f"{stats['win_rate']}%" if stats['win_rate'] is not None else "—"
        deliver.send_text(
            f"🧪 <b>Бумажная торговля</b> (бот сам входит по вердиктам и выходит по цели/стопу)\n\n"
            f"Открыто сейчас:\n{open_lines}\n\n"
            f"Результаты (30 дней):\n"
            f"• закрыто: {stats['closed']} · винрейт: {wr}\n"
            f"• итог: <b>{stats['total_pnl_pct']:+.2f}%</b>", chat_id)

    elif text.startswith("/stats"):
        from . import store
        s = store.trade_stats(30, mode="real")
        if not s["closed"]:
            deliver.send_text("Закрытых сделок пока нет. Открывайте через /trade, "
                              "закрывайте через /close.", chat_id)
            return
        wr = f"{s['win_rate']}%" if s["win_rate"] is not None else "—"
        deliver.send_text(
            f"💼 <b>Статистика сделок (30 дней)</b>\n"
            f"• закрыто: {s['closed']}\n"
            f"• в плюс: {s['wins']} · в минус: {s['losses']}\n"
            f"• винрейт: {wr}\n"
            f"• средняя сделка: {s['avg_pnl_pct']:+.2f}%\n"
            f"• итог: <b>{s['total_pnl_pct']:+.2f}%</b>", chat_id)

    elif text.startswith("/backtest"):
        deliver.send_text("Считаю бэктест сигналов…", chat_id)
        try:
            from . import store, intraday
            hist = store.signals_history(30)
            rows = [h for h in hist if h.get("direction") in ("long", "short")
                    and h.get("payload")]
            if not rows:
                deliver.send_text("Истории сигналов пока нет (она копится с бэктест-логом).", chat_id)
                return
            fee = float(intraday._cfg("fee_pct", 1.0))
            checked = wins = 0
            total = 0.0
            for h in rows:
                p = h.get("payload") or {}
                ticker = p.get("ticker") or h.get("ticker")
                direction = p.get("direction") or h.get("direction")
                entry = db.price_at(ticker, h["created_at"])
                now_p = (db.quote_stats(ticker) or {}).get("last")
                if not entry or not now_p:
                    continue
                chg = (now_p / entry - 1) * 100
                net = chg - fee if direction == "long" else -chg - fee
                checked += 1
                total += net
                if net > 0:
                    wins += 1
            if not checked:
                deliver.send_text("Не хватает цен для бэктеста.", chat_id)
                return
            wr = wins / checked * 100
            deliver.send_text(
                f"📉 <b>Бэктест сигналов (30 дней, комиссия {fee}%)</b>\n"
                f"• проверено: {checked}\n"
                f"• в плюс: {wins} ({wr:.0f}%)\n"
                f"• суммарно: {total:+.2f}%\n"
                f"• в среднем: {total / checked:+.2f}% за сделку", chat_id)
        except Exception as e:
            log.exception("backtest failed")
            deliver.send_text(f"Ошибка бэктеста: {e}", chat_id)

    elif text.startswith("/signals"):
        rows = db.signals_since(hours=24)
        sigs = [s for row in rows if row["kind"] in ("brief", "evening") for s in row["payload"]]
        if not sigs:
            deliver.send_text("Сигналов пока нет — ждите брифинг или пришлите /brief", chat_id)
        else:
            prices = db.latest_prices()
            blocks = [deliver.render_signal(s, prices) for s in sigs]
            msgs, cur = [], ""
            for b in blocks:
                if len(cur) + len(b) + 1 > deliver.TG_LIMIT:
                    msgs.append(cur)
                    cur = ""
                cur += ("\n" if cur else "") + b
            if cur:
                msgs.append(cur)
            # дисклеймер всегда в последнем чанке: если не влезает — отдельным сообщением
            if len(msgs[-1]) + len(deliver.DISCLAIMER) <= deliver.TG_LIMIT:
                msgs[-1] = msgs[-1] + deliver.DISCLAIMER
            else:
                msgs.append(deliver.DISCLAIMER.lstrip("\n"))
            if len(msgs) > 1:
                deliver.send_text(f"Сигналов за 24ч много — шлю в {len(msgs)} частях", chat_id)
            for m in msgs:
                deliver.send_text(m, chat_id=chat_id)
    elif text.startswith("/brief"):
        deliver.send_text("Запускаю анализ…", chat_id)
        try:
            fetch_quotes.fetch_snapshot()
            signals = analyze.run_analysis("brief")
            deliver.send_report("brief", signals, chat_id=chat_id)
        except Exception as e:
            log.exception("brief failed")
            deliver.send_text(f"Ошибка анализа: {e}", chat_id)
    else:
        deliver.send_text(_answer(text), chat_id)


# ---------------- Long polling ----------------

_offset = 0


def _poll_loop():
    global _offset
    log.info("polling запущен")
    while True:
        try:
            r = requests_session.get(f"{API}/getUpdates",
                             params={"timeout": 25, "offset": _offset,
                                     "allowed_updates": ["message"]},
                             timeout=40)
            for u in r.json().get("result", []):
                _offset = u["update_id"] + 1
                msg = u.get("message")
                if msg:
                    try:
                        handle_update(msg)
                    except Exception:
                        log.exception("update failed")
        except Exception as e:
            log.warning("poll: %s", e)
            import time
            time.sleep(5)


def start_polling():
    if not TELEGRAM_BOT_TOKEN:
        log.warning("TELEGRAM_BOT_TOKEN не задан — бот выключен")
        return
    threading.Thread(target=_poll_loop, daemon=True, name="tg-poll").start()
