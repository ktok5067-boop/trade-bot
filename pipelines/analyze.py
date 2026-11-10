"""Аналитический пайплайн: новости + котировки + теханализ -> LLM -> сигналы."""
from __future__ import annotations

import logging

from . import db, fetch_quotes, indicators, llm
from .settings import CONFIG

log = logging.getLogger("analyze")

SYSTEM_PROMPT = """Ты — опытный аналитик российского фондового рынка (Мосбиржа).
Тебе дают по каждому тикеру: свежие новости и техническую картину (тренд, RSI,
MACD, SMA20/50, Боллинджер, ATR, диапазон 52 недель, объёмы), плюс индекс IMOEX.

Задача: взвесить НОВОСТНОЙ и ТЕХНИЧЕСКИЙ факторы и выдать по каждому тикеру сигнал.
Торговля только СПОТ (покупка акций): шорт недоступен.

Правила:
1. direction: "long" (рост вероятен, есть смысл покупать) или "flat" (нет ясного
   сигнала / покупать не стоит). "short" НЕ используй — на споте его не исполнить.
2. confidence: "low" | "medium" | "high".
   high — только когда новостной драйвер И техника смотрят в одну сторону;
   расхождение факторов — не выше medium; противоречие — low или flat.
   Учитывай блок "Качество прошлых сигналов": при слабом hit-rate занижай уверенность.
3. horizon: например "1-2 дня", "до конца недели", "1-3 месяца".
4. tech_view: 1 предложение — что говорит техника (тренд, RSI, уровни 52 недель).
5. rationale: 1-2 предложения: как новости сошлись (или разошлись) с техникой.
   НИКОГДА не выдумывай факты и источники — используй только предоставленные данные.
6. risks: главная угроза для идеи, 1 предложение (отмечай перекупленность/перепроданность
   и близость к 52-недельным экстремумам, если это существенно).
7. sources: до 3 ссылок ТОЛЬКО из предоставленного списка новостей.

Ответ — строго JSON:
{"signals": [{"ticker": "SBER", "direction": "long|flat",
  "confidence": "low|medium|high", "horizon": "...", "tech_view": "...",
  "rationale": "...", "risks": "...", "sources": [{"title": "...", "url": "..."}]}]}
Все тикеры из списка обязательны."""


def _news_block(news: list[dict], ticker: str, max_n: int) -> str:
    matched = [n for n in news if ticker in n["tickers"]]
    other = [n for n in news if n["impact"] >= 4 and ticker not in n["tickers"]][:3]
    lines = [f"- [{n['impact']}] {n['title']} ({n['source']}) {n['url']}"
             for n in matched[:max_n]]
    if other:
        lines.append("Фон рынка (важные новости без прямой привязки):")
        lines += [f"- [{n['impact']}] {n['title']} ({n['source']})" for n in other]
    return "\n".join(lines) if lines else "- релевантных новостей нет"


def _tech_block(pool: list[str]) -> list[str]:
    """Строки техкартины по пулу внимания."""
    lines = []
    for t in pool:
        candles = db.get_candles(t)
        if not candles:  # после деплоя кеш пуст — догружаем дневные свечи
            try:
                fetch_quotes.fetch_candles(t, days=120)
                candles = db.get_candles(t)
            except Exception:
                pass
        try:
            ind = indicators.compute(candles)
        except Exception:
            ind = None
        lines.append(indicators.to_prompt_line(t, ind))
    return lines


def _performance_line() -> str:
    """Hit-rate направленных сигналов за 7 дней — для калибровки уверенности."""
    checked, hit = 0, 0
    for row in db.signals_since(hours=24 * 7):
        if row["kind"] not in ("brief", "evening"):
            continue
        for s in row["payload"]:
            if s["direction"] != "long":
                continue
            # цена на момент сигнала сохранена в payload — не зависит от эфемерных котировок
            price_then = s.get("price_at_signal") or db.price_at(s["ticker"], row["created_at"])
            price_now = (db.quote_stats(s["ticker"]) or {}).get("last")
            if not price_then or not price_now:
                continue
            chg = price_now / price_then - 1
            checked += 1
            hit += chg > 0
    if not checked:
        return "Качество прошлых сигналов: данных пока нет."
    rate = hit / checked
    advice = ("Калибруй уверенность консервативно." if rate < 0.5
              else "Уверенность обосновывай осторожно: рынок переменчив.")
    return f"Качество прошлых сигналов (7 дней): {hit}/{checked} верных ({rate * 100:.0f}%). {advice}"


def build_user_prompt(news: list[dict], prices: dict, idx: dict, pool: list[str]) -> str:
    names = {w["ticker"]: w.get("name", t) for w in fetch_quotes.watchlist()}
    parts = [f"Дата анализа (МСК): {db.now_iso()}",
             f"Индекс IMOEX: {idx.get('value', 'n/a')} ({idx.get('chg_pct', 'n/a')}%)",
             _performance_line(),
             "", "== ТЕХНИЧЕСКАЯ КАРТИНА =="]
    parts += _tech_block(pool)
    parts += ["", "== НОВОСТИ И КОНТЕКСТ =="]
    for t in pool:
        p = prices.get(t, {})
        price_line = f"цена сейчас: {p.get('last', 'n/a')} ₽ (за день {p.get('chg_pct', 'n/a')}%)"
        parts.append(f"### {t} ({names.get(t, t)})\nКотировки: {price_line}")
        parts.append(_news_block(news, t, CONFIG["llm"]["max_news_per_ticker"]))
        parts.append("")
    return "\n".join(parts)


def _validate(signals: list, allowed: set[str]) -> list[dict]:
    out = []
    for s in signals:
        if not isinstance(s, dict) or s.get("ticker") not in allowed:
            continue
        direction = str(s.get("direction", "flat")).lower()
        conf = str(s.get("confidence", "low")).lower()
        if direction not in ("long", "flat"):
            direction = "flat"  # спот: шорт неисполним — переводим в flat
        if conf not in ("low", "medium", "high"):
            conf = "low"
        sources = [x for x in (s.get("sources") or [])[:3]
                   if isinstance(x, dict) and x.get("url")]
        out.append({
            "ticker": s["ticker"],
            "direction": direction,
            "confidence": conf,
            "horizon": str(s.get("horizon", ""))[:60],
            "tech_view": str(s.get("tech_view", ""))[:200],
            "rationale": str(s.get("rationale", ""))[:500],
            "risks": str(s.get("risks", ""))[:300],
            "sources": sources,
        })
    return out


def _fallback_signals(prices: dict, pool: list[str]) -> list[dict]:
    """Если LLM недоступен — сигналы чисто по теханализу (спот: long/flat)."""
    out = []
    for t in pool:
        candles = db.get_candles(t)
        try:
            ind = indicators.compute(candles)
        except Exception:
            ind = None
        if ind:
            direction = "flat"
            if ind["trend"] == "восходящий" and (ind["rsi14"] or 50) < 70:
                direction = "long"
            rationale = (f"LLM недоступен; техника: {ind['trend']} тренд, "
                         f"RSI {ind['rsi14']}, 20д {ind['chg_20d']:+.1f}%"
                         if ind["chg_20d"] is not None else
                         f"LLM недоступен; техника: {ind['trend']} тренд.")
            if ind["trend"] == "нисходящий":
                rationale += " Нисходящий тренд — для спота не подходит, ждём разворота."
            tech = indicators.to_prompt_line(t, ind)
        else:
            direction = "flat"
            rationale = "LLM недоступен, техданных недостаточно."
            tech = ""
        out.append({
            "ticker": t, "direction": direction, "confidence": "low",
            "horizon": "1-3 дня", "tech_view": tech,
            "rationale": rationale,
            "risks": "Анализ новостей не выполнялся.",
            "sources": [],
        })
    return out


def _analyze_batch(news: list[dict], prices: dict, idx: dict, batch: list[str]) -> list[dict]:
    """LLM-анализ одного батча тикеров. При пропуске тикеров — один ретрай."""
    allowed = set(batch)
    user = build_user_prompt(news, prices, idx, batch)
    answer = llm.chat_json(
        [{"role": "system", "content": SYSTEM_PROMPT},
         {"role": "user", "content": user}],
        max_tokens=CONFIG["llm"]["max_tokens"],
        temperature=CONFIG["llm"]["temperature"],
    )
    signals = _validate(answer.get("signals", []), allowed)
    if len(signals) < len(allowed):  # модель пропустила тикеры — один ретрай
        missing = allowed - {s["ticker"] for s in signals}
        fix = llm.chat_json(
            [{"role": "system", "content": SYSTEM_PROMPT},
             {"role": "user", "content": user +
              f"\n\nТы пропустил тикеры: {', '.join(sorted(missing))}. Дай сигналы ТОЛЬКО по ним."}],
            max_tokens=2500, temperature=0.1)
        signals += _validate(fix.get("signals", []), missing)
    return signals


def run_analysis(kind: str) -> list[dict]:
    """Собрать контекст, вызвать LLM батчами по ВСЕМУ watchlist, сохранить сигналы.

    Каждый сигнал несёт price_at_signal и signal_ts — для честного бэктеста
    даже после рестартов (эфемерные котировки SQLite обнуляются при деплое).
    """
    prices = db.latest_prices() or {}
    idx = fetch_quotes.fetch_index()
    news = db.recent_news(hours=CONFIG["llm"]["news_hours"])
    tickers = fetch_quotes.tickers()
    batch_size = max(1, int(CONFIG["llm"].get("batch_size", 8)))

    signals: list[dict] = []
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        part: list[dict] = []
        try:
            part = _analyze_batch(news, prices, idx, batch)
            log.info("LLM батч %d-%d: %d сигналов", i + 1, i + len(batch), len(part))
        except Exception as e:
            log.error("LLM батч %s не удался: %s — fallback на технику", batch, e)
        if len(part) < len(batch):  # недостающие тикеры закрываем техническим fallback
            have = {s["ticker"] for s in part}
            part += [s for s in _fallback_signals(prices, batch) if s["ticker"] not in have]
        signals += part

    # Фиксируем цену и время сигнала в payload — бэктест не зависит от кеша котировок.
    ts = db.now_iso()
    for s in signals:
        p = prices.get(s["ticker"]) or {}
        s["price_at_signal"] = p.get("last")
        s["signal_ts"] = ts

    db.save_signal(kind, signals)
    try:  # дублируем в постоянное хранилище для бэктеста (переживает рестарты)
        from . import store
        store.log_signal(kind, signals)
    except Exception as e:
        log.warning("store.log_signal: %s", e)
    return signals
