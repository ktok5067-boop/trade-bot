"""Интрадей-сканер волатильности: 10-минутные свечи MOEX + эвристики смарт-мани.

Задача — ловить резкие движения и зоны входа для спот-торговли на волатильности.
Никаких внешних библиотек, только MOEX ISS + чистый Python.
"""
from __future__ import annotations

from . import db
from .http import get_json
from .settings import CONFIG

SEC_BASE = ("https://iss.moex.com/iss/engines/stock/markets/shares"
            "/boards/TQBR/securities")


def _cfg(key, default):
    return (CONFIG.get("intraday") or {}).get(key, default)


def fetch_bars(secid: str, interval: int | None = None, days: int = 3) -> list[dict]:
    """10-минутные свечи (новые в конце)."""
    from datetime import date, timedelta
    interval = interval or int(_cfg("interval", 10))
    frm = (date.today() - timedelta(days=days)).isoformat()
    d = get_json(f"{SEC_BASE}/{secid}/candles.json", {
        "iss.meta": "off", "interval": interval, "from": frm,
    })
    cols = d["candles"]["columns"]
    bars = []
    for r in d["candles"]["data"]:
        m = dict(zip(cols, r))
        bars.append({"ts": m["begin"], "open": m["open"], "close": m["close"],
                     "high": m["high"], "low": m["low"], "volume": m["volume"] or 0})
    return bars


def _rsi(closes: list[float], period: int = 14) -> float | None:
    if len(closes) < period + 1:
        return None
    gains, losses = [], []
    for i in range(1, len(closes)):
        d = closes[i] - closes[i - 1]
        gains.append(max(d, 0))
        losses.append(max(-d, 0))
    ag = sum(gains[:period]) / period
    al = sum(losses[:period]) / period
    for i in range(period, len(gains)):
        ag = (ag * (period - 1) + gains[i]) / period
        al = (al * (period - 1) + losses[i]) / period
    if al == 0:
        return 100.0
    return round(100 - 100 / (1 + ag / al), 1)


def analyze_bars(bars: list[dict]) -> dict | None:
    """Сводка по сегодняшним 10-минуткам: VWAP, объёмы, RSI, диапазон."""
    if not bars:
        return None
    today = bars[-1]["ts"][:10]
    tb = [b for b in bars if b["ts"][:10] == today]
    if len(tb) < 5:
        return None
    closes = [b["close"] for b in tb]
    highs = [b["high"] for b in tb]
    lows = [b["low"] for b in tb]
    vols = [b["volume"] for b in tb]
    last = closes[-1]

    # VWAP по сегодняшним барам
    pv = sum(((b["high"] + b["low"] + b["close"]) / 3) * b["volume"] for b in tb)
    v = sum(vols)
    vwap = round(pv / v, 4) if v else None

    vols_hist = [x for x in vols[:-1] if x]
    avg_vol = (sum(vols_hist[-30:]) / len(vols_hist[-30:])) if vols_hist else 0
    last_vol = vols[-1]
    vol_ratio = round(last_vol / avg_vol, 1) if avg_vol else None

    # объём последнего импульса: сумма объёмов последних 3 баров против среднего
    pulse_vol = sum(vols[-3:])
    pulse_ratio = round(pulse_vol / (avg_vol * 3), 1) if avg_vol else None

    hi, lo = max(highs), min(lows)
    rng_pos = round((last - lo) / (hi - lo) * 100, 1) if hi > lo else None
    chg_day = round((last / tb[0]["open"] - 1) * 100, 2) if tb[0]["open"] else None
    move_15m = round((last / closes[-2] - 1) * 100, 2) if len(closes) >= 2 else None
    move_30m = round((last / closes[-4] - 1) * 100, 2) if len(closes) >= 4 else None

    sma5 = round(sum(closes[-5:]) / 5, 4) if len(closes) >= 5 else None
    sma20 = round(sum(closes[-20:]) / 20, 4) if len(closes) >= 20 else None

    return {
        "last": last, "vwap": vwap, "sma5": sma5, "sma20": sma20,
        "rsi14": _rsi(closes), "vol_ratio": vol_ratio, "pulse_ratio": pulse_ratio,
        "day_high": hi, "day_low": lo, "range_pos": rng_pos,
        "chg_day": chg_day, "move_15m": move_15m, "move_30m": move_30m,
        "bars": len(tb),
    }


def detect_events(secid: str, ind: dict) -> list[dict]:
    """Детектор событий: скачки, объёмные импульсы, пробои, экстремумы RSI."""
    events = []
    spike_pct = float(_cfg("spike_pct", 1.2))
    vol_mult = float(_cfg("vol_mult", 2.0))
    rsi_low = float(_cfg("rsi_low", 25))
    rsi_high = float(_cfg("rsi_high", 75))

    m15 = ind["move_15m"] or 0
    m30 = ind["move_30m"] or 0
    move = m15 if abs(m15) >= abs(m30) / 2 else m30
    pr = ind["pulse_ratio"] or 0

    # 1. Резкий скачок ценой + объёмом (смарт-мани импульс)
    if abs(move) >= spike_pct and pr >= vol_mult:
        events.append({
            "type": "SPIKE_UP" if move > 0 else "SPIKE_DOWN",
            "move": move, "vol": pr,
            "power": abs(move) * pr,
        })

    # 2. Пробой диапазона дня с объёмом
    if ind["range_pos"] is not None and ind["range_pos"] >= 99 and pr >= 1.5:
        events.append({"type": "BREAK_HIGH", "move": ind["chg_day"], "vol": pr,
                       "power": (ind["move_15m"] or 0) * pr})
    if ind["range_pos"] is not None and ind["range_pos"] <= 1 and pr >= 1.5:
        events.append({"type": "BREAK_LOW", "move": ind["chg_day"], "vol": pr,
                       "power": abs(ind["move_15m"] or 0) * pr})

    # 3. Экстремум RSI на интрадее — откат/отскок
    if ind["rsi14"] is not None:
        if ind["rsi14"] <= rsi_low:
            events.append({"type": "RSI_LOW", "move": m15, "vol": pr, "power": rsi_low - ind["rsi14"]})
        elif ind["rsi14"] >= rsi_high:
            events.append({"type": "RSI_HIGH", "move": m15, "vol": pr, "power": ind["rsi14"] - rsi_high})

    # 4. Чистый объёмный всплеск без большого движения (набор позиции тихим айсбергом)
    if (ind["vol_ratio"] or 0) >= vol_mult + 0.5 and abs(m15) < spike_pct / 2:
        events.append({"type": "ABSORPTION", "move": m15, "vol": ind["vol_ratio"],
                       "power": (ind["vol_ratio"] or 0)})

    events.sort(key=lambda e: -e["power"])
    return events


RUS = {
    "SPIKE_UP": "🚀 РЕЗКИЙ СКАЧОК ВВЕРХ",
    "SPIKE_DOWN": "📉 РЕЗКИЙ СКАЧОК ВНИЗ",
    "BREAK_HIGH": "🔥 ПРОБОЙ МАКСИМУМА ДНЯ",
    "BREAK_LOW": "💧 ПРОБОЙ МИНИМУМА ДНЯ",
    "RSI_LOW": "🩸 ПЕРЕПРОДАННОСТЬ (интрадей RSI)",
    "RSI_HIGH": "♨️ ПЕРЕКУПЛЕННОСТЬ (интрадей RSI)",
    "ABSORPTION": "🐘 ОБЪЁМ БЕЗ ДВИЖЕНИЯ (похоже на набор позиции)",
}


def _idea(secid: str, ev: dict, ind: dict) -> str:
    """Эвристическая спот-идея по событию. Не рекомендация, а разбор картины."""
    last, vwap = ind["last"], ind["vwap"]
    hi, lo = ind["day_high"], ind["day_low"]
    rsi = ind["rsi14"]
    above_vwap = vwap and last > vwap

    if ev["type"] in ("SPIKE_UP", "BREAK_HIGH"):
        if rsi and rsi >= 75 and ind["range_pos"] is not None and ind["range_pos"] >= 97:
            return ("импульс на объёме, но цена у потолка дня и RSI перегрет — "
                    f"риск климакса. Спот: не догонять; вход только на откате к "
                    f"{'VWAP ' + format(vwap, '.4g') if vwap else 'SMA20'}, "
                    f"стоп под минимум импульса.")
        return (f"сильный импульс с объёмом x{ev['vol']} — вероятное продолжение. "
                f"Спот: вход на откате к VWAP {vwap:.4g}₽" + ("" if above_vwap else " (цена ниже VWAP — осторожнее)")
                + f"; цель — продолжение тренда, стоп под {lo:.4g}₽ (минимум дня).")
    if ev["type"] in ("SPIKE_DOWN", "BREAK_LOW"):
        if rsi and rsi <= 30:
            return (f"слив с объёмом x{ev['vol']} и RSI {rsi} — возможна капитуляция. "
                    f"Спот: ждать знак разворота (бар с длинной тенью/объём на отскок), "
                    f"зона интереса у {lo:.4g}₽; стоп под минимум.")
        return (f"давление продаж x{ev['vol']}. Спот: не ловить ножи; интерес к покупке "
                f"появится при возврате выше VWAP {vwap and format(vwap, '.4g')}₽.")
    if ev["type"] == "RSI_LOW":
        return (f"интрадей RSI {rsi} — перепроданность. Спот: скальп-отскок возможен от "
                f"{lo:.4g}₽, цель VWAP {vwap and format(vwap, '.4g')}₽, стоп под минимум дня.")
    if ev["type"] == "RSI_HIGH":
        return (f"интрадей RSI {rsi} — перегрев. Спот: покупки здесь опасны; "
                f"фиксировать прибыль / ждать откат к VWAP {vwap and format(vwap, '.4g')}₽.")
    if ev["type"] == "ABSORPTION":
        return (f"объём x{ev['vol']} без движения цены — похоже, крупный игрок набирает "
                f"или раздаёт позицию. Спот: следить за пробоем; при уходе вверх с объёмом — импульс.")
    return "наблюдаем."


def _cooldown_key(secid: str, ev_type: str) -> str:
    return f"alarm_{secid}_{ev_type}"


def scan(secid: str) -> list[dict]:
    """Полный скан одной бумаги: возвращает события с текстом (с учётом кулдауна)."""
    cooldown = int(_cfg("cooldown_min", 30))
    bars = fetch_bars(secid)
    ind = analyze_bars(bars)
    if not ind:
        return []
    out = []
    for ev in detect_events(secid, ind):
        key = _cooldown_key(secid, ev["type"])
        if db.minutes_since_meta(key, default=1e9) < cooldown:
            continue
        db.set_meta(key, db.now_iso())
        out.append({
            "secid": secid,
            "title": RUS.get(ev["type"], ev["type"]),
            "ev": ev,
            "ind": ind,
            "idea": _idea(secid, ev, ind),
        })
    return out


def summary_line(secid: str, ind: dict) -> str:
    """Однострочная сводка для /quote и брифингов."""
    vw = f", VWAP {ind['vwap']:.4g}" if ind["vwap"] else ""
    rp = f", {ind['range_pos']}% диапазона дня" if ind["range_pos"] is not None else ""
    return (f"{secid}: {ind['last']:.4g}₽ за день {ind['chg_day']:+.2f}%{vw}{rp}, "
            f"15м {ind['move_15m']:+.2f}%, объём x{ind['vol_ratio']}, RSI(10м) {ind['rsi14']}")
