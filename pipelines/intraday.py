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


# ---------------------------------------------------------------------------
# Дельта объёма: куда «отработал» объём внутри бара (покупка/продажа).
# ---------------------------------------------------------------------------

def bar_delta(b: dict) -> float:
    """0..1: доля объёма, отработанная вверх (close ближе к high → ближе к 1)."""
    rng = b["high"] - b["low"]
    if rng <= 0:
        return 0.0
    return (b["close"] - b["low"]) / rng


def _avg_delta(bars: list[dict], n: int = 3) -> float | None:
    if len(bars) < n:
        return None
    return round(sum(bar_delta(b) for b in bars[-n:]) / n, 3)


# ---------------------------------------------------------------------------
# Кластеры объёма (POC) по typical price.
# ---------------------------------------------------------------------------

def _poc_buckets(tb: list[dict], n_buckets: int = 15) -> dict:
    """Разбивает диапазон дня на корзины по typical price, суммирует объём.

    Возвращает:
      buckets: список dict {low, high, vol} (по возрастанию цены)
      poc_idx: индекс корзины с максимальным объёмом (POC)
      tp_range: (min, max) typical price за день
    """
    tps = [(b["high"] + b["low"] + b["close"]) / 3 for b in tb]
    lo, hi = min(tps), max(tps)
    if hi <= lo:
        buckets = [{"low": lo, "high": lo, "vol": 0.0}]
        return {"buckets": buckets, "poc_idx": 0, "tp_range": (lo, lo)}
    width = (hi - lo) / n_buckets
    buckets = [{"low": lo + i * width, "high": lo + (i + 1) * width, "vol": 0.0}
               for i in range(n_buckets)]
    for b in tb:
        tp = (b["high"] + b["low"] + b["close"]) / 3
        idx = min(int((tp - lo) / width), n_buckets - 1)
        buckets[idx]["vol"] += b["volume"]
    poc_idx = max(range(len(buckets)), key=lambda i: buckets[i]["vol"])
    return {"buckets": buckets, "poc_idx": poc_idx, "tp_range": (lo, hi)}


# ---------------------------------------------------------------------------
# Уровни поддержки/сопротивления по 3-дневным барам (кластеризация хаёв/лоёв).
# ---------------------------------------------------------------------------

def _cluster_levels(values: list[float], tol_pct: float) -> list[float]:
    """Кластеризует близкие экстремумы (допуск tol_pct %), возвращает уровни
    с >= 2 касаниями, по убыванию числа касаний."""
    if not values:
        return []
    clusters: list[dict] = []  # {center, n, sum}
    for v in values:
        placed = False
        for c in clusters:
            if abs(v - c["center"]) / c["center"] * 100 <= tol_pct:
                c["n"] += 1
                c["sum"] += v
                c["center"] = c["sum"] / c["n"]
                placed = True
                break
        if not placed:
            clusters.append({"center": v, "n": 1, "sum": v})
    # сортируем по числу касаний (убыв.), затем по цене
    return [round(c["center"], 4) for c in
            sorted(clusters, key=lambda c: (-c["n"], c["center"])) if c["n"] >= 2]


def compute_levels(bars: list[dict], tol_pct: float | None = None) -> dict:
    """S/R уровни по последним 3 торговым дням (исключая сегодня).

    Возвращает {"support": [...], "resistance": [...], "nearest_support": ...,
                "nearest_resistance": ...}.
    """
    tol = float(tol_pct if tol_pct is not None else _cfg("levels_tolerance_pct", 0.2))
    if not bars:
        return {"support": [], "resistance": [], "nearest_support": None,
                "nearest_resistance": None}

    today = bars[-1]["ts"][:10]
    days: dict[str, list[dict]] = {}
    for b in bars:
        days.setdefault(b["ts"][:10], []).append(b)
    # 3 последних полных дня, исключая текущий (незавершённый)
    past_days = [d for d in sorted(days) if d < today][-3:]
    past = [b for d in past_days for b in days[d]]

    highs = [b["high"] for b in past]
    lows = [b["low"] for b in past]
    supports = _cluster_levels(lows, tol)
    resistances = _cluster_levels(highs, tol)

    last = bars[-1]["close"]
    # поддержка — уровни ПОД ценой, сопротивление — НАД ценой; до 3 ближайших
    sup_below = sorted((l for l in supports if l <= last),
                       key=lambda x: -x)[:3]          # ближайшие снизу
    res_above = sorted((r for r in resistances if r >= last))[:3]  # сверху
    return {
        "support": sorted(sup_below),
        "resistance": sorted(res_above),
        "nearest_support": sup_below[0] if sup_below else None,
        "nearest_resistance": res_above[0] if res_above else None,
    }


def _stop_below_level(levels: dict, fallback: float | None) -> float | None:
    """Ближайший уровень поддержки ниже цены как «стоп под X», иначе fallback."""
    sup = levels.get("nearest_support")
    if sup is not None and (fallback is None or sup >= fallback):
        return sup
    return fallback


def analyze_bars(bars: list[dict]) -> dict | None:
    """Сводка по сегодняшним 10-минуткам: VWAP, объёмы, RSI, диапазон, POC, уровни."""
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

    # дельта объёма: средняя по последним 3 и 5 барам
    delta_ratio_3 = _avg_delta(tb, 3)
    delta_ratio_5 = _avg_delta(tb, 5)

    # кластеры объёма (POC)
    poc = _poc_buckets(tb, int(_cfg("poc_buckets", 15)))

    # уровни S/R по 3 прошлым дням
    levels = compute_levels(bars)

    return {
        "last": last, "vwap": vwap, "sma5": sma5, "sma20": sma20,
        "rsi14": _rsi(closes), "vol_ratio": vol_ratio, "pulse_ratio": pulse_ratio,
        "day_high": hi, "day_low": lo, "range_pos": rng_pos,
        "chg_day": chg_day, "move_15m": move_15m, "move_30m": move_30m,
        "delta_ratio": delta_ratio_5, "delta_ratio_3": delta_ratio_3,
        "poc": poc, "levels": levels,
        "avg_vol": round(avg_vol, 4),
        "_tb": tb,          # сегодняшние бары (для смарт-мани эвристик)
        "bars": len(tb),
    }


def _is_lower_third(b: dict, hi: float, lo: float) -> bool:
    """Низ бара дошёл до нижней зоны дневного диапазона (сам смыв у лоу дня)."""
    if hi <= lo:
        return False
    return b["low"] <= lo + (hi - lo) / 6


def _is_upper_third(b: dict, hi: float, lo: float) -> bool:
    """Верх бара дошёл до верхней зоны дневного диапазона (смыв у хая дня)."""
    if hi <= lo:
        return False
    return b["high"] >= hi - (hi - lo) / 6


def detect_events(secid: str, ind: dict) -> list[dict]:
    """Детектор событий: скачки, импульсы, пробои, смарт-мани, POC, уровни."""
    events = []
    spike_pct = float(_cfg("spike_pct", 1.2))
    vol_mult = float(_cfg("vol_mult", 2.0))
    rsi_low = float(_cfg("rsi_low", 25))
    rsi_high = float(_cfg("rsi_high", 75))

    m15 = ind["move_15m"] or 0
    m30 = ind["move_30m"] or 0
    move = m15 if abs(m15) >= abs(m30) / 2 else m30
    pr = ind["pulse_ratio"] or 0
    vol_ratio = ind["vol_ratio"] or 0

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
    if vol_ratio >= vol_mult + 0.5 and abs(m15) < spike_pct / 2:
        events.append({"type": "ABSORPTION", "move": m15, "vol": vol_ratio,
                       "power": vol_ratio})

    # --- новые эвристики смарт-мани (нужны сырые бары) ---
    tb = ind.get("_tb") or []
    avg_vol = ind.get("avg_vol") or 0
    hi = ind.get("day_high")
    lo = ind.get("day_low")

    # 5. SPRING / UPTHRUST: последний бар с длинной тенью + объём x2 в трети диапазона
    if tb and avg_vol:
        b = tb[-1]
        rng = b["high"] - b["low"]
        lower_shadow = b["close"] - b["low"] if b["close"] >= b["open"] else b["open"] - b["low"]
        upper_shadow = b["high"] - b["close"] if b["close"] >= b["open"] else b["high"] - b["open"]
        mid = (b["high"] + b["low"]) / 2
        closes_up = b["close"] > b["open"] or b["close"] > mid
        vol_x2 = b["volume"] >= 2 * avg_vol if avg_vol else False
        if rng > 0 and hi is not None and lo is not None:
            if (lower_shadow / rng > 0.6 and closes_up and vol_x2
                    and _is_lower_third(b, hi, lo)):
                events.append({"type": "SPRING", "move": m15,
                               "vol": round(b["volume"] / avg_vol, 1) if avg_vol else 0,
                               "power": vol_mult * 2})
            if (upper_shadow / rng > 0.6 and (b["close"] < b["open"] or b["close"] < mid)
                    and vol_x2 and _is_upper_third(b, hi, lo)):
                events.append({"type": "UPTHRUST", "move": m15,
                               "vol": round(b["volume"] / avg_vol, 1) if avg_vol else 0,
                               "power": vol_mult * 2})

    # 6. Дельта-дивергенция: цена вниз, дельта растёт (скрытая покупка) и зеркально
    dr3 = ind.get("delta_ratio_3")
    if tb and dr3 is not None and len(tb) >= 5:
        new_day_low = tb[-1]["low"] == lo and all(b["low"] >= lo for b in tb[:-1])
        new_day_high = tb[-1]["high"] == hi and all(b["high"] <= hi for b in tb[:-1])
        if new_day_low and dr3 > 0.55:
            events.append({"type": "DELTA_DIV_POS", "move": m15,
                           "vol": round(vol_ratio, 1), "power": dr3 * 3})
        if new_day_high and dr3 < 0.45:
            events.append({"type": "DELTA_DIV_NEG", "move": m15,
                           "vol": round(vol_ratio, 1), "power": (1 - dr3) * 3})

    # 7. POC: улучшенный ABSORPTION (тихий набор у POC) + VOL_NODE (пробой узла)
    poc = ind.get("poc")
    if poc and tb:
        buckets = poc["buckets"]
        poc_idx = poc["poc_idx"]
        n_b = len(buckets)
        # зона POC = сам POC ± 1 корзина
        poc_lo_i = max(0, poc_idx - 1)
        poc_hi_i = min(n_b - 1, poc_idx + 1)
        poc_lo = buckets[poc_lo_i]["low"]
        poc_hi = buckets[poc_hi_i]["high"]
        last5 = tb[-5:]
        last5_vol = sum(b["volume"] for b in last5)
        last5_hi = max(b["high"] for b in last5)
        last5_lo = min(b["low"] for b in last5)
        inside_poc = poc_lo <= last5_lo and last5_hi <= poc_hi
        # улучшенный ABSORPTION: объём x4+ и диапазон внутри POC-зоны
        if avg_vol and last5_vol >= 4 * avg_vol and inside_poc:
            events.append({"type": "ABSORPTION", "move": m15,
                           "vol": round(last5_vol / avg_vol, 1),
                           "power": vol_mult * 3})
        # VOL_NODE: 2 бара подряд закрылись за пределами POC-зоны с объёмом x2+
        if len(tb) >= 2 and avg_vol:
            b1, b2 = tb[-2], tb[-1]
            out1 = b1["close"] < poc_lo or b1["close"] > poc_hi
            out2 = b2["close"] < poc_lo or b2["close"] > poc_hi
            big1 = b1["volume"] >= 2 * avg_vol
            big2 = b2["volume"] >= 2 * avg_vol
            detail = ""
            if out1 and out2 and big1 and big2:
                # чистое удержание за узлом
                detail = ""
            elif out1 and big1 and not out2 and big2:
                # пробили узел, но закрылись обратно внутрь — ложный пробой
                detail = "false_breakout"
            else:
                detail = None
            if detail is not None:
                events.append({"type": "VOL_NODE", "move": m15,
                               "vol": round(max(b1["volume"], b2["volume"]) / avg_vol, 1),
                               "detail": detail, "power": vol_mult * 2.5})

    # 8. LEVEL_TEST: тень за уровень, close вернулся обратно + объём
    levels = ind.get("levels") or {}
    if tb and avg_vol:
        b = tb[-1]
        sup = levels.get("nearest_support")
        res = levels.get("nearest_resistance")
        test_level = None
        if sup is not None and b["low"] < sup <= b["close"]:
            test_level = sup
        elif res is not None and b["high"] > res >= b["close"]:
            test_level = res
        if test_level is not None and b["volume"] >= 1.5 * avg_vol:
            events.append({"type": "LEVEL_TEST", "move": m15,
                           "level": round(test_level, 4),
                           "vol": round(b["volume"] / avg_vol, 1),
                           "power": vol_mult * 1.5})

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
    "SPRING": "🌱 СМЫВ ПРОДАВЦОВ (SPRING) — вероятный разворот вверх",
    "UPTHRUST": "🌊 СМЫВ ПОКУПАТЕЛЕЙ (UPTHRUST) — вероятный разворот вниз",
    "DELTA_DIV_POS": "🫥 СКРЫТАЯ ПОКУПКА (цена вниз, дельта растёт)",
    "DELTA_DIV_NEG": "🫥 СКРЫТАЯ ПРОДАЖА (цена вверх, дельта падает)",
    "VOL_NODE": "📦 ПРОБОЙ ОБЪЁМНОГО УЗЛА (POC)",
    "LEVEL_TEST": "🧱 ТЕСТ УРОВНЯ С ПОГЛОЩЕНИЕМ",
}


def _idea(secid: str, ev: dict, ind: dict) -> str:
    """Эвристическая спот-идея по событию. Не рекомендация, а разбор картины."""
    last, vwap = ind["last"], ind["vwap"]
    hi, lo = ind["day_high"], ind["day_low"]
    rsi = ind["rsi14"]
    above_vwap = vwap and last > vwap
    levels = ind.get("levels") or {}
    # «стоп под X» — ближайший уровень поддержки, если он ниже цены, иначе минимум дня
    stop = _stop_below_level(levels, lo)
    stop_s = f"{stop:.4g}₽" if stop is not None else "минимум импульса"

    if ev["type"] in ("SPIKE_UP", "BREAK_HIGH"):
        if rsi and rsi >= 75 and ind["range_pos"] is not None and ind["range_pos"] >= 97:
            return ("импульс на объёме, но цена у потолка дня и RSI перегрет — "
                    f"риск климакса. Спот: не догонять; вход только на откате к "
                    f"{'VWAP ' + format(vwap, '.4g') if vwap else 'SMA20'}, "
                    f"стоп под {stop_s}.")
        return (f"сильный импульс с объёмом x{ev['vol']} — вероятное продолжение. "
                f"Спот: вход на откате к VWAP {format(vwap, '.4g') if vwap else 'SMA20'}₽"
                + ("" if above_vwap else " (цена ниже VWAP — осторожнее)")
                + f"; цель — продолжение тренда, стоп под {stop_s} (минимум дня).")
    if ev["type"] in ("SPIKE_DOWN", "BREAK_LOW"):
        if rsi and rsi <= 30:
            return (f"слив с объёмом x{ev['vol']} и RSI {rsi} — возможна капитуляция. "
                    f"Спот: ждать знак разворота (бар с длинной тенью/объём на отскок), "
                    f"зона интереса у {lo:.4g}₽; стоп под {stop_s}.")
        return (f"давление продаж x{ev['vol']}. Спот: не ловить ножи; интерес к покупке "
                f"появится при возврате выше VWAP {format(vwap, '.4g') if vwap else '—'}₽.")
    if ev["type"] == "RSI_LOW":
        return (f"интрадей RSI {rsi} — перепроданность. Спот: скальп-отскок возможен от "
                f"{lo:.4g}₽, цель VWAP {format(vwap, '.4g') if vwap else '—'}₽, стоп под {stop_s}.")
    if ev["type"] == "RSI_HIGH":
        return (f"интрадей RSI {rsi} — перегрев. Спот: покупки здесь опасны; "
                f"фиксировать прибыль / ждать откат к VWAP {format(vwap, '.4g') if vwap else '—'}₽.")
    if ev["type"] == "ABSORPTION":
        return (f"объём x{ev['vol']} без движения цены — похоже, крупный игрок набирает "
                f"или раздаёт позицию. Спот: следить за пробоем; при уходе вверх с объёмом — импульс.")
    if ev["type"] == "SPRING":
        return (f"бар с длинной нижней тенью на объёме x{ev.get('vol', 0)} в нижней трети дня — "
                f"смыв продавцов, возможен разворот вверх. Спот: вход после подтверждающего "
                f"закрытия, стоп под {stop_s}, цель VWAP {format(vwap, '.4g') if vwap else '—'}₽.")
    if ev["type"] == "UPTHRUST":
        return (f"бар с длинной верхней тенью на объёме x{ev.get('vol', 0)} у хая — "
                f"смыв покупателей, возможен разворот вниз. Спот: не догонять, "
                f"фиксировать лонг / ждать откат к VWAP {format(vwap, '.4g') if vwap else '—'}₽.")
    if ev["type"] == "DELTA_DIV_POS":
        return (f"цена обновила лоу дня, но дельта объёма растёт (средняя {ind.get('delta_ratio_3')}) — "
                f"скрытая покупка крупным игроком. Спот: интерес к набору от {lo:.4g}₽, "
                f"стоп под {stop_s}.")
    if ev["type"] == "DELTA_DIV_NEG":
        res = levels.get("nearest_resistance")
        res_s = f"{res:.4g}₽" if res is not None else f"{hi:.4g}₽"
        return (f"цена обновила хай дня, но дельта объёма падает — скрытая продажа. "
                f"Спот: не покупать на хае, ждать подтверждения разворота, "
                f"сопротивление у {res_s}.")
    if ev["type"] == "VOL_NODE":
        if ev.get("detail") == "false_breakout":
            return (f"пробой объёмного узла (POC) с объёмом x{ev.get('vol', 0)}, но цена "
                    f"вернулась под узел — ложный пробой. Спот: не входить против узла, "
                    f"ждать повторного удержания над POC.")
        return (f"два бара подряд закрылись за пределами POC-зоны на объёме x{ev.get('vol', 0)} — "
                f"пробой объёмного узла. Спот: удержание за узлом — импульс в сторону пробоя, "
                f"стоп под {stop_s}.")
    if ev["type"] == "LEVEL_TEST":
        return (f"тень за уровень {ev.get('level')} с возвратом close обратно и объёмом — "
                f"тест уровня с поглощением. Спот: удержание уровня — отскок; пробой с "
                f"закреплением — импульс.")
    return "наблюдаем."


def _cooldown_key(secid: str, ev_type: str) -> str:
    return f"alarm_{secid}_{ev_type}"


def scan(secid: str) -> list[dict]:
    """Полный скан одной бумаги: возвращает ТОП-1 событие (с учётом кулдауна).

    Cooldown-метку НЕ ставим здесь — это делается в scheduler только ПОСЛЕ
    успешной отправки (claim-then-send), чтобы неудачная отправка не «съедала»
    событие без алерта.
    """
    cooldown = int(_cfg("cooldown_min", 30))
    bars = fetch_bars(secid)
    ind = analyze_bars(bars)
    if not ind:
        return []
    events = detect_events(secid, ind)
    # отсекаем кулдаун-события, затем берём ТОП-1 по power (events уже отсортированы)
    for ev in events:
        key = _cooldown_key(secid, ev["type"])
        if db.minutes_since_meta(key, default=1e9) < cooldown:
            continue
        return [{
            "secid": secid,
            "title": RUS.get(ev["type"], ev["type"]),
            "ev": ev,
            "ind": ind,
            "idea": _idea(secid, ev, ind),
        }]
    return []


def summary_line(secid: str, ind: dict) -> str:
    """Однострочная сводка для /quote и брифингов."""
    vw = f", VWAP {ind['vwap']:.4g}" if ind["vwap"] else ""
    rp = f", {ind['range_pos']}% диапазона дня" if ind["range_pos"] is not None else ""
    return (f"{secid}: {ind['last']:.4g}₽ за день {ind['chg_day']:+.2f}%{vw}{rp}, "
            f"15м {ind['move_15m']:+.2f}%, объём x{ind['vol_ratio']}, RSI(10м) {ind['rsi14']}")


# ---------------------------------------------------------------------------
# Вердикт «входить / не входить» для спота (лонг-фокус: шорт на споте недоступен)
# ---------------------------------------------------------------------------

def verdict(secid: str, ind: dict, daily_ind: dict | None = None) -> dict:
    """Скоринговый вердикт по бумаге.

    Возвращает dict:
      call: "enter" | "wait" | "avoid"
      side: "long" (для спота иных нет; avoid = не входить вовсе)
      entry / stop / target / risk_pct (в рублях и %)
      score: float, reasons: [str]
    """
    last = ind["last"]
    vwap = ind["vwap"]
    rsi = ind["rsi14"]
    rng_pos = ind["range_pos"]
    levels = ind.get("levels") or {}
    lo, hi = ind["day_low"], ind["day_high"]

    score = 0.0
    reasons: list[str] = []

    # 1. Дневной тренд (контекст 3-4 дня)
    if daily_ind:
        tr = daily_ind.get("trend")
        if tr == "восходящий":
            score += 1.5
            reasons.append("дневной тренд восходящий (+1.5)")
        elif tr == "нисходящий":
            score -= 1.5
            reasons.append("дневной тренд нисходящий (−1.5)")
        else:
            reasons.append("дневной тренд боковой (0)")
        if daily_ind.get("rsi14") is not None:
            if daily_ind["rsi14"] > 70:
                score -= 0.5
                reasons.append("дневной RSI >70 — перегрев (−0.5)")
            elif daily_ind["rsi14"] < 35:
                score += 0.5
                reasons.append("дневной RSI <35 — перепроданность (+0.5)")

    # 2. Позиция относительно VWAP
    if vwap:
        if last > vwap:
            score += 0.5
            reasons.append("цена выше VWAP (+0.5)")
        else:
            score -= 0.5
            reasons.append("цена ниже VWAP (−0.5)")

    # 3. Интрадей RSI
    if rsi is not None:
        if rsi <= 25:
            score += 0.5
            reasons.append(f"RSI(10м) {rsi} — перепродано, отскок возможен (+0.5)")
        elif rsi >= 75:
            score -= 1.0
            reasons.append(f"RSI(10м) {rsi} — перегрев, догонять опасно (−1)")
        elif 40 <= rsi <= 60:
            score += 0.5
            reasons.append(f"RSI(10м) {rsi} — нейтральная зона, есть топливо (+0.5)")

    # 4. Положение в диапазоне дня
    if rng_pos is not None:
        if rng_pos >= 97:
            score -= 0.5
            reasons.append(f"цена у потолка дня ({rng_pos}%) (−0.5)")
        elif 50 <= rng_pos < 90:
            score += 0.5
            reasons.append(f"цена в верхней половине дня ({rng_pos}%) (+0.5)")

    # 5. События смарт-мани
    events = detect_events(secid, ind)
    pos_ev = {"SPRING", "DELTA_DIV_POS", "LEVEL_TEST", "ABSORPTION"}
    neg_ev = {"UPTHRUST", "DELTA_DIV_NEG", "SPIKE_DOWN", "BREAK_LOW"}
    for e in events:
        t = e["type"]
        if t == "SPRING":
            score += 1.5
            reasons.append("SPRING — смыв продавцов у лоу (+1.5)")
        elif t == "DELTA_DIV_POS":
            score += 1.0
            reasons.append("скрытая покупка (дельта растёт) (+1)")
        elif t in ("BREAK_HIGH",) and rng_pos is not None and rng_pos >= 99:
            score += 1.0
            reasons.append("пробой максимума дня (+1)")
        elif t == "VOL_NODE":
            if e.get("detail") == "false_breakout":
                score -= 1.0
                reasons.append("ложный пробой POC (−1)")
            else:
                score += 1.0
                reasons.append("пробой объёмного узла (+1)")
        elif t in neg_ev:
            score -= 1.0
            reasons.append(f"{RUS.get(t, t)} — против лонга (−1)")
        elif t in pos_ev:
            score += 0.5
            reasons.append(f"{RUS.get(t, t)} (+0.5)")

    # 6. Участие объёма
    vol = ind.get("vol_ratio")
    if vol is not None:
        if vol >= 1.5:
            score += 0.5
            reasons.append(f"объём x{vol} — участие крупных (+0.5)")
        elif vol <= 0.4:
            score -= 0.5
            reasons.append(f"объём x{vol} — рынок выдохся (−0.5)")

    # Вход/стоп/цель: стоп по ATR (не по ближайшему тику, чтобы шум не выбивал)
    entry = last
    atr = None
    if daily_ind and daily_ind.get("atr14"):
        atr = daily_ind["atr14"]
    if not atr:  # фолбек: размах дня как прокси волатильности
        atr = (hi - lo) or (last * 0.01)
    stop = round(entry - atr, 4)
    # цель растянута (target_atr × ATR), чтобы после комиссии оставалась прибыль
    target = round(entry + float(_cfg("target_atr", 2.0)) * atr, 4)

    # --- учёт издержек (комиссия вход+выход, % от движения) ---
    fee = float(_cfg("fee_pct", 1.0))
    min_net = float(_cfg("min_net_pct", 1.5))
    gross_pct = (target - entry) / entry * 100 if entry else 0
    net_pct = gross_pct - fee  # чистая прибыль после комиссии
    # если сетап не окупает комиссию — вход бессмыслен
    fee_blocked = net_pct < min_net
    if fee_blocked:
        score -= 2.0
        reasons.append(f"издержки {fee}% съедают потенциал: чистая ≈ {net_pct:.1f}% "
                       f"(нужно ≥ {min_net:.1f}%) (−2)")

    # ближайший уровень поддержки — доп. ориентир (не сам стоп)
    sup_ref = levels.get("nearest_support")
    res_ref = levels.get("nearest_resistance")
    risk_pct = round((entry - stop) / entry * 100, 2) if stop and entry else None

    if score >= 2.0 and not fee_blocked:
        call = "enter"
    elif score >= 0.5:
        call = "wait"
    else:
        call = "avoid"

    return {
        "call": call, "side": "long", "score": round(score, 1),
        "entry": entry, "stop": stop, "target": target,
        "risk_pct": risk_pct, "reasons": reasons,
        "sup_ref": sup_ref, "res_ref": res_ref, "atr": atr,
        "fee_pct": fee, "gross_pct": round(gross_pct, 2),
        "net_pct": round(net_pct, 2),
    }


def _verdict_line(v: dict) -> str:
    call, side = v["call"], v["side"]
    if call == "enter":
        head = "✅ ВХОД рекомендован (лонг)"
    elif call == "wait":
        head = "⏸ ЖДАТЬ подтверждения"
    else:
        head = "⛔ НЕ входить"
    parts = [head, f"Балл сетапа: {v['score']}"]
    if v["entry"]:
        parts.append(f"Вход: {v['entry']:.4g} ₽")
    if v["stop"]:
        parts.append(f"Стоп: {v['stop']:.4g} ₽")
    if v["target"]:
        parts.append(f"Цель: {v['target']:.4g} ₽")
    if v["risk_pct"] is not None:
        rr = ""
        if v["stop"] and v["target"] and v["entry"]:
            rw = (v["target"] - v["entry"]) / (v["entry"] - v["stop"])
            rr = f" · R/R ≈ 1:{max(rw, 0.1):.1f}"
        parts.append(f"Риск: {v['risk_pct']}%{rr}")
    # издержки и чистая прибыль
    if v.get("net_pct") is not None:
        parts.append(f"Движение: {v.get('gross_pct', 0)}% − комиссия {v.get('fee_pct', 0)}% "
                     f"= чистая ≈ {v['net_pct']}%")
    parts.append("Почему: " + "; ".join(v["reasons"]) if v["reasons"] else "аргументов мало")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Скринер «что взять на интрадей 3-4 дня»
# ---------------------------------------------------------------------------

def screener(tickers: list[str], top_n: int = 3) -> list[dict]:
    """Ранжирует бумаги по качеству сетапа (свинг 3-4 дня).

    Для каждой: дневные свечи (тренд) + интрадей-картина + вердикт.
    Возвращает список {ticker, verdict, daily_trend, note} по убыванию score.
    """
    from . import fetch_quotes, indicators
    scored = []
    for t in tickers:
        try:
            daily_ind = None
            try:
                fetch_quotes.fetch_candles(t, days=90)  # свежие дневные свечи
                daily_ind = indicators.compute(db.get_candles(t, limit=80))
            except Exception:
                pass
            bars = fetch_bars(t)
            ind = analyze_bars(bars)
            if not ind:
                continue
            v = verdict(t, ind, daily_ind)
            scored.append({
                "ticker": t,
                "verdict": v,
                "ind": ind,
                "daily_trend": (daily_ind or {}).get("trend", "—"),
            })
        except Exception as e:
            log.warning("screener %s: %s", t, e)
    scored.sort(key=lambda x: -x["verdict"]["score"])
    return scored[:top_n]
