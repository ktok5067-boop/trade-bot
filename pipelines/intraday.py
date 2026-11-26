"""Интрадей-сканер волатильности: 10-минутные свечи MOEX + эвристики смарт-мани.

Задача — ловить резкие движения и зоны входа для спот-торговли на волатильности.
Никаких внешних библиотек, только MOEX ISS + чистый Python.
"""
from __future__ import annotations

import logging

from . import db
from .http import get_json
from .settings import CONFIG

log = logging.getLogger("intraday")

SEC_BASE = ("https://iss.moex.com/iss/engines/stock/markets/shares"
            "/boards/TQBR/securities")


def _cfg(key, default):
    return (CONFIG.get("intraday") or {}).get(key, default)


# Веса компонентов скоринга. Вынесены в конфиг (intraday.score_weights), чтобы
# калибровать их на истории, а не править код руками. Значения по умолчанию —
# те, что были захардкожены до 2026-11-27.
DEFAULT_WEIGHTS: dict[str, float] = {
    "trend_up": 1.5, "trend_down": -1.5,
    "rsi_d_overbought": -0.5, "rsi_d_oversold": 0.5,
    "adx_flat": -0.5, "adx_trend": 0.5,
    "supertrend_up": 0.5, "supertrend_down": -0.5,
    "obv_bull": 0.5, "obv_bear": -0.5,
    "div_daily_bull": 1.0, "div_daily_bear": -1.0,
    "vwap_above": 0.5, "vwap_below": -0.5,
    "rsi_oversold": 0.5, "rsi_overbought": -1.0, "rsi_neutral": 0.5,
    "rng_top": -0.5, "rng_upper": 0.5,
    "spring": 1.5, "delta_div": 1.0, "break_high": 1.0,
    "volnode_up": 1.0, "volnode_down": -1.0, "false_break_up": -0.5,
    "level_support": 0.5, "level_resistance": -0.5,
    "bos_up": 1.0, "bos_down": -1.0,
    "sweep_low": 0.5, "sweep_high": -0.5,
    "va_break_up": 1.0, "va_break_down": -1.0,
    "div_intraday_bull": 1.0, "div_intraday_bear": -1.0,
    "profile_above": 0.5, "profile_below": -0.5,
    "neg_event": -1.0,
    "vol_high": 0.5, "vol_low": -0.5,
    "idx_down": -1.0, "idx_up": 0.5,
    "event_risk": -1.0, "fee_blocked": -2.0,
}


def _weights() -> dict[str, float]:
    """Веса компонентов.

    Если в конфиге задан score_weights — он считается ПОЛНЫМ набором:
    отсутствующие компоненты получают вес 0 (так их и обнуляет калибровка).
    Исключение — fee_blocked: это жёсткий фильтр издержек, он не отключается.
    Если конфиг пуст — работают значения по умолчанию (историческое поведение).
    """
    custom = (CONFIG.get("intraday") or {}).get("score_weights") or {}
    if not custom:
        return dict(DEFAULT_WEIGHTS)
    w = {k: 0.0 for k in DEFAULT_WEIGHTS}
    w.update({k: float(v) for k, v in custom.items() if k in DEFAULT_WEIGHTS})
    w["fee_blocked"] = float(custom.get("fee_blocked", DEFAULT_WEIGHTS["fee_blocked"]))
    return w


def fetch_bars(secid: str, interval: int | None = None, days: int | None = None) -> list[dict]:
    """10-минутные свечи (новые в конце).

    days по умолчанию берём с запасом (5 календарных), чтобы compute_levels
    видел 3 полных прошлых дня даже после выходных/понедельника.
    """
    from datetime import date, timedelta
    interval = interval or int(_cfg("interval", 10))
    days = int(days if days is not None else _cfg("history_days", 5))
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


def volume_profile(bars: list[dict], n_buckets: int = 30,
                   va_pct: float = 0.70) -> dict | None:
    """Многодневный Volume Profile: POC + зона стоимости (VAH/VAL).

    В отличие от _poc_buckets (только сегодняшний день), считает по всему
    окну баров (обычно ~5 календарных дней). VAH/VAL — границы области,
    где проторговано va_pct (70%) объёма; работают как уровни S/R.
    """
    if not bars:
        return None
    tps = [(b["high"] + b["low"] + b["close"]) / 3 for b in bars]
    lo, hi = min(tps), max(tps)
    if hi <= lo:
        return None
    width = (hi - lo) / n_buckets
    buckets = [{"low": lo + i * width, "high": lo + (i + 1) * width, "vol": 0.0}
               for i in range(n_buckets)]
    for b, tp in zip(bars, tps):
        idx = min(int((tp - lo) / width), n_buckets - 1)
        buckets[idx]["vol"] += b["volume"]
    total = sum(x["vol"] for x in buckets)
    if total <= 0:
        return None
    poc_idx = max(range(len(buckets)), key=lambda i: buckets[i]["vol"])
    # зона стоимости: расширяемся от POC в сторону более «тяжёлого» соседа
    acc = buckets[poc_idx]["vol"]
    target = total * va_pct
    lo_i = hi_i = poc_idx
    while acc < target and (lo_i > 0 or hi_i < n_buckets - 1):
        up = buckets[hi_i + 1]["vol"] if hi_i < n_buckets - 1 else -1.0
        dn = buckets[lo_i - 1]["vol"] if lo_i > 0 else -1.0
        if up >= dn:
            hi_i += 1
            acc += buckets[hi_i]["vol"]
        else:
            lo_i -= 1
            acc += buckets[lo_i]["vol"]
    poc_b = buckets[poc_idx]
    days = len({b["ts"][:10] for b in bars})
    return {
        "poc": round((poc_b["low"] + poc_b["high"]) / 2, 4),
        "vah": round(buckets[hi_i]["high"], 4),
        "val": round(buckets[lo_i]["low"], 4),
        "days": days,
    }


# ---------------------------------------------------------------------------
# Свинг-структура (SMC-lite): BOS/CHoCH + свипы ликвидности.
# ---------------------------------------------------------------------------

def _smc_pivots(values: list[float], kind: str, k: int = 2) -> list[tuple[int, float]]:
    """Подтверждённые фрактальные свинги (k баров с каждой стороны)."""
    out = []
    for i in range(k, len(values) - k):
        w = values[i - k:i + k + 1]
        if kind == "high" and values[i] == max(w):
            out.append((i, values[i]))
        elif kind == "low" and values[i] == min(w):
            out.append((i, values[i]))
    return out


def smc_structure(tb: list[dict]) -> dict | None:
    """Микроструктура дня по 10-мин барам.

    Возвращает {"trend", "break": {dir, choc, level}|None,
                "sweep": {dir, level}|None, "last_sh", "last_sl"}.
    trend — по последовательности свингов (HH+HL / LH+LL / range).
    break — пробой последнего свинга последним баром (BOS по тренду,
            CHoCH — против тренда). sweep — укол за свинг с возвратом.
    """
    if len(tb) < 12:
        return None
    highs = [b["high"] for b in tb]
    lows = [b["low"] for b in tb]
    closes = [b["close"] for b in tb]
    k = 2
    ph = _smc_pivots(highs, "high", k)
    pl = _smc_pivots(lows, "low", k)
    if not ph or not pl:
        return None
    last_sh = ph[-1][1]
    last_sl = pl[-1][1]
    trend = "range"
    if len(ph) >= 2 and len(pl) >= 2:
        hh = ph[-1][1] > ph[-2][1]
        hl = pl[-1][1] > pl[-2][1]
        if hh and hl:
            trend = "up"
        elif not hh and not hl:
            trend = "down"
    last, prev = closes[-1], closes[-2]
    brk = None
    if ph[-1][0] <= len(tb) - 2 and prev <= last_sh < last:
        brk = {"dir": "up", "choc": trend == "down", "level": last_sh}
    elif pl[-1][0] <= len(tb) - 2 and prev >= last_sl > last:
        brk = {"dir": "down", "choc": trend == "up", "level": last_sl}
    # свип ликвидности: тень за последний свинг, close вернулся обратно
    sweep = None
    b = tb[-1]
    if pl[-1][0] <= len(tb) - 2 and b["low"] < last_sl <= b["close"]:
        sweep = {"dir": "low", "level": last_sl}
    elif ph[-1][0] <= len(tb) - 2 and b["high"] > last_sh >= b["close"]:
        sweep = {"dir": "high", "level": last_sh}
    return {"trend": trend, "break": brk, "sweep": sweep,
            "last_sh": last_sh, "last_sl": last_sl}


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

    # многодневный Volume Profile (POC + VAH/VAL) по всему окну баров
    profile = volume_profile(bars, int(_cfg("profile_buckets", 30)),
                             float(_cfg("va_pct", 0.70)))

    # уровни S/R по 3 прошлым дням + подмешиваем VAH/VAL многодневного профиля
    levels = compute_levels(bars)
    if profile:
        vah, val = profile["vah"], profile["val"]
        if val <= last and val not in levels["support"]:
            levels["support"] = sorted(levels["support"] + [val])
            levels["nearest_support"] = max(
                [l for l in levels["support"] if l <= last], default=None)
        if vah >= last and vah not in levels["resistance"]:
            levels["resistance"] = sorted(levels["resistance"] + [vah])
            levels["nearest_resistance"] = min(
                [r for r in levels["resistance"] if r >= last], default=None)

    # интрадей-дивергенции RSI/MACD по 10-мин свингам
    from . import indicators as _ind
    intraday_divs = _ind.detect_divergences(highs, lows, closes, k=2)

    return {
        "last": last, "vwap": vwap, "sma5": sma5, "sma20": sma20,
        "rsi14": _rsi(closes), "vol_ratio": vol_ratio, "pulse_ratio": pulse_ratio,
        "day_high": hi, "day_low": lo, "range_pos": rng_pos,
        "chg_day": chg_day, "move_15m": move_15m, "move_30m": move_30m,
        "delta_ratio": delta_ratio_5, "delta_ratio_3": delta_ratio_3,
        "poc": poc, "levels": levels, "profile": profile,
        "divergences": intraday_divs,
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
    tb = ind.get("_tb") or []

    # 1. Резкий скачок ценой + объёмом (смарт-мани импульс)
    if abs(move) >= spike_pct and pr >= vol_mult:
        events.append({
            "type": "SPIKE_UP" if move > 0 else "SPIKE_DOWN",
            "move": move, "vol": pr,
            "power": abs(move) * pr,
        })

    # 2. Пробой ЭКСТРЕМУМА предыдущих баров дня (не просто касание хая/лоу,
    #    который сам входит в дневной диапазон). Нужно закрытие за уровнем + объём.
    if len(tb) >= 2 and pr >= 1.5:
        prev_hi = max(b["high"] for b in tb[:-1])
        prev_lo = min(b["low"] for b in tb[:-1])
        if ind["last"] > prev_hi:
            events.append({"type": "BREAK_HIGH", "move": ind["chg_day"], "vol": pr,
                           "power": (ind["move_15m"] or 0) * pr})
        if ind["last"] < prev_lo:
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
        # VOL_NODE: 2 бара подряд закрылись за пределами POC-зоны с объёмом x2+.
        # ВАЖНО: направление пробоя. Пробой ВНИЗ не должен помогать лонгу.
        if len(tb) >= 2 and avg_vol:
            b1, b2 = tb[-2], tb[-1]
            above1, below1 = b1["close"] > poc_hi, b1["close"] < poc_lo
            above2, below2 = b2["close"] > poc_hi, b2["close"] < poc_lo
            inside2 = not above2 and not below2
            big1 = b1["volume"] >= 2 * avg_vol
            big2 = b2["volume"] >= 2 * avg_vol
            detail = None
            if big1 and big2:
                if above1 and above2:
                    detail = "up"                 # удержание над узлом — за лонг
                elif below1 and below2:
                    detail = "down"               # удержание под узлом — против лонга
                elif above1 and inside2:
                    detail = "false_breakout_up"  # не смогли удержать верх — слабо
                elif below1 and inside2:
                    detail = "false_breakout_down"  # возврат из-под узла — осторожно
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
        level_side = None
        if sup is not None and b["low"] < sup <= b["close"]:
            test_level = sup
            level_side = "support"      # отскок от поддержки — за лонг
        elif res is not None and b["high"] > res >= b["close"]:
            test_level = res
            level_side = "resistance"   # отказ от сопротивления — против лонга
        if test_level is not None and b["volume"] >= 1.5 * avg_vol:
            events.append({"type": "LEVEL_TEST", "move": m15,
                           "level": round(test_level, 4), "level_side": level_side,
                           "vol": round(b["volume"] / avg_vol, 1),
                           "power": vol_mult * 1.5})

    # 9. Интрадей-дивергенции RSI/MACD по 10-мин свингам
    for d in ind.get("divergences") or []:
        kind = d["kind"]
        bull = "BULL" in kind
        events.append({"type": kind, "move": m15, "vol": round(vol_ratio, 1),
                       "text": d["text"], "power": vol_mult * 1.5 if bull else vol_mult * 1.5})

    # 10. Свинг-структура (SMC-lite): BOS/CHoCH + свипы ликвидности
    if tb:
        smc = smc_structure(tb)
        if smc:
            brk = smc.get("break")
            if brk:
                if brk["choc"]:
                    t = "CHOC_UP" if brk["dir"] == "up" else "CHOC_DOWN"
                else:
                    t = "BOS_UP" if brk["dir"] == "up" else "BOS_DOWN"
                events.append({"type": t, "move": m15, "vol": round(vol_ratio, 1),
                               "level": round(brk["level"], 4),
                               "smc_trend": smc["trend"], "power": vol_mult * 2})
            sw = smc.get("sweep")
            if sw:
                t = "SWEEP_LOW" if sw["dir"] == "low" else "SWEEP_HIGH"
                events.append({"type": t, "move": m15, "vol": round(vol_ratio, 1),
                               "level": round(sw["level"], 4),
                               "power": vol_mult * 1.8})

    # 11. Пробой границ зоны стоимости многодневного профиля (VAH/VAL)
    prof = ind.get("profile")
    if prof and len(tb) >= 2:
        vah, val = prof["vah"], prof["val"]
        c1, c2 = tb[-2]["close"], tb[-1]["close"]
        if c1 <= vah < c2:
            events.append({"type": "VA_BREAK_UP", "move": m15, "vol": round(vol_ratio, 1),
                           "level": vah, "power": vol_mult * 2})
        elif c1 >= val > c2:
            events.append({"type": "VA_BREAK_DOWN", "move": m15, "vol": round(vol_ratio, 1),
                           "level": val, "power": vol_mult * 2})

    events.sort(key=lambda e: -e["power"])
    return events


RUS = {
    "SPIKE_UP": "🚀 РЕЗКИЙ СКАЧОК ВВЕРХ",
    "SPIKE_DOWN": "📉 РЕЗКИЙ СКАЧОК ВНИЗ",
    "BREAK_HIGH": "🔥 ПРОБОЙ МАКСИМУМА ДНЯ",
    "BREAK_LOW": "💧 ПРОБОЙ МИНИМУМА ДНЯ",
    "RSI_LOW": "🩸 ПЕРЕПРОДАННОСТЬ (интрадей RSI)",
    "RSI_HIGH": "♨️ ПЕРЕКУПЛЕННОСТЬ (интрадей RSI)",
    "ABSORPTION": "🐘 ОБЪЁМ БЕЗ ДВИЖЕНИЯ (набор или раздача — направление неясно)",
    "SPRING": "🌱 СМЫВ ПРОДАВЦОВ (SPRING) — вероятный разворот вверх",
    "UPTHRUST": "🌊 СМЫВ ПОКУПАТЕЛЕЙ (UPTHRUST) — вероятный разворот вниз",
    "DELTA_DIV_POS": "🫥 ВОЗМОЖНАЯ ПОКУПКА (эвристика по закрытиям свечей)",
    "DELTA_DIV_NEG": "🫥 ВОЗМОЖНАЯ ПРОДАЖА (эвристика по закрытиям свечей)",
    "VOL_NODE": "📦 ПРОБОЙ ОБЪЁМНОГО УЗЛА (POC)",
    "LEVEL_TEST": "🧱 ТЕСТ УРОВНЯ С ПОГЛОЩЕНИЕМ",
    "RSI_BULL_DIV": "🔄 БЫЧЬЯ ДИВЕРГЕНЦИЯ RSI (интрадей)",
    "RSI_BEAR_DIV": "🔄 МЕВЕЖЬЯ ДИВЕРГЕНЦИЯ RSI (интрадей)",
    "MACD_BULL_DIV": "🔄 БЫЧЬЯ ДИВЕРГЕНЦИЯ MACD (интрадей)",
    "MACD_BEAR_DIV": "🔄 МЕВЕЖЬЯ ДИВЕРГЕНЦИЯ MACD (интрадей)",
    "BOS_UP": "🏗 ПРОБОЙ СТРУКТУРЫ ВВЕРХ (BOS)",
    "BOS_DOWN": "🏗 ПРОБОЙ СТРУКТУРЫ ВНИЗ (BOS)",
    "CHOC_UP": "🔀 РАЗВОРОТ СТРУКТУРЫ ВВЕРХ (CHoCH)",
    "CHOC_DOWN": "🔀 РАЗВОРОТ СТРУКТУРЫ ВНИЗ (CHoCH)",
    "SWEEP_LOW": "🪤 СВИП ЛИКВИДНОСТИ ПОД ЛОЯМИ",
    "SWEEP_HIGH": "🪤 СВИП ЛИКВИДНОСТИ НАД ХАЯМИ",
    "VA_BREAK_UP": "📈 ПРОБОЙ ВЕРХНЕЙ ГРАНИЦЫ ЗОНЫ СТОИМОСТИ (VAH)",
    "VA_BREAK_DOWN": "📉 ПРОБОЙ НИЖНЕЙ ГРАНИЦЫ ЗОНЫ СТОИМОСТИ (VAL)",
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
        return (f"цена обновила лоу дня, закрытия в верхней части свечей (эвристика {ind.get('delta_ratio_3')}) — "
                f"возможная покупка, но нужно подтверждение. Спот: интерес к набору от {lo:.4g}₽, "
                f"стоп под {stop_s}.")
    if ev["type"] == "DELTA_DIV_NEG":
        res = levels.get("nearest_resistance")
        res_s = f"{res:.4g}₽" if res is not None else f"{hi:.4g}₽"
        return (f"цена обновила хай дня, закрытия в нижней части свечей (эвристика) — "
                f"возможная продажа. Спот: не покупать на хае, ждать подтверждения разворота, "
                f"сопротивление у {res_s}.")
    if ev["type"] == "VOL_NODE":
        detail = ev.get("detail")
        if detail == "down":
            return (f"два бара закрылись ПОД объёмным узлом (POC) на объёме x{ev.get('vol', 0)} — "
                    f"давление продаж. Спот: не покупать против пробоя вниз, ждать возврата над POC.")
        if detail in ("false_breakout_up", "false_breakout_down"):
            return (f"пробой объёмного узла (POC) на объёме x{ev.get('vol', 0)}, но цена "
                    f"вернулась в узел — ложный пробой. Спот: ждать повторного удержания за узлом.")
        return (f"два бара подряд закрылись НАД объёмным узлом на объёме x{ev.get('vol', 0)} — "
                f"пробой вверх. Спот: удержание над POC — импульс вверх, "
                f"стоп под {stop_s}.")
    if ev["type"] == "LEVEL_TEST":
        if ev.get("level_side") == "resistance":
            return (f"тень за сопротивление {ev.get('level')} с возвратом close обратно и объёмом — "
                    f"отказ от уровня. Спот: покупки здесь опасны, ждать пробоя с закреплением.")
        return (f"тень за поддержку {ev.get('level')} с возвратом close обратно и объёмом — "
                f"тест уровня с поглощением. Спот: удержание уровня — отскок; пробой с "
                f"закреплением — импульс.")
    if ev["type"].endswith("_DIV"):
        return f"{ev.get('text', 'дивергенция')}. Спот: сигнал ранний — ждать подтверждающий бар, стоп под {stop_s}."
    if ev["type"] in ("BOS_UP", "CHOC_UP"):
        what = "разворот структуры вверх (CHoCH)" if ev["type"] == "CHOC_UP" else "продолжение структуры вверх (BOS)"
        return (f"закрытие над свинг-хаем {ev.get('level')} — {what}. "
                f"Спот: вход на откате к VWAP {format(vwap, '.4g') if vwap else '—'}₽, стоп под {stop_s}.")
    if ev["type"] in ("BOS_DOWN", "CHOC_DOWN"):
        what = "разворот структуры вниз (CHoCH)" if ev["type"] == "CHOC_DOWN" else "продолжение структуры вниз (BOS)"
        return (f"закрытие под свинг-лоу {ev.get('level')} — {what}. "
                f"Спот: против лонга; интерес к покупке — только после возврата выше VWAP.")
    if ev["type"] == "SWEEP_LOW":
        return (f"укол под свинг-лоу {ev.get('level')} с возвратом close выше — сняли ликвидность. "
                f"Спот: частый предвестник разворота вверх; стоп под {stop_s}.")
    if ev["type"] == "SWEEP_HIGH":
        return (f"укол над свинг-хаем {ev.get('level')} с возвратом close ниже — сняли ликвидность. "
                f"Спот: против лонга, возможен откат к VWAP {format(vwap, '.4g') if vwap else '—'}₽.")
    if ev["type"] == "VA_BREAK_UP":
        return (f"цена закрепилась над VAH {ev.get('level')} многодневного профиля — выход из зоны стоимости вверх. "
                f"Спот: удержание над VAH — импульс; стоп под {stop_s}.")
    if ev["type"] == "VA_BREAK_DOWN":
        return (f"цена ушла под VAL {ev.get('level')} многодневного профиля — выход из зоны стоимости вниз. "
                f"Спот: против лонга; возврат в зону (над VAL) — сигнал осторожности снят.")
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
    prof = ind.get("profile")
    pf = f", VA {prof['val']:.4g}–{prof['vah']:.4g}" if prof else ""
    return (f"{secid}: {ind['last']:.4g}₽ за день {ind['chg_day']:+.2f}%{vw}{rp}{pf}, "
            f"15м {ind['move_15m']:+.2f}%, объём x{ind['vol_ratio']}, RSI(10м) {ind['rsi14']}")


# ---------------------------------------------------------------------------
# Вердикт «входить / не входить» для спота (лонг-фокус: шорт на споте недоступен)
# ---------------------------------------------------------------------------

_EVENT_RISK_WORDS = ["дивиденд", "отчёт", "отчет", "собрание", "заседание", "цб",
                     "ставк", "решение", "санкци", "выкуп", "допэмисси", "сплит",
                     "делистинг", "спб бирж"]


def _event_risk(secid: str, news: list[dict] | None) -> str | None:
    """Риск события: важная новость-катализатор по бумаге в ближайшие дни.

    Возвращает строку-причину риска или None.
    """
    if not news:
        return None
    q = secid.lower()
    for n in news:
        title = (n.get("title") or "").lower()
        tickers = [t.lower() for t in (n.get("tickers") or [])]
        if secid.lower() not in tickers and q not in title:
            continue
        for w in _EVENT_RISK_WORDS:
            if w in title:
                return n.get("title", "")[:60]
    return None


def avg_daily_turnover(secid: str, days: int = 20) -> float | None:
    """Средний дневной оборот бумаги, ₽ (volume × close по дневным свечам).

    Денежного оборота в дневных свечах db нет, поэтому объём в штуках
    переводим в рубли по закрытию — это тот же порядок величины, что
    в фильтре бэктеста (там value из ISS).
    """
    try:
        candles = db.get_candles(secid, limit=days)
    except Exception:
        return None
    vals = [float(c.get("volume") or 0) * float(c.get("close") or 0) for c in candles]
    vals = [v for v in vals if v > 0]
    return sum(vals) / len(vals) if vals else None


def verdict(secid: str, ind: dict, daily_ind: dict | None = None,
            idx: dict | None = None, news: list[dict] | None = None) -> dict:
    """Скоринговый вердикт по бумаге (лонг-фокус для спота).

    Возвращает dict:
      call: "enter" | "wait" | "avoid"
      entry / stop / target / risk_pct / score / reasons
      confirmations: число независимых подтверждений (конфлюэнс)
    """
    last = ind["last"]
    vwap = ind["vwap"]
    rsi = ind["rsi14"]
    rng_pos = ind["range_pos"]
    levels = ind.get("levels") or {}
    lo, hi = ind["day_low"], ind["day_high"]

    score = 0.0
    reasons: list[str] = []
    confirmations: set[str] = set()  # независимые подтверждения для конфлюэнса
    trace: dict[str, int] = {}       # сработавшие компоненты — для калибровки весов
    w = _weights()

    def add(key: str, reason: str | None = None):
        """Сработал компонент `key`: прибавить его вес и записать в trace."""
        nonlocal score
        pts = float(w.get(key, DEFAULT_WEIGHTS.get(key, 0.0)))
        score += pts
        trace[key] = trace.get(key, 0) + 1
        if reason:
            reasons.append(f"{reason} ({pts:+g})")

    # 1. Дневной тренд (контекст 3-4 дня)
    if daily_ind:
        tr = daily_ind.get("trend")
        if tr == "восходящий":
            add("trend_up", "дневной тренд восходящий")
            confirmations.add("trend")
        elif tr == "нисходящий":
            add("trend_down", "дневной тренд нисходящий")
        else:
            reasons.append("дневной тренд боковой (0)")
        if daily_ind.get("rsi14") is not None:
            if daily_ind["rsi14"] > 70:
                add("rsi_d_overbought", "дневной RSI >70 — перегрев")
            elif daily_ind["rsi14"] < 35:
                add("rsi_d_oversold", "дневной RSI <35 — перепроданность")
                confirmations.add("trend")
        # ADX: сила дневного тренда — фильтр «тренд vs флэт»
        adx = daily_ind.get("adx14")
        if adx is not None:
            if adx < 20:
                add("adx_flat", f"ADX {adx} <20 — на дневке флэт, пробои рискованны")
            elif adx >= 25 and tr in ("восходящий", "умеренно-восходящий"):
                add("adx_trend", f"ADX {adx} ≥25 — дневной тренд реальный")
        # Supertrend (дневной)
        st_dir = daily_ind.get("supertrend_dir")
        if st_dir == 1:
            add("supertrend_up", "Supertrend бычий на дневке")
            confirmations.add("supertrend")
        elif st_dir == -1:
            add("supertrend_down", "Supertrend медвежий на дневке")
        # OBV-дивергенция (дневная)
        obv_div = daily_ind.get("obv_div")
        if obv_div == "bull":
            add("obv_bull", "OBV-дивергенция: скрытая покупка на дневке")
            confirmations.add("obv")
        elif obv_div == "bear":
            add("obv_bear", "OBV-дивергенция: скрытая продажа на дневке")
        # Дневные дивергенции RSI/MACD по свингам
        for d in daily_ind.get("divergences") or []:
            if "BULL" in d["kind"]:
                add("div_daily_bull", d["text"])
                confirmations.add("divergence")
            else:
                add("div_daily_bear", d["text"])

    # 2. Позиция относительно VWAP
    if vwap:
        if last > vwap:
            add("vwap_above", "цена выше VWAP")
            confirmations.add("vwap")
        else:
            add("vwap_below", "цена ниже VWAP")

    # 3. Интрадей RSI
    if rsi is not None:
        if rsi <= 25:
            add("rsi_oversold", f"RSI(10м) {rsi} — перепродано, отскок возможен")
            confirmations.add("rsi")
        elif rsi >= 75:
            add("rsi_overbought", f"RSI(10м) {rsi} — перегрев, догонять опасно")
        elif 40 <= rsi <= 60:
            add("rsi_neutral", f"RSI(10м) {rsi} — нейтральная зона, есть топливо")

    # 4. Положение в диапазоне дня
    if rng_pos is not None:
        if rng_pos >= 97:
            add("rng_top", f"цена у потолка дня ({rng_pos}%)")
        elif 50 <= rng_pos < 90:
            add("rng_upper", f"цена в верхней половине дня ({rng_pos}%)")

    # 5. События смарт-мани (только направленные; беснаправленный объём — не плюс)
    events = detect_events(secid, ind)
    neg_ev = {"UPTHRUST", "DELTA_DIV_NEG", "SPIKE_DOWN", "BREAK_LOW"}
    for e in events:
        t = e["type"]
        if t == "SPRING":
            add("spring", "SPRING — смыв продавцов у лоу")
            confirmations.add("event")
        elif t == "DELTA_DIV_POS":
            add("delta_div", "возможная покупка (эвристика по закрытиям свечей)")
            confirmations.add("event")
        elif t == "BREAK_HIGH":
            add("break_high", "пробой максимума предыдущих баров дня")
            confirmations.add("event")
        elif t == "VOL_NODE":
            detail = e.get("detail")
            if detail == "up":
                add("volnode_up", "пробой объёмного узла ВВЕРХ")
                confirmations.add("event")
            elif detail == "down":
                add("volnode_down", "пробой объёмного узла ВНИЗ — против лонга")
            elif detail == "false_breakout_up":
                add("false_break_up", "ложный пробой POC вверх")
            else:
                reasons.append("возврат в POC после пробоя вниз — направление неясно (0)")
        elif t == "LEVEL_TEST":
            if e.get("level_side") == "support":
                add("level_support", "отскок от поддержки")
                confirmations.add("event")
            elif e.get("level_side") == "resistance":
                add("level_resistance", "отказ от сопротивления — против лонга")
            else:
                reasons.append("тест уровня — направление неясно (0)")
        elif t == "ABSORPTION":
            reasons.append("объём без движения — набор или раздача, не подтверждение (0)")
        elif t in ("BOS_UP", "CHOC_UP"):
            add("bos_up", f"{'разворот' if t == 'CHOC_UP' else 'продолжение'} структуры вверх")
            confirmations.add("structure")
        elif t in ("BOS_DOWN", "CHOC_DOWN"):
            add("bos_down", f"{'разворот' if t == 'CHOC_DOWN' else 'продолжение'} структуры вниз — против лонга")
        elif t == "SWEEP_LOW":
            add("sweep_low", "свип ликвидности под лоями с возвратом")
            confirmations.add("structure")
        elif t == "SWEEP_HIGH":
            add("sweep_high", "свип ликвидности над хаями с возвратом — против лонга")
        elif t == "VA_BREAK_UP":
            add("va_break_up", "пробой VAH многодневного профиля вверх")
            confirmations.add("profile")
        elif t == "VA_BREAK_DOWN":
            add("va_break_down", "пробой VAL многодневного профиля вниз — против лонга")
        elif t in ("RSI_BULL_DIV", "MACD_BULL_DIV"):
            add("div_intraday_bull", "бычья интрадей-дивергенция")
            confirmations.add("divergence")
        elif t in ("RSI_BEAR_DIV", "MACD_BEAR_DIV"):
            add("div_intraday_bear", "медвежья интрадей-дивергенция — против лонга")
        elif t in neg_ev:
            add("neg_event", f"{RUS.get(t, t)} — против лонга")

    # 5b. Положение относительно зоны стоимости многодневного профиля
    profile = ind.get("profile")
    if profile:
        if last > profile["vah"]:
            add("profile_above", "цена выше VAH многодневного профиля — сильнее рынка")
            confirmations.add("profile")
        elif last < profile["val"]:
            add("profile_below", "цена ниже VAL многодневного профиля — слабее рынка")

    # 6. Участие объёма
    vol = ind.get("vol_ratio")
    if vol is not None:
        if vol >= 1.5:
            add("vol_high", f"объём x{vol} — участие крупных")
            confirmations.add("volume")
        elif vol <= 0.4:
            add("vol_low", f"объём x{vol} — рынок выдохся")

    # 7. Контекст индекса (IMOEX) — лонг против падающего рынка опасен
    idx_chg = None
    if idx and idx.get("chg_pct") is not None:
        idx_chg = float(idx["chg_pct"])
        if idx_chg <= -1.0:
            add("idx_down", f"IMOEX {idx_chg:+.2f}% — рынок падает, против тренда")
        elif idx_chg >= 1.0:
            add("idx_up", f"IMOEX {idx_chg:+.2f}% — рынок растёт, попутный ветер")
            confirmations.add("index")

    # 8. Риск события (event risk): новости-катализаторы в ближайшие дни
    event_risk = _event_risk(secid, news)
    if event_risk:
        add("event_risk", f"риск события: {event_risk}")

    # Вход/стоп/цель. Горизонт владельца — свинг (неделя-две) или пересиживание,
    # поэтому стоп/цель считаем от ДНЕВНОГО ATR, а не от размаха текущего дня:
    # внутридневной размах шумный и выбивает из позиции на неделю.
    entry = last
    daily_atr = (daily_ind or {}).get("atr14")
    atr = daily_atr or (hi - lo) or (last * 0.01)
    horizon = str(_cfg("horizon", "swing")).lower()

    if horizon == "hold":
        # «Пересиживание»: широкий стоп, цель дальше — позиция живёт месяцами
        stop_mult = float(_cfg("stop_atr_hold", 2.5))
        target_mult = float(_cfg("target_atr_hold", 4.0))
    else:
        # Свинг: стоп ~1.5 дневных ATR, цель ~3 ATR (R/R 1:2)
        stop_mult = float(_cfg("stop_atr_swing", 1.5))
        target_mult = float(_cfg("target_atr_swing", 3.0))

    stop = round(entry - stop_mult * atr, 4)
    target = round(entry + target_mult * atr, 4)

    # Ближайшая поддержка НИЖЕ цены может уточнить стоп — но только расширить
    # его (поставить под уровень), никогда не сузить: иначе стоп в 0.5% будет
    # выбит первым же шумом, что для свинга на неделю бессмысленно.
    sup_for_stop = levels.get("nearest_support")
    if sup_for_stop and sup_for_stop < entry:
        level_stop = round(sup_for_stop - atr * 0.25, 4)
        if level_stop < stop:  # ниже ATR-стопа — уровень держится лучше, берём его
            stop = level_stop

    # --- учёт издержек: комиссия round-trip + проскальзывание на обе стороны ---
    # Тариф владельца: 0.15% вход+выход. Раньше стояло 1.0% «на глаз» — при таком
    # допущении резалась половина рабочих сетапов, а R/R считался неправильно.
    fee = float(_cfg("fee_pct", 0.15))
    slip = float(_cfg("slip_pct", 0.05))
    cost = fee + 2 * slip
    min_net = float(_cfg("min_net_pct", 1.0))
    min_rr = float(_cfg("min_rr", 1.2))
    gross_pct = (target - entry) / entry * 100 if entry else 0
    risk_gross_pct = (entry - stop) / entry * 100 if entry else 0
    net_pct = gross_pct - cost                 # чистая прибыль при достижении цели
    net_loss_pct = risk_gross_pct + cost       # чистый убыток при срабатывании стопа
    net_rr = net_pct / net_loss_pct if net_loss_pct > 0 else 0.0
    # Сетап отклоняется, если не окупает издержки ИЛИ не даёт чистого R/R после них.
    fee_blocked = net_pct < min_net or net_rr < min_rr
    if fee_blocked:
        add("fee_blocked",
            f"после издержек: цель {net_pct:.1f}%, риск {net_loss_pct:.1f}%, "
            f"R/R {net_rr:.2f} (нужно ≥{min_net:.1f}% и R/R ≥{min_rr})")

    # ближайший уровень поддержки — доп. ориентир (не сам стоп)
    sup_ref = levels.get("nearest_support")
    res_ref = levels.get("nearest_resistance")
    risk_pct = round((entry - stop) / entry * 100, 2) if stop and entry else None

    # конфлюэнс: вход только при >= 2 независимых подтверждений (иначе слабый сигнал)
    n_conf = len(confirmations)
    need_conf = int(_cfg("min_confirmations", 2))
    enter_score = float(_cfg("enter_score", 2.0))
    wait_score = float(_cfg("wait_score", 0.5))
    confluence_cap = float(_cfg("confluence_cap", 1.5))
    weak_confluence = n_conf < need_conf
    if weak_confluence and score >= enter_score:
        score = min(score, confluence_cap)  # не дотягивает до "enter" без конфлюэнса
        reasons.append(f"мало независимых подтверждений ({n_conf}<{need_conf}) — ослабляю")

    if score >= enter_score and not fee_blocked:
        call = "enter"
    elif score >= wait_score:
        call = "wait"
    else:
        call = "avoid"

    return {
        "call": call, "side": "long", "score": round(score, 1),
        "entry": entry, "stop": stop, "target": target,
        "risk_pct": risk_pct, "reasons": reasons,
        "sup_ref": sup_ref, "res_ref": res_ref, "atr": atr,
        "fee_pct": fee, "slip_pct": slip, "cost_pct": round(cost, 2),
        "gross_pct": round(gross_pct, 2),
        "net_pct": round(net_pct, 2), "net_loss_pct": round(net_loss_pct, 2),
        "net_rr": round(net_rr, 2),
        "confirmations": sorted(confirmations),
        "trace": trace,          # сработавшие компоненты: для калибровки весов
        "idx_chg": idx_chg,
        "vol_ratio": ind.get("vol_ratio"),
        "pulse_ratio": ind.get("pulse_ratio"),
        "horizon": horizon,
        # ориентир удержания: сколько дней цена проходит цель при текущем ATR
        "hold_days": round((target - entry) / atr, 1) if atr else None,
        # уровень инвалидации идеи — закрытие ниже него отменяет сценарий
        "invalidate": round(min(stop, sup_for_stop or stop), 2) if stop else None,
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
    # горизонт удержания: свинг или пересиживание
    horizon = v.get("horizon") or "swing"
    hname = "пересиживание" if horizon == "hold" else "свинг 1-2 недели"
    hold = v.get("hold_days")
    if hold is not None:
        parts.append(f"Горизонт: {hname} (~{hold:g} дней до цели)")
    else:
        parts.append(f"Горизонт: {hname}")
    if v.get("invalidate"):
        parts.append(f"Идея отменяется ниже {v['invalidate']:.4g} ₽")
    # издержки, чистая прибыль и чистое R/R
    if v.get("net_pct") is not None:
        cost = v.get("cost_pct", v.get("fee_pct", 0))
        parts.append(f"Цель {v.get('gross_pct', 0)}% − издержки {cost}% "
                     f"= чистая ≈ {v['net_pct']}% · риск {v.get('net_loss_pct', 0)}% "
                     f"· R/R {v.get('net_rr', 0)}")
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
    idx = None
    try:
        idx = fetch_quotes.fetch_index()
    except Exception:
        pass
    scored = []
    news = None
    try:
        news = db.recent_news(hours=48, limit=40)
    except Exception:
        pass
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
            v = verdict(t, ind, daily_ind, idx=idx, news=news)
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


def rank_paper(tickers: list[str], top_n: int = 5) -> tuple[list[dict], dict]:
    """Лёгкое ранжирование для бумажной торговли (без повторного скачивания свечей).

    Использует уже закешированные дневные свечи (их обновляет update_all при
    опросе котировок), чтобы не дёргать MOEX каждые 10 минут по 34 бумагам.
    Возвращает (scored, idx) — индекс нужен для фиксации альфы на входе.
    """
    from . import fetch_quotes, indicators
    idx = None
    try:
        idx = fetch_quotes.fetch_index()
    except Exception:
        pass
    news = None
    try:
        news = db.recent_news(hours=48, limit=40)
    except Exception:
        pass
    scored = []
    for t in tickers:
        try:
            daily_ind = None
            try:
                daily_ind = indicators.compute(db.get_candles(t, limit=80))
                if daily_ind is None:
                    # после деплоя кеш свечей пуст — догружаем, иначе теряем тренд/ATR
                    fetch_quotes.fetch_candles(t, days=90)
                    daily_ind = indicators.compute(db.get_candles(t, limit=80))
            except Exception:
                pass
            bars = fetch_bars(t)
            ind = analyze_bars(bars)
            if not ind:
                continue
            v = verdict(t, ind, daily_ind, idx=idx, news=news)
            scored.append({"ticker": t, "verdict": v})
        except Exception as e:
            log.warning("rank_paper %s: %s", t, e)
    scored.sort(key=lambda x: -x["verdict"]["score"])
    return scored[:top_n], (idx or {})
