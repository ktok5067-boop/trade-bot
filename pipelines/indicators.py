"""Технический анализ по дневным свечам: классические индикаторы без pandas.

На вход — свечи из SQLite (новые в конце). На выход — компактная техкартина
для промпта LLM и для fallback-сигналов.
"""
from __future__ import annotations


def _sma(values: list[float], period: int) -> float | None:
    if len(values) < period:
        return None
    return sum(values[-period:]) / period


def _ema(values: list[float], period: int) -> float | None:
    if len(values) < period:
        return None
    k = 2 / (period + 1)
    e = sum(values[:period]) / period
    for v in values[period:]:
        e = v * k + e * (1 - k)
    return e


def _rsi(closes: list[float], period: int = 14) -> float | None:
    if len(closes) < period + 1:
        return None
    gains, losses = [], []
    for i in range(1, len(closes)):
        d = closes[i] - closes[i - 1]
        gains.append(max(d, 0))
        losses.append(max(-d, 0))
    avg_g = sum(gains[:period]) / period
    avg_l = sum(losses[:period]) / period
    for i in range(period, len(gains)):
        avg_g = (avg_g * (period - 1) + gains[i]) / period
        avg_l = (avg_l * (period - 1) + losses[i]) / period
    if avg_l == 0:
        return 100.0
    rs = avg_g / avg_l
    return round(100 - 100 / (1 + rs), 1)


def _macd(closes: list[float]) -> tuple[float, float, float] | None:
    if len(closes) < 35:
        return None
    ema12 = _ema(closes, 12)
    ema26 = _ema(closes, 26)
    if ema12 is None or ema26 is None:
        return None
    line = ema12 - ema26
    # сигнальная линия: EMA9 по истории MACD-линии
    hist_series = []
    for i in range(26, len(closes) + 1):
        e12 = _ema(closes[:i], 12)
        e26 = _ema(closes[:i], 26)
        if e12 is None or e26 is None:
            continue
        hist_series.append(e12 - e26)
    signal = _ema(hist_series, 9) if len(hist_series) >= 9 else None
    sig = signal if signal is not None else line
    return round(line, 4), round(sig, 4), round(line - sig, 4)


def _atr(highs: list[float], lows: list[float], closes: list[float],
         period: int = 14) -> float | None:
    if len(closes) < period + 1:
        return None
    trs = []
    for i in range(1, len(closes)):
        tr = max(highs[i] - lows[i],
                 abs(highs[i] - closes[i - 1]),
                 abs(lows[i] - closes[i - 1]))
        trs.append(tr)
    atr = sum(trs[:period]) / period
    for tr in trs[period:]:
        atr = (atr * (period - 1) + tr) / period
    return round(atr, 4)


def compute(candles: list[dict]) -> dict | None:
    """candles: [{'ts','open','close','high','low','volume'}], старые -> новые."""
    if len(candles) < 30:
        return None
    closes = [c["close"] for c in candles]
    highs = [c["high"] for c in candles]
    lows = [c["low"] for c in candles]
    vols = [c["volume"] or 0 for c in candles]
    last = closes[-1]

    sma20 = _sma(closes, 20)
    sma50 = _sma(closes, 50)
    rsi = _rsi(closes)
    macd = _macd(closes)
    atr = _atr(highs, lows, closes)

    # Боллинджер %B (20, 2)
    pct_b = None
    if len(closes) >= 20:
        mid = sma20
        window = closes[-20:]
        mean = mid
        var = sum((x - mean) ** 2 for x in window) / 20
        sd = var ** 0.5
        if sd:
            pct_b = round((last - (mean - 2 * sd)) / (4 * sd), 2)

    # 52-недельная позиция (в пределах имеющейся истории, до ~380 дней)
    hi = max(highs)
    lo = min(lows)
    pos_52w = round((last - lo) / (hi - lo) * 100, 1) if hi > lo else None

    # объёмный тренд: средние 5 дней против 20 дней
    vol_note = None
    if len(vols) >= 20 and sum(vols[-20:]) > 0:
        v5 = sum(vols[-5:]) / 5
        v20 = sum(vols[-20:]) / 20
        if v20:
            ratio = v5 / v20
            vol_note = f"x{ratio:.1f}" + (" (внимание к объёмам)" if ratio >= 1.5 else "")

    # простая классификация тренда
    trend = "боковик"
    if sma20 and sma50:
        if last > sma20 > sma50:
            trend = "восходящий"
        elif last < sma20 < sma50:
            trend = "нисходящий"
        elif last > sma50:
            trend = "умеренно-восходящий"
        elif last < sma50:
            trend = "умеренно-нисходящий"

    chg_20d = round((last / closes[-21] - 1) * 100, 1) if len(closes) >= 21 else None

    return {
        "last": last,
        "trend": trend,
        "chg_20d": chg_20d,
        "sma20": round(sma20, 4) if sma20 else None,
        "sma50": round(sma50, 4) if sma50 else None,
        "rsi14": rsi,
        "macd_line": macd[0] if macd else None,
        "macd_signal": macd[1] if macd else None,
        "macd_hist": macd[2] if macd else None,
        "bollinger_pctb": pct_b,
        "atr14": atr,
        "atr_pct": round(atr / last * 100, 2) if atr and last else None,
        "hi_52w": hi,
        "lo_52w": lo,
        "pos_52w_pct": pos_52w,
        "vol_5d_vs_20d": vol_note,
    }


def to_prompt_line(t: str, ind: dict | None) -> str:
    """Компактная строка техкартины для промпта LLM."""
    if not ind:
        return f"{t}: техданных недостаточно (мало истории свечей)"
    parts = [f"{ind['trend']} тренд"]
    if ind["sma20"]:
        parts.append(f"цена {ind['last']:.4g} vs SMA20 {ind['sma20']:.4g}")
    if ind["rsi14"] is not None:
        rsi = ind["rsi14"]
        mark = " (перекупленность)" if rsi >= 70 else (" (перепроданность)" if rsi <= 30 else "")
        parts.append(f"RSI14={rsi}{mark}")
    if ind["macd_hist"] is not None:
        parts.append(f"MACD-гистограмма {'+' if ind['macd_hist'] >= 0 else ''}{ind['macd_hist']:.4g}")
    if ind["chg_20d"] is not None:
        parts.append(f"за 20 сессий {ind['chg_20d']:+.1f}%")
    if ind["pos_52w_pct"] is not None:
        parts.append(f"диапазон 52н: {ind['pos_52w_pct']}% от минимума (max {ind['hi_52w']:.4g}, min {ind['lo_52w']:.4g})")
    if ind["atr_pct"] is not None:
        parts.append(f"волатильность ATR {ind['atr_pct']}%/день")
    if ind["vol_5d_vs_20d"]:
        parts.append(f"объёмы 5д/20д: {ind['vol_5d_vs_20d']}")
    if ind["bollinger_pctb"] is not None:
        parts.append(f"Боллинджер %B={ind['bollinger_pctb']}")
    return f"{t}: " + "; ".join(parts)
