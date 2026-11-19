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


# ---------------------------------------------------------------------------
# Серии значений (для дивергенций по свингам)
# ---------------------------------------------------------------------------

def _rsi_series(closes: list[float], period: int = 14) -> list[float | None]:
    """RSI для каждого бара (индексы совпадают с closes), None где данных не хватает."""
    out: list[float | None] = [None] * len(closes)
    if len(closes) < period + 1:
        return out
    gains, losses = [], []
    for i in range(1, len(closes)):
        d = closes[i] - closes[i - 1]
        gains.append(max(d, 0))
        losses.append(max(-d, 0))
    ag = sum(gains[:period]) / period
    al = sum(losses[:period]) / period
    out[period] = 100.0 if al == 0 else round(100 - 100 / (1 + ag / al), 1)
    for i in range(period, len(gains)):
        ag = (ag * (period - 1) + gains[i]) / period
        al = (al * (period - 1) + losses[i]) / period
        out[i + 1] = 100.0 if al == 0 else round(100 - 100 / (1 + ag / al), 1)
    return out


def _ema_series(values: list[float], period: int) -> list[float | None]:
    out: list[float | None] = [None] * len(values)
    if len(values) < period:
        return out
    k = 2 / (period + 1)
    e = sum(values[:period]) / period
    out[period - 1] = e
    for i in range(period, len(values)):
        e = values[i] * k + e * (1 - k)
        out[i] = e
    return out


def _macd_series(closes: list[float]) -> tuple[list[float | None], list[float | None]]:
    """(линия MACD, гистограмма) для каждого бара."""
    e12 = _ema_series(closes, 12)
    e26 = _ema_series(closes, 26)
    line: list[float | None] = [None] * len(closes)
    for i in range(len(closes)):
        if e12[i] is not None and e26[i] is not None:
            line[i] = e12[i] - e26[i]
    vals = [v for v in line if v is not None]
    sig = _ema_series(vals, 9)
    hist: list[float | None] = [None] * len(closes)
    j = 0
    for i in range(len(closes)):
        if line[i] is not None:
            if sig[j] is not None:
                hist[i] = line[i] - sig[j]
            j += 1
    return line, hist


def _pivot_highs(highs: list[float], k: int = 3) -> list[tuple[int, float]]:
    """Локальные максимумы: k баров слева и справа ниже."""
    out = []
    for i in range(k, len(highs) - k):
        w = highs[i - k:i + k + 1]
        if highs[i] == max(w) and w.count(highs[i]) == 1:
            out.append((i, highs[i]))
    return out


def _pivot_lows(lows: list[float], k: int = 3) -> list[tuple[int, float]]:
    out = []
    for i in range(k, len(lows) - k):
        w = lows[i - k:i + k + 1]
        if lows[i] == min(w) and w.count(lows[i]) == 1:
            out.append((i, lows[i]))
    return out


def detect_divergences(highs: list[float], lows: list[float],
                       closes: list[float], k: int = 3) -> list[dict]:
    """Классические дивергенции цена/RSI и цена/MACD-гистограмма по двум
    последним подтверждённым свингам."""
    if len(closes) < 35 or len(closes) < 2 * k + 5:
        return []
    rsi_s = _rsi_series(closes)
    _, macd_hist = _macd_series(closes)
    divs: list[dict] = []
    ph = _pivot_highs(highs, k)
    pl = _pivot_lows(lows, k)
    if len(ph) >= 2:
        (i1, v1), (i2, v2) = ph[-2], ph[-1]
        if v2 > v1:  # цена выше — новый хай
            r1, r2 = rsi_s[i1], rsi_s[i2]
            if r1 is not None and r2 is not None and r2 < r1:
                divs.append({"kind": "RSI_BEAR_DIV",
                             "text": f"медвежья RSI-дивергенция: хай {v1:.4g}→{v2:.4g}, RSI {r1}→{r2}"})
            m1, m2 = macd_hist[i1], macd_hist[i2]
            if m1 is not None and m2 is not None and m2 < m1:
                divs.append({"kind": "MACD_BEAR_DIV",
                             "text": f"медвежья MACD-дивергенция: хай {v1:.4g}→{v2:.4g}, гистограмма слабеет"})
    if len(pl) >= 2:
        (i1, v1), (i2, v2) = pl[-2], pl[-1]
        if v2 < v1:  # цена ниже — новый лоу
            r1, r2 = rsi_s[i1], rsi_s[i2]
            if r1 is not None and r2 is not None and r2 > r1:
                divs.append({"kind": "RSI_BULL_DIV",
                             "text": f"бычья RSI-дивергенция: лоу {v1:.4g}→{v2:.4g}, RSI {r1}→{r2}"})
            m1, m2 = macd_hist[i1], macd_hist[i2]
            if m1 is not None and m2 is not None and m2 > m1:
                divs.append({"kind": "MACD_BULL_DIV",
                             "text": f"бычья MACD-дивергенция: лоу {v1:.4g}→{v2:.4g}, гистограмма растёт"})
    return divs


# ---------------------------------------------------------------------------
# ADX/DMI, Supertrend, OBV
# ---------------------------------------------------------------------------

def _adx(highs: list[float], lows: list[float], closes: list[float],
         period: int = 14) -> tuple[float, float, float] | None:
    """(ADX, +DI, −DI) по Уайлдеру."""
    if len(closes) < 2 * period + 1:
        return None
    trs, plus_dm, minus_dm = [], [], []
    for i in range(1, len(closes)):
        up = highs[i] - highs[i - 1]
        down = lows[i - 1] - lows[i]
        plus_dm.append(up if up > down and up > 0 else 0.0)
        minus_dm.append(down if down > up and down > 0 else 0.0)
        trs.append(max(highs[i] - lows[i],
                       abs(highs[i] - closes[i - 1]),
                       abs(lows[i] - closes[i - 1])))
    atr = sum(trs[:period])
    pdm = sum(plus_dm[:period])
    mdm = sum(minus_dm[:period])
    dxs: list[float] = []
    pdi = mdi = 0.0
    for i in range(period, len(trs)):
        atr = atr - atr / period + trs[i]
        pdm = pdm - pdm / period + plus_dm[i]
        mdm = mdm - mdm / period + minus_dm[i]
        if atr == 0:
            continue
        pdi = 100 * pdm / atr
        mdi = 100 * mdm / atr
        s = pdi + mdi
        dxs.append(100 * abs(pdi - mdi) / s if s else 0.0)
    if len(dxs) < period:
        return None
    adx = sum(dxs[:period]) / period
    for d in dxs[period:]:
        adx = (adx * (period - 1) + d) / period
    return round(adx, 1), round(pdi, 1), round(mdi, 1)


def _supertrend(highs: list[float], lows: list[float], closes: list[float],
                period: int = 10, mult: float = 3.0) -> dict | None:
    """Supertrend: направление (1 = бычий, −1 = медвежий) и линия-стоп."""
    n = len(closes)
    if n < period + 5:
        return None
    trs = []
    for i in range(1, n):
        trs.append(max(highs[i] - lows[i],
                       abs(highs[i] - closes[i - 1]),
                       abs(lows[i] - closes[i - 1])))
    atrs: list[float | None] = [None] * n
    atr = sum(trs[:period]) / period
    atrs[period] = atr
    for i in range(period + 1, n):
        atr = (atr * (period - 1) + trs[i - 1]) / period
        atrs[i] = atr
    upper: list[float | None] = [None] * n
    lower: list[float | None] = [None] * n
    trend = [0] * n
    for i in range(period, n):
        hl2 = (highs[i] + lows[i]) / 2
        atr_i = atrs[i] or 0.0
        bu = hl2 + mult * atr_i
        bl = hl2 - mult * atr_i
        if upper[i - 1] is None:
            upper[i], lower[i] = bu, bl
            trend[i] = 1 if closes[i] >= hl2 else -1
            continue
        upper[i] = bu if (bu < upper[i - 1] or closes[i - 1] > upper[i - 1]) else upper[i - 1]
        lower[i] = bl if (bl > lower[i - 1] or closes[i - 1] < lower[i - 1]) else lower[i - 1]
        prev = trend[i - 1]
        if prev == 1:
            trend[i] = -1 if closes[i] < (lower[i - 1] or bl) else 1
        else:
            trend[i] = 1 if closes[i] > (upper[i - 1] or bu) else -1
    line = lower[-1] if trend[-1] == 1 else upper[-1]
    return {"dir": trend[-1], "line": round(line, 4) if line else None}


def _obv_note(closes: list[float], vols: list[float], lookback: int = 10) -> dict | None:
    """Наклон OBV за lookback баров + дивергенция OBV/цена."""
    if len(closes) < lookback + 1 or sum(vols[-lookback:]) == 0:
        return None
    obv = [0.0]
    for i in range(1, len(closes)):
        if closes[i] > closes[i - 1]:
            obv.append(obv[-1] + vols[i])
        elif closes[i] < closes[i - 1]:
            obv.append(obv[-1] - vols[i])
        else:
            obv.append(obv[-1])
    slope = obv[-1] - obv[-1 - lookback]
    price_chg = closes[-1] - closes[-1 - lookback]
    div = None
    if price_chg > 0 and slope < 0:
        div = "bear"   # цена растёт, объёмный поток слабеет
    elif price_chg < 0 and slope > 0:
        div = "bull"   # цена падает, но OBV набирает — скрытая покупка
    return {"slope_sign": 1 if slope > 0 else (-1 if slope < 0 else 0), "div": div}


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
    adx = _adx(highs, lows, closes)
    st = _supertrend(highs, lows, closes)
    obv = _obv_note(closes, vols)
    divs = detect_divergences(highs, lows, closes)

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

    # ADX уточняет силу тренда (не направление): <20 — флэт, пробои рискованны.
    adx_val = adx[0] if adx else None
    trend_strength = None
    if adx_val is not None:
        if adx_val >= 25:
            trend_strength = "сильный"
        elif adx_val >= 20:
            trend_strength = "умеренный"
        else:
            trend_strength = "слабый/флэт"

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
        "adx14": adx[0] if adx else None,
        "plus_di": adx[1] if adx else None,
        "minus_di": adx[2] if adx else None,
        "trend_strength": trend_strength,
        "supertrend_dir": st["dir"] if st else None,
        "supertrend_line": st["line"] if st else None,
        "obv_slope": obv["slope_sign"] if obv else None,
        "obv_div": obv["div"] if obv else None,
        "divergences": divs,
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
    if ind.get("adx14") is not None:
        di = ""
        if ind.get("plus_di") is not None and ind.get("minus_di") is not None:
            di = f" (+DI {ind['plus_di']}/−DI {ind['minus_di']})"
        parts.append(f"ADX {ind['adx14']}{di} — тренд {ind.get('trend_strength', '?')}")
    if ind.get("supertrend_dir") is not None:
        st_dir = "бычий" if ind["supertrend_dir"] == 1 else "медвежий"
        line = ind.get("supertrend_line")
        parts.append(f"Supertrend {st_dir}" + (f" (стоп {line:.4g})" if line else ""))
    if ind.get("obv_div"):
        parts.append("OBV-дивергенция: " + ("скрытая покупка" if ind["obv_div"] == "bull" else "скрытая продажа"))
    elif ind.get("obv_slope") is not None:
        parts.append("OBV " + ("растёт" if ind["obv_slope"] > 0 else "падает" if ind["obv_slope"] < 0 else "flat"))
    for d in ind.get("divergences") or []:
        parts.append(d["text"])
    return f"{t}: " + "; ".join(parts)
