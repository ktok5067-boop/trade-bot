r"""Walk-forward бэктест сигналов бота на истории MOEX.

Задача — не «показать красивую кривую», а ответить на один вопрос:
различает ли текущий скоринг хорошие входы от плохих. Поэтому прогоняются
ВСЕ вердикты (enter / wait / avoid) по одинаковому правилу выхода, и
сравниваются между собой, плюс разбивка по половинам периода.

Правила честности:
  * индикаторы считаются только по барам до момента решения t;
  * дневной контекст — только по дням, закрытым ДО дня решения (без подглядывания);
  * вход — по открытию следующего бара после сигнала;
  * внутри бара стоп проверяется раньше цели (консервативно);
  * издержки: fee_pct (комиссия вход+выход) + 2 × slip_pct.

Запуск (из корня trade_bot):
    $env:PYTHONPATH="C:\DSH\trade_bot"
    python -X utf8 -m pipelines.backtest --days 200
"""
from __future__ import annotations

import argparse
import bisect
import csv
import logging
import statistics
from datetime import datetime

from . import history, indicators, intraday
from .settings import CONFIG, DATA_DIR

log = logging.getLogger("backtest")

WARMUP_DAYS = 60          # дневной истории до первого сигнала
CTX_DAYS_BACK = 4         # сколько прошлых дней видит analyze_bars (уровни S/R)


def _cfg(key, default):
    return (CONFIG.get("intraday") or {}).get(key, default)


def _day(ts: str) -> str:
    return ts[:10]


def _pct(a: float, b: float) -> float:
    return (a / b - 1) * 100 if b else 0.0


def _day_index(bars: list[dict]) -> list[tuple[str, int, int]]:
    """[(день, первый индекс, последний индекс)] по 10-мин барам."""
    out: list[tuple[str, int, int]] = []
    for i, b in enumerate(bars):
        d = _day(b["ts"])
        if out and out[-1][0] == d:
            out[-1] = (d, out[-1][1], i)
        else:
            out.append((d, i, i))
    return out


def _daily_before(daily: list[dict], day: str, limit: int = 80) -> list[dict]:
    """Дневные свечи строго до дня `day` (последние `limit`)."""
    return [c for c in daily if _day(c["ts"]) < day][-limit:]


class _IdxSeries:
    """IMOEX для контекста: % изменения от закрытия прошлого дня на момент ts.

    Поиск — бинарный поиск по времени: наивный перебор 17k баров на каждый
    сигнал превратил бы бэктест в часы.
    """

    def __init__(self, idx10: list[dict], idx_daily: list[dict]):
        self.ts = [b["ts"] for b in idx10]
        self.close = [b["close"] for b in idx10]
        self.prev_by_day: dict[str, float] = {}
        for i, c in enumerate(idx_daily):
            if i:
                self.prev_by_day[_day(c["ts"])] = idx_daily[i - 1]["close"]

    def ctx(self, ts: str) -> dict:
        if not self.ts:
            return {}
        i = bisect.bisect_right(self.ts, ts) - 1
        if i < 0:
            return {"value": None, "chg_pct": None}
        last = self.close[i]
        prev = self.prev_by_day.get(_day(ts))
        return {"value": last, "chg_pct": round(_pct(last, prev), 2) if prev else None}


def _simulate(bars: list[dict], i: int, stop_frac: float, target_frac: float,
              max_hold_days: int, atr_abs: float = 0.0, policy: dict | None = None,
              cost_pct: float = 0.0) -> dict | None:
    """Выход после входа по открытию бара i+1.

    policy:
      be_atr     — перевести стоп в безубыток после хода на N ATR (с учётом издержек);
      trail_atr  — трейлинг-стоп на N ATR ниже максимума с момента входа;
      time_days  — свой лимит удержания (иначе max_hold_days).

    Без подглядывания: стоп и цель проверяются уровнями, посчитанными по прошлым
    барам, трейлинг обновляется ПОСЛЕ проверок текущего бара. Стоп проверяется
    раньше цели; гэп ниже стопа исполняется по открытию (то есть хуже).
    """
    policy = policy or {}
    if i + 1 >= len(bars):
        return None
    entry_bar = bars[i + 1]
    entry = float(entry_bar["open"] or entry_bar["close"])
    if not entry:
        return None
    initial_stop = entry * (1 - stop_frac)
    stop = initial_stop
    target = entry * (1 + target_frac)
    be_atr = float(policy.get("be_atr") or 0.0)
    trail_atr = float(policy.get("trail_atr") or 0.0)
    hold_days_limit = int(policy.get("time_days") or max_hold_days)
    breakeven = entry * (1 + cost_pct / 100.0)   # безубыток с издержками
    entry_day = _day(entry_bar["ts"])
    days_seen = 0
    cur_day = entry_day
    last_close = entry
    max_high = entry
    for j in range(i + 1, len(bars)):
        b = bars[j]
        d = _day(b["ts"])
        if d != cur_day:
            cur_day = d
            days_seen += 1
            if days_seen > hold_days_limit:
                return {"entry": entry, "exit": last_close, "reason": "time",
                        "hold_days": days_seen - 1, "exit_ts": bars[j - 1]["ts"]}
        lo, hi = float(b["low"]), float(b["high"])
        # стоп раньше цели — консервативно; гэп через стоп = исполнение по открытию
        if lo <= stop:
            fill = stop
            op = float(b["open"] or stop)
            if op < stop:
                fill = op
            return {"entry": entry, "exit": fill,
                    "reason": "trail" if stop > initial_stop else "stop",
                    "hold_days": days_seen, "exit_ts": b["ts"]}
        if hi >= target:
            return {"entry": entry, "exit": target, "reason": "target",
                    "hold_days": days_seen, "exit_ts": b["ts"]}
        last_close = float(b["close"])
        # обновление трейлинга только по факту текущего бара
        max_high = max(max_high, hi)
        if be_atr and atr_abs and max_high >= entry + be_atr * atr_abs:
            stop = max(stop, breakeven)
        if trail_atr and atr_abs:
            stop = max(stop, max_high - trail_atr * atr_abs)
    return {"entry": entry, "exit": last_close, "reason": "eod",
            "hold_days": days_seen, "exit_ts": bars[-1]["ts"]}


# Политики выхода, которые сравниваются на одних и тех же входах.
EXIT_POLICIES: dict[str, dict] = {
    "base": {},
    "be1": {"be_atr": 1.0},
    "be1_trail2": {"be_atr": 1.0, "trail_atr": 2.0},
    "trail1.5": {"trail_atr": 1.5},
    "trail2.5": {"trail_atr": 2.5},
    "time5": {"time_days": 5},
    "time15": {"time_days": 15},
}


def _simulate_short(bars: list[dict], i: int, stop_frac: float, target_frac: float,
                    max_hold_days: int, atr_abs: float = 0.0,
                    policy: dict | None = None, cost_pct: float = 0.0,
                    borrow_pct_day: float = 0.0) -> dict | None:
    """Шорт: зеркально лонгу, но гэп вверх через стоп исполняется по открытию.

    Плата за заём бумаг начисляется за каждый торговый день удержания —
    она вычитается из результата отдельной строкой (borrow_pct).
    """
    policy = policy or {}
    if i + 1 >= len(bars):
        return None
    entry_bar = bars[i + 1]
    entry = float(entry_bar["open"] or entry_bar["close"])
    if not entry:
        return None
    initial_stop = entry * (1 + stop_frac)
    stop = initial_stop
    target = entry * (1 - target_frac)
    be_atr = float(policy.get("be_atr") or 0.0)
    trail_atr = float(policy.get("trail_atr") or 0.0)
    hold_limit = int(policy.get("time_days") or max_hold_days)
    breakeven = entry * (1 - cost_pct / 100.0)
    cur_day = _day(entry_bar["ts"])
    days_seen = 0
    last_close = entry
    min_low = entry

    def out(px: float, reason: str, days: int, ts: str) -> dict:
        borrow = borrow_pct_day * max(0, days + 1)
        return {"entry": entry, "exit": px, "reason": reason,
                "hold_days": days, "exit_ts": ts, "borrow_pct": round(borrow, 3)}

    for j in range(i + 1, len(bars)):
        b = bars[j]
        d = _day(b["ts"])
        if d != cur_day:
            cur_day = d
            days_seen += 1
            if days_seen > hold_limit:
                return out(last_close, "time", days_seen - 1, bars[j - 1]["ts"])
        hi, lo = float(b["high"]), float(b["low"])
        # стоп (рост цены) раньше цели — консервативно; гэп вверх = открытие
        if hi >= stop:
            fill = stop
            op = float(b["open"] or stop)
            if op > stop:
                fill = op
            return out(fill, "trail" if stop < initial_stop else "stop",
                       days_seen, b["ts"])
        if lo <= target:
            return out(target, "target", days_seen, b["ts"])
        last_close = float(b["close"])
        min_low = min(min_low, lo)
        if be_atr and atr_abs and min_low <= entry - be_atr * atr_abs:
            stop = min(stop, breakeven)
        if trail_atr and atr_abs:
            stop = min(stop, min_low + trail_atr * atr_abs)
    return out(last_close, "eod", days_seen, bars[-1]["ts"])


def run(days: int = 200, tickers: list[str] | None = None,
        eval_every: int | None = None, cost_override: float | None = None,
        min_turnover: float | None = None,
        policies: dict[str, dict] | None = None,
        side: str = "long", borrow_pct_day: float | None = None) -> dict:
    cfg = CONFIG.get("intraday") or {}
    fee = float(cost_override if cost_override is not None else cfg.get("fee_pct", 0.15))
    slip = float(cfg.get("slip_pct", 0.05))
    cost = fee + 2 * slip
    every = int(eval_every or max(1, int(cfg.get("paper_interval_min", 60)) // int(cfg.get("interval", 10))))
    max_hold = int(cfg.get("paper_exit_days", 10))
    horizon = str(cfg.get("horizon", "swing")).lower()
    stop_mult = float(cfg.get(f"stop_atr_{horizon}", 1.5))
    target_mult = float(cfg.get(f"target_atr_{horizon}", 3.0))
    min_turn = float(min_turnover if min_turnover is not None
                     else cfg.get("min_turnover_rub", 0) or 0)
    pols = policies or EXIT_POLICIES
    enter_score = float(cfg.get("enter_score", 2.0))
    min_net = float(cfg.get("min_net_pct", 1.0))
    min_rr = float(cfg.get("min_rr", 1.2))
    borrow = float(borrow_pct_day if borrow_pct_day is not None
                   else cfg.get("short_borrow_pct_day", 0.05))
    if side == "short":
        log.info("шорт: зеркальный сигнал (балл ≤ −%.1f), заём %.3f%%/день",
                 enter_score, borrow)

    tickers = tickers or [w["ticker"] for w in CONFIG["watchlist"]]
    idx = _IdxSeries(history.load("IMOEX", 10), history.load("IMOEX", 24))
    log.info("издержки: комиссия %.2f%% + 2×%.2f%% проскальзывания = %.2f%% на сделку",
             fee, slip, cost)
    log.info("выход: стоп %.1f ATR / цель %.1f ATR / время %d торг. дней",
             stop_mult, target_mult, max_hold)
    if min_turn:
        log.info("фильтр ликвидности: средний дневной оборот ≥ %.0f млн ₽", min_turn / 1e6)

    rows: list[dict] = []
    skipped_illiquid = 0
    daily_cache: dict[tuple[str, str], dict | None] = {}
    for n, ticker in enumerate(tickers, 1):
        bars = history.load(ticker, 10)
        daily = history.load(ticker, 24)
        if len(bars) < 500 or len(daily) < WARMUP_DAYS:
            log.warning("[%d/%d] %s: мало истории (10м=%d, дн=%d) — пропуск",
                        n, len(tickers), ticker, len(bars), len(daily))
            continue
        days_idx = _day_index(bars)
        for dpos, (day, dstart, dend) in enumerate(days_idx):
            if dpos < 5:
                continue
            daily_slice = _daily_before(daily, day)
            if len(daily_slice) < WARMUP_DAYS:
                continue
            dkey = (ticker, day)
            if dkey not in daily_cache:
                daily_cache[dkey] = indicators.compute(daily_slice)
            daily_ind = daily_cache[dkey]
            if not daily_ind or not daily_ind.get("atr14"):
                continue
            atr = float(daily_ind["atr14"])
            # ликвидность: средний денежный оборот за 20 прошлых дней
            vals = [float(c.get("value") or 0) for c in daily_slice[-20:]]
            turn20 = sum(vals) / len(vals) if vals else 0.0
            if min_turn and turn20 < min_turn:
                skipped_illiquid += 1
                continue
            ctx_start = max(0, days_idx[max(0, dpos - CTX_DAYS_BACK)][1])
            for i in range(dstart, dend + 1, every):
                if i - dstart < 5:  # analyze_bars требует >=5 баров дня
                    continue
                window = bars[ctx_start:i + 1]
                ind = intraday.analyze_bars(window)
                if not ind or not ind.get("last"):
                    continue
                idx_ctx = idx.ctx(bars[i]["ts"])
                v = intraday.verdict(ticker, ind, daily_ind, idx=idx_ctx, news=None)
                if not v or not v.get("entry"):
                    continue
                entry_px = float(v["entry"])
                stop_frac = max(0.005, (entry_px - float(v["stop"])) / entry_px)
                target_frac = max(stop_frac, (float(v["target"]) - entry_px) / entry_px)
                is_short = side == "short"
                if is_short:
                    # Шорт входим на зеркальном сигнале: балл настолько же
                    # отрицательный, насколько лонговый вход положителен.
                    fee_blocked = (float(v.get("net_pct") or 0) < min_net
                                   or float(v.get("net_rr") or 0) < min_rr)
                    if not (float(v["score"]) <= -enter_score and not fee_blocked):
                        continue
                    sim = _simulate_short(bars, i, stop_frac, target_frac, max_hold,
                                          atr_abs=atr, cost_pct=cost,
                                          borrow_pct_day=borrow)
                    if not sim:
                        continue
                    pnl = -_pct(sim["exit"], sim["entry"]) - cost - sim["borrow_pct"]
                else:
                    sim = _simulate(bars, i, stop_frac, target_frac, max_hold,
                                    atr_abs=atr, cost_pct=cost)
                    if not sim:
                        continue
                    pnl = _pct(sim["exit"], sim["entry"]) - cost
                # варианты выхода считаем только для входов — это они интересуют
                alt: dict[str, float] = {}
                entered = (float(v["score"]) <= -enter_score) if is_short else (v["call"] == "enter")
                if entered:
                    for pname, pol in pols.items():
                        if pname == "base":
                            continue
                        if is_short:
                            s2 = _simulate_short(bars, i, stop_frac, target_frac, max_hold,
                                                 atr_abs=atr, policy=pol, cost_pct=cost,
                                                 borrow_pct_day=borrow)
                            if s2:
                                alt[pname] = round(-_pct(s2["exit"], s2["entry"])
                                                   - cost - s2["borrow_pct"], 2)
                        else:
                            s2 = _simulate(bars, i, stop_frac, target_frac, max_hold,
                                           atr_abs=atr, policy=pol, cost_pct=cost)
                            if s2:
                                alt[pname] = round(_pct(s2["exit"], s2["entry"]) - cost, 2)
                    alt["base"] = round(pnl, 2)
                # альфа к IMOEX за период удержания (для шорта знак беты обратный)
                ie, ix = idx_ctx, idx.ctx(sim["exit_ts"])
                alpha = None
                if ie.get("value") and ix.get("value"):
                    idx_ret = _pct(ix["value"], ie["value"])
                    alpha = round(pnl + idx_ret if is_short else pnl - idx_ret, 2)
                rows.append({
                    "date": day, "time": bars[i]["ts"][11:16], "ticker": ticker,
                    "side": side, "entered": bool(entered),
                    "call": v["call"], "score": v["score"],
                    "nconf": len(v.get("confirmations") or []),
                    "price": round(entry_px, 4), "stop": round(float(v["stop"]), 4),
                    "target": round(float(v["target"]), 4),
                    "exit": round(sim["exit"], 4), "reason": sim["reason"],
                    "hold_days": sim["hold_days"], "pnl_pct": round(pnl, 2),
                    "alpha_pct": alpha, "cost_pct": round(cost, 2),
                    "net_pct": v.get("net_pct"), "net_rr": v.get("net_rr"),
                    "trace": v.get("trace") or {}, "turn20": round(turn20),
                    "alt": alt,
                    "rsi": ind.get("rsi14"), "vol_ratio": ind.get("vol_ratio"),
                    "rng_pos": ind.get("range_pos"),
                    "idx_chg": idx_ctx.get("chg_pct"), "trend": daily_ind.get("trend"),
                    "atr_pct": daily_ind.get("atr_pct"),
                })
        log.info("[%d/%d] %s: %d сигналов", n, len(tickers), ticker,
                 sum(1 for r in rows if r["ticker"] == ticker))
    return {"rows": rows, "cost": cost, "fee": fee, "slip": slip,
            "stop_mult": stop_mult, "target_mult": target_mult, "max_hold": max_hold,
            "min_turn": min_turn, "skipped_illiquid": skipped_illiquid,
            "side": side, "borrow": borrow}


def _mean_t(xs: list[float]) -> tuple[float, float | None, float]:
    n = len(xs)
    if not n:
        return 0.0, None, 0.0
    m = statistics.fmean(xs)
    sd = statistics.pstdev(xs) if n > 1 else 0.0
    t = m / (sd / n ** 0.5) if sd and n > 1 else None
    return m, t, sd


# ---------------------------- статистика ----------------------------

def _stats(rows: list[dict]) -> dict:
    if not rows:
        return {"n": 0}
    pnl = [r["pnl_pct"] for r in rows]
    n = len(pnl)
    mean = statistics.fmean(pnl)
    sd = statistics.pstdev(pnl) if n > 1 else 0.0
    wins = sum(1 for x in pnl if x > 0)
    cum = peak = dd = 0.0
    for x in pnl:  # в порядке появления = порядок времени
        cum += x
        peak = max(peak, cum)
        dd = max(dd, peak - cum)
    alpha = [r["alpha_pct"] for r in rows if r["alpha_pct"] is not None]
    return {
        "n": n, "win_rate": round(wins / n * 100, 1),
        "avg_pnl": round(mean, 2), "median_pnl": round(statistics.median(pnl), 2),
        "sum_pnl": round(sum(pnl), 1), "std": round(sd, 2),
        "t_stat": round(mean / (sd / n ** 0.5), 2) if sd and n > 1 else None,
        "max_dd": round(dd, 1),
        "avg_alpha": round(statistics.fmean(alpha), 2) if alpha else None,
        "avg_hold": round(statistics.fmean([r["hold_days"] for r in rows]), 1),
    }


def _by(rows: list[dict], key) -> dict:
    out: dict = {}
    for r in rows:
        out.setdefault(key(r), []).append(r)
    return {k: _stats(v) for k, v in sorted(out.items(), key=lambda kv: str(kv[0]))}


def report(res: dict) -> str:
    rows = res["rows"]
    if not rows:
        return "Нет сигналов: проверьте кеш истории (pipelines.history)."
    dates = sorted({r["date"] for r in rows})
    mid = dates[len(dates) // 2]
    lines: list[str] = []
    add = lines.append
    add("# Бэктест сигналов (walk-forward)\n")
    add(f"* период: {dates[0]} … {dates[-1]} ({len(dates)} торговых дней)")
    add(f"* издержки на сделку: **{res['cost']:.2f}%** "
        f"(комиссия {res['fee']:.2f}% + проскальзывание 2×{res['slip']:.2f}%)")
    add(f"* выход: стоп {res['stop_mult']} ATR / цель {res['target_mult']} ATR / "
        f"время {res['max_hold']} торг. дней")
    add(f"* всего наблюдений: {len(rows)}\n")
    if res.get("min_turn"):
        add(f"* фильтр ликвидности: оборот ≥ {res['min_turn'] / 1e6:.0f} млн ₽/день "
            f"(пропущено дней: {res.get('skipped_illiquid', 0)})\n")
    add("## Различает ли скоринг входы?\n")
    add("| вердикт | n | win% | avg pnl% | медиана | t-stat | сумма% | alpha% | maxDD |")
    add("|---|---|---|---|---|---|---|---|---|")
    for call, s in _by(rows, lambda r: r["call"]).items():
        add(f"| {call} | {s['n']} | {s['win_rate']} | {s['avg_pnl']} | {s['median_pnl']} "
            f"| {s['t_stat']} | {s['sum_pnl']} | {s['avg_alpha']} | {s['max_dd']} |")
    add("\n## По баллу сетапа (все вердикты)\n")
    add("| балл | n | win% | avg pnl% | t-stat | alpha% |")
    add("|---|---|---|---|---|---|")

    def bucket(r):
        s = r["score"]
        if s < 0:
            return "<0"
        if s < 1:
            return "0..1"
        if s < 2:
            return "1..2"
        if s < 3:
            return "2..3"
        return ">=3"

    for b, s in _by(rows, bucket).items():
        add(f"| {b} | {s['n']} | {s['win_rate']} | {s['avg_pnl']} | {s['t_stat']} | {s['avg_alpha']} |")
    add("\n## Только входы: первая и вторая половина периода\n")
    add("| период | n | win% | avg pnl% | t-stat | сумма% | alpha% |")
    add("|---|---|---|---|---|---|---|")
    enters = [r for r in rows if r.get("entered", r["call"] == "enter")]
    for name, sub in (("H1", [r for r in enters if r["date"] < mid]),
                      ("H2", [r for r in enters if r["date"] >= mid])):
        s = _stats(sub)
        if s.get("n"):
            add(f"| {name} | {s['n']} | {s['win_rate']} | {s['avg_pnl']} | {s['t_stat']} "
                f"| {s['sum_pnl']} | {s['avg_alpha']} |")
    add("\n## По дневному тренду (только входы)\n")
    add("| тренд | n | win% | avg pnl% | t-stat |")
    add("|---|---|---|---|---|")
    for k, s in _by(enters, lambda r: r["trend"]).items():
        add(f"| {k} | {s['n']} | {s['win_rate']} | {s['avg_pnl']} | {s['t_stat']} |")
    add("\n## По бумагам (только входы, худшие и лучшие)\n")
    tk = _by(enters, lambda r: r["ticker"])
    ordered = sorted(tk.items(), key=lambda kv: kv[1]["avg_pnl"])
    add("| тикер | n | win% | avg pnl% | t-stat |")
    add("|---|---|---|---|---|")
    for k, s in ordered[:5] + ordered[-5:]:
        add(f"| {k} | {s['n']} | {s['win_rate']} | {s['avg_pnl']} | {s['t_stat']} |")
    add("\n## Причины выхода (только входы)\n")
    add("| причина | n | доля% | avg pnl% |")
    add("|---|---|---|---|")
    total = sum(1 for _ in enters)
    for k, s in _by(enters, lambda r: r["reason"]).items():
        add(f"| {k} | {s['n']} | {round(s['n'] / total * 100, 1)} | {s['avg_pnl']} |")
    add("\n---\n")
    add("Читать так: если `enter` не лучше `wait`/`avoid` по avg pnl и t-stat, "
        "скоринг не содержит информации; |t| < 2 — разница в пределах шума.")

    # --- сравнение политик выхода на одних и тех же входах ---
    if any(r.get("alt") for r in enters):
        add("\n## Политики выхода (те же входы, разное управление позицией)\n")
        add("| политика | n | win% | avg pnl% | t | сумма% | maxDD |")
        add("|---|---|---|---|---|---|---|")
        names = ["base"] + sorted({k for r in enters for k in (r.get("alt") or {})
                                  if k != "base"})
        for name in names:
            vals = [(r["alt"] or {}).get(name) for r in enters]
            vals = [v for v in vals if v is not None]
            if not vals:
                continue
            m, t, _ = _mean_t(vals)
            cum = peak = dd = 0.0
            for x in vals:
                cum += x
                peak = max(peak, cum)
                dd = max(dd, peak - cum)
            add(f"| {name} | {len(vals)} | "
                f"{round(sum(1 for x in vals if x > 0) / len(vals) * 100, 1)} | "
                f"{m:+.2f} | {t if t is None else round(t, 2)} | "
                f"{round(sum(vals), 1)} | {round(dd, 1)} |")

    add("\n## Оговорки (важно)\n")
    add("* сигналы внутри дня и по бумагам **перекрываются**: 5 оценок в день × 31 бумага,")
    add("  поэтому n завышено, а |t| оптимистичны — смотреть на знак и устойчивость (H1/H2),")
    add("  а не на величину t;")
    add("* `сумма%` и `maxDD` — арифметическая сумма по сделкам, а не кривая капитала:")
    add("  при одновременных позициях это не эквити;")
    add("* параметры (1.5/3.0 ATR, порог 2.0) НЕ подбирались под этот период —")
    add("  это первый честный замер, а не результат подгонки.")
    return "\n".join(lines)


def features_run(days: int = 730, tickers: list[str] | None = None,
                 eval_every: int | None = None,
                 min_turnover: float | None = None):
    """Таблица фич для калибровки весов: одна строка = сработавшие компоненты + исход.

    Широкая CSV (по колонке на компонент) — чтобы калибровка считалась в секунды
    и не зависела от numpy.
    """
    res = run(days=days, tickers=tickers, eval_every=eval_every,
              min_turnover=min_turnover)
    rows = res["rows"]
    keys = sorted({k for r in rows for k in (r.get("trace") or {})})
    out = DATA_DIR / "features.csv"
    base = ["date", "ticker", "call", "score", "nconf", "net_pct", "net_rr",
            "trend", "atr_pct", "pnl_pct", "alpha_pct", "hold_days", "reason", "turn20"]
    pol_names = sorted({k for r in rows for k in (r.get("alt") or {}) if k != "base"})
    base += [f"pnl_{n}" for n in pol_names]
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(base + keys)
        for r in rows:
            tr = r.get("trace") or {}
            alt = r.get("alt") or {}
            w.writerow([r.get("date"), r.get("ticker"), r.get("call"), r.get("score"),
                        r.get("nconf"), r.get("net_pct"), r.get("net_rr"),
                        r.get("trend"), r.get("atr_pct"), r.get("pnl_pct"),
                        r.get("alpha_pct"), r.get("hold_days"), r.get("reason"),
                        r.get("turn20")]
                       + [alt.get(n, "") for n in pol_names]
                       + [tr.get(k, 0) for k in keys])
    log.info("features: %d строк, %d компонентов → %s", len(rows), len(keys), out)
    return out, keys, res


def write_signals_csv(res: dict, suffix: str = ""):
    """Сырые сигналы в CSV (для ручного анализа в Excel)."""
    if not res["rows"]:
        return
    out = DATA_DIR / f"backtest_signals{suffix}.csv"
    fields = ["date", "time", "ticker", "call", "score", "nconf", "price", "stop",
              "target", "exit", "reason", "hold_days", "pnl_pct", "alpha_pct",
              "cost_pct", "net_pct", "net_rr", "rsi", "vol_ratio", "rng_pos",
              "idx_chg", "trend", "atr_pct", "turn20"]
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter=";", extrasaction="ignore")
        w.writeheader()
        w.writerows(res["rows"])
    log.info("сигналы: %s", out)


def emit(res: dict) -> str:
    """Записать отчёт + CSV и вернуть текст отчёта (файлы различаются по стороне)."""
    side = res.get("side") or "long"
    suffix = "" if side == "long" else f"_{side}"
    text = report(res)
    (DATA_DIR / f"backtest_report{suffix}.md").write_text(text, encoding="utf-8")
    write_signals_csv(res, suffix)
    return text


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=200)
    ap.add_argument("--tickers", default="")
    ap.add_argument("--cost", type=float, default=None, help="комиссия round-trip, %%")
    ap.add_argument("--every", type=int, default=None, help="шаг оценки в барах")
    ap.add_argument("--min-turnover", type=float, default=None,
                    help="фильтр ликвидности: средний дневной оборот, руб")
    ap.add_argument("--side", default="long", choices=["long", "short"],
                    help="сторона: long (обычный) или short (зеркальный сигнал)")
    ap.add_argument("--borrow", type=float, default=None,
                    help="плата за заём бумаг, %% в день (для шорта)")
    ap.add_argument("--features", action="store_true",
                    help="собрать таблицу фич для калибровки весов")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()] or None
    if args.features:
        path, keys, res = features_run(days=args.days, tickers=tickers,
                                       eval_every=args.every,
                                       min_turnover=args.min_turnover)
        text = emit(res)
        print(text)
        print(f"features: {path} ({len(keys)} компонентов)")
        return 0
    res = run(days=args.days, tickers=tickers, eval_every=args.every,
              cost_override=args.cost, min_turnover=args.min_turnover,
              side=args.side, borrow_pct_day=args.borrow)
    print(emit(res))
    suffix = "" if res.get("side") == "long" else f"_{res.get('side')}"
    print(f"\nотчёты: {DATA_DIR / f'backtest_report{suffix}.md'}"
          f"  |  {DATA_DIR / f'backtest_signals{suffix}.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
