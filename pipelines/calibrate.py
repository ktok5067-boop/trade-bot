r"""Калибровка весов скоринга на истории — по компонентам, а не подгонкой сетки.

Идея: каждый сигнал в features.csv уже содержит, какие компоненты сработали
(0/1) и чем закончилась сделка. Значит можно честно спросить — какие
компоненты предсказывают результат, а какие только шумят, и проверить это
на отложенной половине периода.

Что выдаёт:
  1. таблицу «компонент → средний pnl при срабатывании vs без» с устойчивостью
     знака на train/test;
  2. монотонность текущего балла (децили);
  3. готовый блок score_weights для config.yaml;
  4. сравнение текущих и предложенных весов на отложенной половине.

Запуск:
    $env:PYTHONPATH="C:\DSH\trade_bot"
    python -X utf8 -m pipelines.calibrate
"""
from __future__ import annotations

import argparse
import csv
import logging
import statistics

from . import intraday
from .settings import DATA_DIR, CONFIG

log = logging.getLogger("calibrate")

META = ["date", "ticker", "call", "score", "nconf", "net_pct", "net_rr",
        "trend", "atr_pct", "pnl_pct", "alpha_pct", "hold_days", "reason", "turn20"]

def load(path=None, min_turnover: float = 0.0) -> tuple[list[dict], list[str]]:
    path = path or (DATA_DIR / "features.csv")
    rows: list[dict] = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        r = csv.reader(f, delimiter=";")
        head = next(r)
        pols = [h for h in head if h.startswith("pnl_")]
        comps = [h for h in head if h not in META and not h.startswith("pnl_")]
        for line in r:
            d = dict(zip(head, line))
            row = {
                "date": d["date"], "ticker": d["ticker"], "call": d["call"],
                "nconf": int(float(d["nconf"] or 0)),
                "net_pct": float(d["net_pct"] or 0),
                "net_rr": float(d["net_rr"] or 0),
                "trend": d["trend"],
                "atr_pct": float(d["atr_pct"] or 0),
                "pnl": float(d["pnl_pct"]),
                "alpha": float(d["alpha_pct"]) if d["alpha_pct"] else None,
                "hold": float(d["hold_days"] or 0),
                "trace": {c: int(float(d.get(c) or 0)) for c in comps},
                "turn20": float(d.get("turn20") or 0),
                "pol": {p: (float(d[p]) if d.get(p) not in ("", None) else None)
                        for p in pols},
            }
            if min_turnover and row["turn20"] < min_turnover:
                continue
            rows.append(row)
    return rows, comps


def _mean_t(xs: list[float]) -> tuple[float, float | None, float]:
    n = len(xs)
    if not n:
        return 0.0, None, 0.0
    m = statistics.fmean(xs)
    sd = statistics.pstdev(xs) if n > 1 else 0.0
    t = m / (sd / n ** 0.5) if sd and n > 1 else None
    return m, t, sd


def _welch(a: list[float], b: list[float]) -> float | None:
    if len(a) < 30 or len(b) < 30:
        return None
    ma, _, sa = _mean_t(a)
    mb, _, sb = _mean_t(b)
    se = (sa ** 2 / len(a) + sb ** 2 / len(b)) ** 0.5
    return (ma - mb) / se if se else None


def component_effects(rows: list[dict], comps: list[str]) -> list[dict]:
    out = []
    for c in comps:
        fired = [r["pnl"] for r in rows if r["trace"].get(c)]
        off = [r["pnl"] for r in rows if not r["trace"].get(c)]
        if not fired:
            continue
        mf, _, _ = _mean_t(fired)
        mo, _, _ = _mean_t(off)
        out.append({"key": c, "n": len(fired), "delta": mf - mo,
                    "mean_fired": mf, "t": _welch(fired, off)})
    return sorted(out, key=lambda d: -(abs(d["t"]) if d["t"] is not None else 0))


def fee_ok(r: dict) -> bool:
    cfg = CONFIG.get("intraday") or {}
    return (r["net_pct"] >= float(cfg.get("min_net_pct", 1.0))
            and r["net_rr"] >= float(cfg.get("min_rr", 1.2)))


def score_of(r: dict, weights: dict[str, float]) -> float:
    s = sum(w * r["trace"].get(k, 0) for k, w in weights.items())
    cfg = CONFIG.get("intraday") or {}
    if r["nconf"] < int(cfg.get("min_confirmations", 2)) and s >= float(cfg.get("enter_score", 2.0)):
        s = min(s, float(cfg.get("confluence_cap", 1.5)))
    return s


def evaluate(rows: list[dict], weights: dict[str, float], enter_score: float) -> dict:
    picked = [r for r in rows if fee_ok(r) and score_of(r, weights) >= enter_score]
    if not picked:
        return {"n": 0}
    pnl = [r["pnl"] for r in picked]
    m, t, _ = _mean_t(pnl)
    alpha = [r["alpha"] for r in picked if r["alpha"] is not None]
    return {
        "n": len(picked),
        "win": round(sum(1 for x in pnl if x > 0) / len(pnl) * 100, 1),
        "avg": round(m, 3), "t": round(t, 2) if t is not None else None,
        "alpha": round(statistics.fmean(alpha), 2) if alpha else None,
        "share": round(len(picked) / len(rows) * 100, 1),
    }


def deciles(rows: list[dict], weights: dict[str, float]) -> list[dict]:
    scored = [(score_of(r, weights), r) for r in rows]
    scored.sort(key=lambda x: x[0])
    n = len(scored)
    size = max(1, n // 10)
    out = []
    for i in range(10):
        chunk = scored[i * size:(i + 1) * size] if i < 9 else scored[9 * size:]
        if not chunk:
            continue
        pnl = [r["pnl"] for _, r in chunk]
        m, _, _ = _mean_t(pnl)
        out.append({"decile": i + 1, "n": len(chunk),
                    "score_lo": round(chunk[0][0], 2), "score_hi": round(chunk[-1][0], 2),
                    "avg": round(m, 3)})
    return out


def propose(rows: list[dict], comps: list[str], dates: list[str], cut: str) -> dict:
    """Веса по устойчивости знака: train и test должны согласиться."""
    train = [r for r in rows if r["date"] < cut]
    test = [r for r in rows if r["date"] >= cut]
    cur = intraday._weights()
    tr = {d["key"]: d for d in component_effects(train, comps)}
    te = {d["key"]: d for d in component_effects(test, comps)}
    out: dict[str, float] = {}
    detail = []
    for c in comps:
        if c == "fee_blocked":
            out[c] = cur["fee_blocked"]   # это жёсткий фильтр, его не калибруем
            detail.append((c, out[c], "жёсткий фильтр"))
            continue
        t_, e_ = tr.get(c), te.get(c)
        if not t_ or not e_ or t_["n"] < 200:
            detail.append((c, 0.0, "мало данных"))
            out[c] = 0.0
            continue
        stable = (t_["delta"] > 0) == (e_["delta"] > 0)
        strong = t_["t"] is not None and abs(t_["t"]) >= 2
        if stable and strong:
            sign = 1.0 if t_["delta"] > 0 else -1.0
            w = sign * abs(cur.get(c, 1.0))
            out[c] = round(w, 2)
            detail.append((c, out[c], "устойчив"))
        else:
            out[c] = 0.0
            detail.append((c, 0.0, "нестабилен/слаб"))
    return {"weights": out, "detail": detail, "train_n": len(train), "test_n": len(test)}


def scale_to_min_entries(rows: list[dict], weights: dict[str, float], enter_score: float,
                         min_n: int = 300) -> tuple[float, dict]:
    """Подобрать общий множитель: чтобы входов было достаточно для статистики.

    Веса задают только относительную важность; масштаб определяет «планку».
    """
    for s in (1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0):
        cand = {k: round(v * s, 2) for k, v in weights.items()}
        e = evaluate(rows, cand, enter_score)
        if e.get("n", 0) >= min_n:
            return s, cand
    return 5.0, {k: round(v * 5.0, 2) for k, v in weights.items()}


def report(rows: list[dict], comps: list[str], cut: str | None = None) -> str:
    dates = sorted({r["date"] for r in rows})
    cut = cut or dates[int(len(dates) * 0.6)]
    train = [r for r in rows if r["date"] < cut]
    test = [r for r in rows if r["date"] >= cut]
    cur = intraday._weights()
    cfg = CONFIG.get("intraday") or {}
    enter_score = float(cfg.get("enter_score", 2.0))

    L: list[str] = []
    add = L.append
    add("# Калибровка весов скоринга\n")
    add(f"* данные: {dates[0]} … {dates[-1]} ({len(dates)} дней, {len(rows)} наблюдений)")
    add(f"* split: train < {cut} ({len(train)}), test ≥ {cut} ({len(test)})")
    add(f"* порог входа (текущий): {enter_score}\n")

    add("## 1. Какие компоненты предсказывают результат\n")
    add("| компонент | n срабатываний | Δ pnl% (сработал − нет) | t (Welch) | train Δ | test Δ | знак согласован |")
    add("|---|---|---|---|---|---|---|")
    tr = {d["key"]: d for d in component_effects(train, comps)}
    te = {d["key"]: d for d in component_effects(test, comps)}
    for d in component_effects(rows, comps):
        k = d["key"]
        a, b = tr.get(k, {}), te.get(k, {})
        agree = (a.get("delta", 0) > 0) == (b.get("delta", 0) > 0) if a and b else None
        add(f"| {k} | {d['n']} | {d['delta']:+.3f} | "
            f"{d['t']:+.1f} | {a.get('delta', float('nan')):+.3f} | "
            f"{b.get('delta', float('nan')):+.3f} | "
            f"{'да' if agree else 'НЕТ'} |")

    add("\n## 2. Монотонен ли балл (децили по текущим весам, test)\n")
    add("| дециль | балл от | балл до | n | avg pnl% |")
    add("|---|---|---|---|---|")
    for d in deciles(test, cur):
        add(f"| {d['decile']} | {d['score_lo']} | {d['score_hi']} | {d['n']} | {d['avg']:+.3f} |")

    prop = propose(rows, comps, dates, cut)
    scale, w = scale_to_min_entries(train, prop["weights"], enter_score)
    add("\n## 3. Предложенные веса (по устойчивости знака на train/test)\n")
    add(f"Множитель к весам: **×{scale}** — подобран так, чтобы на train было "
        f"≥ 300 входов (иначе выборка слишком мала для выводов).\n")
    add("```yaml")
    add("score_weights:")
    for c, val, _ in prop["detail"]:
        if c == "fee_blocked":
            continue
        if w.get(c):
            add(f"  {c}: {w[c]}")
    add("```")
    zeroed = [c for c, v, _ in prop["detail"] if not v and c != "fee_blocked"]
    add(f"\nОбнулены как нестабильные/слабые ({len(zeroed)}): "
        + (", ".join(zeroed) if zeroed else "нет"))

    add("\n## 4. Что это даёт на отложенной половине\n")
    add("| вариант | набор | n входов | доля от всех | win% | avg pnl% | t | alpha% |")
    add("|---|---|---|---|---|---|---|---|")
    for name, wts in (("текущие веса", cur), ("предложенные", w)):
        for part, sub in (("train", train), ("test", test)):
            e = evaluate(sub, wts, enter_score)
            if e.get("n"):
                add(f"| {name} | {part} | {e['n']} | {e['share']} | {e['win']} "
                    f"| {e['avg']:+.3f} | {e['t']} | {e['alpha']} |")

    # --- сравнение политик выхода на выбранных входах (отложенная половина) ---
    pol_keys = sorted({k for r in test for k in (r.get("pol") or {})})
    if pol_keys:
        add("\n## 5. Политики выхода на выбранных входах (test)\n")
        add("| политика | n | win% | avg pnl% | t |")
        add("|---|---|---|---|---|")
        for p in pol_keys:
            vals = [r["pol"][p] for r in test
                    if (r.get("pol") or {}).get(p) is not None
                    and fee_ok(r) and score_of(r, w) >= enter_score]
            if len(vals) < 20:
                continue
            m, t, _ = _mean_t(vals)
            add(f"| {p} | {len(vals)} | "
                f"{round(sum(1 for x in vals if x > 0) / len(vals) * 100, 1)} | "
                f"{m:+.2f} | {t if t is None else round(t, 2)} |")

    add("\n## Оговорки\n")
    add("* сигналы перекрываются (5 оценок в день × бумага) — n завышено, |t| оптимистичны;")
    add("* «устойчивость знака» — грубый фильтр от подгонки, а не гарантия;")
    add("* веса из блока 3 вставлять в config.yaml осознанно: компоненты с обнулённым")
    add("  весом перестают влиять на вердикт, но остаются в trace и в причинах.")
    return "\n".join(L)

def main() -> int:
    ap = argparse.ArgumentParser(description="Калибровка весов скоринга")
    ap.add_argument("--features", default=None, help="путь к features.csv")
    ap.add_argument("--cut", default=None,
                    help="дата начала теста YYYY-MM-DD (по умолчанию 60%% периода)")
    ap.add_argument("--min-turnover", type=float, default=0.0,
                    help="фильтр ликвидности: средний дневной оборот, руб")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    rows, comps = load(args.features, min_turnover=args.min_turnover)
    if not rows:
        print("features.csv пуст — сначала: python -m pipelines.backtest --features")
        return 1
    text = report(rows, comps, cut=args.cut)
    out = DATA_DIR / "calibrate_report.md"
    out.write_text(text, encoding="utf-8")
    print(text)
    print(f"\nотчёт: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
