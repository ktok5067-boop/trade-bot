"""Исторические свечи MOEX ISS → локальный кеш.

Зачем: бэктест и оценка сигналов требуют много истории, а живой бот тянет
только последние дни. Качаем один раз, дальше работаем офлайн из data/history.

Особенности площадки:
  * MOEX ISS отдаёт максимум 500 строк на страницу → пагинация через `start`;
  * TLS: локально (Windows + корпоративный/антивирусный перехват) проверка
    сертификата может падать. Обычный путь — requests с проверкой; аварийный —
    системный curl. Отключать проверку разрешено ТОЛЬКО явно:
        INSECURE_TLS=1
    На Render эта переменная не задаётся, там работает штатная проверка.
"""
from __future__ import annotations

import json
import logging
import os
import pathlib
import subprocess
import urllib.parse
from datetime import date, timedelta

from .http import UA
from .settings import DATA_DIR

log = logging.getLogger("history")

HISTORY_DIR: pathlib.Path = DATA_DIR / "history"
ISS = "https://iss.moex.com/iss"
PAGE = 500  # максимум строк на страницу у ISS
FRESH_HOURS = 12

SHARES = {"engine": "stock", "market": "shares", "board": "TQBR"}
INDEX = {"engine": "stock", "market": "index", "board": "SNDX"}


# ---------------------------- транспорт ----------------------------

def _full_url(path: str, params: dict) -> str:
    qs = urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
    return f"{ISS}{path}" + (f"?{qs}" if qs else "")


def _curl_json(url: str, timeout: int = 40) -> dict:
    cmd = ["curl", "-sS", "--fail", "--max-time", str(timeout), "-A", UA]
    if os.environ.get("INSECURE_TLS", "") == "1":
        cmd.append("-k")
    cmd.append(url)
    out = subprocess.run(cmd, capture_output=True, text=True,
                         encoding="utf-8", errors="replace")
    if out.returncode != 0:
        raise RuntimeError(f"curl failed: {out.stderr.strip()[:300]}")
    return json.loads(out.stdout)


def _get_json(url: str) -> dict:
    """curl → requests. Локально TLS перехвачен, и requests на части страниц
    висит до таймаута (40 с) — при сборке истории это несопоставимо дороже,
    поэтому HISTORY_CURL_ONLY=1 идёт сразу через curl."""
    if os.environ.get("HISTORY_CURL_ONLY", "") == "1":
        return _curl_json(url)
    try:
        import requests

        s = requests.Session()
        s.trust_env = os.environ.get("FORCE_SYSTEM_PROXY", "") == "1"
        r = s.get(url, timeout=40, headers={"User-Agent": UA})
        r.raise_for_status()
        return r.json()
    except Exception as e:  # noqa: BLE001 — любой сетевой/SSL сбой → curl
        log.debug("requests failed (%s), curl fallback", e)
        return _curl_json(url)


# ---------------------------- выборка ----------------------------

def candles(secid: str, interval: int, frm: str, till: str | None = None,
            market: dict | None = None) -> list[dict]:
    """Свечи за период [frm, till] с пагинацией. interval: 1|10|60|24 (мин/день)."""
    m = market or SHARES
    path = (f"/engines/{m['engine']}/markets/{m['market']}/boards/{m['board']}"
            f"/securities/{secid}/candles.json")
    bars: list[dict] = []
    start = 0
    while True:
        d = _get_json(_full_url(path, {
            "iss.meta": "off", "interval": interval,
            "from": frm, "till": till, "start": start,
        }))
        block = d.get("candles") or {}
        cols, rows = block.get("columns") or [], block.get("data") or []
        if not cols:
            break
        for r in rows:
            m2 = dict(zip(cols, r))
            bars.append({
                "ts": m2["begin"], "open": m2["open"], "close": m2["close"],
                "high": m2["high"], "low": m2["low"],
                "volume": m2["volume"] or 0,
                # денежный оборот бара, ₽ — нужен для фильтра ликвидности
                "value": m2.get("value") or 0,
            })
        if len(rows) < PAGE:
            break
        start += len(rows)
    # дедуп и порядок: ISS иногда отдаёт стык страниц дважды
    uniq = {b["ts"]: b for b in bars}
    return [uniq[k] for k in sorted(uniq)]


def path_for(secid: str, interval: int) -> pathlib.Path:
    return HISTORY_DIR / f"{secid}_{interval}.json"


def _is_fresh(p: pathlib.Path, hours: int = FRESH_HOURS) -> bool:
    if not p.exists():
        return False
    import time
    return (time.time() - p.stat().st_mtime) / 3600 < hours


def sync(secids: list[str], interval: int, days: int, market: dict | None = None,
         force: bool = False, workers: int = 8) -> dict:
    """Скачать/обновить кеш по списку бумаг. Возвращает {secid: баров}.

    Параллельно (по умолчанию 8 потоков): ISS ограничивает 500 строк на ответ,
    поэтому 200 дней 10-мин баров — это ~35 запросов на бумагу.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    frm = (date.today() - timedelta(days=days)).isoformat()
    result: dict[str, int] = {}

    def one(secid: str) -> tuple[str, int]:
        p = path_for(secid, interval)
        if not force and _is_fresh(p):
            try:
                return secid, len(json.loads(p.read_text(encoding="utf-8"))["bars"])
            except Exception:
                pass
        bars = candles(secid, interval, frm, market=market)
        p.write_text(json.dumps({"secid": secid, "interval": interval,
                                 "from": frm, "bars": bars},
                                ensure_ascii=False), encoding="utf-8")
        return secid, len(bars)

    done = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futs = {ex.submit(one, s): s for s in secids}
        for fut in as_completed(futs):
            secid = futs[fut]
            done += 1
            try:
                _, n = fut.result()
                result[secid] = n
                log.info("[%d/%d] %s i=%s: %d баров", done, len(secids), secid, interval, n)
            except Exception as e:  # noqa: BLE001 — одна бумага не должна ронять синк
                result[secid] = -1
                log.warning("[%d/%d] %s i=%s: FAILED %s: %s", done, len(secids), secid,
                            interval, type(e).__name__, e)
    return result


def load(secid: str, interval: int) -> list[dict]:
    """Свечи из кеша (старые → новые). Пусто, если кеша нет."""
    p = path_for(secid, interval)
    if not p.exists():
        return []
    return json.loads(p.read_text(encoding="utf-8")).get("bars", [])


def sync_all(secids: list[str], days: int = 200) -> dict:
    """10-мин + дневные по бумагам и IMOEX (10-мин + дневные) для контекста.

    HISTORY_WORKERS: сколько запросов параллельно (по умолчанию 3). ISS заметно
    проседает при 8 потоках — ответы уходят в таймаут, поэтому умеренно.
    """
    import os as _os

    workers = int(_os.environ.get("HISTORY_WORKERS", "3"))
    out: dict[str, dict] = {}
    out["intraday"] = sync(secids, 10, days, workers=workers)
    out["daily"] = sync(secids, 24, days + 200, workers=workers)  # дневным нужно больше истории (SMA50/ATR)
    out["index_10"] = sync(["IMOEX"], 10, days, market=INDEX, workers=workers)
    out["index_d"] = sync(["IMOEX"], 24, days + 200, market=INDEX, workers=workers)
    return out


if __name__ == "__main__":
    import argparse
    import logging as _l

    ap = argparse.ArgumentParser(description="Кеш исторических свечей MOEX")
    ap.add_argument("--days", type=int, default=200, help="календарных дней истории")
    ap.add_argument("--force", action="store_true", help="перекачать, игнорируя свежесть кеша")
    args = ap.parse_args()

    _l.basicConfig(level=_l.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    from .settings import CONFIG

    tickers = [w["ticker"] for w in CONFIG["watchlist"]]
    if args.force:
        for t in tickers + ["IMOEX"]:
            for interval in (10, 24):
                p = path_for(t, interval)
                if p.exists():
                    p.unlink()
    res = sync_all(tickers, days=args.days)
    print(json.dumps(res, ensure_ascii=False, indent=2))
