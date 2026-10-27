"""SQLite-хранилище: котировки, свечи, новости, сигналы, служебные метки."""
from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from .settings import DATA_DIR, MSK

DB_PATH = DATA_DIR / "bot.db"


@contextmanager
def conn():
    c = sqlite3.connect(DB_PATH, timeout=30)
    c.row_factory = sqlite3.Row
    try:
        c.execute("PRAGMA journal_mode=WAL")
        yield c
        c.commit()
    finally:
        c.close()


def now_iso() -> str:
    """UTC в формате SQLite 'YYYY-MM-DD HH:MM:SS' — чтобы работали datetime('now', ...)."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def minutes_since_meta(key: str, default: float = 1e9) -> float:
    """Минуты с момента, записанного в meta (now_iso). Иначе default."""
    v = get_meta(key)
    if not v:
        return default
    try:
        dt = datetime.fromisoformat(v)
    except ValueError:
        return default
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - dt).total_seconds() / 60


def get_pool() -> list[str]:
    """Пул внимания для интрадей-сканера (по умолчанию SPBE)."""
    raw = get_meta("focus_pool")
    if not raw:
        return ["SPBE"]
    try:
        pool = json.loads(raw)
        return pool if isinstance(pool, list) and pool else ["SPBE"]
    except (ValueError, TypeError):
        return ["SPBE"]


def set_pool(pool: list[str]):
    set_meta("focus_pool", json.dumps(pool))


def init_db():
    with conn() as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS prices (
            secid TEXT, ts TEXT, shortname TEXT,
            last REAL, open REAL, high REAL, low REAL, prev REAL,
            chg_pct REAL, voltoday INTEGER,
            PRIMARY KEY (secid, ts));
        CREATE TABLE IF NOT EXISTS candles (
            secid TEXT, ts TEXT, open REAL, close REAL, high REAL, low REAL,
            volume INTEGER, PRIMARY KEY (secid, ts));
        CREATE TABLE IF NOT EXISTS news (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            hash TEXT UNIQUE, source TEXT, title TEXT, url TEXT,
            summary TEXT, published TEXT, tickers TEXT,
            impact INTEGER DEFAULT 0, added_at TEXT);
        CREATE INDEX IF NOT EXISTS idx_news_added ON news(added_at);
        CREATE INDEX IF NOT EXISTS idx_news_impact ON news(impact);
        CREATE TABLE IF NOT EXISTS signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT, kind TEXT, payload TEXT);
        CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
        """)


# ---------- meta ----------

def get_meta(key: str, default: str = "") -> str:
    with conn() as c:
        row = c.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row["value"] if row else default


def set_meta(key: str, value: str):
    with conn() as c:
        c.execute("INSERT INTO meta(key,value) VALUES(?,?) "
                  "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, value))


# ---------- prices / candles ----------

def save_prices(rows: list[dict]):
    with conn() as c:
        c.executemany(
            "INSERT OR REPLACE INTO prices(secid,ts,shortname,last,open,high,low,prev,chg_pct,voltoday) "
            "VALUES(:secid,:ts,:shortname,:last,:open,:high,:low,:prev,:chg_pct,:voltoday)", rows)


def save_candles(secid: str, rows: list[dict]):
    with conn() as c:
        c.executemany(
            "INSERT OR REPLACE INTO candles(secid,ts,open,close,high,low,volume) "
            "VALUES(?,?,?,?,?,?,?)",
            [(secid, r["ts"], r["open"], r["close"], r["high"], r["low"], r["volume"]) for r in rows])


def latest_prices() -> dict[str, dict]:
    """Последний снапшот цен по каждому тикеру."""
    with conn() as c:
        rows = c.execute(
            "SELECT p.* FROM prices p "
            "JOIN (SELECT secid, MAX(ts) AS mts FROM prices GROUP BY secid) m "
            "ON p.secid=m.secid AND p.ts=m.mts").fetchall()
        return {r["secid"]: dict(r) for r in rows}


def quote_stats(secid: str) -> dict | None:
    """Техсводка по дневным свечам: последний close, изменение за 1 и 5 дней, объём."""
    with conn() as c:
        rows = c.execute(
            "SELECT ts, close, volume FROM candles WHERE secid=? ORDER BY ts DESC LIMIT 6",
            (secid,)).fetchall()
    if len(rows) < 2:
        return None
    last = rows[0]["close"]
    prev = rows[1]["close"]
    chg_1d = round((last / prev - 1) * 100, 2) if prev else None
    chg_5d = None
    if len(rows) >= 6 and rows[5]["close"]:
        chg_5d = round((last / rows[5]["close"] - 1) * 100, 2)
    vols = [r["volume"] for r in rows[1:6] if r["volume"]]
    vol_note = ""
    if vols and rows[0]["volume"]:
        ratio = rows[0]["volume"] / (sum(vols) / len(vols))
        if ratio >= 1.5:
            vol_note = f", объём повышенный (x{ratio:.1f})"
    return {"last": last, "chg_1d": chg_1d, "chg_5d": chg_5d, "volume": vol_note}


def price_at(secid: str, ts: str) -> float | None:
    """Ближайшая зафиксированная цена ПОСЛЕ момента ts (для проверки сигналов)."""
    with conn() as c:
        row = c.execute(
            "SELECT last FROM prices WHERE secid=? AND ts>=? ORDER BY ts LIMIT 1",
            (secid, ts)).fetchone()
        return row["last"] if row else None


def get_candles(secid: str, limit: int = 400) -> list[dict]:
    """Свечи по тикеру, старые -> новые (для индикаторов)."""
    with conn() as c:
        rows = c.execute(
            "SELECT ts, open, close, high, low, volume FROM candles "
            "WHERE secid=? ORDER BY ts DESC LIMIT ?", (secid, limit)).fetchall()
        return [dict(r) for r in reversed(rows)]


# ---------- news ----------

def news_hash(title: str, url: str) -> str:
    return hashlib.sha256((title.strip().lower() + url).encode()).hexdigest()[:20]


def add_news(items: list[dict]) -> int:
    """Дедуп по хэшу. Возвращает число реально новых записей."""
    added = 0
    with conn() as c:
        for it in items:
            try:
                cur = c.execute(
                    "INSERT OR IGNORE INTO news(hash,source,title,url,summary,published,tickers,impact,added_at) "
                    "VALUES(?,?,?,?,?,?,?,?,?)",
                    (news_hash(it["title"], it["url"]), it["source"], it["title"], it["url"],
                     it.get("summary", "")[:800], it.get("published", ""),
                     json.dumps(it.get("tickers", []), ensure_ascii=False),
                     it.get("impact", 0), now_iso()))
                added += cur.rowcount
            except sqlite3.Error:
                continue
    return added


def recent_news(hours: int = 48, limit: int = 200) -> list[dict]:
    with conn() as c:
        rows = c.execute(
            "SELECT * FROM news WHERE added_at >= datetime('now', ?) ORDER BY published DESC LIMIT ?",
            (f"-{hours} hours", limit)).fetchall()
        return [dict(r) | {"tickers": json.loads(r["tickers"] or "[]")} for r in rows]


def high_impact_unalerted(threshold: int) -> list[dict]:
    with conn() as c:
        rows = c.execute(
            "SELECT * FROM news WHERE impact >= ? AND added_at >= datetime('now', '-3 hours') "
            "ORDER BY impact DESC", (threshold,)).fetchall()
        return [dict(r) | {"tickers": json.loads(r["tickers"] or "[]")} for r in rows]


def mark_news_alerted(ids: list[int]):
    with conn() as c:
        c.executemany("UPDATE news SET impact = -impact WHERE id=?", [(i,) for i in ids])


# ---------- signals ----------

def save_signal(kind: str, payload: dict | list):
    with conn() as c:
        c.execute("INSERT INTO signals(created_at,kind,payload) VALUES(?,?,?)",
                  (now_iso(), kind, json.dumps(payload, ensure_ascii=False)))


def signals_since(hours: int = 24) -> list[dict]:
    with conn() as c:
        rows = c.execute(
            "SELECT * FROM signals WHERE created_at >= datetime('now', ?) ORDER BY id",
            (f"-{hours} hours",)).fetchall()
        return [dict(r) | {"payload": json.loads(r["payload"])} for r in rows]


def alerts_last_hour() -> int:
    with conn() as c:
        row = c.execute(
            "SELECT COUNT(*) AS n FROM signals WHERE kind='alert' "
            "AND created_at >= datetime('now', '-1 hours')").fetchone()
        return row["n"]


# ---------- usage (бюджет LLM) ----------

def bump_usage(prompt_tokens: int, completion_tokens: int):
    day = datetime.now(ZoneInfo(MSK)).strftime("%Y-%m-%d")
    with conn() as c:
        row = c.execute("SELECT value FROM meta WHERE key=?", (f"usage:{day}",)).fetchone()
        u = json.loads(row["value"]) if row else {"in": 0, "out": 0}
        u["in"] += prompt_tokens
        u["out"] += completion_tokens
        c.execute("INSERT INTO meta(key,value) VALUES(?,?) "
                  "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                  (f"usage:{day}", json.dumps(u)))


def usage_today() -> dict:
    day = datetime.now(ZoneInfo(MSK)).strftime("%Y-%m-%d")
    with conn() as c:
        row = c.execute("SELECT value FROM meta WHERE key=?", (f"usage:{day}",)).fetchone()
    u = json.loads(row["value"]) if row else {"in": 0, "out": 0}
    return u
