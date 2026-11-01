"""Постоянное хранилище на Supabase Postgres (переживает рестарты Render).

Что хранится НАВСЕГДА (в отличие от эфемерного SQLite):
  - trades    — сделки пользователя (трекер + статистика P&L)
  - positions — активные позиции со стопом/целью (алерты о срабатывании)
  - signals_log — история сигналов (для бэктеста с учётом комиссии)
  - meta      — выборочные постоянные настройки (focus_pool)

Если Supabase недоступен (нет пароля/сети) — падаем на SQLite-фолбек
(те же таблицы в data/persist.db), чтобы бот продолжал работать локально.
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
from datetime import datetime, timezone

from .settings import DATA_DIR  # noqa: E402  (грузит .env ДО чтения env-переменных)

log = logging.getLogger("store")

# --- Supabase connection (из .env) ---
SUPABASE_PASSWORD = os.getenv("SUPABASE_PASSWORD", "")
SUPABASE_REF = os.getenv("SUPABASE_REF", "rapqyiylqtxkbzupoqdc")
SUPABASE_REGION = os.getenv("SUPABASE_REGION", "eu-west-1")

SQLITE_PATH = DATA_DIR / "persist.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS trades (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    ticker TEXT NOT NULL,
    side TEXT NOT NULL DEFAULT 'long',
    entry_price NUMERIC,
    exit_price NUMERIC,
    qty NUMERIC,
    fee_pct NUMERIC DEFAULT 1.0,
    mode TEXT NOT NULL DEFAULT 'real',
    note TEXT
);
CREATE TABLE IF NOT EXISTS positions (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    ticker TEXT NOT NULL,
    entry_price NUMERIC,
    stop NUMERIC,
    target NUMERIC,
    mode TEXT NOT NULL DEFAULT 'real',
    status TEXT NOT NULL DEFAULT 'open'
);
CREATE TABLE IF NOT EXISTS signals_log (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    kind TEXT NOT NULL,
    ticker TEXT,
    direction TEXT,
    confidence TEXT,
    payload JSONB
);
CREATE TABLE IF NOT EXISTS meta_persist (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

# SQLite-эквивалент схемы (fallback)
_SCHEMA_SQLITE = """
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    ticker TEXT NOT NULL,
    side TEXT NOT NULL DEFAULT 'long',
    entry_price REAL,
    exit_price REAL,
    qty REAL,
    fee_pct REAL DEFAULT 1.0,
    mode TEXT NOT NULL DEFAULT 'real',
    note TEXT
);
CREATE TABLE IF NOT EXISTS positions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    ticker TEXT NOT NULL,
    entry_price REAL,
    stop REAL,
    target REAL,
    mode TEXT NOT NULL DEFAULT 'real',
    status TEXT NOT NULL DEFAULT 'open'
);
CREATE TABLE IF NOT EXISTS signals_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    kind TEXT NOT NULL,
    ticker TEXT,
    direction TEXT,
    confidence TEXT,
    payload TEXT
);
CREATE TABLE IF NOT EXISTS meta_persist (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

_conn = None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _pg_dsn() -> str:
    return (f"host=aws-0-{SUPABASE_REGION}.pooler.supabase.com port=6543 "
            f"dbname=postgres user=postgres.{SUPABASE_REF} "
            f"password={SUPABASE_PASSWORD} connect_timeout=10 sslmode=require")


def available() -> bool:
    return bool(SUPABASE_PASSWORD)


def init():
    """Инициализация: подключение + создание таблиц."""
    global _conn
    if _conn is not None:
        return
    if available():
        try:
            import psycopg2
            import psycopg2.extras
            _conn = psycopg2.connect(_pg_dsn())
            _conn.autocommit = True
            cur = _conn.cursor()
            for stmt in _SCHEMA.split(";"):
                if stmt.strip():
                    cur.execute(stmt)
            # миграция старых таблиц (добавляем колонку mode, если её нет)
            for stmt in ("ALTER TABLE trades ADD COLUMN IF NOT EXISTS mode TEXT NOT NULL DEFAULT 'real'",
                         "ALTER TABLE positions ADD COLUMN IF NOT EXISTS mode TEXT NOT NULL DEFAULT 'real'"):
                cur.execute(stmt)
            cur.close()
            log.info("store: подключено к Supabase Postgres")
            return
        except Exception as e:
            log.warning("store: Supabase недоступен (%s) — фолбек на SQLite", str(e)[:120])
            _conn = None
    # fallback SQLite
    _conn = sqlite3.connect(SQLITE_PATH, timeout=30)
    _conn.row_factory = sqlite3.Row
    cur = _conn.cursor()
    for stmt in _SCHEMA_SQLITE.split(";"):
        if stmt.strip():
            cur.execute(stmt)
    # миграция SQLite (старые таблицы без mode)
    for col, tbl in (("mode", "trades"), ("mode", "positions")):
        try:
            cur.execute(f"ALTER TABLE {tbl} ADD COLUMN {col} TEXT NOT NULL DEFAULT 'real'")
        except sqlite3.OperationalError:
            pass  # колонка уже есть
    cur.close()
    _conn.commit()
    log.info("store: локальный SQLite persist.db")


def _is_pg() -> bool:
    return _conn is not None and not isinstance(_conn, sqlite3.Connection)


def _exec(sql: str, params: tuple = ()):
    cur = _conn.cursor()
    cur.execute(sql, params)
    if not _is_pg():
        _conn.commit()
    cur.close()


def _query(sql: str, params: tuple = ()) -> list[dict]:
    cur = _conn.cursor()
    cur.execute(sql, params)
    if _is_pg():
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description]
        out = [dict(zip(cols, r)) for r in rows]
    else:
        out = [dict(r) for r in cur.fetchall()]
    cur.close()
    return out


# ----------------------- trades -----------------------

def add_trade(ticker: str, entry_price: float, side: str = "long",
              qty: float | None = None, fee_pct: float = 1.0, note: str = "",
              mode: str = "real") -> int:
    init()
    _exec("INSERT INTO trades(ticker, side, entry_price, qty, fee_pct, mode, note, created_at) "
          "VALUES(%s,%s,%s,%s,%s,%s,%s,%s)" if _is_pg() else
          "INSERT INTO trades(ticker, side, entry_price, qty, fee_pct, mode, note, created_at) "
          "VALUES(?,?,?,?,?,?,?,?)",
          (ticker, side, entry_price, qty, fee_pct, mode, note, _now()))
    row = _query("SELECT MAX(id) AS id FROM trades")
    return row[0]["id"] if row and row[0]["id"] else 0


def close_trade(trade_id: int, exit_price: float, qty: float | None = None, note: str = "") -> dict | None:
    init()
    row = _query("SELECT * FROM trades WHERE id=%s", (trade_id,)) if _is_pg() \
        else _query("SELECT * FROM trades WHERE id=?", (trade_id,))
    if not row:
        return None
    t = row[0]
    entry = float(t["entry_price"])
    q = float(qty if qty is not None else (t["qty"] or 0))
    fee = float(t["fee_pct"] if t["fee_pct"] is not None else 1.0)
    pnl_pct = ((exit_price - entry) / entry * 100 - fee) if entry else 0
    _exec("UPDATE trades SET exit_price=%s, qty=%s, note=%s WHERE id=%s" if _is_pg() else
          "UPDATE trades SET exit_price=?, qty=?, note=? WHERE id=?",
          (exit_price, q, note, trade_id))
    return {"id": trade_id, "ticker": t["ticker"], "entry": entry,
            "exit": exit_price, "pnl_pct": round(pnl_pct, 2)}


def list_trades(days: int = 30, mode: str | None = None) -> list[dict]:
    init()
    if _is_pg():
        if mode:
            return _query("SELECT * FROM trades WHERE created_at > now() - interval '%s days' "
                          "AND mode=%s ORDER BY id", (days, mode))
        return _query("SELECT * FROM trades WHERE created_at > now() - interval '%s days' ORDER BY id", (days,))
    from datetime import timedelta
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    if mode:
        return _query("SELECT * FROM trades WHERE created_at > ? AND mode=? ORDER BY id", (cutoff, mode))
    return _query("SELECT * FROM trades WHERE created_at > ? ORDER BY id", (cutoff,))


def trade_stats(days: int = 30, mode: str | None = None) -> dict:
    trades = [t for t in list_trades(days, mode) if t["exit_price"] is not None]
    if not trades:
        return {"closed": 0, "open": 0, "wins": 0, "losses": 0, "win_rate": None,
                "avg_pnl_pct": None, "total_pnl_pct": 0.0}
    wins = losses = 0
    total = 0.0
    for t in trades:
        entry = float(t["entry_price"])
        exitp = float(t["exit_price"])
        fee = float(t["fee_pct"] if t["fee_pct"] is not None else 1.0)
        pnl = (exitp - entry) / entry * 100 - fee
        total += pnl
        if pnl > 0:
            wins += 1
        else:
            losses += 1
    n = len(trades)
    return {"closed": n, "open": 0, "wins": wins, "losses": losses,
            "win_rate": round(wins / n * 100, 1),
            "avg_pnl_pct": round(total / n, 2),
            "total_pnl_pct": round(total, 2)}


# ----------------------- positions -----------------------

def open_position(ticker: str, entry_price: float, stop: float, target: float,
                  mode: str = "real") -> int:
    init()
    _exec("INSERT INTO positions(ticker, entry_price, stop, target, mode, status, created_at) "
          "VALUES(%s,%s,%s,%s,%s,'open',%s)" if _is_pg() else
          "INSERT INTO positions(ticker, entry_price, stop, target, mode, status, created_at) "
          "VALUES(?,?,?,?,?,'open',?)",
          (ticker, entry_price, stop, target, mode, _now()))
    row = _query("SELECT MAX(id) AS id FROM positions")
    return row[0]["id"] if row and row[0]["id"] else 0


def close_position(pos_id: int, reason: str = "manual"):
    init()
    _exec("UPDATE positions SET status=%s WHERE id=%s" if _is_pg() else
          "UPDATE positions SET status=? WHERE id=?",
          (reason, pos_id))


def active_positions(mode: str | None = None) -> list[dict]:
    init()
    if mode and _is_pg():
        return _query("SELECT * FROM positions WHERE status='open' AND mode=%s ORDER BY id", (mode,))
    if mode:
        return _query("SELECT * FROM positions WHERE status='open' AND mode=? ORDER BY id", (mode,))
    return _query("SELECT * FROM positions WHERE status='open' ORDER BY id")


# ----------------------- signals (backtest) -----------------------

def log_signal(kind: str, payload: dict | list):
    init()
    if isinstance(payload, list):
        for s in payload:
            if isinstance(s, dict):
                _log_one_signal(kind, s)
    elif isinstance(payload, dict):
        _log_one_signal(kind, payload)


def _log_one_signal(kind: str, s: dict):
    ticker = s.get("ticker")
    direction = s.get("direction")
    confidence = s.get("confidence")
    payload = json.dumps(s, ensure_ascii=False)
    if _is_pg():
        _exec("INSERT INTO signals_log(kind, ticker, direction, confidence, payload, created_at) "
              "VALUES(%s,%s,%s,%s,%s::jsonb,%s)",
              (kind, ticker, direction, confidence, payload, _now()))
    else:
        _exec("INSERT INTO signals_log(kind, ticker, direction, confidence, payload, created_at) "
              "VALUES(?,?,?,?,?,?)",
              (kind, ticker, direction, confidence, payload, _now()))


def signals_history(days: int = 30) -> list[dict]:
    init()
    if _is_pg():
        rows = _query("SELECT * FROM signals_log WHERE created_at > now() - interval '%s days' ORDER BY id", (days,))
    else:
        from datetime import timedelta
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        rows = _query("SELECT * FROM signals_log WHERE created_at > ? ORDER BY id", (cutoff,))
    out = []
    for r in rows:
        d = dict(r)
        try:
            d["payload"] = json.loads(d["payload"])
        except Exception:
            pass
        out.append(d)
    return out


# ----------------------- meta (постоянные настройки) -----------------------

def get_meta(key: str, default: str = "") -> str:
    init()
    row = _query("SELECT value FROM meta_persist WHERE key=%s" if _is_pg() else
                 "SELECT value FROM meta_persist WHERE key=?", (key,))
    return row[0]["value"] if row else default


def set_meta(key: str, value: str):
    init()
    if _is_pg():
        _exec("INSERT INTO meta_persist(key, value) VALUES(%s, %s) "
              "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value", (key, value))
    else:
        _exec("INSERT INTO meta_persist(key, value) VALUES(?, ?) "
              "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, value))
