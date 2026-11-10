"""Постоянное хранилище на Supabase Postgres (переживает рестарты Render).

Что хранится НАВСЕГДА (в отличие от эфемерного SQLite):
  - trades    — сделки пользователя (трекер + статистика P&L)
  - positions — активные позиции со стопом/целью (алерты о срабатывании)
  - signals_log — история сигналов (для бэктеста с учётом комиссии)
  - meta_persist — постоянные настройки (focus_pool, кулдауны, старт эксперимента)

Если Supabase недоступен при СТАРТЕ — SQLite-фолбек (data/persist.db).
Если соединение оборвалось в работе — переподключаемся, а не молча уходим в SQLite.
Все операции защищены блокировкой (бот и планировщик работают в разных потоках).
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
from datetime import datetime, timezone

from .settings import CONFIG, DATA_DIR  # noqa: E402  (грузит .env ДО чтения env)

try:
    import psycopg2
    from psycopg2 import InterfaceError, OperationalError
    _PG_RETRY_ERRORS = (OperationalError, InterfaceError)
except ImportError:  # pragma: no cover
    psycopg2 = None
    _PG_RETRY_ERRORS = ()

log = logging.getLogger("store")

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
_backend = None  # "pg" | "sqlite"
_lock = threading.RLock()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _fee_pct(fee_pct: float | None = None) -> float:
    if fee_pct is not None:
        return float(fee_pct)
    return float((CONFIG.get("intraday") or {}).get("fee_pct", 1.0))


def _pg_dsn() -> str:
    return (f"host=aws-0-{SUPABASE_REGION}.pooler.supabase.com port=6543 "
            f"dbname=postgres user=postgres.{SUPABASE_REF} "
            f"password={SUPABASE_PASSWORD} connect_timeout=10 sslmode=require")


def available() -> bool:
    return bool(SUPABASE_PASSWORD)


def _connect_pg():
    import psycopg2.extras
    c = psycopg2.connect(_pg_dsn())
    c.autocommit = True
    cur = c.cursor()
    for stmt in _SCHEMA.split(";"):
        if stmt.strip():
            cur.execute(stmt)
    for stmt in ("ALTER TABLE trades ADD COLUMN IF NOT EXISTS mode TEXT NOT NULL DEFAULT 'real'",
                 "ALTER TABLE positions ADD COLUMN IF NOT EXISTS mode TEXT NOT NULL DEFAULT 'real'"):
        cur.execute(stmt)
    cur.close()
    return c


def _connect_sqlite():
    c = sqlite3.connect(SQLITE_PATH, timeout=30, check_same_thread=False)
    c.row_factory = sqlite3.Row
    cur = c.cursor()
    for stmt in _SCHEMA_SQLITE.split(";"):
        if stmt.strip():
            cur.execute(stmt)
    for col, tbl in (("mode", "trades"), ("mode", "positions")):
        try:
            cur.execute(f"ALTER TABLE {tbl} ADD COLUMN {col} TEXT NOT NULL DEFAULT 'real'")
        except sqlite3.OperationalError:
            pass
    cur.close()
    c.commit()
    return c


def init():
    """Подключение + создание таблиц. Потокобезопасно, с переподключением PG."""
    global _conn, _backend
    with _lock:
        if _conn is not None:
            if _backend == "pg" and getattr(_conn, "closed", 0):
                _conn = None
            else:
                return
        if available() and psycopg2 is not None:
            last_err = None
            for attempt in range(3):
                try:
                    _conn = _connect_pg()
                    _backend = "pg"
                    log.info("store: подключено к Supabase Postgres")
                    return
                except Exception as e:
                    last_err = e
                    _conn = None
                    if attempt < 2:
                        time.sleep(2)
            # Если уже работали на PG — не подменяем историю молчаливым SQLite.
            if _backend == "pg":
                raise RuntimeError(f"Supabase connection lost: {last_err}")
            log.warning("store: Supabase недоступен (%s) — фолбек на SQLite", str(last_err)[:120])
        _conn = _connect_sqlite()
        _backend = "sqlite"
        log.info("store: локальный SQLite persist.db")


def _is_pg() -> bool:
    return _conn is not None and _backend == "pg"


def _reset_conn():
    global _conn
    try:
        if _conn is not None:
            _conn.close()
    except Exception:
        pass
    _conn = None


def _exec(sql: str, params: tuple = ()):
    with _lock:
        init()
        for attempt in range(2):
            cur = _conn.cursor()
            try:
                cur.execute(sql, params)
                if not _is_pg():
                    _conn.commit()
                return
            except _PG_RETRY_ERRORS as e:
                if attempt == 0 and _is_pg():
                    log.warning("store: обрыв PG (%s) — переподключение", str(e)[:120])
                    _reset_conn()
                    init()
                    continue
                raise
            finally:
                try:
                    cur.close()
                except Exception:
                    pass


def _query(sql: str, params: tuple = ()) -> list[dict]:
    with _lock:
        init()
        for attempt in range(2):
            cur = _conn.cursor()
            try:
                cur.execute(sql, params)
                if _is_pg():
                    rows = cur.fetchall()
                    cols = [d[0] for d in cur.description]
                    return [dict(zip(cols, r)) for r in rows]
                return [dict(r) for r in cur.fetchall()]
            except _PG_RETRY_ERRORS as e:
                if attempt == 0 and _is_pg():
                    log.warning("store: обрыв PG (%s) — переподключение", str(e)[:120])
                    _reset_conn()
                    init()
                    continue
                raise
            finally:
                try:
                    cur.close()
                except Exception:
                    pass


def _insert_id(sql_pg: str, sql_sqlite: str, params: tuple) -> int:
    """INSERT с надёжным id: RETURNING для PG, last_insert_rowid для SQLite."""
    with _lock:
        init()
        if _is_pg():
            rows = _query(sql_pg + " RETURNING id", params)
            return int(rows[0]["id"]) if rows else 0
        _exec(sql_sqlite, params)
        rows = _query("SELECT last_insert_rowid() AS id")
        return int(rows[0]["id"]) if rows else 0


# ----------------------- trades -----------------------

def add_trade(ticker: str, entry_price: float, side: str = "long",
              qty: float | None = None, fee_pct: float | None = None, note: str = "",
              mode: str = "real") -> int:
    fee = _fee_pct(fee_pct)
    params = (ticker, side, entry_price, qty, fee, mode, note, _now())
    return _insert_id(
        "INSERT INTO trades(ticker, side, entry_price, qty, fee_pct, mode, note, created_at) "
        "VALUES(%s,%s,%s,%s,%s,%s,%s,%s)",
        "INSERT INTO trades(ticker, side, entry_price, qty, fee_pct, mode, note, created_at) "
        "VALUES(?,?,?,?,?,?,?,?)",
        params)


def close_trade(trade_id: int, exit_price: float, qty: float | None = None, note: str | None = None) -> dict | None:
    init()
    ph = "%s" if _is_pg() else "?"
    rows = _query(f"SELECT * FROM trades WHERE id={ph}", (trade_id,))
    if not rows:
        return None
    t = rows[0]
    entry = float(t["entry_price"])
    q = float(qty if qty is not None else (t["qty"] or 0))
    fee = float(t["fee_pct"] if t["fee_pct"] is not None else _fee_pct())
    pnl_pct = ((exit_price - entry) / entry * 100 - fee) if entry else 0
    # Пустая note НЕ затирает причину входа/выхода.
    _exec(f"UPDATE trades SET exit_price={ph}, qty={ph}, "
          f"note=COALESCE(NULLIF({ph}, ''), note) WHERE id={ph}",
          (exit_price, q, note or "", trade_id))
    return {"id": trade_id, "ticker": t["ticker"], "entry": entry,
            "exit": exit_price, "pnl_pct": round(pnl_pct, 2)}


def record_paper_exit(position: dict, exit_price: float, reason: str) -> dict | None:
    """Атомарно: сделка + закрытие позиции. Чтобы обрыв не терял результат."""
    ticker = position["ticker"]
    entry = float(position["entry_price"])
    pos_id = position["id"]
    fee = _fee_pct()
    note = f"paper:{reason}"
    pnl_pct = round(((exit_price - entry) / entry * 100 - fee) if entry else 0, 2)
    with _lock:
        init()
        if _is_pg():
            old_autocommit = _conn.autocommit
            _conn.autocommit = False
            cur = _conn.cursor()
            try:
                cur.execute(
                    "INSERT INTO trades(ticker, side, entry_price, exit_price, qty, fee_pct, mode, note, created_at) "
                    "VALUES(%s,'long',%s,%s,0,%s,'paper',%s,%s) RETURNING id",
                    (ticker, entry, exit_price, fee, note, _now()))
                tid = int(cur.fetchone()[0])
                cur.execute("UPDATE positions SET status=%s WHERE id=%s",
                            (reason + "_hit", pos_id))
                _conn.commit()
            except Exception:
                _conn.rollback()
                raise
            finally:
                _conn.autocommit = old_autocommit
                cur.close()
        else:
            cur = _conn.cursor()
            try:
                cur.execute(
                    "INSERT INTO trades(ticker, side, entry_price, exit_price, qty, fee_pct, mode, note, created_at) "
                    "VALUES(?,?,?,?,?,?,?,?,?)",
                    (ticker, "long", entry, exit_price, 0, fee, "paper", note, _now()))
                tid = int(cur.lastrowid)
                cur.execute("UPDATE positions SET status=? WHERE id=?",
                            (reason + "_hit", pos_id))
                _conn.commit()
            except Exception:
                _conn.rollback()
                raise
            finally:
                cur.close()
    return {"id": tid, "ticker": ticker, "entry": entry,
            "exit": exit_price, "pnl_pct": pnl_pct}


def list_trades(days: int = 30, mode: str | None = None) -> list[dict]:
    init()
    if _is_pg():
        # %s вне кавычек: psycopg2 не подставляет параметры внутри строковых литералов
        if mode:
            return _query("SELECT * FROM trades WHERE created_at > now() - %s * interval '1 day' "
                          "AND mode=%s ORDER BY id", (days, mode))
        return _query("SELECT * FROM trades WHERE created_at > now() - %s * interval '1 day' ORDER BY id", (days,))
    from datetime import timedelta
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    if mode:
        return _query("SELECT * FROM trades WHERE created_at > ? AND mode=? ORDER BY id", (cutoff, mode))
    return _query("SELECT * FROM trades WHERE created_at > ? ORDER BY id", (cutoff,))


def trade_stats(days: int = 30, mode: str | None = None) -> dict:
    all_rows = list_trades(days, mode)
    trades = [t for t in all_rows if t["exit_price"] is not None]
    try:
        open_n = len(active_positions(mode))
    except Exception:
        open_n = 0
    base = {"closed": 0, "open": open_n, "wins": 0, "losses": 0, "win_rate": None,
            "avg_pnl_pct": None, "total_pnl_pct": 0.0, "max_drawdown_pct": 0.0,
            "avg_fee_pct": None, "first_trade_at": None, "last_trade_at": None}
    if not trades:
        return base
    wins = losses = 0
    total = 0.0
    fees = 0.0
    cum = 0.0
    peak = 0.0
    max_dd = 0.0
    for t in trades:
        entry = float(t["entry_price"])
        exitp = float(t["exit_price"])
        fee = float(t["fee_pct"] if t["fee_pct"] is not None else _fee_pct())
        pnl = (exitp - entry) / entry * 100 - fee
        total += pnl
        fees += fee
        cum += pnl
        peak = max(peak, cum)
        max_dd = max(max_dd, peak - cum)
        if pnl > 0:
            wins += 1
        else:
            losses += 1
    n = len(trades)
    base.update({
        "closed": n, "wins": wins, "losses": losses,
        "win_rate": round(wins / n * 100, 1),
        "avg_pnl_pct": round(total / n, 2),
        "total_pnl_pct": round(total, 2),
        "max_drawdown_pct": round(max_dd, 2),
        "avg_fee_pct": round(fees / n, 2),
        "first_trade_at": trades[0].get("created_at"),
        "last_trade_at": trades[-1].get("created_at"),
    })
    return base


# ----------------------- positions -----------------------

def open_position(ticker: str, entry_price: float, stop: float, target: float,
                  mode: str = "real") -> int:
    params = (ticker, entry_price, stop, target, mode, _now())
    return _insert_id(
        "INSERT INTO positions(ticker, entry_price, stop, target, mode, status, created_at) "
        "VALUES(%s,%s,%s,%s,%s,'open',%s)",
        "INSERT INTO positions(ticker, entry_price, stop, target, mode, status, created_at) "
        "VALUES(?,?,?,?,?,'open',?)",
        params)


def close_position(pos_id: int, reason: str = "manual"):
    init()
    ph = "%s" if _is_pg() else "?"
    _exec(f"UPDATE positions SET status={ph} WHERE id={ph}", (reason, pos_id))


def active_positions(mode: str | None = None) -> list[dict]:
    init()
    ph = "%s" if _is_pg() else "?"
    if mode:
        return _query(f"SELECT * FROM positions WHERE status='open' AND mode={ph} ORDER BY id", (mode,))
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
        rows = _query("SELECT * FROM signals_log WHERE created_at > now() - %s * interval '1 day' ORDER BY id", (days,))
    else:
        from datetime import timedelta
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        rows = _query("SELECT * FROM signals_log WHERE created_at > ? ORDER BY id", (cutoff,))
    out = []
    for r in rows:
        d = dict(r)
        if isinstance(d.get("payload"), str):
            try:
                d["payload"] = json.loads(d["payload"])
            except Exception:
                pass
        out.append(d)
    return out


# ----------------------- meta (постоянные настройки) -----------------------

def get_meta(key: str, default: str = "") -> str:
    init()
    ph = "%s" if _is_pg() else "?"
    rows = _query(f"SELECT value FROM meta_persist WHERE key={ph}", (key,))
    return rows[0]["value"] if rows else default


def set_meta(key: str, value: str):
    init()
    if _is_pg():
        _exec("INSERT INTO meta_persist(key, value) VALUES(%s, %s) "
              "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value", (key, value))
    else:
        _exec("INSERT INTO meta_persist(key, value) VALUES(?, ?) "
              "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, value))
