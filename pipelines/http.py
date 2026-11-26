"""HTTP-хелпер.

Важно: MOEX ISS (и некоторые другие российские сервисы) режут TLS-отпечаток
Python/OpenSSL (ошибка UNEXPECTED_EOF_WHILE_READING). Поэтому все GET идут
сначала через requests, а при SSL/сетевой ошибке — через системный curl,
который использует другой TLS-стек и проходит.

Вторая беда (2026-11-21…27): у части egress-точек цепочка сертификатов MOEX
не проходит проверку вообще (`curl (60) self-signed certificate in chain`),
и бот остаётся без котировок. Для таких хостов есть третий, явно
разрешённый конфигом запасной путь без проверки сертификата:
общий ключ `moex_insecure_tls: true` в config.yaml. Риск осознанный:
подмены цен это не исключает, зато возвращает боту данные.
"""
from __future__ import annotations

import json
import os
import subprocess
import urllib.parse

import requests

UA = "Mozilla/5.0 (X11; Linux x86_64) trade-bot/1.0"
TIMEOUT = 30

# Хосты, для которых разрешён запасной путь без проверки сертификата
# (публичные рыночные данные, без ключей и авторизации).
INSECURE_HOSTS = ("iss.moex.com",)
_WARNED: set[str] = set()

# Системный прокси (частая причина обрывов с российскими хостами) игнорируем.
# Если прокси реально нужен — задайте переменную окружения FORCE_SYSTEM_PROXY=1.
_SESSION = requests.Session()
_SESSION.trust_env = os.environ.get("FORCE_SYSTEM_PROXY", "") != "1"

session = _SESSION  # общий прокси-независимый клиент для остальных модулей


def _url_with_params(url: str, params: dict | None) -> str:
    if not params:
        return url
    qs = urllib.parse.urlencode(params, doseq=True)
    return url + ("&" if "?" in url else "?") + qs


def _insecure_allowed(url: str) -> bool:
    """Разрешён ли для этого URL запасной путь без проверки сертификата."""
    if not any(h in url for h in INSECURE_HOSTS):
        return False
    try:
        from .settings import CONFIG
        return bool(CONFIG.get("moex_insecure_tls"))
    except Exception:  # noqa: BLE001 — конфиг недоступен: считаем, что нельзя
        return False


def _curl_get(url: str, timeout: int, insecure: bool = False) -> str:
    cmd = ["curl", "-sS", "--fail", "--silent", "--show-error",
           "--max-time", str(timeout), "-A", UA]
    if insecure:
        cmd.append("-k")
    cmd.append(url)
    out = subprocess.run(cmd, capture_output=True, text=True,
                         encoding="utf-8", errors="replace")
    if out.returncode != 0:
        raise RuntimeError(f"curl failed: {out.stderr.strip()[:300]}")
    return out.stdout


def get_json(url: str, params: dict | None = None, timeout: int = TIMEOUT):
    """GET -> JSON: requests → curl → (для MOEX при разрешении) curl -k."""
    full = _url_with_params(url, params)
    err: Exception | None = None
    try:
        r = _SESSION.get(full, timeout=timeout, headers={"User-Agent": UA})
        r.raise_for_status()
        return r.json()
    except (requests.RequestException, json.JSONDecodeError) as e:
        err = e
    try:
        return json.loads(_curl_get(full, timeout))
    except Exception as e:  # noqa: BLE001
        last = e
    if _insecure_allowed(full):
        try:
            import logging
            host = urllib.parse.urlsplit(full).netloc
            if host not in _WARNED:
                _WARNED.add(host)
                logging.getLogger("http").warning(
                    "TLS не проверяется для %s: %s", host, str(last)[:120])
            return json.loads(_curl_get(full, timeout, insecure=True))
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"GET {url} failed: {err}; insecure fallback: {e}") from e
    raise RuntimeError(f"GET {url} failed: {err}; curl: {last}") from last


def post_json(url: str, payload: dict, headers: dict | None = None,
              timeout: int = 90) -> dict:
    r = _SESSION.post(url, json=payload, headers=headers, timeout=timeout)
    r.raise_for_status()
    return r.json()


def post_multipart(url: str, data: dict, files: dict, headers: dict | None = None,
                   timeout: int = 120) -> dict:
    r = _SESSION.post(url, data=data, files=files, headers=headers, timeout=timeout)
    r.raise_for_status()
    return r.json()
