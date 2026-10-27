"""HTTP-хелпер.

Важно: MOEX ISS (и некоторые другие российские сервисы) режут TLS-отпечаток
Python/OpenSSL (ошибка UNEXPECTED_EOF_WHILE_READING). Поэтому все GET идут
сначала через requests, а при SSL/сетевой ошибке — через системный curl,
который использует другой TLS-стек и проходит.
"""
from __future__ import annotations

import json
import subprocess
import urllib.parse

import requests

UA = "Mozilla/5.0 (X11; Linux x86_64) trade-bot/1.0"
TIMEOUT = 30

# Системный прокси (частая причина обрывов с российскими хостами) игнорируем.
# Если прокси реально нужен — задайте переменную окружения FORCE_SYSTEM_PROXY=1.
import os
_SESSION = requests.Session()
_SESSION.trust_env = os.environ.get("FORCE_SYSTEM_PROXY", "") != "1"

session = _SESSION  # общий прокси-независимый клиент для остальных модулей


def _url_with_params(url: str, params: dict | None) -> str:
    if not params:
        return url
    qs = urllib.parse.urlencode(params, doseq=True)
    return url + ("&" if "?" in url else "?") + qs


def _curl_get(url: str, timeout: int) -> str:
    out = subprocess.run(
        ["curl", "-sS", "--fail", "--silent", "--show-error",
         "--max-time", str(timeout), "-A", UA, url],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    if out.returncode != 0:
        raise RuntimeError(f"curl failed: {out.stderr.strip()[:300]}")
    return out.stdout


def get_json(url: str, params: dict | None = None, timeout: int = TIMEOUT):
    """GET -> JSON, с fallback на curl при сетевых/SSL ошибках."""
    full = _url_with_params(url, params)
    try:
        r = _SESSION.get(full, timeout=timeout, headers={"User-Agent": UA})
        r.raise_for_status()
        return r.json()
    except (requests.RequestException, json.JSONDecodeError) as e:
        raw = _curl_get(full, timeout)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            raise RuntimeError(f"GET {url} failed: {e}; curl returned non-JSON") from e


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
