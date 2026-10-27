"""Клиент LLM (OpenAI-совместимый API, по умолчанию DeepSeek) + контроль бюджета."""
from __future__ import annotations

import json
import logging
import time

from . import db
from .http import post_json
from .settings import (DAILY_USD_BUDGET, DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL,
                       DEEPSEEK_MODEL, PRICE_IN_PER_M, PRICE_OUT_PER_M)

log = logging.getLogger("llm")


class BudgetExceeded(RuntimeError):
    pass


def _cost_usd(u: dict) -> float:
    return u["in"] / 1e6 * PRICE_IN_PER_M + u["out"] / 1e6 * PRICE_OUT_PER_M


def check_budget():
    if not DAILY_USD_BUDGET:
        return
    u = db.usage_today()
    spent = _cost_usd(u)
    if spent >= DAILY_USD_BUDGET:
        raise BudgetExceeded(
            f"Дневной бюджет LLM исчерпан: ${spent:.2f} >= ${DAILY_USD_BUDGET:.2f}")


def chat(messages: list[dict], json_mode: bool = False,
         max_tokens: int = 2500, temperature: float = 0.2) -> str:
    """Один вызов chat/completions с ретраем и учётом токенов."""
    if not DEEPSEEK_API_KEY:
        raise RuntimeError("DEEPSEEK_API_KEY не задан (.env)")
    check_budget()

    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    last_err = None
    for attempt in range(3):
        try:
            d = post_json(
                f"{DEEPSEEK_BASE_URL}/chat/completions",
                payload=payload,
                headers={"Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                         "Content-Type": "application/json"},
            )
            u = d.get("usage", {})
            db.bump_usage(u.get("prompt_tokens", 0), u.get("completion_tokens", 0))
            return d["choices"][0]["message"]["content"]
        except Exception as e:
            last_err = e
            wait = 3 * (attempt + 1)
            log.warning("LLM attempt %d failed: %s — retry in %ds", attempt + 1, e, wait)
            time.sleep(wait)
    raise RuntimeError(f"LLM failed after retries: {last_err}")


def chat_json(messages: list[dict], **kw) -> dict:
    """Вызов с json-режимом + страховочный парсинг (некоторые модели любят ```json)."""
    text = chat(messages, json_mode=True, **kw)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # вырезать JSON из ответа с обрамлением
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start:end + 1])
        raise


def usage_report() -> str:
    u = db.usage_today()
    cost = _cost_usd(u)
    budget = f" / ${DAILY_USD_BUDGET:.2f}" if DAILY_USD_BUDGET else ""
    return f"Токены за сегодня: {u['in'] + u['out']:,} (~${cost:.3f}{budget})"
