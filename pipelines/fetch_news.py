"""Сбор новостей: RSS-ленты + опционально Exa-поиск.

- дедупликация по хэшу (title+url) в SQLite
- привязка новостей к тикерам watchlist по ключевым словам
- эвристический impact-скоринг 1..5 (важность для российского рынка)
"""
from __future__ import annotations

import html
import json
import logging
import re
import time
from datetime import datetime, timezone

import feedparser

from . import db
from .http import post_json
from .settings import CONFIG, EXA_API_KEY

log = logging.getLogger("news")

NEWS_MAX_AGE_H = 72

# Эвристический скоринг: суммарный вес совпадений -> impact 1..5
IMPACT_KEYWORDS: list[tuple[str, int]] = [
    (r"дефолт|банкрот|отзыв лицензи|обвал|крах", 3),
    (r"ставк\w* цб|ключев\w* ставк|решени\w* цб|совет директоров банка россии", 3),
    (r"санкц", 3),
    (r"дивиденд", 3),
    (r"удар\w*|ракет|обстрел|атак\w* дрон|вторжен|мобилиз", 3),
    (r"перемири|мирн\w* договор|мирных переговор|прекращени\w* огня", 3),
    (r"отчетност|мсфо|рсбу|выручк|прибыл\w*|годуем", 2),
    (r"инфляц|ключевая ставка|дефлятор", 2),
    (r"слияни|поглощен|выкуп\w* акций|байбэк|допэмисси|размещени\w* акций", 2),
    (r"рейтинг (повысил|понизил)|прогноз|цель.*цен|подорожал|подешевел", 1),
    (r"нефт|газ|opec|uran|золот|металл|экспортн\w* пошлин", 1),
    (r"эмбарго|потолок цен|трубопровод", 2),
]
COMPILED = [(re.compile(p, re.IGNORECASE), w) for p, w in IMPACT_KEYWORDS]


def score_impact(text: str) -> int:
    score = sum(w for rx, w in COMPILED if rx.search(text))
    return max(1, min(5, 1 + score // 2)) if score else 1


def match_tickers(text: str) -> list[str]:
    t = text.lower()
    out = []
    for w in CONFIG["watchlist"]:
        for kw in w.get("keywords", []):
            if kw.lower() in t:
                out.append(w["ticker"])
                break
    return out


def _clean(s: str) -> str:
    s = re.sub(r"<[^>]+>", " ", s or "")
    return html.unescape(re.sub(r"\s+", " ", s)).strip()


def fetch_feeds() -> list[dict]:
    items = []
    for feed in CONFIG.get("feeds", []):
        try:
            parsed = feedparser.parse(feed["url"])
            for e in parsed.entries[:30]:
                title = _clean(getattr(e, "title", ""))
                link = getattr(e, "link", "")
                if not title or not link:
                    continue
                pub = ""
                if getattr(e, "published_parsed", None):
                    pub = datetime(*e.published_parsed[:6], tzinfo=timezone.utc) \
                        .strftime("%Y-%m-%d %H:%M:%S")
                    age_h = (datetime.now(timezone.utc) -
                             datetime(*e.published_parsed[:6], tzinfo=timezone.utc)).total_seconds() / 3600
                    if age_h > NEWS_MAX_AGE_H:
                        continue
                text = title + " " + _clean(getattr(e, "summary", ""))[:300]
                items.append({
                    "source": feed["name"], "title": title, "url": link,
                    "summary": _clean(getattr(e, "summary", ""))[:400],
                    "published": pub,
                    "tickers": match_tickers(text),
                    "impact": score_impact(text),
                })
        except Exception as e:
            log.warning("feed %s: %s", feed["name"], e)
    return items


def fetch_exa() -> list[dict]:
    """Семантический новостной поиск через Exa (если есть ключ)."""
    if not EXA_API_KEY:
        return []
    items = []
    for q in CONFIG.get("exa_queries", []):
        try:
            d = post_json(
                "https://api.exa.ai/search",
                payload={"query": q, "numResults": 8, "category": "news",
                         "startPublishedDaysAgo": 2},
                headers={"x-api-key": EXA_API_KEY},
            )
            for r in d.get("results", []):
                title = _clean(r.get("title", ""))
                url = r.get("url", "")
                if not title or not url:
                    continue
                text = title + " " + (r.get("text") or "")[:300]
                items.append({
                    "source": "Exa", "title": title, "url": url,
                    "summary": (r.get("text") or "")[:400],
                    "published": (r.get("publishedDate") or "")[:19].replace("T", " "),
                    "tickers": match_tickers(text),
                    "impact": score_impact(text),
                })
            time.sleep(1)  # вежливость к API
        except Exception as e:
            log.warning("exa %r: %s", q, e)
    return items


def run() -> list[dict]:
    """Собрать все источники, положить новые в БД. Возвращает новые записи."""
    items = fetch_feeds() + fetch_exa()
    # дедуп внутри батча
    seen, uniq = set(), []
    for it in items:
        h = db.news_hash(it["title"], it["url"])
        if h in seen:
            continue
        seen.add(h)
        uniq.append(it)
    added = db.add_news(uniq)
    log.info("новостей собрано %d, новых %d", len(uniq), added)
    return _only_new(uniq)


def _only_new(uniq: list[dict]) -> list[dict]:
    """Вернуть только те, что реально сохранились как новые (для алертов)."""
    fresh_hashes = set()
    from .db import conn
    with conn() as c:
        for it in uniq:
            row = c.execute("SELECT id FROM news WHERE hash=? AND added_at >= datetime('now','-5 minutes')",
                            (db.news_hash(it["title"], it["url"]),)).fetchone()
            if row:
                fresh_hashes.add(db.news_hash(it["title"], it["url"]))
    return [it for it in uniq if db.news_hash(it["title"], it["url"]) in fresh_hashes]


if __name__ == "__main__":
    db.init_db()
    for it in run():
        print(f"[{it['impact']}] {it['source']}: {it['title'][:90]} -> {it['tickers']}")
