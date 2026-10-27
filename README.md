# Trade Bot — анализ MOEX + сигналы в Telegram

Торговый ассистент: собирает новости (RSS) и котировки Мосбиржи (MOEX ISS API),
считает теханализ (RSI, MACD, SMA, ATR, Боллинджер, 52-недельные уровни) и
отправляет в Telegram брифинги с сигналами, а также срочные алерты по важным новостям.

## Возможности

- **Брифинг 09:00 (ВЛТ)** — сигнал по каждой бумаге watchlist: long/short/flat,
  уверенность, горизонт, обоснование (новости + техника), риски, источники
- **Итоги дня 22:00 (ВЛТ)** — отчёт + проверка сбылась ли каждая идея (hit-rate)
- **Алерты** — важные новости (важность >= 3) с антиспамом (квота в час)
- **Самокалибровка** — hit-rate сигналов за 7 дней подставляется в промпт LLM
- **Q&A в Telegram** — спросите «что по спб бирже?» — ответит с живой котировкой
  и техкартиной; команды /quote /signals /watchlist /brief /status
- Голосовые вопросы (Groq Whisper) — опционально

## Запуск на Render.com

1. Создайте **Web Service** из этого репозитория (render.yaml подхватится автоматически).
2. Задайте переменные окружения (Secrets):
   - `TELEGRAM_BOT_TOKEN` — токен от @BotFather
   - `DEEPSEEK_API_KEY` — ключ LLM-провайдера
   - `TELEGRAM_ALLOWED_IDS` — ваш chat_id (узнать: @userinfobot)
   - `TELEGRAM_REPORT_CHAT_ID` — куда слать отчёты (обычно тот же chat_id)
   - `DEEPSEEK_BASE_URL`, `DEEPSEEK_MODEL` — уже предзаполнены в render.yaml
3. Free-план Render «засыпает» без трафика: повесьте UptimeRobot (ping каждые 5 мин)
   на URL сервиса — бот держит HTTP-эндпоинт `/` именно для этого.
4. База SQLite эфемерна на free-плане: при рестарте сервиса история сигналов
   обнуляется (есть защита от повторных алертов после холодного старта).

## Локальный запуск

```bash
pip install -r requirements.txt
# заполните .env (см. переменные выше)
python -m pipelines.scheduler
```

## Структура

```
pipelines/
  scheduler.py    — главный цикл (новости/котировки/алерты/брифинги)
  fetch_news.py   — RSS + опционально Exa
  fetch_quotes.py — MOEX ISS: котировки, свечи, индекс
  indicators.py   — теханализ: RSI, MACD, SMA, ATR, Боллинджер, 52н
  analyze.py      — LLM-анализ: новости + техника -> сигналы (JSON)
  bot.py          — Telegram long polling, команды и Q&A
  deliver.py      — отправка отчётов/алертов
  db.py           — SQLite (цены, свечи, новости, сигналы)
  llm.py          — OpenAI-совместимый клиент + бюджет
config/config.yaml — watchlist (22 бумаги MOEX), RSS, расписание (Asia/Vladivostok)
```

Расписание: `brief_time` и `evening_time` задаются в локальной зоне владельца
(`schedule.tz`), торговое окно MOEX — по МСК.
