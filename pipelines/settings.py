"""Загрузка настроек: .env (секреты) + config/config.yaml (конфигурация)."""
from __future__ import annotations

import logging
import os
import pathlib

import yaml
from dotenv import load_dotenv

load_dotenv()

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
LOGS_DIR = ROOT / "logs"
CONFIG_PATH = pathlib.Path(os.environ.get("CONFIG_PATH", ROOT / "config" / "config.yaml"))

DATA_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)


def load_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


CONFIG = load_config()
SHOW_DISCLAIMER = bool(CONFIG.get("show_disclaimer", True))

MSK = "Europe/Moscow"
# Локальная зона владельца для расписаний (schedule.tz в конфиге), по умолчанию МСК
LOCAL_TZ = CONFIG["schedule"].get("tz", MSK)
TZ_LABEL = CONFIG["schedule"].get("tz_label", "МСК")


def env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


# --- Секреты ---
TELEGRAM_BOT_TOKEN = env("TELEGRAM_BOT_TOKEN")
DEEPSEEK_API_KEY = env("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = env("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
DEEPSEEK_MODEL = env("DEEPSEEK_MODEL", "deepseek-chat")
EXA_API_KEY = env("EXA_API_KEY")
GROQ_API_KEY = env("GROQ_API_KEY")
GROQ_TTS_MODEL = env("GROQ_TTS_MODEL", "playai-tts")
GROQ_TTS_VOICE = env("GROQ_TTS_VOICE", "Fritz-PlayAI")
VOICE_REPLIES = env("VOICE_REPLIES", "false").lower() == "true"
HEALTHCHECK_URL = env("HEALTHCHECK_URL")
DAILY_USD_BUDGET = float(env("DAILY_USD_BUDGET", "0") or 0)
PRICE_IN_PER_M = float(env("PRICE_IN_PER_M", "0.4") or 0)
PRICE_OUT_PER_M = float(env("PRICE_OUT_PER_M", "1.5") or 0)

_ALLOWED_IDS: set[int] = set()
if env("TELEGRAM_ALLOWED_IDS"):
    _ALLOWED_IDS = {int(x) for x in env("TELEGRAM_ALLOWED_IDS").split(",") if x.strip()}

# Куда слать плановые отчёты: первый из разрешённых chat_id (или отдельная переменная)
REPORT_CHAT_ID = env("TELEGRAM_REPORT_CHAT_ID") or (
    str(sorted(_ALLOWED_IDS)[0]) if _ALLOWED_IDS else "")


def setup_logging() -> logging.Logger:
    level = getattr(logging, env("LOG_LEVEL", "INFO").upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(LOGS_DIR / "bot.log", encoding="utf-8"),
        ],
    )
    return logging.getLogger("trade-bot")
