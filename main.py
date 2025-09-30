# main.py
# coding: utf-8
import logging
import os
import shutil
import datetime
import toml
import requests

from tqdm import tqdm
from src.utils.config_loader import ConfigLoader
from src.utils.logger_setup import setup_logger
from src.utils.search_utils import greedy_search, exhaustive_search

from src.data_loading.dataset_builder import make_wsm_dataset_and_loader
from src.data_loading.pretrained_extractors import build_extractors_from_config

from transformers import CLIPProcessor, AutoImageProcessor

# Если у тебя есть тренер — подключим. Иначе можешь временно закомментить.
from src.train import train

# ───────────────────── optionally load .env ─────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# ───────────────────── Telegram helper ─────────────────────
def _notify_telegram(text: str, enabled: bool = True) -> bool:
    """Sends a message to TG if enabled and TELEGRAM_BOT_TOKEN/CHAT_ID are set.
       Returns True/False and logs the reason for silence."""
    if not enabled:
        logging.info("TG notify: disabled by config")
        return False
    token   = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        logging.info("TG notify: skipped (no TELEGRAM_BOT_TOKEN/CHAT_ID)")
        return False
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": text, "parse_mode": "HTML"},
            timeout=8,
        )
        # Log what Telegram responded with
        try:
            payload = r.json()
        except Exception:
            payload = {"raw": r.text}
        if r.ok and isinstance(payload, dict) and payload.get("ok"):
            logging.info("TG notify: sent")
            return True
        logging.warning(f"TG notify: API error {r.status_code} -> {payload}")
        return False
    except Exception as e:
        logging.warning(f"TG notify failed: {e}")
        return False

def _any_split_exists(cfg, split_name: str) -> bool:
    """
    Проверяем, есть ли хоть один CSV для данного split среди секций datasets.wsm_*.
    """
    for ds_name, ds_cfg in getattr(cfg, "datasets", {}).items():
        if not ds_name.lower().startswith("wsm_"):
            continue
        csv_path = ds_cfg["csv_path"].format(base_dir=ds_cfg["base_dir"], split=split_name)
        if os.path.exists(csv_path):
            return True
    return False


def main():
    # ──────────────────── 1. Конфиг и директории ────────────────────
    base_config = ConfigLoader("config.toml")

    model_name = base_config.model_name.replace("/", "_").replace(" ", "_").lower()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_dir = f"results/results_{model_name}_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)

    base_config.checkpoint_dir = os.path.join(results_dir, "checkpoints")
    os.makedirs(base_config.checkpoint_dir, exist_ok=True)

    epochlog_dir = os.path.join(results_dir, "metrics_by_epoch")
    os.makedirs(epochlog_dir, exist_ok=True)

    # ──────────────────── 2. Логирование ────────────────────────────
    log_file = os.path.join(results_dir, "session_log.txt")
    setup_logger(logging.INFO, log_file=log_file)
    base_config.show_config()

    use_tg = base_config.use_telegram
    logging.info(f"use_telegram = {use_tg}  (env token={bool(os.getenv('TELEGRAM_BOT_TOKEN'))}, chat={bool(os.getenv('TELEGRAM_CHAT_ID'))})")

    # startup ping — handy to confirm everything is connected
    _notify_telegram(f"🚀 Start: <b>{model_name}</b>\n📁 {results_dir}", enabled=use_tg)

    # Сохраним копию конфига и место для оверрайдов поиска
    shutil.copy("config.toml", os.path.join(results_dir, "config_copy.toml"))
    overrides_file = os.path.join(results_dir, "overrides.txt")
    csv_prefix = os.path.join(epochlog_dir, "metrics_epochlog")

    # ──────────────────── 3. Экстракторы/процессоры ─────────────────
    logging.info("🔧 Инициализация экстракторов по конфигу (только BODY)...")

    # ВНИМАНИЕ: наш build_extractors_from_config должен вернуть ключ 'body'
    modality_extractors = build_extractors_from_config(base_config)

    # Processor под видео: AutoImageProcessor для ViT, CLIPProcessor для CLIP
    if getattr(base_config, "video_extractor", "").lower() == "off":
        raise ValueError("video_extractor='off' не поддержан — требуется processor для 'body'.")

    model_name = base_config.video_extractor
    try:
        if "vit" in model_name.lower():
            body_processor = AutoImageProcessor.from_pretrained(model_name)
        else:
            body_processor = CLIPProcessor.from_pretrained(model_name)
    except Exception as e:
        raise RuntimeError(
            f"Не удалось инициализировать image processor из '{model_name}'. "
            f"Проверь config.video_extractor. Оригинальная ошибка: {e}"
        )

    modality_processors = {"body": body_processor}

    # Положим в конфиг, чтобы билдер мог их прочитать
    base_config.modality_extractors = modality_extractors
    base_config.modality_processors = modality_processors

    enabled = ", ".join(sorted(modality_extractors.keys())) or "—"
    logging.info(f"✅ Активные модальности: {enabled}")

    # ──────────────────── 4. Даталоадеры (WSM) ──────────────────────
    # Определим dev/val: если есть хоть один dev-CSV — берём 'dev', иначе 'val'
    dev_split = "dev" if _any_split_exists(base_config, "dev") else "val"

    logging.info("📦 Загружаем WSM (train/dev/test)...")
    _, train_loader = make_wsm_dataset_and_loader(base_config, "train")
    _, dev_loader   = make_wsm_dataset_and_loader(base_config, dev_split)

    # test: если нет теста — используем dev
    if _any_split_exists(base_config, "test"):
        _, test_loader = make_wsm_dataset_and_loader(base_config, "test")
    else:
        test_loader = dev_loader

    # ──────────────────── 5. Режим prepare_only ────────────────────
    if base_config.prepare_only:
        logging.info("== prepare_only mode: only data preparation, no training ==")
        _notify_telegram(
            f"✅ <b>{model_name}</b>: prepare_only completed\n📁 {results_dir}",
            enabled=use_tg
        )
        return

    # ──────────────────── 6. Поиск/обучение ────────────────────────
    search_type = base_config.search_type

    dev_loaders  = {"wsm": dev_loader}
    test_loaders = {"wsm": test_loader}

    if search_type == "greedy":
        search_config = toml.load("search_params.toml")
        param_grid     = dict(search_config.get("grid", {}))
        default_values = dict(search_config.get("defaults", {}))

        greedy_search(
            base_config    = base_config,
            train_loader   = train_loader,
            dev_loader     = dev_loaders,
            test_loader    = test_loaders,
            train_fn       = train,
            overrides_file = overrides_file,
            param_grid     = param_grid,
            default_values = default_values,
        )
        _notify_telegram(
            f"✅ <b>{model_name}</b>: greedy search finished\n📁 {results_dir}",
            enabled=use_tg
        )

    elif search_type == "exhaustive":
        search_config = toml.load("search_params.toml")
        param_grid     = dict(search_config.get("grid", {}))

        exhaustive_search(
            base_config    = base_config,
            train_loader   = train_loader,
            dev_loader     = dev_loaders,
            test_loader    = test_loaders,
            train_fn       = train,
            overrides_file = overrides_file,
            param_grid     = param_grid,
        )
        _notify_telegram(
            f"✅ <b>{model_name}</b>: exhaustive search finished\n📁 {results_dir}",
            enabled=use_tg
        )

    elif search_type == "none":
        logging.info("== Одиночная тренировка (без поиска параметров) ==")
        train(
            cfg         = base_config,
            mm_loader   = train_loader,   # единый лоадер WSM
            dev_loaders = dev_loaders,
            test_loaders= test_loaders,
        )
        _notify_telegram(
            f"✅ <b>{model_name}</b>: training (no search) completed\n📁 {results_dir}",
            enabled=use_tg
        )

    else:
        raise ValueError(
            f"⛔️ Неверное значение search_type: '{base_config.search_type}'. "
            f"Используй 'greedy', 'exhaustive' или 'none'."
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # crash notification always goes out so you know everything burned down
        _notify_telegram(
            f"❌ Crash: <code>{type(e).__name__}</code>\n{e}",
            enabled=True
        )
        raise
