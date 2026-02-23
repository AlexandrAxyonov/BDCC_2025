# coding: utf-8

import copy
import os
import logging
from itertools import product
from typing import Any

# Порядок показа: сначала базовые, потом все recall_*, потом прочее по алфавиту.
METRIC_ORDER = [
    "UAR", "MF1", "WF1",    # базовые
    "mUAR", "mF1",   # если тренер дублирует такими именами
]

def _pick_score(metrics: dict, metric_name: str) -> float:
    """Берём строго по selection_metric. Никаких подстановок."""
    try:
        val = metrics.get(metric_name, 0.0)
        return float(val) if isinstance(val, (int, float)) else 0.0
    except Exception:
        return 0.0


def _ordered_keys(metrics: dict[str, Any]) -> list[str]:
    """METRIC_ORDER → все recall_* → остальное по алфавиту (без by_dataset)."""
    base = [k for k in METRIC_ORDER if k in metrics]
    recalls = sorted(k for k in metrics.keys() if k.startswith("recall_"))
    rest = sorted(
        k for k in metrics.keys()
        if k not in METRIC_ORDER and not k.startswith("recall_") and k != "by_dataset"
    )
    return base + recalls + rest


def _ordered_keys_ds(ds: dict[str, Any]) -> list[str]:
    """То же для блока датасетов (без поля name)."""
    base = [k for k in METRIC_ORDER if k in ds and k != "name"]
    recalls = sorted(k for k in ds.keys() if k.startswith("recall_") and k != "name")
    rest = sorted(
        k for k in ds.keys()
        if k not in METRIC_ORDER and not k.startswith("recall_") and k != "name"
    )
    return base + recalls + rest


def format_result_box_dual(step_num: int,
                           param_name: str,
                           candidate: Any,
                           fixed_params: dict[str, Any],
                           dev_metrics: dict[str, Any],
                           test_metrics: dict[str, Any],
                           is_best: bool = False,
                           selection_metric: str = "UAR",
                           early_stop_on: str = "dev") -> str:
    """
    Красивый ASCII-бокс. Показываем только то, что реально есть.
    Поддерживаем вывод by_dataset, если его положил тренер.
    """
    title = f"Шаг {step_num}: {param_name} = {candidate}"
    fixed_lines = [f"{k} = {v}" for k, v in fixed_params.items()]

    def format_metrics_block(metrics: dict[str, Any], label: str) -> list[str]:
        lines = [f"  Результаты ({label.upper()}):"]
        for k in _ordered_keys(metrics):
            v = metrics[k]
            line = f"    {k.upper():16} = {v:.4f}" if isinstance(v, (int, float)) else f"    {k.upper():16} = {v}"
            if is_best and label == early_stop_on and k == selection_metric:
                line += " ✅"
            lines.append(line)

        # подробные метрики по датасетам (если есть)
        by_ds = metrics.get("by_dataset")
        if isinstance(by_ds, list):
            lines.append("  По датасетам:")
            for ds in by_ds:
                name = ds.get("name", "unknown")
                lines.append(f"    - {name}:")
                for k in _ordered_keys_ds(ds):
                    v = ds[k]
                    lines.append(f"        {k.upper():14} = {v:.4f}" if isinstance(v, (int, float)) else f"        {k.upper():14} = {v}")
        return lines

    content_lines = [title, "  Фиксировано:"]
    content_lines += [f"    {line}" for line in fixed_lines]
    content_lines += format_metrics_block(dev_metrics or {},  "dev")
    content_lines.append("")
    content_lines += format_metrics_block(test_metrics or {}, "test")

    max_width   = max(len(line) for line in content_lines) if content_lines else 0
    border_top  = "┌" + "─" * (max_width + 2) + "┐"
    border_bot  = "└" + "─" * (max_width + 2) + "┘"

    box = [border_top]
    for line in content_lines:
        box.append(f"│ {line.ljust(max_width)} │")
    box.append(border_bot)
    return "\n".join(box)


# ─────────────────────────── жадный поиск ──────────────────────────────────
def greedy_search(
    base_config,
    train_loader,
    dev_loader,
    test_loader,
    train_fn,
    overrides_file: str,
    param_grid: dict[str, list],
    default_values: dict[str, Any],
):
    """
    Поэтапный перебор. Отбор строго по cfg.selection_metric
    на сплите cfg.early_stop_on ('dev' или 'test').
    Никаких «mean_all» и прочего — только то, что вернул train().
    """
    current_best_params   = copy.deepcopy(default_values)
    all_param_names       = list(param_grid.keys())
    model_name            = base_config.model_name
    selection_metric      = getattr(base_config, "selection_metric", "UAR")
    early_stop_on         = getattr(base_config, "early_stop_on", "dev")

    with open(overrides_file, "a", encoding="utf-8") as f:
        f.write("=== Жадный перебор гиперпараметров (One task) ===\n")
        f.write(f"Модель: {model_name}\n")
        f.write(f"Отбор по: {selection_metric} [{early_stop_on}]\n")

    for i, param_name in enumerate(all_param_names):
        candidates     = param_grid[param_name]
        tried_value    = current_best_params[param_name]
        candidates_now = candidates if i == 0 else [v for v in candidates if v != tried_value]

        best_val_for_param    = tried_value
        best_metric_for_param = float("-inf")

        # оценка текущего лучшего
        cfg_def = copy.copy(base_config)
        for k, v in current_best_params.items():
            setattr(cfg_def, k, v)
        combo_dir = os.path.join(base_config.checkpoint_dir, f"greedy_{param_name}_{tried_value}")
        os.makedirs(combo_dir, exist_ok=True)
        cfg_def.checkpoint_dir = combo_dir

        dev_met_def, test_met_def = train_fn(cfg_def, train_loader, dev_loader, test_loader)
        eval_split_metrics = dev_met_def if early_stop_on == "dev" else test_met_def
        cur_score = _pick_score(eval_split_metrics or {}, selection_metric)

        box = format_result_box_dual(
            i + 1, param_name, tried_value,
            {k: v for k, v in current_best_params.items() if k != param_name},
            dev_met_def or {}, test_met_def or {},
            is_best=True, selection_metric=selection_metric,
            early_stop_on=early_stop_on
        )
        with open(overrides_file, "a", encoding="utf-8") as f:
            f.write("\n" + box + "\n")

        _log_dataset_metrics(dev_met_def or {},  overrides_file, "dev")
        _log_dataset_metrics(test_met_def or {}, overrides_file, "test")

        best_metric_for_param = cur_score

        # бежим по остальным кандидатам
        for cand in candidates_now:
            cfg = copy.copy(base_config)
            for k, v in current_best_params.items():
                setattr(cfg, k, v)
            setattr(cfg, param_name, cand)

            logging.info(f"[ШАГ {i+1}] {param_name} = {cand}, остальные {current_best_params}")

            dev_met, test_met = train_fn(cfg, train_loader, dev_loader, test_loader)
            eval_split_metrics = dev_met if early_stop_on == "dev" else test_met
            cand_score = _pick_score(eval_split_metrics or {}, selection_metric)
            is_better  = cand_score > best_metric_for_param

            box = format_result_box_dual(
                i + 1, param_name, cand,
                {k: v for k, v in current_best_params.items() if k != param_name},
                dev_met or {}, test_met or {},
                is_best=is_better, selection_metric=selection_metric,
                early_stop_on=early_stop_on
            )
            with open(overrides_file, "a", encoding="utf-8") as f:
                f.write("\n" + box + "\n")

            _log_dataset_metrics(dev_met or {},  overrides_file, "dev")
            _log_dataset_metrics(test_met or {}, overrides_file, "test")

            if is_better:
                best_val_for_param    = cand
                best_metric_for_param = cand_score

        current_best_params[param_name] = best_val_for_param
        with open(overrides_file, "a", encoding="utf-8") as f:
            f.write(f"\n>> [Итог Шаг{i+1}] лучший {param_name}={best_val_for_param}, "
                    f"{early_stop_on}_{selection_metric}={best_metric_for_param:.4f}\n")

    with open(overrides_file, "a", encoding="utf-8") as f:
        f.write("\n=== Итоговая комбинация ===\n")
        for k, v in current_best_params.items():
            f.write(f"{k} = {v}\n")
    logging.info("Готово! Жадный поиск завершён.")


# ──────────────────────── полный перебор ───────────────────────────────────
def exhaustive_search(
    base_config,
    train_loader,
    dev_loader,
    test_loader,
    train_fn,
    overrides_file: str,
    param_grid: dict[str, list],
):
    """
    Полный перебор. Отбор строго по cfg.selection_metric на сплите cfg.early_stop_on.
    Ничего дополнительного не считаем.
    """
    all_param_names  = list(param_grid.keys())
    selection_metric = getattr(base_config, "selection_metric", "UAR")
    early_stop_on    = getattr(base_config, "early_stop_on", "dev")

    with open(overrides_file, "a", encoding="utf-8") as f:
        f.write("=== Полный перебор гиперпараметров (One task) ===\n")
        f.write(f"Модель: {base_config.model_name}\n")
        f.write(f"Отбор по: {selection_metric} [{early_stop_on}]\n")

    best_config = None
    best_score  = float("-inf")
    combo_id    = 0

    for combo in product(*(param_grid[p] for p in all_param_names)):
        combo_id += 1
        param_combo = dict(zip(all_param_names, combo))

        cfg = copy.copy(base_config)
        for k, v in param_combo.items():
            setattr(cfg, k, v)

        combo_dir = os.path.join(base_config.checkpoint_dir, f"combo_{combo_id}")
        os.makedirs(combo_dir, exist_ok=True)
        cfg.checkpoint_dir = combo_dir

        logging.info(f"\n[Комбинация #{combo_id}] {param_combo}")

        train_out = train_fn(cfg, train_loader, dev_loader, test_loader)

        # если train() вернул кортеж (dev, test) – распакуем
        if isinstance(train_out, tuple) and len(train_out) == 2:
            dev_met, test_met = train_out
        else:
            dev_met, test_met = train_out, {}

        eval_split_metrics = dev_met if early_stop_on == "dev" else test_met
        cand_score         = _pick_score(eval_split_metrics or {}, selection_metric)
        is_better          = cand_score > best_score

        box = format_result_box_dual(
            combo_id, " + ".join(all_param_names), str(combo),
            {}, dev_met or {}, test_met or {},
            is_best=is_better, selection_metric=selection_metric,
            early_stop_on=early_stop_on
        )
        with open(overrides_file, "a", encoding="utf-8") as f:
            f.write("\n" + box + "\n")

        _log_dataset_metrics(dev_met or {},  overrides_file, "dev")
        _log_dataset_metrics(test_met or {}, overrides_file, "test")

        if is_better:
            best_score  = cand_score
            best_config = param_combo

    with open(overrides_file, "a", encoding="utf-8") as f:
        f.write("\n=== Лучшая комбинация ===\n")
        for k, v in (best_config or {}).items():
            f.write(f"{k} = {v}\n")
    logging.info("Полный перебор завершён. Лучшие параметры выбраны.")
    return best_score, (best_config or {})

# ────────────────────────── дополнительные логи ───────────────────────────
def _log_dataset_metrics(metrics: dict, file_path: str, label: str = "dev") -> None:
    """
    Просто печатаем то, что train положил в metrics['by_dataset'].
    Ни формул, ни домыслов.
    """
    by_ds = metrics.get("by_dataset")
    if not isinstance(by_ds, list):
        return
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(f"\n>>> Подробные метрики по каждому датасету ({label})\n")
        for ds in by_ds:
            name = ds.get("name", "unknown")
            f.write(f"  - {name}:\n")
            for k in _ordered_keys_ds(ds):
                v = ds[k]
                if isinstance(v, (int, float)):
                    f.write(f"      {k.upper():14} = {v:.4f}\n")
                else:
                    f.write(f"      {k.upper():14} = {v}\n")
        f.write(f"<<< Конец подробных метрик ({label})\n")
