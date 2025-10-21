# src/train_wsm.py (aka train.py)
# coding: utf-8
from __future__ import annotations
import logging, os, math
from pathlib import Path
from typing import Dict, List, Optional
from lion_pytorch import Lion

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score, recall_score

from src.models.models import VideoMamba, VideoFormer
from src.utils.logger_setup import color_metric, color_split
from src.utils.schedulers import SmartScheduler

CLASS_LABELS = {
    0: "control",
    1: "depression",
    2: "parkinson",
}

# ───────────────────────── utils ─────────────────────────

def seed_everything(seed: int):
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _stack_body_features(
    features_list: List[Optional[dict]],
    average_mode: str = "mean"
):
    """
    batch["features"] — список:
      {"body": {"mean":[D]} | {"mean":[D],"std":[D]} | {"seq":[T,D]}}
    Возвращаем X=[B,D’] и индексы сохранённых элементов.
    """
    rows: List[torch.Tensor] = []
    keep_idx: List[int] = []

    for i, feats in enumerate(features_list):
        if not feats or "body" not in feats or feats["body"] is None:
            continue
        body = feats["body"]
        if average_mode == "mean_std" and "mean" in body and "std" in body:
            x = torch.cat([body["mean"].view(-1), body["std"].view(-1)], dim=0).to(torch.float32).cpu()
        elif "mean" in body:
            x = body["mean"].view(-1).to(torch.float32).cpu()
        elif "seq" in body:
            s = body["seq"]  # [T,D]
            x = s.mean(dim=0).to(torch.float32).cpu()
        else:
            continue
        rows.append(x)
        keep_idx.append(i)

    if not rows:
        raise RuntimeError("В батче нет пригодных body-фич. Проверь кэш и average_features.")
    X = torch.stack(rows, dim=0)  # [B,D’]
    return X, keep_idx


def _filter_labels(labels: torch.Tensor, keep_idx: List[int]) -> torch.Tensor:
    return labels[keep_idx]


def _gather_all_labels(loader: DataLoader, average_mode: str) -> np.ndarray:
    ys = []
    for batch in loader:
        if batch is None:
            continue
        _, keep = _stack_body_features(batch["features"], average_mode)
        y = _filter_labels(batch["labels"], keep)
        ys.append(y.cpu().numpy())
    if not ys:
        raise RuntimeError("Не удалось собрать метки с train-лоадера.")
    return np.concatenate(ys, axis=0)


def _num_classes_from_loader(loader: DataLoader, average_mode: str) -> int:
    y = _gather_all_labels(loader, average_mode)
    return int(np.max(y) + 1)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> Dict[str, float]:
    mf1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    uar = recall_score(y_true, y_pred, average="macro", zero_division=0)  # UAR = macro recall
    per_cls = recall_score(y_true, y_pred, average=None, labels=list(range(num_classes)), zero_division=0)
    out: Dict[str, float] = {"MF1": float(mf1), "UAR": float(uar)}
    for c, r in enumerate(per_cls):
        name = CLASS_LABELS.get(c, f"class{c}")
        out[f"recall_c{c}_{name}"] = float(r)
    return out


def _build_model(cfg, input_dim: int, seq_len: int, num_classes: int, device: torch.device) -> nn.Module:
    model_name = cfg.model_name.lower()  # "mamba" | "transformer"

    if model_name in ("mamba", "vmamba", "video_mamba"):
        model = VideoMamba(
            input_dim=input_dim,
            hidden_dim=cfg.hidden_dim,
            mamba_d_state=cfg.mamba_d_state,
            mamba_ker_size=cfg.mamba_ker_size,
            mamba_layer_number=cfg.mamba_layers,
            d_discr=getattr(cfg, "mamba_d_discr", None),
            dropout=cfg.dropout,
            seg_len=seq_len,
            out_features=cfg.out_features,
            num_classes=num_classes,
            device=str(device)
        )
    elif model_name in ("transformer", "former", "videoformer", "tr"):
        model = VideoFormer(
            input_dim=input_dim,
            hidden_dim=cfg.hidden_dim,
            num_transformer_heads=cfg.num_transformer_heads,
            positional_encoding=cfg.positional_encoding,
            dropout=cfg.dropout,
            tr_layer_number=cfg.tr_layers,
            seg_len=seq_len,
            out_features=cfg.out_features,
            num_classes=num_classes
        )
    else:
        raise ValueError(f"Неизвестная модель ='{cfg.model_name}'. Используй 'mamba' или 'transformer'.")
    return model.to(device)


# ───────────────────── multi-label helpers ─────────────────────

def _gather_pos_weight_for_ml(loader: DataLoader, average_mode: str) -> torch.Tensor:
    """
    Считает pos_weight для BCE по двум головам: N_neg/N_pos на трейне.
    Ожидает, что в батче есть 'labels_ml': FloatTensor[B,2].
    """
    pos = torch.zeros(2, dtype=torch.float64)
    neg = torch.zeros(2, dtype=torch.float64)
    for batch in loader:
        if batch is None or "labels_ml" not in batch:
            continue
        _, keep = _stack_body_features(batch["features"], average_mode)
        y_ml = batch["labels_ml"][keep]  # [b,2]
        pos += y_ml.double().sum(dim=0)
        neg += (1.0 - y_ml.double()).sum(dim=0)
    pos = torch.clamp(pos, min=1.0)
    neg = torch.clamp(neg, min=1.0)
    return (neg / pos).to(torch.float32)  # shape [2]


def _map_probs_to_single_label(p_dep: np.ndarray, p_park: np.ndarray,
                               t_dep: float = 0.5, t_park: float = 0.5) -> np.ndarray:
    """
    Маппинг двух вероятностей в {0,1,2}:
      оба < t → 0 (control)
      dep ≥ t & park < t → 1
      park ≥ t & dep < t → 2
      оба ≥ t → выбираем по max(p)
    """
    assert p_dep.shape == p_park.shape
    dep = p_dep >= t_dep
    par = p_park >= t_park
    y = np.zeros_like(p_dep, dtype=np.int64)
    only_dep = np.logical_and(dep, ~par)
    only_par = np.logical_and(par, ~dep)
    both     = np.logical_and(dep, par)
    y[only_dep] = 1
    y[only_par] = 2
    if both.any():
        choose_dep = p_dep[both] >= p_park[both]
        idx = np.where(both)[0]
        y[idx[choose_dep]] = 1
        y[idx[~choose_dep]] = 2
    return y


@torch.no_grad()
def _eval_epoch(model: nn.Module, loader: DataLoader, device: torch.device,
                avg_mode: str, metrics_num_classes: int,
                multi_label: bool = False,
                thr_dep: float = 0.5, thr_park: float = 0.5) -> Dict[str, float]:
    """
    metrics_num_classes:
      - single-label: 3 (control/depression/parkinson)
      - multi-label:  3 (після маппинга двух голов в 3 класса)
    """
    model.eval()
    all_y, all_p = [], []

    for batch in tqdm(loader, desc="Eval", leave=False):
        if batch is None:
            continue

        X, keep = _stack_body_features(batch["features"], avg_mode)  # [B,D’]
        y = _filter_labels(batch["labels"], keep).to(device)         # [B]
        if X.ndim == 2:
            X = X.unsqueeze(1)  # → [B,1,D’]

        logits = model(X.to(device))  # [B,C]
        if not multi_label:
            pred = logits.argmax(dim=1)
        else:
            # C=2: dep, park → сигмоида + пороги → маппинг в 3 класса
            probs = torch.sigmoid(logits).cpu().numpy()           # [B,2]
            y_hat = _map_probs_to_single_label(probs[:, 0], probs[:, 1],
                                               t_dep=thr_dep, t_park=thr_park)
            pred = torch.as_tensor(y_hat, dtype=torch.long)

        all_y.append(y.cpu())
        all_p.append(pred.cpu())

    if not all_y:
        return {}
    y_true = torch.cat(all_y).numpy()
    y_pred = torch.cat(all_p).numpy()
    return _metrics(y_true, y_pred, metrics_num_classes)


# ───────────────────────── helpers for early stopping ──────────────────────

def _score_for_split(metrics_map: Dict[str, float], selection_metric: str) -> float:
    """
    Берём среднее по всем ключам вида '{selection_metric}_<dataset>'.
    Если таких ключей нет — пытаемся взять прямой ключ selection_metric.
    Если и его нет — fallback: UAR -> MF1 -> -1.
    """
    if not metrics_map:
        return -1.0
    pref = f"{selection_metric}_"
    vals = [v for k, v in metrics_map.items() if isinstance(v, (int, float)) and k.startswith(pref)]
    if vals:
        return float(np.mean(vals))
    if selection_metric in metrics_map and isinstance(metrics_map[selection_metric], (int, float)):
        return float(metrics_map[selection_metric])
    for k in ("UAR", "MF1"):
        if k in metrics_map and isinstance(metrics_map[k], (int, float)):
            return float(metrics_map[k])
    return -1.0


# ───────────────────────── train loop ─────────────────────

def train(
    cfg,
    mm_loader: DataLoader,                # train
    dev_loaders: Dict[str, DataLoader] | None = None,
    test_loaders: Dict[str, DataLoader] | None = None,
):
    """
    Single-label: CE + class weights + UAR/MF1.
    Multi-label:  BCEWithLogits + pos_weight + маппинг в 3 класса для метрик.
    Ранняя остановка по cfg.selection_metric на cfg.early_stop_on (dev|test).
    """
    seed_everything(cfg.random_seed)
    device = torch.device(cfg.device)
    avg_mode = cfg.average_features.lower()
    multi_label = bool(getattr(cfg, "multi_label", False))

    # первый батч → определяем D’ (и фиктивную длину 1, т.к. уже спулили до вектора)
    first = None
    for b in mm_loader:
        if b is not None:
            first = b
            break
    if first is None:
        raise RuntimeError("train loader пустой (или collate всё фильтрует).")
    X0, _ = _stack_body_features(first["features"], avg_mode)  # [B0, D’]
    in_dim = int(X0.shape[1])
    seq_len = 1

    # число выходов модели и число классов для метрик
    # model_num_classes: 3 (single) / 2 (multi)
    # metrics_num_classes: всегда 3 (control, depression, parkinson)
    if multi_label:
        model_num_classes = 2
        metrics_num_classes = 3
    else:
        try:
            model_num_classes = cfg.num_classes
        except AttributeError:
            model_num_classes = _num_classes_from_loader(mm_loader, avg_mode)
        metrics_num_classes = model_num_classes

    # ─── class weights / pos_weight ─────────────────────────────────────
    if not multi_label:
        if cfg.class_weighting == "none":
            ce_weights = None
            logging.info("⚖️ Class weighting: none (веса отключены)")
        else:
            y_all = _gather_all_labels(mm_loader, avg_mode)
            classes = np.arange(model_num_classes)
            class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_all)
            ce_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)
            logging.info(f"⚖️ Class weighting: balanced → {class_weights.tolist()}")
    else:
        # pos_weight для BCE по двум головам
        pos_weight = _gather_pos_weight_for_ml(mm_loader, avg_mode).to(device)
        logging.info(f"⚖️ pos_weight (BCE) → {pos_weight.cpu().numpy().tolist()}")

    # модель/опт/лосс
    model = _build_model(cfg, in_dim, seq_len, model_num_classes, device)

    # ─── Оптимизатор ──────────────────────────────────────────────────
    if cfg.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == "lion":
        optimizer = Lion(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum)
    elif cfg.optimizer == "rmsprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=cfg.lr)
    else:
        raise ValueError(f"⛔ Неизвестный оптимизатор: {cfg.optimizer}")
    logging.info(f"⚙️ Оптимизатор: {cfg.optimizer}, learning rate: {cfg.lr}")

    # ─── Scheduler ────────────────────────────────────────────────────
    steps_per_epoch = sum(1 for b in mm_loader if b is not None)
    scheduler = SmartScheduler(
        scheduler_type=cfg.scheduler_type,
        optimizer=optimizer,
        config=cfg,
        steps_per_epoch=steps_per_epoch
    )

    # ─── Лосс ─────────────────────────────────────────────────────────
    if not multi_label:
        # criterion = nn.CrossEntropyLoss(weight=(ce_weights if 'ce_weights' in locals() else None))
        criterion = nn.CrossEntropyLoss(weight=ce_weights)

    else:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # конфиг для ранней остановки/выбора лучшего
    selection_metric = cfg.selection_metric
    early_stop_on = cfg.early_stop_on

    best_score = -1.0
    best_dev, best_test = {}, {}
    patience = 0

    for epoch in range(cfg.num_epochs):
        logging.info(f"═══ EPOCH {epoch+1}/{cfg.num_epochs} ═══")
        model.train()
        tot_loss, tot_n = 0.0, 0
        tr_y, tr_p = [], []

        for batch in tqdm(mm_loader, desc="Train"):
            if batch is None:
                continue
            X, keep = _stack_body_features(batch["features"], avg_mode)  # [B,D’]

            # таргеты
            if not multi_label:
                y = _filter_labels(batch["labels"], keep).to(device)  # [b]
            else:
                if "labels_ml" not in batch:
                    raise RuntimeError("multi_label=True, но в батче нет 'labels_ml'. Проверь датасет/коллатер.")
                y = batch["labels_ml"][keep].to(device)               # [b,2] float32

            if X.ndim == 2:
                X = X.unsqueeze(1)  # [B,1,D’]
            logits = model(X.to(device))  # [b,C]
            loss = criterion(logits, y)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step(batch_level=True)

            bs = y.size(0)
            tot_loss += loss.item() * bs
            tot_n += bs

            # онлайновые train-метрики (в single — argmax, в multi — маппинг в 3 класса с порогами по умолчанию)
            if not multi_label:
                tr_y.append(_filter_labels(batch["labels"], keep).cpu())
                tr_p.append(logits.argmax(dim=1).detach().cpu())
            else:
                # целевые метки для отчёта — single-label индексы (0/1/2)
                tr_y.append(_filter_labels(batch["labels"], keep).cpu())
                probs = torch.sigmoid(logits).detach().cpu().numpy()
                y_hat = _map_probs_to_single_label(
                    probs[:, 0], probs[:, 1],
                    t_dep=getattr(cfg, "thr_dep", 0.5),
                    t_park=getattr(cfg, "thr_park", 0.5)
                )
                tr_p.append(torch.as_tensor(y_hat, dtype=torch.long))

        train_loss = tot_loss / max(1, tot_n)
        tr_y_np = torch.cat(tr_y).numpy() if tr_y else np.array([])
        tr_p_np = torch.cat(tr_p).numpy() if tr_p else np.array([])
        if tr_y_np.size > 0:
            m_tr = _metrics(tr_y_np, tr_p_np, metrics_num_classes)
            parts = [
                f"Loss={train_loss:.4f}",
                color_metric("UAR", m_tr["UAR"]),
                color_metric("MF1", m_tr["MF1"]),
            ]
            for c in range(metrics_num_classes):
                key = f"recall_c{c}_{CLASS_LABELS.get(c, f'class{c}')}"
                if key in m_tr:
                    parts.append(color_metric(key, m_tr[key]))
            logging.info(f"[{color_split('TRAIN')}] " + " | ".join(parts))
        else:
            logging.info(f"[{color_split('TRAIN')}] Loss={train_loss:.4f} | (пустые метрики)")

        # ── Оценка DEV/TEST ──
        cur_dev = {}
        if dev_loaders:
            for name, ldr in dev_loaders.items():
                md = _eval_epoch(
                    model, ldr, device, avg_mode,
                    metrics_num_classes,
                    multi_label=multi_label,
                    thr_dep=getattr(cfg, "thr_dep", 0.5),
                    thr_park=getattr(cfg, "thr_park", 0.5),
                )
                if md:
                    cur_dev.update({f"{k}_{name}": v for k, v in md.items()})
                    msg = " · ".join(color_metric(k, v) for k, v in md.items())
                    logging.info(f"[{color_split('DEV')}:{name}] {msg}")

        cur_test = {}
        if test_loaders:
            for name, ldr in test_loaders.items():
                mt = _eval_epoch(
                    model, ldr, device, avg_mode,
                    metrics_num_classes,
                    multi_label=multi_label,
                    thr_dep=getattr(cfg, "thr_dep", 0.5),
                    thr_park=getattr(cfg, "thr_park", 0.5),
                )
                if mt:
                    cur_test.update({f"{k}_{name}": v for k, v in mt.items()})
                    msg = " · ".join(color_metric(k, v) for k, v in mt.items())
                    logging.info(f"[{color_split('TEST')}:{name}] {msg}")

        # ── Ранняя остановка по cfg.selection_metric на сплите cfg.early_stop_on ──
        eval_map = cur_dev if early_stop_on == "dev" else cur_test
        score = _score_for_split(eval_map, selection_metric)

        scheduler.step(score)

        if score > best_score:
            best_score = score
            best_dev, best_test = cur_dev, cur_test
            patience = 0
            os.makedirs(cfg.checkpoint_dir, exist_ok=True)
            ckpt = Path(cfg.checkpoint_dir) / f"best_ep{epoch+1}_{early_stop_on}_{selection_metric}_{best_score:.4f}.pt"
            torch.save(model.state_dict(), ckpt)
            logging.info(f"✔ Saved best model ({early_stop_on}/{selection_metric}={best_score:.4f}): {ckpt.name}")
        else:
            patience += 1
            if patience >= cfg.max_patience:
                logging.info("Early stopping.")
                break

    return best_dev, best_test
