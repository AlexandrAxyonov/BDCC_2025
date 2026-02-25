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
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score, recall_score

from src.models.models import VideoFormer, VideoFormer_with_Prototypes, VideoMamba
from src.utils.logger_setup import color_metric, color_split,dbg_check_logits, dbg_dump_logits
from src.utils.schedulers import SmartScheduler
from src.utils.losses import prototype_contrastive_loss
import pickle

CLASS_LABELS = {
    0: "control",
    1: "depression",
    2: "parkinson",
}
ML_ONEHOT3 = {"onehot3", "3way", "3", "onehot"}



def seed_everything(seed: int):
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _stack_body_features(
    features_list: List[Optional[dict]],
    average_mode: str = "mean",
    segment_length: Optional[int] = None,
):

    if average_mode not in {"mean", "mean_std", "raw"}:
        raise ValueError(f"unknown average_mode={average_mode!r} (expected 'mean'|'mean_std'|'raw')")

    rows: List[torch.Tensor] = []
    keep_idx: List[int] = []
    lengths: List[int] = []

    for i, feats in enumerate(features_list):
        if not feats or "body" not in feats or feats["body"] is None:
            continue
        body = feats["body"]

        if average_mode == "mean_std" and "mean" in body and "std" in body:
            x = torch.cat([body["mean"].view(-1), body["std"].view(-1)], dim=0).to(torch.float32)
            rows.append(x)
            keep_idx.append(i)

        elif average_mode == "mean" and "mean" in body:
            x = body["mean"].view(-1).to(torch.float32)
            rows.append(x)
            keep_idx.append(i)

        elif average_mode == "raw" and "seq" in body:
            s = body["seq"].to(torch.float32)  # [T, D]
            rows.append(s)
            lengths.append(s.size(0))
            keep_idx.append(i)

        else:
            continue

    if not rows:
        raise RuntimeError("No usable body features in the batch. Check cache and average_features.")

    if average_mode == "raw":

        X = pad_sequence(rows, batch_first=True, padding_value=0.0)  # [B, T_max, D]


        T = X.size(1)
        mask = torch.zeros(X.size(0), T, dtype=torch.bool, device=X.device)
        if lengths:
            for bi, L in enumerate(lengths):
                mask[bi, :min(L, T)] = True
        else:

            mask[:] = True

    else:

        X = torch.stack(rows, dim=0)
        mask = None

    return X, keep_idx, mask


def _filter_labels(labels: torch.Tensor, keep_idx: List[int]) -> torch.Tensor:
    return labels[keep_idx]


def _gather_all_labels(loader: DataLoader, average_mode: str, segment_length: Optional[int] = None) -> np.ndarray:

    ys = []
    for batch in loader:
        if batch is None:
            continue
        _, keep, _ = _stack_body_features(batch["features"], average_mode, segment_length=segment_length)

        y = _filter_labels(batch["labels"], keep)
        ys.append(y.cpu().numpy())
    if not ys:
        raise RuntimeError("Failed to collect labels from the train loader.")
    return np.concatenate(ys, axis=0)

def _num_classes_from_loader(loader: DataLoader, average_mode: str, segment_length: Optional[int] = None) -> int:
    y = _gather_all_labels(loader, average_mode, segment_length=segment_length)
    return int(np.max(y) + 1)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> Dict[str, float]:
    mf1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    wf1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    uar = recall_score(y_true, y_pred, average="macro", zero_division=0)  # UAR = macro recall
    per_cls = recall_score(y_true, y_pred, average=None, labels=list(range(num_classes)), zero_division=0)
    out: Dict[str, float] = {"MF1": float(mf1), "WF1": float(wf1), "UAR": float(uar)}
    for c, r in enumerate(per_cls):
        name = CLASS_LABELS.get(c, f"class{c}")
        out[f"recall_c{c}_{name}"] = float(r)
    return out


def _save_eval_protocol_tsv(
    out_path: str,
    split_tag: str,
    epoch_idx: int,
    metrics: Dict[str, float],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    names: List[str],
    keys: List[str],
) -> None:
    """
    Save a reproducible eval protocol as plain TSV:
      - metadata and metrics in commented lines
      - per-sample records with true/pred labels
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"# split={split_tag}\n")
        f.write(f"# epoch={epoch_idx + 1}\n")
        for mk in sorted(metrics.keys()):
            mv = metrics[mk]
            if isinstance(mv, (int, float)):
                f.write(f"# {mk}={mv:.6f}\n")
        f.write("video_path\tsample_name\ty_true\ty_pred\ttrue_name\tpred_name\n")
        for i in range(len(y_true)):
            t = int(y_true[i])
            p = int(y_pred[i])
            t_name = CLASS_LABELS.get(t, f"class{t}")
            p_name = CLASS_LABELS.get(p, f"class{p}")
            k = keys[i] if i < len(keys) else f"idx_{i}"
            n = names[i] if i < len(names) else f"sample_{i}"
            f.write(f"{k}\t{n}\t{t}\t{p}\t{t_name}\t{p_name}\n")


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
            device=str(device),
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
            num_classes=num_classes,
            gate_mode=cfg.gate_mode
        )

    elif model_name == "prototypes":
        model = VideoFormer_with_Prototypes(
            input_dim=input_dim,
            hidden_dim=cfg.hidden_dim,
            num_transformer_heads=cfg.num_transformer_heads,
            positional_encoding=cfg.positional_encoding,
            dropout=cfg.dropout,
            tr_layer_number=cfg.tr_layers,
            seg_len=seq_len,
            out_features=cfg.out_features,
            num_classes=num_classes,
            num_prototypes_per_class=cfg.num_prototypes_per_class,
            proto_similarity=cfg.proto_similarity,
            proto_temperature=cfg.proto_temperature,
            proto_proj_enabled=getattr(cfg, "proto_proj_enabled", False),
            proto_proj_dim=getattr(cfg, "proto_proj_dim", 0),
        )

    else:
        raise ValueError(f"Unknown model='{cfg.model_name}'. Use 'mamba' or 'transformer'.")
    return model.to(device)




def _gather_pos_weight_for_ml(loader: DataLoader, average_mode: str, segment_length: Optional[int] = None) -> torch.Tensor:
    """
    pos_weight per class: N_neg / N_pos for each column of labels_ml.
    """
    pos = None
    neg = None
    for batch in loader:
        if batch is None or "labels_ml" not in batch:
            continue
        _, keep, _ = _stack_body_features(batch["features"], average_mode, segment_length=segment_length)
        y_ml = batch["labels_ml"][keep]
        if y_ml.numel() == 0:
            continue
        if pos is None:
            pos = torch.zeros(y_ml.size(1), dtype=torch.float64)
            neg = torch.zeros_like(pos)
        pos += y_ml.double().sum(dim=0)
        neg += (1.0 - y_ml.double()).sum(dim=0)
    if pos is None:
        raise RuntimeError("labels_ml not found for pos_weight calculation")
    pos = torch.clamp(pos, min=1.0)
    neg = torch.clamp(neg, min=1.0)
    return (neg / pos).to(torch.float32)


def _map_probs_to_single_label(p_dep: np.ndarray, p_park: np.ndarray,
                               t_dep: float = 0.5, t_park: float = 0.5) -> np.ndarray:
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
def _eval_epoch(cfg, model: nn.Module, loader: DataLoader, device: torch.device,
            avg_mode: str, metrics_num_classes: int, model_name: str,
            multi_label: bool = False,
            thr_dep: float = 0.5, thr_park: float = 0.5,
            segment_length: Optional[int] = None,
            protocol_dir: Optional[str] = None,
            split_tag: Optional[str] = None,
            epoch_idx: Optional[int] = None) -> Dict[str, float]:
    model.eval()
    ml_mode = str(getattr(cfg, "multi_label_mode", "2way") or "2way").lower()
    all_y, all_p = [], []
    all_names: List[str] = []
    all_keys: List[str] = []

    # for batch in tqdm(loader, desc="Eval", leave=False):
    for bidx, batch in enumerate(tqdm(loader, desc="Eval", leave=False)):
        if batch is None:
            continue

        # X, keep = _stack_body_features(batch["features"], avg_mode, segment_length=segment_length)
        X, keep, mask = _stack_body_features(batch["features"], avg_mode, segment_length=segment_length)


        y = _filter_labels(batch["labels"], keep).to(device)
        if X.ndim == 2:
            X = X.unsqueeze(1)

        # logits = model(X.to(device))  # [B,C]

        if model_name == 'prototypes':
            final, cls_l, proto_l, embeddings = model(
                X.to(device, non_blocking=True),
                mask=mask.to(device, non_blocking=True) if mask is not None else None
            )

            if bidx == 0:
                dbg_dump_logits(final, cfg.print_logits, prefix="[DBG:VAL:final]", max_rows=5, max_cols=final.size(1))
                dbg_dump_logits(cls_l, cfg.print_logits, prefix="[DBG:VAL:cls]",   max_rows=5, max_cols=cls_l.size(1))
                dbg_dump_logits(proto_l, cfg.print_logits, prefix="[DBG:VAL:proto]", max_rows=5, max_cols=proto_l.size(1))
                dbg_check_logits(final_logits=final, cls_logits=cls_l, proto_logits=proto_l, print_logits=cfg.print_logits,  prefix="[DBG:VAL]")
            logits = final
            # logits = cls_l
        else:
            logits =  model(
                X.to(device, non_blocking=True),
                mask=mask.to(device, non_blocking=True) if mask is not None else None
            )
            if bidx == 0:
                # _dbg_check_logits(final_logits=logits, prefix="[DBG:VAL]")

                dbg_dump_logits(logits, cfg.print_logits, prefix="[DBG:VAL:final]", max_rows=5, max_cols=logits.size(1))

        if not multi_label:
            pred = logits.argmax(dim=1)
        else:
            if ml_mode in ML_ONEHOT3:
                pred = logits.argmax(dim=1)
            else:
                # C=2: dep, park -> map to 3-class label
                probs = torch.sigmoid(logits).cpu().numpy()           # [B,2]
                y_hat = _map_probs_to_single_label(probs[:, 0], probs[:, 1],
                                                   t_dep=thr_dep, t_park=thr_park)
                pred = torch.as_tensor(y_hat, dtype=torch.long)

        all_y.append(y.cpu())
        all_p.append(pred.cpu())
        if "names" in batch:
            all_names.extend([str(batch["names"][i]) for i in keep])
        if "video_paths" in batch:
            all_keys.extend([str(batch["video_paths"][i]) for i in keep])

    if not all_y:
        return {}
    y_true = torch.cat(all_y).numpy()
    y_pred = torch.cat(all_p).numpy()
    met = _metrics(y_true, y_pred, metrics_num_classes)
    if protocol_dir and split_tag is not None and epoch_idx is not None:
        out_file = os.path.join(protocol_dir, f"{split_tag}_epoch{epoch_idx + 1:03d}.tsv")
        _save_eval_protocol_tsv(
            out_path=out_file,
            split_tag=split_tag,
            epoch_idx=epoch_idx,
            metrics=met,
            y_true=y_true,
            y_pred=y_pred,
            names=all_names,
            keys=all_keys,
        )
    return met




def _score_for_split(metrics_map: Dict[str, float], selection_metric: str) -> float:
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

def _probs_from_logits(logits: torch.Tensor, multi_label: bool) -> torch.Tensor:
    # single-label: softmax; multi-label: sigmoid
    return torch.sigmoid(logits) if multi_label else torch.softmax(logits, dim=1)


@torch.no_grad()
def export_logits_to_pkl(
    cfg,
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    avg_mode: str,
    model_name: str,
    out_path: str,
    multi_label: bool,
    segment_length: int | None,
):
    model.eval()
    out = {}  # dict[video_path] -> dict
    export_raw = bool(getattr(cfg, "export_logits_raw", False))

    for batch in tqdm(loader, desc=f"Export logits -> {out_path}", leave=False):
        if batch is None:
            continue

        X, keep, mask = _stack_body_features(batch["features"], avg_mode, segment_length=segment_length)
        if len(keep) == 0:
            continue


        if "names" not in batch or "video_paths" not in batch:
            raise KeyError("export_logits_to_pkl expects batch['names'] and batch['video_paths']")

        if X.ndim == 2:
            X = X.unsqueeze(1)

        names = [str(batch["names"][i]) for i in keep]
        keys  = [str(batch["video_paths"][i]) for i in keep]

        Xd = X.to(device, non_blocking=True)
        md = mask.to(device, non_blocking=True) if mask is not None else None

        if model_name == "prototypes":
            final, cls_l, proto_l, embeddings = model(Xd, mask=md)

            if export_raw:
                final_v = final
                cls_v   = cls_l
                proto_v = proto_l
            else:
                final_v = _probs_from_logits(final,   multi_label)
                cls_v   = _probs_from_logits(cls_l,   multi_label)
                proto_v = _probs_from_logits(proto_l, multi_label)

            final_v = final_v.detach().cpu()
            cls_v   = cls_v.detach().cpu()
            proto_v = proto_v.detach().cpu()
            embeddings = embeddings.detach().cpu()

        else:

            logits, embeddings = model(Xd, mask=md, return_embeddings=True)

            if export_raw:
                final_v = logits.detach().cpu()
            else:
                final_v = _probs_from_logits(logits, multi_label).detach().cpu()
            embeddings = embeddings.detach().cpu()


            C = final_v.size(1)
            cls_v   = torch.full((final_v.size(0), C), float("nan"))
            proto_v = torch.full((final_v.size(0), C), float("nan"))

        for i in range(len(keys)):
            k = keys[i]

            if k in out:
                k = f"{k}__dup{i}"

            if export_raw:
                out[k] = {
                    "name": names[i],
                    "final_logits": final_v[i].numpy().astype(np.float32),
                    "cls_logits":   cls_v[i].numpy().astype(np.float32),
                    "proto_logits": proto_v[i].numpy().astype(np.float32),
                    "embeddings": embeddings[i].numpy().astype(np.float32),
                }
            else:
                out[k] = {
                    "name": names[i],
                    "final_prob": final_v[i].numpy().astype(np.float32),
                    "cls_prob":   cls_v[i].numpy().astype(np.float32),
                    "proto_prob": proto_v[i].numpy().astype(np.float32),
                    "embeddings": embeddings[i].numpy().astype(np.float32),
                }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)

    logging.info(f"[EXPORT] saved {len(out)} files -> {out_path}")






def train(
    cfg,
    mm_loader: DataLoader,                # train
    dev_loaders: Dict[str, DataLoader] | None = None,
    test_loaders: Dict[str, DataLoader] | None = None,
):
    seed_everything(cfg.random_seed)
    device = torch.device(cfg.device)
    avg_mode = cfg.average_features.lower()
    multi_label = bool(getattr(cfg, "multi_label", False))
    ml_mode = str(getattr(cfg, "multi_label_mode", "2way") or "2way").lower()


    first = None
    for b in mm_loader:
        if b is not None:
            first = b
            break
    if first is None:
        raise RuntimeError("Train loader is empty (or collate filters everything).")
    X0, _, _ = _stack_body_features(first["features"], avg_mode, segment_length=cfg.segment_length)

    if X0.ndim == 3:   # raw: [B0, T, D]
        in_dim  = int(X0.shape[2])
        seq_len = int(X0.shape[1])  # == cfg.segment_length
    else:               # mean / mean_std: [B0, D]
        in_dim  = int(X0.shape[1])
        seq_len = 1



    # model_num_classes: 3 (single) / 2 (multi)

    if multi_label:
        if ml_mode in ML_ONEHOT3:
            model_num_classes = 3
        else:
            model_num_classes = 2
        metrics_num_classes = 3
    else:
        try:
            model_num_classes = cfg.num_classes
        except AttributeError:
            model_num_classes = _num_classes_from_loader(mm_loader, avg_mode, segment_length=cfg.segment_length)
        single_task = str(getattr(cfg, "single_task", "none") or "none").lower()
        global CLASS_LABELS
        if single_task.startswith("dep"):
            CLASS_LABELS = {0: "control", 1: "depression"}
            model_num_classes = 2
            metrics_num_classes = 2
        elif single_task.startswith("park"):
            CLASS_LABELS = {0: "control", 1: "parkinson"}
            model_num_classes = 2
            metrics_num_classes = 2
        else:
            CLASS_LABELS = {0: "control", 1: "depression", 2: "parkinson"}
            metrics_num_classes = model_num_classes


    if not multi_label:
        if cfg.class_weighting == "none":
            ce_weights = None
            logging.info("Class weighting: none (disabled)")
        else:
            y_all = _gather_all_labels(mm_loader, avg_mode, segment_length=cfg.segment_length)
            classes = np.arange(model_num_classes)
            class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_all)
            ce_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)
            logging.info(f"Class weighting: balanced -> {class_weights.tolist()}")
    else:

        pos_weight = _gather_pos_weight_for_ml(mm_loader, avg_mode, segment_length=cfg.segment_length).to(device)
        logging.info(f"pos_weight (BCE) -> {pos_weight.cpu().numpy().tolist()}")


    model = _build_model(cfg, in_dim, seq_len, model_num_classes, device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"[MODEL] params total={total_params:,} trainable={trainable_params:,}")


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
        raise ValueError(f"Unknown optimizer: {cfg.optimizer}")
    logging.info(f"Optimizer: {cfg.optimizer}, learning rate: {cfg.lr}")


    steps_per_epoch = sum(1 for b in mm_loader if b is not None)
    # steps_per_epoch = len(mm_loader)
    scheduler = SmartScheduler(
        scheduler_type=cfg.scheduler_type,
        optimizer=optimizer,
        config=cfg,
        steps_per_epoch=steps_per_epoch
    )


    if not multi_label:
        # criterion = nn.CrossEntropyLoss(weight=(ce_weights if 'ce_weights' in locals() else None))
        criterion = nn.CrossEntropyLoss(weight=ce_weights)
        # criterion = nn.CrossEntropyLoss(weight=ce_weights, label_smoothing=0.05)

    else:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)


    selection_metric = cfg.selection_metric
    early_stop_on = cfg.early_stop_on

    best_score = -1.0
    best_dev, best_test = {}, {}
    patience = 0
    best_ckpt_path = None
    protocol_dir = os.path.join(cfg.checkpoint_dir, "eval_protocol")
    os.makedirs(protocol_dir, exist_ok=True)

    for epoch in range(cfg.num_epochs):
        logging.info(f"=== EPOCH {epoch+1}/{cfg.num_epochs} ===")
        model.train()
        tot_loss, tot_n = 0.0, 0
        tr_y, tr_p = [], []

        for batch_idx, batch in enumerate(tqdm(mm_loader, desc="Train")):
            if batch is None:
                continue
            # X, keep = _stack_body_features(batch["features"], avg_mode, segment_length=cfg.segment_length)
            X, keep, mask = _stack_body_features(batch["features"], avg_mode, segment_length=cfg.segment_length)



            y_idx = _filter_labels(batch["labels"], keep).to(device)
            if not multi_label:
                y = y_idx
            else:
                if "labels_ml" not in batch:
                    raise RuntimeError("multi_label=True, but batch has no 'labels_ml'. Check dataset/collate.")
                y = batch["labels_ml"][keep].to(device)

            if X.ndim == 2:
                X = X.unsqueeze(1)

            # logits = model(X.to(device))  # [b,C]
            # logits = model(
            #     X.to(device, non_blocking=True),
            #     mask=mask.to(device, non_blocking=True) if mask is not None else None
            # )

            # loss = criterion(logits, y)

            if cfg.model_name.lower() == 'prototypes':
                # logits, _, _, embeddings =  model(
                # _, logits, _, embeddings =  model(
                final, cls_l, proto_l, embeddings =  model(
                    X.to(device, non_blocking=True),
                    mask=mask.to(device, non_blocking=True) if mask is not None else None
                )
                # logits = cls_l
                logits = final
                w_final = float(getattr(cfg, "loss_final_weight", 1.0))
                w_cls = float(getattr(cfg, "loss_cls_weight", 0.0))
                w_proto = float(getattr(cfg, "loss_proto_weight", 0.0))
                loss_terms = []
                if w_final != 0.0:
                    loss_terms.append(w_final * criterion(final, y))
                if w_cls != 0.0:
                    loss_terms.append(w_cls * criterion(cls_l, y))
                if w_proto != 0.0:
                    loss_terms.append(w_proto * criterion(proto_l, y))
                loss = sum(loss_terms) if loss_terms else torch.zeros((), device=final.device)

                if batch_idx == 0:
                    dbg_dump_logits(final, cfg.print_logits,   prefix="[DBG:TRAIN:final]", max_rows=5, max_cols=final.size(1))
                    dbg_dump_logits(cls_l, cfg.print_logits, prefix="[DBG:TRAIN:cls]",   max_rows=5, max_cols=cls_l.size(1))
                    dbg_dump_logits(proto_l, cfg.print_logits, prefix="[DBG:TRAIN:proto]", max_rows=5, max_cols=proto_l.size(1))
                    dbg_check_logits(final_logits=final, cls_logits=cls_l, proto_logits=proto_l, print_logits= cfg.print_logits, prefix="[DBG:TRAIN]")
            else:
                logits =  model(
                    X.to(device, non_blocking=True),
                    mask=mask.to(device, non_blocking=True) if mask is not None else None
                )
                if batch_idx == 0:
                    # _dbg_check_logits(cls_logits=logits, prefix="[DBG:TRAIN]")
                    dbg_dump_logits(logits, cfg.print_logits, prefix="[DBG:TRAIN:final]", max_rows=5, max_cols=logits.size(1))
                loss = criterion(logits, y)

            if cfg.model_name.lower() == 'prototypes':
                proto_emb = model._proto_project(embeddings)
                proto_bank = model._proto_project(model.prototypes)
                cont_loss = prototype_contrastive_loss(
                    proto_emb, y_idx, proto_bank,
                    num_classes=model.num_classes,
                    temperature=getattr(cfg, "proto_temperature", 0.1),
                    similarity=getattr(cfg, "proto_similarity", "cosine"),
                )

                alpha = cfg.prototype_alpha
                loss = loss + alpha * cont_loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step(batch_level=True)

            bs = y.size(0)
            tot_loss += loss.item() * bs
            tot_n += bs


            if not multi_label:
                tr_y.append(y_idx.cpu())
                tr_p.append(logits.argmax(dim=1).detach().cpu())
            else:
                tr_y.append(y_idx.cpu())
                if ml_mode in ML_ONEHOT3:
                    tr_p.append(logits.argmax(dim=1).detach().cpu())
                else:
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
                color_metric("WF1", m_tr["WF1"]),
            ]
            for c in range(metrics_num_classes):
                key = f"recall_c{c}_{CLASS_LABELS.get(c, f'class{c}')}"
                if key in m_tr:
                    parts.append(color_metric(key, m_tr[key]))
            logging.info(f"[{color_split('TRAIN')}] " + " | ".join(parts))
        else:
            logging.info(f"[{color_split('TRAIN')}] Loss={train_loss:.4f} | (no metrics)")


        cur_dev = {}
        if dev_loaders:
            for name, ldr in dev_loaders.items():

                md = _eval_epoch(
                    cfg, model, ldr, device, avg_mode,
                    metrics_num_classes,
                    model_name = cfg.model_name.lower(),
                    multi_label=multi_label,
                    thr_dep=getattr(cfg,"thr_dep",0.5),
                    thr_park=getattr(cfg,"thr_park",0.5),
                    segment_length=cfg.segment_length,
                    protocol_dir=protocol_dir,
                    split_tag=f"dev_{name}",
                    epoch_idx=epoch,
                )
                if md:
                    cur_dev.update({f"{k}_{name}": v for k, v in md.items()})
                    msg = " | ".join(color_metric(k, v) for k, v in md.items())
                    logging.info(f"[{color_split('DEV')}:{name}] {msg}")

        cur_test = {}
        if test_loaders:
            for name, ldr in test_loaders.items():
                mt = _eval_epoch(
                    cfg, model, ldr, device, avg_mode,
                    metrics_num_classes,
                    model_name = cfg.model_name.lower(),
                    multi_label=multi_label,
                    thr_dep=getattr(cfg, "thr_dep", 0.5),
                    thr_park=getattr(cfg, "thr_park", 0.5),
                    segment_length=cfg.segment_length,
                    protocol_dir=protocol_dir,
                    split_tag=f"test_{name}",
                    epoch_idx=epoch,
                )
                if mt:
                    cur_test.update({f"{k}_{name}": v for k, v in mt.items()})
                    msg = " | ".join(color_metric(k, v) for k, v in mt.items())
                    logging.info(f"[{color_split('TEST')}:{name}] {msg}")


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
            best_ckpt_path = str(ckpt)
            logging.info(f"Saved best model ({early_stop_on}/{selection_metric}={best_score:.4f}): {ckpt.name}")
        else:
            patience += 1
            if patience >= cfg.max_patience:
                logging.info("Early stopping.")
                break

    # ===== ONE-TIME EXPORT AFTER TRAIN =====
    EXPORT_DIR = "pkl_logits"
    os.makedirs(EXPORT_DIR, exist_ok=True)

    if best_ckpt_path is not None:
        state = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(state)
        model.eval()

        # train export
        out_path = os.path.join(EXPORT_DIR, f"{cfg.model_name.lower()}_train_best.pkl")
        export_logits_to_pkl(
            cfg=cfg,
            model=model,
            loader=mm_loader,
            device=device,
            avg_mode=avg_mode,
            model_name=cfg.model_name.lower(),
            out_path=out_path,
            multi_label=multi_label,
            segment_length=cfg.segment_length,
        )

        # dev export
        if dev_loaders:
            for split_name, ldr in dev_loaders.items():
                out_path = os.path.join(
                    EXPORT_DIR, f"{cfg.model_name.lower()}_dev_{split_name}_best.pkl"
                )
                export_logits_to_pkl(
                    cfg=cfg,
                    model=model,
                    loader=ldr,
                    device=device,
                    avg_mode=avg_mode,
                    model_name=cfg.model_name.lower(),
                    out_path=out_path,
                    multi_label=multi_label,
                    segment_length=cfg.segment_length,
                )

        if test_loaders:
            for split_name, ldr in test_loaders.items():
                out_path = os.path.join(EXPORT_DIR, f"{cfg.model_name.lower()}_{split_name}_best.pkl")
                export_logits_to_pkl(
                    cfg=cfg,
                    model=model,
                    loader=ldr,
                    device=device,
                    avg_mode=avg_mode,
                    model_name=cfg.model_name.lower(),
                    out_path=out_path,
                    multi_label=multi_label,
                    segment_length=cfg.segment_length,
                )
    # ======================================

    return best_dev, best_test

