from __future__ import annotations
from typing import Dict, Any, List, Tuple
import os
import torch
from torch.utils.data import DataLoader, ConcatDataset

from .dataset_wsm import WSMBodyDataset

def wsm_collate_fn(batch: List[Dict[str, Any]]):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None

    names  = [b["sample_name"] for b in batch]
    vpaths = [b["video_path"] for b in batch]

    # single-label: LongTensor[B]
    labels = torch.stack(
        [
            torch.as_tensor(b["label"], dtype=torch.long)
            if not isinstance(b["label"], torch.Tensor) else b["label"].to(torch.long)
            for b in batch
        ],
        dim=0
    )


    has_ml = all(("label_ml" in b) and (b["label_ml"] is not None) for b in batch)
    if has_ml:
        labels_ml = torch.stack(
            [
                torch.as_tensor(b["label_ml"], dtype=torch.float32)
                if not isinstance(b["label_ml"], torch.Tensor) else b["label_ml"].to(torch.float32)
                for b in batch
            ],
            dim=0
        )
    else:
        labels_ml = None


    features = [b.get("features") for b in batch]

    out = {
        "video_paths": vpaths,
        "labels": labels,          # LongTensor[B]
        "names": names,
        "features": features,
    }
    if has_ml:
        out["labels_ml"] = labels_ml  # FloatTensor[B, 2] or [B, 3]

    return out


def make_wsm_dataset_and_loader(config, split: str) -> Tuple[ConcatDataset, DataLoader]:
    ds_list = []
    single_task = str(getattr(config, "single_task", "none") or "none").lower()
    def _match_single_task(name: str) -> bool:
        if single_task in {"none", ""}:
            return True
        n = name.lower()
        if single_task.startswith("dep"):
            return "depress" in n
        if single_task.startswith("park"):
            return "parkinson" in n
        return True
    for ds_name, ds_cfg in getattr(config, "datasets", {}).items():
        if not ds_name.lower().startswith("wsm_"):
            continue
        if not _match_single_task(ds_name):
            continue
        csv_path  = ds_cfg["csv_path"].format(base_dir=ds_cfg["base_dir"], split=split)
        video_dir = ds_cfg["video_dir"].format(base_dir=ds_cfg["base_dir"], split=split)
        if not os.path.exists(csv_path):
            print(f"[WSM] skip {ds_name} for split={split}: CSV not found -> {csv_path}")
            continue

        ds = WSMBodyDataset(
            csv_path=csv_path,
            video_dir=video_dir,
            config=config,
            split=split,
            modality_processors=getattr(config, "modality_processors"),
            modality_feature_extractors=getattr(config, "modality_extractors"),
            dataset_name=ds_name,
            device=getattr(config, "device", "cuda"),
        )
        ds_list.append(ds)

    if not ds_list:
        raise ValueError(f"No WSM corpus was found for split='{split}'.")

    dataset = ds_list[0] if len(ds_list) == 1 else ConcatDataset(ds_list)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=(split == "train"),
        num_workers=config.num_workers,
        collate_fn=wsm_collate_fn,
    )
    return dataset, loader
