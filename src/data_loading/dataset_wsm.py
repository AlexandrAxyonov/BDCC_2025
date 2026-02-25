from __future__ import annotations
import os
import logging
from typing import Dict, Any, List, Optional

import torch
from torch.utils.data import Dataset
import pandas as pd
from tqdm import tqdm

from .video_preprocessor import get_body_pixel_values
from src.utils.feature_store import FeatureStore, build_cache_key, need_full_reextract, merge_missing


class WSMBodyDataset(Dataset):

    def __init__(
        self,
        csv_path: str,
        video_dir: str,
        config,
        split: str,
        modality_processors: Dict[str, Any],
        modality_feature_extractors: Dict[str, Any],
        dataset_name: str = "wsm",
        device: str = "cuda",
    ) -> None:
        super().__init__()


        self.csv_path   = csv_path
        self.video_dir  = video_dir
        self.config     = config
        self.split      = split
        self.dataset_name = dataset_name
        self.device     = device

        self.multi_label: bool = bool(getattr(self.config, "multi_label", False))
        self.multi_label_mode: str = str(getattr(self.config, "multi_label_mode", "2way") or "2way")
        self.single_task: str = str(getattr(self.config, "single_task", "none") or "none").lower()
        self.meta_store_name = (
            f"{self.dataset_name}__{self.single_task}"
            if self.single_task not in {"none", ""}
            else self.dataset_name
        )


        self.segment_length   = config.segment_length
        self.subset_size      = config.subset_size
        self.average_features = config.average_features  # 'raw'|'mean'|'mean_std'
        self.yolo_weights     = config.yolo_weights
        self.video_mode       = config.video_mode


        self.proc = modality_processors.get("body", None)
        self.extr = modality_feature_extractors.get("body", None)
        if self.proc is None:
            raise ValueError("An image processor is required for 'body' (CLIPProcessor/AutoImageProcessor).")
        if self.extr is None:
            raise ValueError("An extractor is required for 'body' (CLIP/VIT).")


        self.save_prepared_data = config.save_prepared_data
        self.save_feature_path  = config.save_feature_path
        self.store = FeatureStore(self.save_feature_path)

        # CSV
        df = pd.read_csv(self.csv_path)
        required = {"video_id", "diagnosis", "segment_file"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"CSV must contain columns {sorted(required)}. Missing: {sorted(missing)}")
        if self.subset_size > 0:
            df = df.head(self.subset_size)
        logging.info(
            f"[WSMBodyDataset] {self.dataset_name}/{self.split}: "
            f"subset_size={self.subset_size} -> rows={len(df)}"
        )
        self.df = df

        self.corpus = self._detect_corpus(self.csv_path, self.video_dir)


        self.meta: List[Dict[str, Any]] = []
        if self.save_prepared_data:
            self.meta = self.store.load_meta(
                self.meta_store_name, self.split, getattr(self.config, "random_seed", 0), self.subset_size
            )
        if not self.meta:
            self._build_meta_only()
            if self.save_prepared_data:
                self.store.save_meta(
                    self.meta_store_name, self.split, getattr(self.config, "random_seed", 0),
                    self.subset_size, self.meta
                )


        self._prepare_body_cache()



    @staticmethod
    def _detect_corpus(csv_path: str, video_dir: str) -> str:
        def f(s: str) -> Optional[str]:
            s = (s or "").lower()
            if "parkinson" in s: return "parkinson"
            if "depress"   in s: return "depression"
            return None
        return f(csv_path) or f(video_dir) or "unknown"

    def _map_label(self, raw: int) -> int:
        raw = int(raw)
        if self.single_task.startswith("dep"):
            return 1 if raw == 1 else 0
        if self.single_task.startswith("park"):
            return 1 if raw == 1 else 0
        if self.corpus == "depression":
            return 1 if raw == 1 else 0
        if self.corpus == "parkinson":
            return 2 if raw == 1 else 0
        return 0

    def _segment_path(self, base_dir: str, video_id: str, segment_file: str) -> str:
        p = os.path.join(base_dir, str(video_id), "segments", segment_file)
        if not os.path.exists(p):
            raise FileNotFoundError(f"Expected a segment at path: {p}")
        return p

    def _build_meta_only(self) -> None:
        self.meta = []
        for _, row in tqdm(self.df.iterrows(), total=len(self.df),
                           desc=f"Indexing WSM segments [{self.dataset_name}/{self.split}]"):
            vid = str(row["video_id"])
            seg = str(row["segment_file"])
            vpath = self._segment_path(self.video_dir, vid, seg)

            class_id = self._map_label(int(row["diagnosis"]))


            sample_name = os.path.splitext(os.path.basename(seg))[0]

            self.meta.append({
                "sample_name": sample_name,
                "video_path": vpath,
                "label": int(class_id),
            })

        logging.info(
            f"[WSMBodyDataset] {self.dataset_name}/{self.split}: "
            f"indexed segments={len(self.meta)} / rows={len(self.df)}"
        )

    def _to_multi_label_vec(self, class_id: int) -> torch.Tensor:
        """
        2way: [dep, park] where control -> [0,0]
        onehot3: [control, dep, park]
        """
        mode = self.multi_label_mode.strip().lower()
        if mode in {"onehot3", "3way", "3", "onehot"}:
            if class_id == 0:
                return torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)
            if class_id == 1:
                return torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)
            return torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)

        if class_id == 1:
            return torch.tensor([1.0, 0.0], dtype=torch.float32)
        if class_id == 2:
            return torch.tensor([0.0, 1.0], dtype=torch.float32)
        return torch.tensor([0.0, 0.0], dtype=torch.float32)


    def _prepare_body_cache(self) -> None:
        if not self.meta:
            return
        sample_names = [m["sample_name"] for m in self.meta]

        mod = "body"
        ex = self.extr

        key = build_cache_key(mod, ex, self.config)
        store, header = self.store.load_modality_store(
            self.dataset_name, self.split, key, getattr(self.config, "random_seed", 0), self.subset_size
        )
        if need_full_reextract(self.config, mod, header, key):
            store = {}

        missing = merge_missing(store, sample_names)
        if not missing:
            return


        path_by_name = {m["sample_name"]: m["video_path"] for m in self.meta}

        for name in tqdm(
            missing,
            desc=f"Extracting {mod} [{self.dataset_name}/{self.split}]",
            leave=True
        ):
            try:
                vpath = path_by_name.get(name)
                if not vpath:
                    store[name] = None
                    continue


                _, body_pv = get_body_pixel_values(
                    video_path=vpath,
                    segment_length=self.segment_length,
                    image_processor=self.proc,
                    device=self.device,
                    yolo_weights=self.yolo_weights,
                    mode=self.video_mode,
                )

                feats = ex.extract(pixel_values=body_pv) if body_pv is not None else None
                feats = self._aggregate(feats, self.average_features) if feats is not None else None
                store[name] = feats

            except Exception as e:
                logging.warning(f"{mod} extract error {name}: {e}")
                store[name] = None

        self.store.save_modality_store(
            self.dataset_name, self.split, key, getattr(self.config, "random_seed", 0), self.subset_size, store
        )
        torch.cuda.empty_cache()

    def _aggregate(self, feats: Any, average: str) -> Optional[dict]:
        if not isinstance(feats, dict):
            raise TypeError(f"Expected dict with key 'embedding', got {type(feats)}")
        emb = feats.get("embedding", None)
        if emb is None or not isinstance(emb, torch.Tensor):
            raise TypeError(f"Features dict must contain 'embedding' Tensor, got keys {list(feats.keys())}")

        if emb.ndim == 1:
            emb = emb.unsqueeze(0)  # [1,D]

        if average == "mean_std":
            return {"mean": emb.mean(dim=0), "std": emb.std(dim=0, unbiased=False)}
        elif average == "mean":
            return {"mean": emb.mean(dim=0)}
        else:  # 'raw'
            return {"seq": emb}



    def __len__(self) -> int:
        return len(self.meta)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        base = self.meta[idx]
        name = base["sample_name"]


        features = {}
        key = build_cache_key("body", self.extr, self.config)
        cache = self.store.get_store(
            self.dataset_name, self.split, key, getattr(self.config, "random_seed", 0), self.subset_size
        )
        features["body"] = cache.get(name, None)
        label_idx = int(base["label"])
        out = {
            "sample_name": name,
            "video_path": base["video_path"],
            "label": torch.tensor(label_idx, dtype=torch.long),
            "features": features,
        }
        if self.multi_label:
            out["label_ml"] = self._to_multi_label_vec(label_idx)  # float32, shape [2] or [3]

        return out
