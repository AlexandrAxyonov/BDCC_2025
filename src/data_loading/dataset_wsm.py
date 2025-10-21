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
    """
    Сегмент = независимое видео.
    CSV обязан содержать колонки: video_id, diagnosis, segment_file.
    Пути к сегментам считаются по шаблону:
        <video_dir>/<video_id>/segments/<segment_file>
    где video_dir указывается в config.toml (например: E:/WSM/depression/{split}_labels).
    """

    def __init__(
        self,
        csv_path: str,
        video_dir: str,
        config,
        split: str,
        modality_processors: Dict[str, Any],         # должен содержать 'body'
        modality_feature_extractors: Dict[str, Any], # должен содержать 'body'
        dataset_name: str = "wsm",
        device: str = "cuda",
    ) -> None:
        super().__init__()

        # базовые поля
        self.csv_path   = csv_path
        self.video_dir  = video_dir
        self.config     = config
        self.split      = split
        self.dataset_name = dataset_name
        self.device     = device

        self.multi_label: bool = bool(getattr(self.config, "multi_label", False))

        # параметры извлечения
        self.segment_length   = config.segment_length
        self.subset_size      = config.subset_size
        self.average_features = config.average_features  # 'raw'|'mean'|'mean_std'
        self.yolo_weights     = config.yolo_weights
        self.video_mode       = config.video_mode

        # процессоры/экстракторы (только body)
        self.proc = modality_processors.get("body", None)
        self.extr = modality_feature_extractors.get("body", None)
        if self.proc is None:
            raise ValueError("Нужен image processor для 'body' (CLIPProcessor/AutoImageProcessor).")
        if self.extr is None:
            raise ValueError("Нужен extractor для 'body' (CLIP/VIT).")

        # кэш
        self.save_prepared_data = config.save_prepared_data
        self.save_feature_path  = config.save_feature_path
        self.store = FeatureStore(self.save_feature_path)

        # CSV
        df = pd.read_csv(self.csv_path)
        required = {"video_id", "diagnosis", "segment_file"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"CSV должен содержать столбцы {sorted(required)}. Отсутствуют: {sorted(missing)}")
        if self.subset_size > 0:
            df = df.head(self.subset_size)
        logging.info(
            f"[WSMBodyDataset] {self.dataset_name}/{self.split}: "
            f"subset_size={self.subset_size} -> rows={len(df)}"
        )
        self.df = df

        self.corpus = self._detect_corpus(self.csv_path, self.video_dir)

        # meta (пути + лейбл)
        self.meta: List[Dict[str, Any]] = []
        if self.save_prepared_data:
            self.meta = self.store.load_meta(
                self.dataset_name, self.split, getattr(self.config, "random_seed", 0), self.subset_size
            )
        if not self.meta:
            self._build_meta_only()
            if self.save_prepared_data:
                self.store.save_meta(
                    self.dataset_name, self.split, getattr(self.config, "random_seed", 0),
                    self.subset_size, self.meta
                )

        # подготовим/дозаполним кэш фич
        self._prepare_body_cache()

    # ───────────────────── helpers ───────────────────── #

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
        if self.corpus == "depression":
            return 1 if raw == 1 else 0
        if self.corpus == "parkinson":
            return 2 if raw == 1 else 0
        return 0

    def _segment_path(self, base_dir: str, video_id: str, segment_file: str) -> str:
        """
        Идеальные условия: сегменты всегда по шаблону
            <base_dir>/<video_id>/segments/<segment_file>
        Абсолютные пути не ожидаем и не поддерживаем.
        """
        p = os.path.join(base_dir, str(video_id), "segments", segment_file)
        if not os.path.exists(p):
            raise FileNotFoundError(f"Ожидал сегмент по пути: {p}")
        return p

    def _build_meta_only(self) -> None:
        self.meta = []
        for _, row in tqdm(self.df.iterrows(), total=len(self.df),
                           desc=f"Indexing WSM segments [{self.dataset_name}/{self.split}]"):
            vid = str(row["video_id"])
            seg = str(row["segment_file"])
            vpath = self._segment_path(self.video_dir, vid, seg)

            class_id = self._map_label(int(row["diagnosis"]))

            # У тебя нет коллизий имён — берём stem(segment_file)
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
        Маппинг single-label класса в 2-мерный multi-label вектор [dep, park]:
          0 (control)   -> [0., 0.]
          1 (depression)-> [1., 0.]
          2 (parkinson) -> [0., 1.]
        """
        if class_id == 1:
            return torch.tensor([1.0, 0.0], dtype=torch.float32)
        if class_id == 2:
            return torch.tensor([0.0, 1.0], dtype=torch.float32)
        return torch.tensor([0.0, 0.0], dtype=torch.float32)

    # ─────────────────── feature caching ─────────────────── #

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

        # быстрый индекс name -> vpath
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

                # ROI тела → pixel_values [T,3,H,W]
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
        """
        feats: {'embedding': Tensor [T,D] или [D]}
        'raw'      -> {'seq': [T,D]}
        'mean'     -> {'mean': [D]}
        'mean_std' -> {'mean':[D],'std':[D]}
        """
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

    # ───────────────────── dataset API ───────────────────── #

    def __len__(self) -> int:
        return len(self.meta)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        base = self.meta[idx]
        name = base["sample_name"]

        # достаём из кэша
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
            out["label_ml"] = self._to_multi_label_vec(label_idx)  # float32, форма [2]

        return out
