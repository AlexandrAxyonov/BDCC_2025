# coding: utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F

from .help_layers import TransformerEncoderLayer


class VideoFormer(nn.Module):
    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 128,
        num_transformer_heads: int = 2,
        positional_encoding: bool = True,
        dropout: float = 0.1,
        tr_layer_number: int = 5,
        seg_len: int = 20,
        out_features: int = 128,
        num_classes: int = 7,
        gate_mode: str | None = None,
    ):
        super(VideoFormer, self).__init__()

        if isinstance(gate_mode, str) and gate_mode.lower() in {"none", "", "null"}:
            gate_mode = None

        self.seg_len = seg_len
        self.hidden_dim = hidden_dim
        self.gate_mode = gate_mode

        # Проекция входных фич в hidden_dim
        self.image_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )

        # Трансформерные слои
        self.transformer = nn.ModuleList([
            TransformerEncoderLayer(
                input_dim=hidden_dim,
                num_heads=num_transformer_heads,
                dropout=dropout,
                positional_encoding=positional_encoding,
            )
            for _ in range(tr_layer_number)
        ])

        # if self.gate_mode is not None:
        #     self.time_gates = nn.ModuleList([
        #         nn.Linear(hidden_dim, 1)          # [B, T, D] -> [B, T, 1]
        #         for _ in range(tr_layer_number)
        #     ])
        #     self.feat_gates = nn.ModuleList([
        #         nn.Linear(hidden_dim, hidden_dim) # [B, D] -> [B, D]
        #         for _ in range(tr_layer_number)
        #     ])

        # Классификатор
        self._calculate_classifier_input_dim()
        self.classifier = nn.Sequential(
            nn.Linear(self.classifier_input_dim, out_features),
            nn.LayerNorm(out_features),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_features, num_classes),
        )

        self._init_weights()

    def forward(self, sequences: torch.Tensor, mask: torch.Tensor | None = None):

        # Проекция входа
        sequences = self.image_proj(sequences)  # [B, T, hidden_dim]

        # Фиксируем "память"
        fixed_seq = sequences

        # Последовательные слои трансформера + gated residual
        for i in range(len(self.transformer)):
            att = self.transformer[i](
                sequences,   # Q
                fixed_seq,   # K
                fixed_seq,   # V
                key_padding_mask=(~mask) if mask is not None else None,
            )  # [B, T, hidden_dim]

            if self.gate_mode is None:
                # Стандартный residual без gating
                sequences = sequences + att
            else:
                alpha = self._compute_alpha(i, sequences)  # [B, T, hidden_dim]
                sequences = (1.0 - alpha) * sequences + alpha * att

        # Пулинг по времени
        sequences_pool = self._pool_features(sequences, mask)  # [B, hidden_dim]

        # Классификатор
        output = self.classifier(sequences_pool)  # [B, num_classes]
        return output

    #    GATING: alpha(x)        #

    def _compute_alpha(self, layer_idx: int, sequences: torch.Tensor) -> torch.Tensor:
        """
        Вычисление alpha для заданного слоя в зависимости от режима gate_mode.

        Вход:
            sequences: [B, T, D]
        Выход:
            alpha:     [B, T, D]  (через broadcast)
        """
        if self.gate_mode in ("bt", "t"):
            # time-based gate: сначала [B, T, 1]
            alpha = torch.sigmoid(self.time_gates[layer_idx](sequences))  # [B, T, 1]

            if self.gate_mode == "t":
                # общий по батчу: [1, T, 1]
                alpha = alpha.mean(dim=0, keepdim=True)

        elif self.gate_mode in ("bd", "d"):
            # feature-based gate
            # усредняем по времени -> [B, D]
            seq_mean = sequences.mean(dim=1)                          # [B, D]
            alpha = torch.sigmoid(self.feat_gates[layer_idx](seq_mean))  # [B, D]
            alpha = alpha.unsqueeze(1)                                # [B, 1, D]

            if self.gate_mode == "d":
                # общий по батчу: [1, 1, D]
                alpha = alpha.mean(dim=0, keepdim=True)

        else:
            raise ValueError(f"Unknown gate_mode: {self.gate_mode}")

        # Приводим к размеру [B, T, D] через broadcast
        alpha = alpha.expand_as(sequences)
        return alpha

    def _calculate_classifier_input_dim(self):

        dummy_video = torch.randn(1, self.seg_len, self.hidden_dim)
        video_pool = self._pool_features(dummy_video, mask=None)
        self.classifier_input_dim = video_pool.size(1)

    def _pool_features(self, sequences: torch.Tensor, mask: torch.Tensor | None = None):

        if mask is None:
            mean_temp = sequences.mean(dim=1)  # [B, H]
            return mean_temp

        denom = mask.sum(dim=1).clamp(min=1).unsqueeze(-1).to(sequences.dtype)  # [B,1]
        sequences_masked = sequences.masked_fill(~mask.unsqueeze(-1), 0.0)
        mean_temp = sequences_masked.sum(dim=1) / denom  # [B, H]
        return mean_temp

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class VideoFormerMoE(nn.Module):
    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 128,
        num_transformer_heads: int = 2,
        positional_encoding: bool = True,
        dropout: float = 0.1,
        tr_layer_number: int = 5,      # глубина (число MoE-блоков)
        moe_num_experts: int = 3,      # число экспертов на слой
        moe_top_k: int = 1,            # сколько экспертов реально использовать
        seg_len: int = 20,
        out_features: int = 128,
        num_classes: int = 7,
    ):
        super().__init__()

        self.seg_len = seg_len
        self.hidden_dim = hidden_dim
        self.num_layers = tr_layer_number
        self.moe_num_experts = moe_num_experts
        self.moe_top_k = moe_top_k

        # 1) проекция входа
        self.image_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )

        # 2) MoE-слои: для каждого слоя — список экспертов
        self.moe_layers = nn.ModuleList([
            nn.ModuleList([
                TransformerEncoderLayer(
                    input_dim=hidden_dim,
                    num_heads=num_transformer_heads,
                    dropout=dropout,
                    positional_encoding=positional_encoding,
                )
                for _ in range(moe_num_experts)
            ])
            for _ in range(tr_layer_number)
        ])

        # 3) роутеры: по слою, по токену → веса по экспертам
        self.routers = nn.ModuleList([
            nn.Linear(hidden_dim, moe_num_experts)   # [B,T,D] → [B,T,E]
            for _ in range(tr_layer_number)
        ])

        # 4) классификатор (как у тебя)
        self._calculate_classifier_input_dim()
        self.classifier = nn.Sequential(
            nn.Linear(self.classifier_input_dim, out_features),
            nn.LayerNorm(out_features),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_features, num_classes),
        )

        self._init_weights()

    def forward(self, sequences: torch.Tensor, mask: torch.Tensor | None = None):
        # [B,T,D_in] → [B,T,H]
        sequences = self.image_proj(sequences)
        fixed_seq = sequences

        B, T, H = sequences.shape

        for layer_idx in range(self.num_layers):
            # ── 1) прогоняем через всех экспертов этого слоя ─────────
            expert_outs = []
            for e_idx, expert in enumerate(self.moe_layers[layer_idx]):
                att_e = expert(
                    sequences,   # Q
                    fixed_seq,   # K
                    fixed_seq,   # V
                    key_padding_mask=(~mask) if mask is not None else None,
                )               # [B,T,H]
                expert_outs.append(att_e.unsqueeze(2))  # [B,T,1,H]

            # [B,T,E,H]
            expert_outs = torch.cat(expert_outs, dim=2)

            # ── 2) router: веса по экспертам для каждого токена ──────
            logits = self.routers[layer_idx](sequences)   # [B,T,E]

            if self.moe_top_k is not None and self.moe_top_k < self.moe_num_experts:
                # top-k sparsity
                topk_vals, topk_idx = torch.topk(logits, self.moe_top_k, dim=-1)
                mask_logits = logits.new_full(logits.shape, float("-inf"))
                mask_logits.scatter_(-1, topk_idx, topk_vals)
                logits = mask_logits

            probs = torch.softmax(logits, dim=-1)  # [B,T,E]
            probs = probs.unsqueeze(-1)            # [B,T,E,1]

            # ── 3) смешиваем экспертов ───────────────────────────────
            att_mix = (probs * expert_outs).sum(dim=2)  # [B,T,H]

            # обычный residual (можно сюда потом прикрутить твой alpha-гейтинг)
            sequences = sequences + att_mix

        # pooling + classifier как в твоей модели
        sequences_pool = self._pool_features(sequences, mask)
        return self.classifier(sequences_pool)

    # ── служебные функции такие же, как у тебя ─────────────────────────
    def _calculate_classifier_input_dim(self):
        dummy_video = torch.randn(1, self.seg_len, self.hidden_dim)
        video_pool = self._pool_features(dummy_video, mask=None)
        self.classifier_input_dim = video_pool.size(1)

    def _pool_features(self, sequences, mask=None):
        if mask is None:
            return sequences.mean(dim=1)
        denom = mask.sum(dim=1).clamp(min=1).unsqueeze(-1).to(sequences.dtype)
        sequences_masked = sequences.masked_fill(~mask.unsqueeze(-1), 0.0)
        return sequences_masked.sum(dim=1) / denom

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
