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

        # нормализуем строковые варианты "none"
        if isinstance(gate_mode, str) and gate_mode.lower() in {"none", "", "null"}:
            gate_mode = None

        self.seg_len = seg_len
        self.hidden_dim = hidden_dim
        self.gate_mode = gate_mode
        self.num_layers = tr_layer_number

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

        # ───────────────── gating-параметры ─────────────────
        # 4 режима:
        #   bt: B×T×1  (per-sample, per-time)
        #   bd: B×1×D  (per-sample, per-feature)
        #   t:  1×T×1  (global по батчу, по времени)
        #   d:  1×1×D  (global по батчу, по фичам)
        #
        # Здесь храним ЛОГИТЫ, зависят только от слоя, не от данных.
        if self.gate_mode is not None:
            # bt: логиты по времени (как у t), но потом расширяем до B×T×1
            self.bt_gates = nn.ParameterList([
                nn.Parameter(torch.zeros(self.seg_len, 1))   # [T, 1]
                for _ in range(tr_layer_number)
            ])

            # bd: логиты по фичам (как у d), но потом расширяем до B×1×D
            self.bd_gates = nn.ParameterList([
                nn.Parameter(torch.zeros(hidden_dim))        # [D]
                for _ in range(tr_layer_number)
            ])

            # t: глобальные логиты по времени (как прежде)
            self.t_gates = nn.ParameterList([
                nn.Parameter(torch.zeros(self.seg_len, 1))   # [T, 1]
                for _ in range(tr_layer_number)
            ])

            # d: глобальные логиты по фичам (как прежде)
            self.d_gates = nn.ParameterList([
                nn.Parameter(torch.zeros(hidden_dim))        # [D]
                for _ in range(tr_layer_number)
            ])

            # можно слегка возбудить логиты по времени, а фичи оставить нулями
            for p in self.bt_gates:
                nn.init.xavier_uniform_(p)      # [T,1]
            for p in self.t_gates:
                nn.init.xavier_uniform_(p)      # [T,1]
            # bd_gates и d_gates остаются нулями (равномерные softmax по D)

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

        # Последовательные слои трансформера + (при необходимости) gated residual
        for i in range(len(self.transformer)):
            att = self.transformer[i](
                sequences,   # Q
                fixed_seq,   # K
                fixed_seq,   # V
                key_padding_mask=(~mask) if mask is not None else None,
            )  # [B, T, hidden_dim]

            if self.gate_mode is None:
                # Стандартный residual
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
        sequences: [B, T, D]
        Возвращает alpha: [B, T, D]

        Режимы (форма alpha до expand_as):
          - "bt": [B, T, 1]  (per-sample, per-time)      ← bt_gates (логиты по T)
          - "bd": [B, 1, D]  (per-sample, per-feature)   ← bd_gates (логиты по D)
          - "t":  [1, T, 1]  (global по батчу, по времени) ← t_gates
          - "d":  [1, 1, D]  (global по батчу, по фичам)   ← d_gates

        """
        B, T, D = sequences.shape

        if self.gate_mode == "bt":
            # per-sample, per-time (формально B×T×1 после expand по B)
            logits_t = self.bt_gates[layer_idx]              # [T, 1]
            alpha_t = torch.softmax(logits_t.squeeze(-1), 0) # [T]
            alpha = alpha_t.view(1, T, 1).expand(B, T, 1)    # [B, T, 1]

        elif self.gate_mode == "bd":
            # per-sample, per-feature (формально B×1×D после expand по B)
            logits_d = self.bd_gates[layer_idx]              # [D]
            alpha_d = torch.softmax(logits_d, 0)             # [D]
            alpha = alpha_d.view(1, 1, D).expand(B, 1, D)    # [B, 1, D]

        elif self.gate_mode == "t":
            # global по батчу, по времени: 1×T×1
            logits_t = self.t_gates[layer_idx]               # [T, 1]
            alpha_t = torch.softmax(logits_t.squeeze(-1), 0) # [T]
            alpha = alpha_t.view(1, T, 1)                    # [1, T, 1]

        elif self.gate_mode == "d":
            # global по батчу, по фичам: 1×1×D
            logits_d = self.d_gates[layer_idx]               # [D]
            alpha_d = torch.softmax(logits_d, 0)             # [D]
            alpha = alpha_d.view(1, 1, D)                    # [1, 1, D]

        else:
            raise ValueError(f"Unknown gate_mode: {self.gate_mode}")

        # Приводим к [B,T,D] broadcasting-ом
        alpha = alpha.expand_as(sequences)                   # [B, T, D]
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

        # 4) классификатор
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
            # 1) прогоняем через всех экспертов этого слоя
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

            # 2) router: веса по экспертам для каждого токена
            logits = self.routers[layer_idx](sequences)   # [B,T,E]

            if self.moe_top_k is not None and self.moe_top_k < self.moe_num_experts:
                # top-k sparsity
                topk_vals, topk_idx = torch.topk(logits, self.moe_top_k, dim=-1)
                mask_logits = logits.new_full(logits.shape, float("-inf"))
                mask_logits.scatter_(-1, topk_idx, topk_vals)
                logits = mask_logits

            probs = torch.softmax(logits, dim=-1)  # [B,T,E]
            probs = probs.unsqueeze(-1)            # [B,T,E,1]

            # 3) смешиваем экспертов
            att_mix = (probs * expert_outs).sum(dim=2)  # [B,T,H]

            # обычный residual
            sequences = sequences + att_mix

        # pooling + classifier
        sequences_pool = self._pool_features(sequences, mask)
        return self.classifier(sequences_pool)

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
