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

        if self.gate_mode is not None:
            self.bt_gates = nn.ParameterList([
                nn.Parameter(torch.empty(hidden_dim, 1))      # [D, 1]
                for _ in range(tr_layer_number)
            ])

            self.bd_gates = nn.ParameterList([
                nn.Parameter(torch.empty(hidden_dim, hidden_dim))  # [D, D]
                for _ in range(tr_layer_number)
            ])

            self.t_gates = nn.ParameterList([
                nn.Parameter(torch.empty(self.seg_len, 1))   # [T, 1]
                for _ in range(tr_layer_number)
            ])

            self.d_gates = nn.ParameterList([
                nn.Parameter(torch.empty(hidden_dim))        # [D]
                for _ in range(tr_layer_number)
            ])

            # инициализация
            for plist in (self.bt_gates, self.bd_gates, self.t_gates, self.d_gates):
                for p in plist:
                    if p.dim() >= 2:
                        nn.init.xavier_uniform_(p)
                    else:
                        nn.init.zeros_(p)

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

    def forward(self, sequences: torch.Tensor, mask: torch.Tensor | None = None, return_embeddings: bool = False):

        # Проекция входа
        sequences = self.image_proj(sequences)  # [B, T, hidden_dim]

        # Фиксируем "память"
        # fixed_seq = sequences

        # Последовательные слои трансформера + (при необходимости) gated residual
        for i in range(len(self.transformer)):
            att = self.transformer[i](
                sequences,   # Q
                sequences,   # K
                sequences,   # V
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
        if return_embeddings:
            return output, sequences_pool
        return output

    #    GATING: alpha(x)        #

    def _compute_alpha(self, layer_idx: int, sequences: torch.Tensor) -> torch.Tensor:

        B, T, D = sequences.shape

        if self.gate_mode == "bt":
            # per-sample, per-time
            W_bt = self.bt_gates[layer_idx]                     # [D, 1]
            seq_flat = sequences.reshape(B * T, D)              # [B*T, D]
            alpha_flat = torch.matmul(seq_flat, W_bt)           # [B*T, 1]
            alpha = torch.sigmoid(alpha_flat).view(B, T, 1)     # [B, T, 1]

        elif self.gate_mode == "bd":
            # per-sample, per-feature
            seq_mean = sequences.mean(dim=1)                    # [B, D]
            W_bd = self.bd_gates[layer_idx]                     # [D, D]
            alpha_feat = torch.matmul(seq_mean, W_bd)           # [B, D]
            alpha_feat = torch.sigmoid(alpha_feat)              # [B, D]
            alpha = alpha_feat.unsqueeze(1)                     # [B, 1, D]

        elif self.gate_mode == "t":
            # global по батчу, по времени
            W_t = self.t_gates[layer_idx]                       # [T, 1]
            alpha_t = torch.softmax(W_t.squeeze(-1), dim=0)     # [T]
            alpha = alpha_t.view(1, T, 1)                       # [1, T, 1]

        elif self.gate_mode == "d":
            # global по батчу, по фичам
            W_d = self.d_gates[layer_idx]                       # [D]
            alpha_d = torch.softmax(W_d, dim=0)                 # [D]
            alpha = alpha_d.view(1, 1, D)                       # [1, 1, D]

        else:
            raise ValueError(f"Unknown gate_mode: {self.gate_mode}")

        # Приводим к [B,T,D] broadcasting-ом
        alpha = alpha.expand_as(sequences)                      # [B, T, D]
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

class VideoFormer_with_Prototypes(nn.Module):
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
        num_prototypes_per_class: int = 3,
        proto_similarity: str = "cosine",
        proto_temperature: float = 0.1,
    ):
        super(VideoFormer_with_Prototypes, self).__init__()


        self.seg_len = seg_len
        self.hidden_dim = hidden_dim

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
                dropout=0.1,
                positional_encoding=positional_encoding,
            )
            for _ in range(tr_layer_number)
        ])

        self.num_classes = num_classes
        self.num_prototypes_per_class = num_prototypes_per_class
        self.total_prototypes = num_classes * num_prototypes_per_class
        self.proto_similarity = (proto_similarity or "cosine").lower()
        if self.proto_similarity in {"euclid", "euclidean", "l2"}:
            self.proto_similarity = "inv_euclid"
        if self.proto_similarity not in {"cosine", "inv_euclid"}:
            raise ValueError(f"unknown proto_similarity={self.proto_similarity!r}")


        # Прототипы: [total_prototypes, hidden_dim]
        self.prototypes = nn.Parameter(
            torch.randn(self.total_prototypes, self.hidden_dim)
        )
        # Инициализация как у весов
        nn.init.normal_(self.prototypes, mean=0.0, std=0.02)

        self.class_mix_weights = nn.Parameter(torch.ones(num_classes) * 0.5)  # начальное значение 0.5

        self.proto_temperature = float(proto_temperature)

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
        # fixed_seq = sequences

        # Последовательные слои трансформера + gated residual
        for i in range(len(self.transformer)):
            att = self.transformer[i](
                sequences,   # Q
                sequences,   # K
                sequences,   # V
                key_padding_mask=(~mask) if mask is not None else None,
            )  # [B, T, hidden_dim]

            sequences = sequences + att


        # Пулинг по времени
        sequences_pool = self._pool_features(sequences, mask)  # [B, D]

        classifier_logits = self.classifier(sequences_pool)  # [B, C]
        proto_logits = self._compute_proto_logits(sequences_pool)  # [B, C]

        #softmax на логитсы
        # softmax по классам (dim=1, потому что размерность [B, C])
        # classifier_probs = F.softmax(classifier_logits, dim=1)  # [B, C]
        # proto_probs      = F.softmax(proto_logits, dim=1)       # [B, C]

        # === Смешивание с обучаемыми весами по классам ===
        # Применяем сигмоиду, чтобы веса были в [0,1]
        mix_weights = torch.sigmoid(self.class_mix_weights)  # [C]

        # Расширяем до [B, C]
        mix_weights = mix_weights.unsqueeze(0).expand_as(classifier_logits)

        # final_logits = mix_weights * classifier_logits + (1 - mix_weights) * proto_logits
        final_logits = mix_weights * classifier_logits + (1 - mix_weights) * proto_logits

        # Возвращаем всё, что нужно для лосса
        # return final_logits, classifier_logits, proto_logits, sequences_pool
        return final_logits, classifier_logits, proto_logits, sequences_pool

    def _compute_proto_logits(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, D] pooled features
        Returns class logits based on proto_similarity ("cosine" or "inv_euclid").
        """
        B = x.size(0)
        C = self.num_classes
        N = self.num_prototypes_per_class

        if self.proto_similarity == "cosine":
            x_norm = torch.nn.functional.normalize(x, dim=1)  # [B, D]
            p_norm = torch.nn.functional.normalize(self.prototypes, dim=1)  # [P, D]
            sim = torch.matmul(x_norm, p_norm.t())  # [B, P]
        else:
            dist = torch.cdist(x, self.prototypes, p=2)  # [B, P]
            sim = 1.0 / (1.0 + dist)

        sim = sim.view(B, C, N)  # [B, C, N]
        proto_logits_per_class = sim.max(dim=2).values  # [B, C]
        # proto_logits_per_class = sim.mean(dim=2)  # [B, C]
        # k = 2
        # topk_vals = sim.topk(k, dim=2).values      # [B, C, k]
        # proto_logits_per_class = topk_vals.mean(dim=2)  # [B, C]

        proto_logits_per_class = proto_logits_per_class / self.proto_temperature
        return proto_logits_per_class


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



class VideoFormer_with_Archetypes(nn.Module):
    """
    AURA/VQ-style archetypes:
      - hard assignment to nearest archetype by cosine similarity
      - straight-through quantization for encoder grads
      - VQ-VAE loss: codebook + beta * commitment (with stop-grad)
    Returns: logits, vq_loss, idx
    """
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
        num_classes: int = 3,
        num_archetypes: int = 12,
        commit_beta: float = 0.25,
    ):
        super().__init__()

        self.seg_len = seg_len
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_archetypes = num_archetypes
        self.commit_beta = float(commit_beta)

        # input projection
        self.image_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )

        # transformer stack
        self.transformer = nn.ModuleList([
            TransformerEncoderLayer(
                input_dim=hidden_dim,
                num_heads=num_transformer_heads,
                dropout=0.1,
                positional_encoding=positional_encoding,
            )
            for _ in range(tr_layer_number)
        ])

        # archetype codebook (learnable)
        self.archetypes = nn.Parameter(torch.randn(num_archetypes, hidden_dim))

        # classifier head (same style as your prototypes model)
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
        # [B,T,Din] -> [B,T,H]
        sequences = self.image_proj(sequences)

        # transformer
        for layer in self.transformer:
            att = layer(
                sequences, sequences, sequences,
                key_padding_mask=(~mask) if mask is not None else None,
            )
            sequences = sequences + att

        # pool -> [B,H]
        f = self._pool_features(sequences, mask)

        # normalize for cosine geometry
        f_n = F.normalize(f, dim=1)                       # [B,H]
        A_n = F.normalize(self.archetypes, dim=1)         # [K,H]

        # hard assignment by cosine
        sim = f_n @ A_n.t()                               # [B,K]
        idx = sim.argmax(dim=1)                           # [B]
        e_n = A_n[idx]                                    # [B,H]

        # straight-through quantization in normalized space
        z_q = f_n + (e_n - f_n).detach()

        logits = self.classifier(z_q)

        # VQ loss in the same (normalized) space
        codebook_loss = F.mse_loss(e_n, f_n.detach())
        commit_loss   = F.mse_loss(f_n, e_n.detach())
        vq_loss = codebook_loss + self.commit_beta * commit_loss

        return logits, vq_loss, idx


    def _calculate_classifier_input_dim(self):
        dummy_video = torch.randn(1, self.seg_len, self.hidden_dim)
        video_pool = self._pool_features(dummy_video, mask=None)
        self.classifier_input_dim = video_pool.size(1)

    def _pool_features(self, sequences: torch.Tensor, mask: torch.Tensor | None = None):
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
