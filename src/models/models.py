# coding: utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F

from .help_layers import MambaBlock, TransformerEncoderLayer


class VideoMamba(nn.Module):
    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 128,
        mamba_d_state: int = 8,
        mamba_ker_size: int = 3,
        mamba_layer_number: int = 2,
        d_discr: int | None = None,
        dropout: float = 0.1,
        seg_len: int = 20,
        out_features: int = 128,
        num_classes: int = 7,
        device: str = "cpu",
    ):
        super(VideoMamba, self).__init__()

        mamba_par = {
            "d_input": hidden_dim,
            "d_model": hidden_dim,
            "d_state": mamba_d_state,
            "d_discr": d_discr,
            "ker_size": mamba_ker_size,
            "dropout": dropout,
            "device": device,
        }

        self.seg_len = seg_len
        self.hidden_dim = hidden_dim

        self.image_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )

        self.mamba = nn.ModuleList([
            MambaBlock(**mamba_par) for _ in range(mamba_layer_number)
        ])

        self._calculate_classifier_input_dim()

        self.classifier = nn.Sequential(
            nn.Linear(self.classifier_input_dim, out_features),
            nn.LayerNorm(out_features),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_features, num_classes),
        )

        self._init_weights()

    def forward(
        self,
        sequences: torch.Tensor,
        mask: torch.Tensor | None = None,
        return_embeddings: bool = False,
    ):
        sequences = self.image_proj(sequences)  # [B, T, hidden_dim]

        for i in range(len(self.mamba)):
            att_sequences, _ = self.mamba[i](sequences)
            sequences = sequences + att_sequences

        sequences_pool = self._pool_features(sequences, mask)  # [B, hidden_dim]
        logits = self.classifier(sequences_pool)

        if return_embeddings:
            return logits, sequences_pool
        return logits

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
        self.num_layers = tr_layer_number


        self.image_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )


        self.transformer = nn.ModuleList([
            TransformerEncoderLayer(
                input_dim=hidden_dim,
                num_heads=num_transformer_heads,
                dropout=dropout,
                positional_encoding=positional_encoding,
            )
            for _ in range(tr_layer_number)
        ])



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


            for plist in (self.bt_gates, self.bd_gates, self.t_gates, self.d_gates):
                for p in plist:
                    if p.dim() >= 2:
                        nn.init.xavier_uniform_(p)
                    else:
                        nn.init.zeros_(p)


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


        sequences = self.image_proj(sequences)  # [B, T, hidden_dim]


        # fixed_seq = sequences


        for i in range(len(self.transformer)):
            att = self.transformer[i](
                sequences,   # Q
                sequences,   # K
                sequences,   # V
                key_padding_mask=(~mask) if mask is not None else None,
            )  # [B, T, hidden_dim]

            if self.gate_mode is None:

                sequences = sequences + att
            else:
                alpha = self._compute_alpha(i, sequences)  # [B, T, hidden_dim]
                sequences = (1.0 - alpha) * sequences + alpha * att

        #------------------- FGC --------------------------
        # fixed_seq = sequences
        # for i in range(len(self.transformer)):
        #     alpha = 0.75
        #     mixed = (1 - alpha) * sequences + alpha * fixed_seq

        #     att = self.transformer[i](
        #         sequences,   # Q
        #         mixed,       # K
        #         mixed,       # V
        #         key_padding_mask=(~mask) if mask is not None else None
        #     )
        #     sequences = sequences + att

        #--------------------------------------------------


        sequences_pool = self._pool_features(sequences, mask)  # [B, hidden_dim]


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

            W_t = self.t_gates[layer_idx]                       # [T, 1]
            alpha_t = torch.softmax(W_t.squeeze(-1), dim=0)     # [T]
            alpha = alpha_t.view(1, T, 1)                       # [1, T, 1]

        elif self.gate_mode == "d":

            W_d = self.d_gates[layer_idx]                       # [D]
            alpha_d = torch.softmax(W_d, dim=0)                 # [D]
            alpha = alpha_d.view(1, 1, D)                       # [1, 1, D]

        else:
            raise ValueError(f"Unknown gate_mode: {self.gate_mode}")


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
        proto_proj_enabled: bool = False,
        proto_proj_dim: int | None = None,
    ):
        super(VideoFormer_with_Prototypes, self).__init__()


        self.seg_len = seg_len
        self.hidden_dim = hidden_dim


        self.image_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )


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



        self.prototypes = nn.Parameter(
            torch.randn(self.total_prototypes, self.hidden_dim)
        )

        nn.init.normal_(self.prototypes, mean=0.0, std=0.02)

        self.class_mix_weights = nn.Parameter(torch.ones(num_classes) * 0.5)

        self.proto_temperature = float(proto_temperature)
        self.proto_proj_enabled = bool(proto_proj_enabled)
        proj_dim = self.hidden_dim if proto_proj_dim is None else int(proto_proj_dim)
        if proj_dim <= 0:
            proj_dim = self.hidden_dim
        self.proto_proj_dim = proj_dim
        self.proto_proj = (
            nn.Linear(self.hidden_dim, self.proto_proj_dim, bias=False)
            if self.proto_proj_enabled
            else nn.Identity()
        )


        self._calculate_classifier_input_dim()
        self.classifier = nn.Sequential(
            nn.Linear(self.classifier_input_dim, out_features),
            nn.LayerNorm(out_features),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_features, num_classes),
        )

        self._init_weights()
        if self.proto_proj_enabled:
            if self.proto_proj_dim == self.hidden_dim:
                nn.init.eye_(self.proto_proj.weight)
            else:
                nn.init.xavier_uniform_(self.proto_proj.weight)

    def forward(self, sequences: torch.Tensor, mask: torch.Tensor | None = None):


        sequences = self.image_proj(sequences)  # [B, T, hidden_dim]


        # fixed_seq = sequences


        for i in range(len(self.transformer)):
            att = self.transformer[i](
                sequences,   # Q
                sequences,   # K
                sequences,   # V
                key_padding_mask=(~mask) if mask is not None else None,
            )  # [B, T, hidden_dim]

            sequences = sequences + att



        sequences_pool = self._pool_features(sequences, mask)  # [B, D]

        classifier_logits = self.classifier(sequences_pool)  # [B, C]
        proto_logits = self._compute_proto_logits(sequences_pool)  # [B, C]



        # classifier_probs = F.softmax(classifier_logits, dim=1)  # [B, C]
        # proto_probs      = F.softmax(proto_logits, dim=1)       # [B, C]



        mix_weights = torch.sigmoid(self.class_mix_weights)  # [C]


        mix_weights = mix_weights.unsqueeze(0).expand_as(classifier_logits)

        # final_logits = mix_weights * classifier_logits + (1 - mix_weights) * proto_logits
        final_logits = mix_weights * classifier_logits + (1 - mix_weights) * proto_logits


        # return final_logits, classifier_logits, proto_logits, sequences_pool
        return final_logits, classifier_logits, proto_logits, sequences_pool

    def _proto_project(self, x: torch.Tensor) -> torch.Tensor:
        return self.proto_proj(x)

    def _compute_proto_logits(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, D] pooled features
        Returns class logits based on proto_similarity ("cosine" or "inv_euclid").
        """
        B = x.size(0)
        C = self.num_classes
        N = self.num_prototypes_per_class

        x = self._proto_project(x)
        protos = self._proto_project(self.prototypes)

        if self.proto_similarity == "cosine":
            x_norm = torch.nn.functional.normalize(x, dim=1)  # [B, D']
            p_norm = torch.nn.functional.normalize(protos, dim=1)  # [P, D']
            sim = torch.matmul(x_norm, p_norm.t())  # [B, P]
        else:
            x_norm = torch.nn.functional.normalize(x, dim=1)  # [B, D']
            p_norm = torch.nn.functional.normalize(protos, dim=1)  # [P, D']
            dist = torch.cdist(x_norm, p_norm, p=2)  # [B, P]
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
