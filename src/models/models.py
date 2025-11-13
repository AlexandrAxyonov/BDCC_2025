# coding: utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from .help_layers import TransformerEncoderLayer, MambaBlock
from .attention.crossmpt.Model_CrossMPT import (
    MultiHeadedAttention,
    PositionwiseFeedForward,
    Encoder,
    EncoderLayer,
)

class VideoMamba(nn.Module):
    def __init__(
        self,
        input_dim=512,
        hidden_dim=128,
        mamba_d_state=8,
        mamba_ker_size=3,
        mamba_layer_number=2,
        d_discr=None,
        dropout=0.1,
        seg_len=20,
        out_features=128,
        num_classes=7,
        device='cpu'
    ):
        super(VideoMamba, self).__init__()

        mamba_par = {
            'd_input': hidden_dim,
            'd_model': hidden_dim,
            'd_state': mamba_d_state,
            'd_discr': d_discr,
            'ker_size': mamba_ker_size,
            'dropout': dropout,
            'device': device
        }

        self.seg_len = seg_len
        self.hidden_dim = hidden_dim

        self.image_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout)
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
            nn.Linear(out_features, num_classes)
        )

        self._init_weights()

    def forward(self, sequences, mask=None, features=False):
        """
        sequences: [B, T, D_in]
        mask:      [B, T] (bool) — True для реальных кадров, False — паддинг
        """
        sequences = self.image_proj(sequences)  # [B, T, hidden_dim]

        # основной проход через Mamba-блоки (без явной маски, если её не поддерживает блок)
        for i in range(len(self.mamba)):
            att_sequences, _ = self.mamba[i](sequences)
            sequences = sequences + att_sequences

        # усредняем ТОЛЬКО по реальным кадрам
        sequences_pool = self._pool_features(sequences, mask)  # [B, hidden_dim]
        out = self.classifier(sequences_pool)

        if features:
            return {'prob': out, 'features': sequences_pool}
        else:
            return out

    def _calculate_classifier_input_dim(self):
        """Calculates input feature size for classifier"""
        dummy_video = torch.randn(1, self.seg_len, self.hidden_dim)
        video_pool = self._pool_features(dummy_video, mask=None)
        self.classifier_input_dim = video_pool.size(1)

    def _pool_features(self, sequences, mask=None):
        """
        sequences: [B, T, H]
        mask: [B, T] (bool)
        """
        if mask is None:
            mean_temp = sequences.mean(dim=1)  # [B, H]
            return mean_temp

        # masked mean: sum(valid) / count(valid)
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


class VideoFormer(nn.Module):
    def __init__(
        self,
        input_dim=512,
        hidden_dim=128,
        num_transformer_heads=2,
        positional_encoding=True,
        dropout=0.1,
        tr_layer_number=2,
        seg_len=20,
        out_features=128,
        num_classes=7
    ):
        super(VideoFormer, self).__init__()

        self.seg_len = seg_len
        self.hidden_dim = hidden_dim

        self.image_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout)
        )

        self.transformer = nn.ModuleList([
            TransformerEncoderLayer(
                input_dim=hidden_dim,
                num_heads=num_transformer_heads,
                dropout=dropout,
                positional_encoding=positional_encoding
            ) for _ in range(tr_layer_number)
        ])

        self._calculate_classifier_input_dim()

        self.classifier = nn.Sequential(
            nn.Linear(self.classifier_input_dim, out_features),
            nn.LayerNorm(out_features),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_features, num_classes)
        )

        self._init_weights()

    def forward(self, sequences, mask=None):
        """
        sequences: [B, T, D_in]
        mask:      [B, T] (bool) — True для реальных кадров, False — паддинг
        """
        sequences = self.image_proj(sequences)  # [B, T, hidden_dim]

        fixed_seq = sequences

        for i in range(len(self.transformer)):
            att = self.transformer[i](
                sequences,   # Q
                fixed_seq,         # K
                fixed_seq,         # V
                key_padding_mask=(~mask) if mask is not None else None
            )
            sequences = sequences + att  # residual

        sequences_pool = self._pool_features(sequences, mask)
        output = self.classifier(sequences_pool)
        return output

    def _calculate_classifier_input_dim(self):
        """Calculates input feature size for classifier"""
        dummy_video = torch.randn(1, self.seg_len, self.hidden_dim)
        video_pool = self._pool_features(dummy_video, mask=None)
        self.classifier_input_dim = video_pool.size(1)

    def _pool_features(self, sequences, mask=None):
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
