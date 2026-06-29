import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import random


# ── Shared building blocks ────────────────────────────────────────────────────

class ResidualBlock(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(feat_dim, feat_dim, 3, 1, 1),
            nn.LeakyReLU(0.1),
            nn.Conv1d(feat_dim, feat_dim, 3, 1, 1),
        )

    def forward(self, inp):
        return self.block(inp)


class SparseSampleEmbedder(nn.Module):
    """Hierarchical video context via random sparse sampling + CNN pyramid.

    Randomly samples num_frames from the segment sequence, processes through a
    log2(num_frames)-deep residual pyramid with halving max-pool at each stage,
    then averages across samples to produce a single global video feature vector.
    """

    def __init__(self, in_dim, feat_dim, num_frames, num_samples):
        super().__init__()
        self.num_frames = num_frames
        self.num_samples = num_samples
        self.init_conv = nn.Conv1d(in_dim, feat_dim, 3, 1, 1)
        self.blocks = nn.ModuleList(
            [ResidualBlock(feat_dim) for _ in range(int(np.log2(self.num_frames)))]
        )
        self.maxpool = nn.MaxPool1d(2, 2)
        self.relu = nn.LeakyReLU(0.1)

    def forward(self, inp):
        B, L, D = inp.shape
        sampled_inps = []
        for _ in range(self.num_samples):
            if L >= self.num_frames:
                idx = sorted(random.sample(range(L), self.num_frames))
            else:
                idx = sorted([random.randint(0, L - 1) for _ in range(self.num_frames)])
            sampled_inps.append(inp[:, idx, :])

        sampled = torch.cat(sampled_inps, dim=0)          # (num_samples*B, num_frames, D)
        out = self.init_conv(sampled.permute(0, 2, 1))    # (num_samples*B, feat_dim, num_frames)

        for block in self.blocks:
            out = self.relu(out + block(out))
            out = self.maxpool(out)

        out = out.squeeze(-1)                              # (num_samples*B, feat_dim)
        out = out.reshape(self.num_samples, B, -1)         # (num_samples, B, feat_dim)
        return torch.mean(out, dim=0)                      # (B, feat_dim)


# ── Base class: shared segment extraction ────────────────────────────────────

class BaseHASR(nn.Module):
    """Common segment-level feature extraction shared by GRU and Transformer refiners.

    Implements:
      - Query-attention pooling to compute per-segment features
      - SparseSampleEmbedder for global video context
      - Rollout to expand segment predictions back to frame level
    """

    def __init__(self, num_actions, input_dim, feat_dim,
                 num_highlevel_frames, num_highlevel_samples, device):
        super().__init__()
        self.key_embedding   = nn.Linear(input_dim, feat_dim, bias=False)
        self.value_embedding = nn.Linear(input_dim, feat_dim, bias=False)
        self.query_embedding = nn.Embedding(num_actions, feat_dim)
        self.label_embedding = nn.Embedding(num_actions, feat_dim)
        self.video_embedding = SparseSampleEmbedder(
            feat_dim * 2, feat_dim,
            num_frames=num_highlevel_frames,
            num_samples=num_highlevel_samples,
        )
        self.feat_dim    = feat_dim
        self.num_actions = num_actions
        self.device      = device

    def get_segment_info(self, action_idx, batch_input, batch_target=None):
        """Group consecutive same-label frames into segments; compute attention features.

        Args:
            action_idx:   (T,) backbone predicted class indices
            batch_input:  (B, D, T) frame features
            batch_target: (B, T) GT labels (training only)

        Returns:
            segment_idx:    list of boundary frame indices (length S+1)
            segment_feat:   (B, S, feat_dim) attention-pooled segment features
            PREDlabel_list: (1, S) predicted label per segment
            GTlabel_list:   (1, S) majority-vote GT label per segment
        """
        segment_idx = [0]
        prev = action_idx[0]
        for ii, idx in enumerate(action_idx):
            if idx != prev:
                segment_idx.append(ii)
            prev = idx
        segment_idx.append(len(action_idx))

        GTlabel_list, PREDlabel_list, segment_feat = [], [], []

        for s_i in range(len(segment_idx) - 1):
            p, c = segment_idx[s_i], segment_idx[s_i + 1]
            curr_seg = batch_input[:, :, p:c]

            if self.training and batch_target is not None:
                gt_seg = batch_target[:, p:c]
                GTlabel_list.append(torch.argmax(torch.bincount(gt_seg[0])))

            pred_label = torch.mean(action_idx[p:c].float()).long()
            PREDlabel_list.append(pred_label)

            q     = self.query_embedding(pred_label).view(1, -1, 1)
            k     = self.key_embedding(curr_seg.permute(0, 2, 1))
            v     = self.value_embedding(curr_seg.permute(0, 2, 1))
            score = F.softmax(torch.bmm(k, q) / np.sqrt(self.feat_dim), dim=1)
            feat  = torch.bmm(v.permute(0, 2, 1), score)
            segment_feat.append(feat.permute(0, 2, 1))    # (B, 1, feat_dim)

        GTlabel_list   = torch.LongTensor(GTlabel_list).view(1, -1).to(self.device)
        PREDlabel_list = torch.LongTensor(PREDlabel_list).view(1, -1).to(self.device)
        segment_feat   = torch.cat(segment_feat, dim=1)   # (B, S, feat_dim)

        return segment_idx, segment_feat, PREDlabel_list, GTlabel_list

    def rollout(self, segment_idx, refine_pred):
        """Expand (1, S, num_actions) segment logits to (1, num_actions, T) frame logits."""
        parts = []
        for s_i in range(len(segment_idx) - 1):
            p, c = segment_idx[s_i], segment_idx[s_i + 1]
            parts.append(refine_pred[0, s_i, :].view(1, -1).repeat(c - p, 1))
        return torch.cat(parts, dim=0).unsqueeze(0).transpose(2, 1)


# ── GRU refiner (original HASR architecture) ─────────────────────────────────

class GRURefiner(BaseHASR):
    """Original HASR bidirectional GRU refiner (ICCV 2021).

    Segment features, label embeddings, and global video context are
    concatenated and passed through a bidirectional GRU to produce
    refined per-segment predictions.
    """

    def __init__(
        self,
        num_actions,
        input_dim,
        feat_dim=512,
        num_gru_layers=1,
        dropout=0.0,
        num_highlevel_frames=32,
        num_highlevel_samples=64,
        device='cpu',
    ):
        super().__init__(num_actions, input_dim, feat_dim,
                         num_highlevel_frames, num_highlevel_samples, device)

        # Bidirectional GRU: input = [seg_feat | label_embed | global_feat]
        self.refiner     = nn.GRU(
            feat_dim * 3, feat_dim,
            num_layers=num_gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_gru_layers > 1 else 0.0,
        )
        self.output_proj = nn.Linear(feat_dim * 2, num_actions)  # *2 for bidirectional

    def forward(self, action_idx, batch_input, batch_target=None):
        segment_idx, segment_feat, PREDlabel_list, GTlabel_list = self.get_segment_info(
            action_idx, batch_input, batch_target
        )
        S = segment_feat.shape[1]

        label_embed  = self.label_embedding(PREDlabel_list)          # (B, S, feat_dim)
        global_feat  = self.video_embedding(
            torch.cat([segment_feat, label_embed], dim=-1)
        )                                                             # (B, feat_dim)

        gru_input    = torch.cat([
            segment_feat,
            label_embed,
            global_feat.unsqueeze(1).expand(-1, S, -1),
        ], dim=-1)                                                    # (B, S, feat_dim*3)

        gru_out, _   = self.refiner(gru_input)                       # (B, S, feat_dim*2)
        refine_pred  = self.output_proj(gru_out)                     # (B, S, num_actions)

        refine_rollout = self.rollout(segment_idx, refine_pred)      # (B, num_actions, T)
        return refine_pred, refine_rollout, GTlabel_list


# ── Transformer refiner ───────────────────────────────────────────────────────

class TransformerHASR(BaseHASR):
    """HASR with bidirectional GRU replaced by a Transformer encoder.

    Segment features, label embeddings, and global video context are projected
    to d_model, augmented with learned positional encodings, and processed by
    a multi-head self-attention Transformer encoder.
    """

    def __init__(
        self,
        num_actions,
        input_dim,
        feat_dim=512,
        d_model=512,
        nhead=8,
        num_transformer_layers=4,
        dim_feedforward=2048,
        dropout=0.1,
        num_highlevel_frames=32,
        num_highlevel_samples=64,
        max_segments=512,
        device='cpu',
    ):
        super().__init__(num_actions, input_dim, feat_dim,
                         num_highlevel_frames, num_highlevel_samples, device)

        self.input_proj    = nn.Linear(feat_dim * 3, d_model)
        self.pos_embedding = nn.Embedding(max_segments, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,   # pre-LN for training stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)
        self.output_proj  = nn.Linear(d_model, num_actions)

    def forward(self, action_idx, batch_input, batch_target=None):
        segment_idx, segment_feat, PREDlabel_list, GTlabel_list = self.get_segment_info(
            action_idx, batch_input, batch_target
        )
        S = segment_feat.shape[1]

        label_embed  = self.label_embedding(PREDlabel_list)          # (B, S, feat_dim)
        global_feat  = self.video_embedding(
            torch.cat([segment_feat, label_embed], dim=-1)
        )                                                             # (B, feat_dim)

        tokens = self.input_proj(torch.cat([
            segment_feat,
            label_embed,
            global_feat.unsqueeze(1).expand(-1, S, -1),
        ], dim=-1))                                                   # (B, S, d_model)

        tokens       = tokens + self.pos_embedding(
            torch.arange(S, device=self.device)
        )                                                             # (B, S, d_model)

        refined      = self.transformer(tokens)                      # (B, S, d_model)
        refine_pred  = self.output_proj(refined)                     # (B, S, num_actions)

        refine_rollout = self.rollout(segment_idx, refine_pred)      # (B, num_actions, T)
        return refine_pred, refine_rollout, GTlabel_list
