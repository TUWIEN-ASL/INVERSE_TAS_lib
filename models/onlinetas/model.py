import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import numpy as np
import math


# ---------------------------------------------------------------------------
# Causal TCN
# ---------------------------------------------------------------------------

class CausalDilatedResidualLayer(nn.Module):
    def __init__(self, dilation, in_channels, out_channels):
        super().__init__()
        self.dilation = dilation
        self.conv_dilated = nn.Conv1d(in_channels, out_channels, 3, padding=0, dilation=dilation)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()

    def forward(self, x, mask):
        # left-only padding keeps causality: only past/present frames in receptive field
        pad = 2 * self.dilation
        x_padded = F.pad(x, (pad, 0))
        out = F.relu(self.conv_dilated(x_padded))
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return (x + out) * mask[:, 0:1, :]


class CausalSingleStageModel(nn.Module):
    def __init__(self, num_layers, num_f_maps, input_dim, num_classes):
        super().__init__()
        self.conv_1x1 = nn.Conv1d(input_dim, num_f_maps, 1)
        self.layers = nn.ModuleList([
            CausalDilatedResidualLayer(2 ** i, num_f_maps, num_f_maps)
            for i in range(num_layers)
        ])
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x, mask):
        out = self.conv_1x1(x)
        for layer in self.layers:
            out = layer(out, mask)
        out = self.conv_out(out) * mask[:, 0:1, :]
        return out


# ---------------------------------------------------------------------------
# Attention helpers
# ---------------------------------------------------------------------------

class WindowedSelfAttention(nn.Module):
    """Swin-style: split into 2 windows, SA within each, relative position bias."""

    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert dim % num_heads == 0
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

        # relative position bias — sized for any window up to 256 frames
        # we store a table and compute indices dynamically based on actual window size
        self._max_window = 256
        self.rel_pos_bias = nn.Embedding(2 * self._max_window - 1, num_heads)
        nn.init.trunc_normal_(self.rel_pos_bias.weight, std=0.02)

    def _attn_window(self, x):
        """x: (B, T, D) — self-attention within the T-length window."""
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)           # each (B, T, H, head_dim)
        q = q.transpose(1, 2)             # (B, H, T, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, H, T, T)

        # relative position bias
        coords = torch.arange(T, device=x.device)
        rel = coords[:, None] - coords[None, :] + (self._max_window - 1)  # (T, T)
        rel = rel.clamp(0, 2 * self._max_window - 2)
        bias = self.rel_pos_bias(rel)      # (T, T, H)
        attn = attn + bias.permute(2, 0, 1).unsqueeze(0)  # (1, H, T, T)

        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)        # (B, H, T, head_dim)
        out = out.transpose(1, 2).reshape(B, T, D)
        return self.proj(out)

    def forward(self, x):
        """x: (B, D, T) — returns (B, D, T)."""
        B, D, T = x.shape
        x_t = x.permute(0, 2, 1)          # (B, T, D)
        x_t = self.norm(x_t)

        half = T // 2
        w1 = self._attn_window(x_t[:, :half, :])
        w2 = self._attn_window(x_t[:, half:, :])
        out = torch.cat([w1, w2], dim=1)   # (B, T, D)
        return (x_t + out).permute(0, 2, 1)  # residual, back to (B, D, T)


class TransformerDecoderLayer(nn.Module):
    """SA on memory + CA with clip GRU features + FFN."""

    def __init__(self, dim, num_heads):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.ReLU(),
            nn.Linear(4 * dim, dim),
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

    def forward(self, memory, clip_gru):
        """
        memory:   (B, D, Tm) — current memory state
        clip_gru: (B, D, w)  — GRU-processed clip features
        Returns:  (B, D, Tm) — updated memory encoding
        """
        B, D, Tm = memory.shape
        m = memory.permute(0, 2, 1)       # (B, Tm, D)
        g = clip_gru.permute(0, 2, 1)     # (B, w, D)

        # self-attention on memory
        m2, _ = self.self_attn(m, m, m)
        m = self.norm1(m + m2)

        # cross-attention: Q=memory, K/V=clip_gru
        m2, _ = self.cross_attn(m, g, g)
        m = self.norm2(m + m2)

        # FFN
        m = self.norm3(m + self.ffn(m))
        return m.permute(0, 2, 1)          # (B, D, Tm)


# ---------------------------------------------------------------------------
# CFA Module
# ---------------------------------------------------------------------------

class CFAModule(nn.Module):
    """Context-Aware Feature Augmentation (Sec. 3.2 of OnlineTAS paper)."""

    def __init__(self, input_dim, w, num_td_layers=2, td_heads=8,
                 sa_heads=4, num_iterations=2):
        super().__init__()
        self.w = w
        self.I = num_iterations

        self.gru = nn.GRU(input_dim, input_dim, num_layers=1, batch_first=True)

        self.sa_layers = nn.ModuleList([
            WindowedSelfAttention(input_dim, sa_heads) for _ in range(num_iterations)
        ])
        self.td_layers = nn.ModuleList([
            nn.ModuleList([
                TransformerDecoderLayer(input_dim, td_heads)
                for _ in range(num_td_layers)
            ])
            for _ in range(num_iterations)
        ])
        self.ca_layers = nn.ModuleList([
            nn.MultiheadAttention(input_dim, sa_heads, batch_first=True)
            for _ in range(num_iterations)
        ])
        self.ca_norms = nn.ModuleList([
            nn.LayerNorm(input_dim) for _ in range(num_iterations)
        ])

        # collapses w temporal frames to a single memory token
        self.memory_compressor = nn.Conv1d(input_dim, input_dim, kernel_size=w, stride=w)

    def forward(self, clip, M_prev):
        """
        clip:   (B, D, w)
        M_prev: (B, D, w) — current memory (long + short concatenated)
        Returns c_tilde: (B, D, w)
        """
        B, D, w = clip.shape

        # pass clip through GRU (reset hidden state per clip)
        c_gru, _ = self.gru(clip.permute(0, 2, 1))   # (B, w, D)
        c_gru = c_gru.permute(0, 2, 1)                # (B, D, w)

        c_aug = c_gru
        for i in range(self.I):
            # --- self-attention on clip ---
            c_sa = self.sa_layers[i](c_aug)            # (B, D, w)

            # --- transformer decoder on memory ---
            M_td = M_prev
            for td in self.td_layers[i]:
                M_td = td(M_td, c_gru)                 # (B, D, Tm)

            # --- cross-attention: Q=c_sa, K/V=M_td ---
            q = c_sa.permute(0, 2, 1)                  # (B, w, D)
            kv = M_td.permute(0, 2, 1)                 # (B, Tm, D)
            ca_out, _ = self.ca_layers[i](q, kv, kv)  # (B, w, D)
            ca_out = ca_out.permute(0, 2, 1)           # (B, D, w)

            c_aug = self.ca_norms[i](
                (ca_out + c_gru).permute(0, 2, 1)
            ).permute(0, 2, 1)                         # (B, D, w)

        return c_aug

    def compress(self, c_tilde):
        """Compress a clip into a single memory token: (B, D, w) -> (B, D, 1)."""
        return self.memory_compressor(c_tilde)


# ---------------------------------------------------------------------------
# Full OnlineTAS Model
# ---------------------------------------------------------------------------

class OnlineTASModel(nn.Module):
    def __init__(self, input_dim, num_classes, num_layers=10, num_f_maps=64,
                 w=128, num_iterations=2, num_td_layers=2, td_heads=8, sa_heads=4):
        super().__init__()
        self.w = w
        self.input_dim = input_dim

        self.cfa = CFAModule(
            input_dim=input_dim, w=w,
            num_td_layers=num_td_layers, td_heads=td_heads,
            sa_heads=sa_heads, num_iterations=num_iterations,
        )
        self.tcn = CausalSingleStageModel(num_layers, num_f_maps, input_dim, num_classes)

    def forward_clip(self, clip, M_prev):
        """
        clip:   (B, D, w) — raw input features for this clip
        M_prev: (B, D, w) — memory from previous clips
        Returns:
            logits:  (B, num_classes, w)
            c_tilde: (B, D, w) — enhanced clip features (for memory update)
        """
        c_tilde = self.cfa(clip, M_prev)
        mask = torch.ones(clip.shape[0], 1, clip.shape[2], device=clip.device)
        logits = self.tcn(c_tilde, mask)
        return logits, c_tilde


# ---------------------------------------------------------------------------
# Memory bank (stateful, not an nn.Module)
# ---------------------------------------------------------------------------

class MemoryBank:
    """Manages M_long and M_short as described in Algorithm 1."""

    def __init__(self, w, input_dim, device):
        self.w = w
        self.input_dim = input_dim
        self.device = device
        self.M_long = None      # (B, D, n_tokens)
        self.prev_c_tilde = None  # (B, D, w)

    def initialize(self, first_clip):
        """Initialize with first clip features (before any CFA)."""
        B, D, w = first_clip.shape
        self.M_long = torch.zeros(B, D, 0, device=self.device)
        self.prev_c_tilde = first_clip.detach()

    def get(self):
        """Return M = [M_long, M_short] with total temporal size w."""
        n = self.M_long.shape[-1]
        M_short = self.prev_c_tilde[:, :, n:]   # (B, D, w-n)
        if n == 0:
            return M_short
        return torch.cat([self.M_long, M_short], dim=-1)  # (B, D, w)

    def update(self, c_tilde, m_k):
        """
        c_tilde: (B, D, w) — enhanced features of current clip
        m_k:     (B, D, 1) — compressed memory token for current clip
        """
        max_long = int(math.floor(2.0 / 3.0 * self.w))
        if self.M_long.shape[-1] < max_long:
            self.M_long = torch.cat([self.M_long, m_k.detach()], dim=-1)
        else:
            # FIFO: drop oldest, append newest
            self.M_long = torch.cat([self.M_long[:, :, 1:], m_k.detach()], dim=-1)
        self.prev_c_tilde = c_tilde.detach()


# ---------------------------------------------------------------------------
# Post-processing (Algorithm 2)
# ---------------------------------------------------------------------------

def post_process(predictions, confidences, theta, l_min):
    """
    predictions: np.array (T,) int — raw frame-level class predictions
    confidences: np.array (T,) float — max softmax probability per frame
    theta:       float — confidence threshold
    l_min:       int   — minimum segment length
    Returns refined predictions (T,) int
    """
    T = len(predictions)
    refined = predictions.copy()
    seg_len = 0
    for t in range(T):
        if t == 0:
            refined[t] = predictions[t]
            seg_len = 1
        else:
            if confidences[t] < theta and seg_len < l_min:
                refined[t] = refined[t - 1]
                seg_len += 1
            else:
                refined[t] = predictions[t]
                if predictions[t] == refined[t - 1]:
                    seg_len += 1
                else:
                    seg_len = 1
    return refined


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    def __init__(self, input_dim, num_classes, num_layers=10, num_f_maps=64,
                 w=128, num_iterations=2, num_td_layers=2, td_heads=8, sa_heads=4):
        self.model = OnlineTASModel(
            input_dim=input_dim, num_classes=num_classes,
            num_layers=num_layers, num_f_maps=num_f_maps, w=w,
            num_iterations=num_iterations, num_td_layers=num_td_layers,
            td_heads=td_heads, sa_heads=sa_heads,
        )
        self.w = w
        self.num_classes = num_classes
        self.ce = nn.CrossEntropyLoss(ignore_index=-100)
        self.mse = nn.MSELoss(reduction='none')

    def _split_into_clips(self, features, labels, mask):
        """
        features: (1, D, T)
        labels:   (1, T)   long
        mask:     (1, C, T) float
        Returns lists of (clip, lbl, msk) each of length w (zero-padded last).
        """
        T = features.shape[2]
        clips, lbls, msks = [], [], []
        for start in range(0, T, self.w):
            end = min(start + self.w, T)
            clip = features[:, :, start:end]
            lbl = labels[:, start:end]
            msk = mask[:, :, start:end]
            if end - start < self.w:
                pad = self.w - (end - start)
                clip = F.pad(clip, (0, pad))
                lbl = F.pad(lbl, (0, pad), value=-100)
                msk = F.pad(msk, (0, pad))
            clips.append(clip)
            lbls.append(lbl)
            msks.append(msk)
        return clips, lbls, msks

    def train(self, save_dir, batch_gen, num_epochs, batch_size, learning_rate, device,
              wandb_run=None, eval_data=None):
        self.model.train()
        self.model.to(device)
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            correct = 0
            total = 0
            n_videos = 0

            while batch_gen.has_next():
                batch_input, batch_target, mask = batch_gen.next_batch(batch_size)
                batch_input = batch_input.to(device)
                batch_target = batch_target.to(device)
                mask = mask.to(device)

                optimizer.zero_grad()
                video_loss = 0.0

                clips, lbls, msks = self._split_into_clips(batch_input, batch_target, mask)

                mem = MemoryBank(self.w, batch_input.shape[1], device)
                mem.initialize(clips[0])

                for clip, lbl, msk in zip(clips, lbls, msks):
                    M_prev = mem.get()
                    logits, c_tilde = self.model.forward_clip(clip, M_prev)

                    loss_ce = self.ce(
                        logits.transpose(2, 1).contiguous().view(-1, self.num_classes),
                        lbl.view(-1),
                    )
                    loss_sm = torch.mean(
                        torch.clamp(
                            self.mse(
                                F.log_softmax(logits[:, :, 1:], dim=1),
                                F.log_softmax(logits.detach()[:, :, :-1], dim=1),
                            ),
                            min=0, max=16,
                        ) * msk[:, :, 1:]
                    )
                    video_loss = video_loss + loss_ce + 0.15 * loss_sm

                    # update memory (detached)
                    m_k = self.model.cfa.compress(c_tilde.detach())
                    mem.update(c_tilde, m_k)

                    with torch.no_grad():
                        _, pred = torch.max(logits, 1)
                        valid = (lbl != -100)
                        correct += (pred[valid] == lbl[valid]).sum().item()
                        total += valid.sum().item()

                video_loss = video_loss / len(clips)
                video_loss.backward()
                optimizer.step()

                epoch_loss += video_loss.item()
                n_videos += 1

            batch_gen.reset()
            import os
            torch.save(self.model.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + ".model")
            # torch.save(optimizer.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + ".opt")
            acc = float(correct) / total if total > 0 else 0.0
            loss_avg = epoch_loss / max(n_videos, 1)
            print("[epoch %d]: epoch loss = %f,   acc = %f" % (epoch + 1, loss_avg, acc))
            if wandb_run is not None:
                wandb_run.log({"train/loss": loss_avg, "train/acc": acc}, step=epoch + 1)
            if eval_data is not None:
                metrics = self.evaluate(*eval_data, device)
                print("[epoch %d] eval: %s" % (epoch + 1, metrics))
                if wandb_run is not None:
                    wandb_run.log(metrics, step=epoch + 1)

    def evaluate(self, features_path, vid_list_file, gt_path, actions_dict, sample_rate, device):
        from configs.eval_utils import compute_metrics
        idx_to_action = {v: k for k, v in actions_dict.items()}
        self.model.eval()
        predictions = {}
        with torch.no_grad():
            self.model.to(device)
            with open(vid_list_file, 'r') as f:
                vid_list = f.read().split('\n')[:-1]
            for vid in vid_list:
                features = np.load(features_path + vid.split('.')[0] + '.npy').T
                features = features[:, ::sample_rate]
                T = features.shape[1]
                input_x = torch.tensor(features, dtype=torch.float).unsqueeze(0).to(device)
                all_logits, _ = self._infer_semi_online(input_x)
                predicted = np.argmax(all_logits[:T], axis=1)
                recognition = [label for idx in predicted
                                for label in [idx_to_action[int(idx)]] * sample_rate]
                predictions[vid.split('.')[0]] = recognition
        self.model.train()
        return compute_metrics(predictions, gt_path, vid_list, actions_dict)

    def predict(self, model_dir, results_dir, features_path, vid_list_file,
                epoch, actions_dict, device, sample_rate,
                inference_mode='semi_online', theta=0.9, sigma=1.0 / 16, T_max=None):
        self.model.eval()
        self.model.to(device)
        self.model.load_state_dict(torch.load(model_dir + "/epoch-" + str(epoch) + ".model",
                                              map_location=device))

        l_min = int(sigma * T_max) if T_max is not None else 0

        with open(vid_list_file, 'r') as f:
            list_of_vids = f.read().split('\n')[:-1]

        idx_to_action = {v: k for k, v in actions_dict.items()}

        with torch.no_grad():
            for vid in list_of_vids:
                features = np.load(features_path + vid.split('.')[0] + '.npy').T
                features = features[:, ::sample_rate]
                T = features.shape[1]

                input_x = torch.tensor(features, dtype=torch.float).unsqueeze(0).to(device)

                if inference_mode == 'semi_online':
                    all_logits, all_confs = self._infer_semi_online(input_x)
                else:
                    all_logits, all_confs = self._infer_online(input_x)

                all_logits = all_logits[:T]
                all_confs = all_confs[:T]
                predictions = np.argmax(all_logits, axis=1)

                if l_min > 0:
                    predictions = post_process(predictions, all_confs, theta, l_min)

                recognition = []
                for idx in predictions:
                    recognition.extend([idx_to_action[int(idx)]] * sample_rate)

                import os
                f_name = vid.split('/')[-1].split('.')[0]
                os.makedirs(results_dir, exist_ok=True)
                with open(results_dir + "/" + f_name, "w") as fp:
                    fp.write("### Frame level recognition: ###\n")
                    fp.write(' '.join(recognition))

    def _infer_semi_online(self, input_x):
        """Non-overlapping clips, all predictions kept. Returns (T_padded, C) logits and (T_padded,) confs."""
        D = input_x.shape[1]
        T = input_x.shape[2]
        device = input_x.device

        all_logits = []
        mem = MemoryBank(self.w, D, device)

        for start in range(0, T, self.w):
            end = min(start + self.w, T)
            clip = input_x[:, :, start:end]
            actual_len = clip.shape[2]
            if actual_len < self.w:
                clip = F.pad(clip, (0, self.w - actual_len))
            if start == 0:
                mem.initialize(clip)
            M_prev = mem.get()
            logits, c_tilde = self.model.forward_clip(clip, M_prev)
            m_k = self.model.cfa.compress(c_tilde)
            mem.update(c_tilde, m_k)
            all_logits.append(logits[0].cpu().numpy().T)  # (w, C)

        all_logits = np.concatenate(all_logits, axis=0)  # (T_padded, C)
        probs = np.exp(all_logits) / np.exp(all_logits).sum(axis=1, keepdims=True)
        confs = probs.max(axis=1)
        return all_logits, confs

    def _infer_online(self, input_x):
        """Stride-1 online inference: one prediction per frame (only last frame of each window)."""
        D = input_x.shape[1]
        T = input_x.shape[2]
        device = input_x.device

        all_logits = []
        mem = MemoryBank(self.w, D, device)

        for t in range(T):
            start = max(0, t - self.w + 1)
            clip = input_x[:, :, start:t + 1]
            if clip.shape[2] < self.w:
                clip = F.pad(clip, (self.w - clip.shape[2], 0))
            if t == 0:
                mem.initialize(clip)
            M_prev = mem.get()
            logits, c_tilde = self.model.forward_clip(clip, M_prev)
            # only keep last frame
            all_logits.append(logits[0, :, -1].cpu().numpy())

            # update memory every w frames
            if (t + 1) % self.w == 0:
                m_k = self.model.cfa.compress(c_tilde)
                mem.update(c_tilde, m_k)

        all_logits = np.array(all_logits)  # (T, C)
        probs = np.exp(all_logits) / np.exp(all_logits).sum(axis=1, keepdims=True)
        confs = probs.max(axis=1)
        return all_logits, confs


# ---------------------------------------------------------------------------
# Segmentator (for predict.py / deploy_TAS.py compatibility)
# ---------------------------------------------------------------------------

class Segmentator:
    def __init__(self, model_path, input_dim, num_classes, num_layers=10,
                 num_f_maps=64, w=128, device='cpu',
                 inference_mode='semi_online', theta=0.9, sigma=1.0/16, T_max=None):
        self.model = OnlineTASModel(
            input_dim=input_dim, num_classes=num_classes,
            num_layers=num_layers, num_f_maps=num_f_maps, w=w,
        )
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.to(device)
        self.model.eval()
        self.device = device
        self.w = w
        self.inference_mode = inference_mode
        self.theta = theta
        self.l_min = int(sigma * T_max) if T_max is not None else 0
        self._trainer = Trainer.__new__(Trainer)
        self._trainer.model = self.model
        self._trainer.w = w
        self._trainer.num_classes = num_classes

    def predict(self, features):
        """features: (D, T) numpy array. Returns (T,) int tensor."""
        input_x = torch.tensor(features, dtype=torch.float).unsqueeze(0).to(self.device)
        with torch.no_grad():
            if self.inference_mode == 'semi_online':
                logits, confs = self._trainer._infer_semi_online(input_x)
            else:
                logits, confs = self._trainer._infer_online(input_x)

        T = features.shape[1]
        logits = logits[:T]
        confs = confs[:T]
        predictions = np.argmax(logits, axis=1)
        if self.l_min > 0:
            predictions = post_process(predictions, confs, self.theta, self.l_min)
        return torch.tensor(predictions)
