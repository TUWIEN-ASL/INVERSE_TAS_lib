"""StreamingHASRRefiner — online inference wrapper.

Maintains a growing buffer of frame features and backbone predictions.
At each call to step(), the HASR refiner is re-applied to the entire history,
producing refined predictions for ALL frames seen so far.

Typical usage with OnlineTAS backbone:

    mem = MemoryBank(w, D, device)
    mem.initialize(first_clip)
    refiner_stream = StreamingHASRRefiner(hasr_model, device)

    id2action = {v: k for k, v in actions_dict.items()}
    all_refined = []

    for clip in clips:           # (1, D, w) each
        M_prev = mem.get()
        logits, c_tilde = onlinetas.forward_clip(clip, M_prev)  # (1, C, w)
        preds_clip = torch.argmax(logits, dim=1).squeeze()       # (w,)

        refined_so_far = refiner_stream.step(clip.squeeze(0), preds_clip)
        all_refined = refined_so_far   # (T_so_far,) — revised every clip

        m_k = onlinetas.cfa.compress(c_tilde)
        mem.update(c_tilde, m_k)

    # all_refined now holds the final frame-level refined predictions (T,)
    recognition = [id2action[int(p)] for p in all_refined]
"""

import os
import torch
import torch.nn.functional as F
import numpy as np


class StreamingHASRRefiner:
    """Stateful online HASR inference.

    The refiner model itself is stateless; all state is kept in the feature
    and prediction buffers maintained by this wrapper.

    Args:
        refiner: GRURefiner or TransformerHASR (already loaded, on device)
        device:  torch device
    """

    def __init__(self, refiner, device):
        self.refiner = refiner
        self.device  = device
        self._feat_cols  = []   # list of (D,) tensors
        self._pred_vals  = []   # list of int

    def reset(self):
        """Clear all accumulated state (call between videos)."""
        self._feat_cols = []
        self._pred_vals = []

    def step(self, features_clip, preds_clip):
        """Extend history with a new clip and return refined predictions for all frames.

        Args:
            features_clip: (D, n_new) raw frame features for the new clip
            preds_clip:    (n_new,) backbone argmax predictions for the new clip
                           (already trimmed to actual frames — no padding)

        Returns:
            (T_total,) LongTensor — refined predictions for every frame 0..T_total-1
        """
        n_new = preds_clip.shape[0]

        # Append new frames to buffers
        for i in range(n_new):
            self._feat_cols.append(features_clip[:, i].cpu())
            self._pred_vals.append(int(preds_clip[i].item()))

        # Build tensors from full history
        X_t = torch.stack(self._feat_cols, dim=1).unsqueeze(0).to(self.device)  # (1, D, T)
        A_t = torch.tensor(self._pred_vals, dtype=torch.long, device=self.device)  # (T,)

        # Run refiner on full history (no gradient needed)
        self.refiner.eval()
        with torch.no_grad():
            _, refine_rollout, _ = self.refiner(A_t, X_t)

        return torch.argmax(refine_rollout, dim=1).squeeze(0)  # (T,)


# ── Per-video online inference ────────────────────────────────────────────────

def run_online_inference(
    hasr_model,
    onlinetas_model,
    MemoryBank,
    features_path,
    vid_list_file,
    result_dir,
    actions_dict,
    device,
    sample_rate,
    clip_width=128,
):
    """Run full online inference for every video in vid_list_file.

    For each video, OnlineTAS processes non-overlapping clips of width w.
    After each clip the StreamingHASRRefiner refines all predictions so far.
    Final frame-level predictions (at t=T) are written to result_dir.

    Args:
        hasr_model:      GRURefiner or TransformerHASR (checkpoint already loaded)
        onlinetas_model: OnlineTASModel (checkpoint already loaded)
        MemoryBank:      MemoryBank class from onlinetas/model.py
        features_path:   path to directory containing .npy feature files
        vid_list_file:   text file listing video names (one per line)
        result_dir:      directory to write prediction files
        actions_dict:    {action_name: class_id}
        device:          torch device
        sample_rate:     temporal subsampling rate
        clip_width:      w (default: 128)
    """
    os.makedirs(result_dir, exist_ok=True)
    id2action = {v: k for k, v in actions_dict.items()}

    onlinetas_model.eval()
    onlinetas_model.to(device)
    hasr_model.eval()
    hasr_model.to(device)

    with open(vid_list_file, 'r') as f:
        videos = [v for v in f.read().split('\n') if v.strip()]

    for vid in videos:
        features = np.load(features_path + vid.split('.')[0] + '.npy').T
        features  = features[:, ::sample_rate]                        # (D, T)
        T         = features.shape[1]
        D         = features.shape[0]

        feat_t    = torch.tensor(features, dtype=torch.float, device=device)  # (D, T)
        feat_bt   = feat_t.unsqueeze(0)                                        # (1, D, T)

        mem    = MemoryBank(clip_width, D, device)
        stream = StreamingHASRRefiner(hasr_model, device)
        stream.reset()

        refined_final = None

        for start in range(0, T, clip_width):
            end         = min(start + clip_width, T)
            actual_len  = end - start
            clip        = feat_bt[:, :, start:end]                    # (1, D, actual_len)

            if actual_len < clip_width:
                clip_padded = F.pad(clip, (0, clip_width - actual_len))
            else:
                clip_padded = clip

            if start == 0:
                mem.initialize(clip_padded)

            M_prev = mem.get()
            with torch.no_grad():
                logits, c_tilde = onlinetas_model.forward_clip(clip_padded, M_prev)

            preds_clip = torch.argmax(logits, dim=1).squeeze(0)[:actual_len]  # (actual_len,)
            refined_final = stream.step(clip.squeeze(0), preds_clip)           # (T_so_far,)

            m_k = onlinetas_model.cfa.compress(c_tilde)
            mem.update(c_tilde, m_k)

        # Write final predictions
        recognition = []
        for p in refined_final.cpu().tolist():
            recognition.extend([id2action[p]] * sample_rate)

        fname = vid.split('/')[-1].split('.')[0]
        with open(os.path.join(result_dir, fname), 'w') as f:
            f.write('### Frame level recognition: ###\n')
            f.write(' '.join(recognition))
