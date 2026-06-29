"""Online HASR refiner training.

Two prefix-sampling strategies:

  random          — sample one clip-boundary t* ~ Uniform(1, ⌈T/w⌉) per video
                    per iteration.  Same per-step cost as offline training.

  clip_boundaries — train on every clip boundary t ∈ {w, 2w, …, T} per video.
                    Accumulates ⌈T/w⌉ forward passes into one gradient step.
                    More faithful to the inference cadence (default).
"""

import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from train import _load_random_backbone_epoch  # models/HASR/train.py


# ── OnlineTAS backbone inference ─────────────────────────────────────────────

def _get_onlinetas_predictions(backbone, MemoryBank, batch_input, w, device):
    """Run OnlineTAS in semi-online (non-overlapping clips) mode.

    Args:
        backbone:    OnlineTASModel, already loaded and on device, in eval mode
        MemoryBank:  MemoryBank class (loaded from onlinetas/model.py)
        batch_input: (1, D, T) feature tensor on device
        w:           clip width
        device:      torch device

    Returns:
        (T,) LongTensor of argmax backbone predictions
    """
    D = batch_input.shape[1]
    T = batch_input.shape[2]

    mem = MemoryBank(w, D, device)
    all_preds = []

    with torch.no_grad():
        for start in range(0, T, w):
            end = min(start + w, T)
            clip = batch_input[:, :, start:end]
            actual_len = clip.shape[2]
            if actual_len < w:
                clip = F.pad(clip, (0, w - actual_len))

            if start == 0:
                mem.initialize(clip)

            M_prev = mem.get()
            logits, c_tilde = backbone.forward_clip(clip, M_prev)  # (1, C, w)
            preds = torch.argmax(logits, dim=1).squeeze(0)          # (w,)
            all_preds.append(preds[:actual_len])                    # trim padding

            m_k = backbone.cfa.compress(c_tilde)
            mem.update(c_tilde, m_k)

    return torch.cat(all_preds, dim=0)  # (T,)


# ── Training loop ─────────────────────────────────────────────────────────────

def train_online_refiner(
    model,
    backbone,
    MemoryBank,
    backbone_model_dir,
    backbone_epochs,
    batch_gen,
    num_epochs,
    lr,
    model_dir,
    device,
    prefix_strategy='clip_boundaries',
    clip_width=128,
    wandb_run=None,
):
    """Train a HASR refiner (GRURefiner or TransformerHASR) in online mode.

    At each iteration the refiner sees only a prefix of the video — either a
    random clip boundary (prefix_strategy='random') or every clip boundary
    (prefix_strategy='clip_boundaries') — simulating the growing-history
    condition of online inference.

    Args:
        model:             GRURefiner or TransformerHASR
        backbone:          OnlineTASModel (weights replaced each iteration)
        MemoryBank:        MemoryBank class from onlinetas/model.py
        backbone_model_dir: directory containing OnlineTAS epoch-N.model files
        backbone_epochs:   list of valid epoch indices to sample from
        batch_gen:         BatchGenerator
        num_epochs:        number of refiner training epochs
        lr:                Adam learning rate
        model_dir:         directory to save refiner checkpoints
        device:            torch device
        prefix_strategy:   'random' | 'clip_boundaries'  (default: clip_boundaries)
        clip_width:        w — clip size used by OnlineTAS (default: 128)
    """
    os.makedirs(model_dir, exist_ok=True)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        model.train()
        backbone.eval()
        total_loss = 0.0
        n_samples = 0

        batch_gen.reset()
        while batch_gen.has_next():
            batch_input, batch_target, _ = batch_gen.next_batch(1)
            batch_input  = batch_input.to(device)
            batch_target = batch_target.to(device)

            # Fresh random backbone epoch each sample (HASR robustness strategy)
            _load_random_backbone_epoch(backbone, backbone_model_dir, backbone_epochs, device)
            action_idx = _get_onlinetas_predictions(backbone, MemoryBank, batch_input, clip_width, device)

            T = batch_input.shape[2]

            # ── Build prefix set P ────────────────────────────────────────────
            if prefix_strategy == 'random':
                n_clips = max(1, (T + clip_width - 1) // clip_width)
                k_star  = random.randint(1, n_clips)
                t_star  = min(k_star * clip_width, T)
                prefixes = [t_star]

            elif prefix_strategy == 'clip_boundaries':
                prefixes = list(range(clip_width, T, clip_width))
                prefixes.append(T)   # always include the full sequence
                prefixes = [p for p in prefixes if p >= 1]

            else:
                raise ValueError(f'Unknown prefix_strategy: {prefix_strategy!r}')

            # ── Accumulate loss over prefix set ───────────────────────────────
            loss = torch.tensor(0.0, device=device, requires_grad=False)
            valid_prefixes = 0

            for t in prefixes:
                prefix_input     = batch_input[:, :, :t]
                prefix_target    = batch_target[:, :t]
                prefix_action    = action_idx[:t]

                refine_pred, _, GTlabel_list = model(
                    prefix_action, prefix_input, prefix_target
                )

                if GTlabel_list.numel() == 0:
                    continue

                loss = loss + ce(refine_pred[0], GTlabel_list.view(-1))
                valid_prefixes += 1

            if valid_prefixes == 0:
                continue

            loss = loss / valid_prefixes

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_samples  += 1

        avg_loss = total_loss / max(n_samples, 1)
        print(f'[epoch {epoch + 1:3d}] strategy={prefix_strategy}  loss={avg_loss:.4f}',
              flush=True)
        if wandb_run is not None:
            wandb_run.log({"train/loss": avg_loss}, step=epoch + 1)
        torch.save(model.state_dict(), os.path.join(model_dir, f'epoch-{epoch + 1}.model'))
        torch.save(optimizer.state_dict(), os.path.join(model_dir, f'epoch-{epoch + 1}.opt'))
