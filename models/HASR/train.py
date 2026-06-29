import os
import torch
import torch.nn as nn
from torch import optim
import random
import numpy as np
from typing import List


def _evaluate_refiner(model, backbone, backbone_model_dir, backbone_epochs, eval_data, device):
    """Run backbone (last epoch) + refiner on test set and return eval metrics."""
    from configs.eval_utils import compute_metrics
    features_path, vid_list_file, gt_path, actions_dict, sample_rate = eval_data
    idx_to_action = {v: k for k, v in actions_dict.items()}

    eval_epoch = backbone_epochs[-1]
    ckpt = os.path.join(backbone_model_dir, f'epoch-{eval_epoch}.model')
    backbone.load_state_dict(torch.load(ckpt, map_location=device))
    backbone.to(device)
    backbone.eval()
    model.eval()

    predictions = {}
    with open(vid_list_file, 'r') as f:
        vid_list = f.read().split('\n')[:-1]

    with torch.no_grad():
        for vid in vid_list:
            features = np.load(features_path + vid.split('.')[0] + '.npy').T
            features = features[:, ::sample_rate]
            x = torch.tensor(features, dtype=torch.float).unsqueeze(0).to(device)
            mask = torch.ones(x.size(), device=device)
            action_idx = torch.argmax(backbone(x, mask)[-1], dim=1).squeeze().detach()
            _, refine_rollout, _ = model(action_idx, x)
            _, predicted = torch.max(refine_rollout.data, 1)
            predicted = predicted.squeeze()
            recognition = [label for i in range(len(predicted))
                           for label in [idx_to_action[predicted[i].item()]] * sample_rate]
            predictions[vid.split('.')[0]] = recognition

    model.train()
    return compute_metrics(predictions, gt_path, vid_list, actions_dict)


def _get_backbone_predictions(backbone, batch_input, device):
    """Run frozen backbone and return argmax action indices (T,).

    Both mstcn and ASFormer share the same forward signature:
    backbone(x, mask) -> tensor list; take [-1] for the final-stage output.
    """
    mask = torch.ones(batch_input.size(), device=device)
    with torch.no_grad():
        preds = backbone(batch_input, mask)
    return torch.argmax(preds[-1], dim=1).squeeze().detach()


def _load_random_backbone_epoch(backbone, model_dir, backbone_epochs, device):
    """Load a random backbone checkpoint from the provided epoch list.

    Sampling across epochs exposes the refiner to varied prediction quality,
    matching the original HASR training strategy.
    """
    epoch = random.choice(backbone_epochs)
    ckpt = os.path.join(model_dir, f'epoch-{epoch}.model')
    backbone.load_state_dict(torch.load(ckpt, map_location=device))
    backbone.to(device)
    backbone.eval()


def train_refiner(
    model,
    backbone,
    backbone_model_dir,
    backbone_epochs: List[int],
    batch_gen,
    num_epochs,
    learning_rate,
    model_dir,
    device,
    wandb_run=None,
    eval_data=None,
):
    """Train TransformerHASR on top of a frozen backbone.

    Each iteration samples a random checkpoint from backbone_epochs so the
    refiner learns to correct predictions across the full quality spectrum.

    Args:
        model: TransformerHASR instance
        backbone: backbone model (mstcn or ASFormer)
        backbone_model_dir: directory containing epoch-N.model files
        backbone_epochs: list of valid epoch indices to sample from
        batch_gen: BatchGenerator
        num_epochs: number of training epochs for the refiner
        learning_rate: Adam learning rate
        model_dir: directory to save refiner checkpoints
        device: torch device
    """
    os.makedirs(model_dir, exist_ok=True)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    ce = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        model.train()
        backbone.eval()

        total_loss = 0.0
        n_samples = 0

        batch_gen.reset()
        while batch_gen.has_next():
            batch_input, batch_target, _ = batch_gen.next_batch(1)
            batch_input = batch_input.to(device)
            batch_target = batch_target.to(device)

            _load_random_backbone_epoch(backbone, backbone_model_dir, backbone_epochs, device)
            action_idx = _get_backbone_predictions(backbone, batch_input, device)

            refine_pred, _, GTlabel_list = model(action_idx, batch_input, batch_target)

            loss = ce(refine_pred[0], GTlabel_list.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_samples += 1

        avg_loss = total_loss / max(n_samples, 1)
        print(f"[epoch {epoch + 1:3d}]: loss = {avg_loss:.4f}")
        if wandb_run is not None:
            wandb_run.log({"train/loss": avg_loss}, step=epoch + 1)
        torch.save(model.state_dict(), os.path.join(model_dir, f'epoch-{epoch + 1}.model'))
        torch.save(optimizer.state_dict(), os.path.join(model_dir, f'epoch-{epoch + 1}.opt'))
        if eval_data is not None:
            metrics = _evaluate_refiner(model, backbone, backbone_model_dir, backbone_epochs, eval_data, device)
            print(f"[epoch {epoch + 1:3d}] eval: {metrics}")
            if wandb_run is not None:
                wandb_run.log(metrics, step=epoch + 1)
