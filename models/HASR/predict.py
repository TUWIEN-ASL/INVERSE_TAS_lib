import os
import torch
import numpy as np


def predict_refiner(
    model,
    backbone,
    backbone_model_dir,
    backbone_epoch,
    model_dir,
    refiner_epoch,
    result_dir,
    features_path,
    vid_list_file,
    actions_dict,
    device,
    sample_rate,
):
    """Run backbone + TransformerHASR refiner and write frame-level predictions.

    Args:
        model: TransformerHASR instance
        backbone: mstcn MultiStageModel
        backbone_model_dir: directory with backbone checkpoints
        backbone_epoch: which backbone epoch to load
        model_dir: directory with HASR refiner checkpoints
        refiner_epoch: which refiner epoch to load
        result_dir: directory to write per-video prediction files
        features_path: path to directory containing .npy feature files
        vid_list_file: text file listing video names (one per line)
        actions_dict: {action_name: class_id} mapping
        device: torch device
        sample_rate: temporal subsampling rate
    """
    os.makedirs(result_dir, exist_ok=True)

    # Load backbone
    ckpt = os.path.join(backbone_model_dir, f'epoch-{backbone_epoch}.model')
    backbone.load_state_dict(torch.load(ckpt, map_location=device))
    backbone.to(device)
    backbone.eval()

    # Load refiner
    model.load_state_dict(
        torch.load(os.path.join(model_dir, f'epoch-{refiner_epoch}.model'), map_location=device)
    )
    model.to(device)
    model.eval()

    id2action = {v: k for k, v in actions_dict.items()}

    with open(vid_list_file, 'r') as f:
        videos = f.read().split('\n')[:-1]

    with torch.no_grad():
        for vid in videos:
            features = np.load(features_path + vid.split('.')[0] + '.npy').T
            features = features[:, ::sample_rate]
            x = torch.tensor(features, dtype=torch.float).unsqueeze(0).to(device)

            # Backbone forward
            mask = torch.ones(x.size(), device=device)
            action_idx = torch.argmax(backbone(x, mask)[-1], dim=1).squeeze().detach()

            # Refiner forward
            _, refine_rollout, _ = model(action_idx, x)
            _, predicted = torch.max(refine_rollout.data, 1)
            predicted = predicted.squeeze()

            recognition = []
            for i in range(len(predicted)):
                recognition.extend([id2action[predicted[i].item()]] * sample_rate)

            fname = vid.split('/')[-1].split('.')[0]
            with open(os.path.join(result_dir, fname), 'w') as f:
                f.write('### Frame level recognition: ###\n')
                f.write(' '.join(recognition))
