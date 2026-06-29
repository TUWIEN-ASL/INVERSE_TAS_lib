import torch
import numpy as np
import random
import os


class BatchGenerator(object):
    """
    Loads one full video at a time (batch_size=1).
    Returns features as (1, D, T), labels as (1, T), mask as (1, C, T).
    Supports both:
      - .npy annotations (integer-string labels, no header) — used for REASSEMBLEmm
      - .txt annotations (action-name labels, optional header) — used for 50Salads etc.
    """

    def __init__(self, num_classes, actions_dict, gt_path, features_path,
                 sample_rate, downsample_annot=False):
        self.list_of_examples = []
        self.index = 0
        self.num_classes = num_classes
        self.actions_dict = actions_dict
        self.gt_path = gt_path
        self.features_path = features_path
        self.sample_rate = sample_rate
        self.downsample_annot = downsample_annot

    def reset(self):
        self.index = 0
        random.shuffle(self.list_of_examples)

    def has_next(self):
        return self.index < len(self.list_of_examples)

    def read_data(self, vid_list_file):
        with open(vid_list_file, 'r') as f:
            self.list_of_examples = f.read().split('\n')[:-1]
        random.shuffle(self.list_of_examples)

    def _load_annotation(self, vid):
        """Load frame-level integer class labels for a video."""
        base = vid.split('.')[0]
        npy_path = self.gt_path + base + '.npy'
        txt_path = self.gt_path + (vid if '.txt' in vid else base + '.txt')

        if os.path.exists(npy_path):
            ann = np.load(npy_path, allow_pickle=True)
            # Labels stored as integer strings ('0', '1', ...) — convert directly
            return ann.astype(int)
        else:
            with open(txt_path, 'r') as f:
                content = f.read().split('\n')
            # Remove trailing empty string
            content = [c for c in content if c]
            # Detect header: if first line is not a known action, skip it
            if content and content[0] not in self.actions_dict:
                content = content[1:]
            if self.downsample_annot:
                content = content[::2]
            return np.array([self.actions_dict[c] for c in content])

    def next_batch(self, batch_size):
        # batch_size is ignored — always returns one video
        vid = self.list_of_examples[self.index]
        self.index += 1

        features = np.load(self.features_path + vid.split('.')[0] + '.npy').T
        # features: (D, T) after transpose
        annotation = self._load_annotation(vid)

        if self.downsample_annot and not os.path.exists(self.gt_path + vid.split('.')[0] + '.npy'):
            pass  # already downsampled in txt path

        T = min(features.shape[1], len(annotation))
        features = features[:, :T:self.sample_rate]
        classes = annotation[:T:self.sample_rate].astype(np.int64)
        T_sampled = features.shape[1]

        feat_tensor = torch.tensor(features, dtype=torch.float).unsqueeze(0)  # (1, D, T)
        lbl_tensor = torch.tensor(classes, dtype=torch.long).unsqueeze(0)     # (1, T)
        mask = torch.ones(1, self.num_classes, T_sampled, dtype=torch.float)  # (1, C, T)

        return feat_tensor, lbl_tensor, mask
