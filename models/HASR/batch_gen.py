import torch
import numpy as np
import random


class BatchGenerator:
    def __init__(self, num_classes, actions_dict, gt_path, features_path, sample_rate):
        self.list_of_examples = []
        self.index = 0
        self.num_classes = num_classes
        self.actions_dict = actions_dict
        self.gt_path = gt_path
        self.features_path = features_path
        self.sample_rate = sample_rate

    def reset(self):
        self.index = 0
        random.shuffle(self.list_of_examples)

    def has_next(self):
        return self.index < len(self.list_of_examples)

    def read_data(self, vid_list_file):
        with open(vid_list_file, 'r') as f:
            self.list_of_examples = f.read().split('\n')[:-1]
        random.shuffle(self.list_of_examples)

    def next_batch(self, batch_size):
        batch = self.list_of_examples[self.index:self.index + batch_size]
        self.index += batch_size

        batch_input, batch_target = [], []
        for vid in batch:
            # Features stored as (T, D) in .npy, load and transpose to (D, T)
            features = np.load(self.features_path + vid.split('.')[0] + '.npy').T

            gt_file = self.gt_path + (vid if '.txt' in vid else vid + '.txt')
            with open(gt_file, 'r') as f:
                content = f.read().split('\n')[:-1]
            content = content[1:]  # skip header line

            min_len = min(features.shape[1], len(content))
            features = features[:, :min_len]
            classes = np.array([self.actions_dict[content[i]] for i in range(min_len)])

            batch_input.append(features[:, ::self.sample_rate])
            batch_target.append(classes[::self.sample_rate])

        lengths = [t.shape[0] for t in batch_target]
        max_len = max(lengths)
        D = batch_input[0].shape[0]

        input_tensor = torch.zeros(len(batch), D, max_len, dtype=torch.float)
        target_tensor = torch.full((len(batch), max_len), -100, dtype=torch.long)
        mask = torch.zeros(len(batch), self.num_classes, max_len, dtype=torch.float)

        for i in range(len(batch)):
            T = batch_target[i].shape[0]
            input_tensor[i, :, :T] = torch.from_numpy(batch_input[i])
            target_tensor[i, :T] = torch.from_numpy(batch_target[i])
            mask[i, :, :T] = 1.0

        return input_tensor, target_tensor, mask
