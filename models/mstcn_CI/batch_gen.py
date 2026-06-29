#!/usr/bin/python2.7

import torch
import numpy as np
import random

from scipy.signal import medfilt


class BatchGenerator(object):
    def __init__(self, num_classes, actions_dict, gt_path, features_path, prio_path, sample_rate, downsample_annot=False):
        self.list_of_examples = list()
        self.index = 0
        self.num_classes = num_classes
        self.actions_dict = actions_dict
        self.gt_path = gt_path
        self.features_path = features_path
        self.sample_rate = sample_rate
        self.downsample_annot = downsample_annot
        self.prio_path = prio_path

    def reset(self):
        self.index = 0
        random.shuffle(self.list_of_examples)

    def has_next(self):
        if self.index < len(self.list_of_examples):
            return True
        return False

    def read_data(self, vid_list_file):
        file_ptr = open(vid_list_file, 'r')
        self.list_of_examples = file_ptr.read().split('\n')[:-1]
        file_ptr.close()
        random.shuffle(self.list_of_examples)

    def add_noise_to_binary_tensor(self,tensor, column_flip_probability=0.0):
        """
        Add noise to a binary tensor by applying the same random flip pattern within each matrix,
        but different patterns across matrices.
        
        Args:
            tensor: A binary tensor with shape (n_matrices, height, width)
            matrix_flip_probability: Probability of flipping each element (between 0 and 1)
        
        Returns:
            Noisy tensor with same shape as input
        """
        n_matrices, height, width = tensor.shape
        noisy_tensor = tensor.copy()
        
        for i in range(n_matrices):
            # Decide whether to flip a column in this matrix
            if np.random.random() < column_flip_probability:
                # Randomly select one column to flip (out of the 12)
                column_to_flip = np.random.randint(0, width)
                
                # Flip all values in the selected column
                noisy_tensor[i, :, column_to_flip] = 1 - tensor[i, :, column_to_flip]
        
        return noisy_tensor

    def next_batch(self, batch_size):
        batch = self.list_of_examples[self.index:self.index + batch_size]
        self.index += batch_size

        batch_input = []
        batch_target = []
        constraint_input = []
        constraint_target = []
        for vid in batch:
            features = np.load(self.features_path + vid.split('.')[0] + '.npy').T
            prio_data = np.load(self.prio_path + vid.split('.')[0] + '.npy', allow_pickle=True)[()]
            constraint_data = np.concatenate(prio_data["constraint_state"])
            kernel_size = 601  # Adjust as needed
            filtered_data = np.zeros_like(constraint_data)
            for i in range(constraint_data.shape[1]):
                filtered_data[:, i] = medfilt(constraint_data[:, i], kernel_size)
            constraint_gt = filtered_data.reshape(prio_data["constraint_state"].shape)

            constraint_noise = self.add_noise_to_binary_tensor(prio_data["constraint_state"])
            # constraint_gt = constraint_gt[:, -1, :][:, np.newaxis, :]
            # constraint_noise = constraint_noise[:, -1, :][:, np.newaxis, :]
            # import pdb; pdb.set_trace()
            # import matplotlib.pyplot as plt
            # plt.plot(np.concatenate(constraint_gt))
            # plt.figure()
            # plt.plot(np.concatenate(constraint_noise))
            # plt.show()
            file_ptr = open(self.gt_path + vid + ".txt", 'r')
            content = file_ptr.read().split('\n')[:-1]
            if self.downsample_annot:
                content = content[::2]
            classes = np.zeros(min(np.shape(features)[1], len(content)))
            for i in range(len(classes)):
                classes[i] = self.actions_dict[content[i]]
            batch_input .append(features[:, ::self.sample_rate])
            batch_target.append(classes[::self.sample_rate])
            constraint_input .append(constraint_noise[:, ::self.sample_rate])
            constraint_target.append(constraint_gt[::self.sample_rate])


        length_of_sequences = [len(x) for x in batch_target]
        batch_input_tensor = torch.zeros(len(batch_input), np.shape(batch_input[0])[0], max(length_of_sequences), dtype=torch.float)
        batch_target_tensor = torch.ones(len(batch_input), max(length_of_sequences), dtype=torch.long)*(-100)
        constraint_input_tensor = torch.zeros(len(batch_input), np.shape(constraint_input[0])[1], np.shape(constraint_input[0])[2], max(length_of_sequences), dtype=torch.float)
        constraint_target_tensor = torch.ones(len(batch_input), np.shape(constraint_input[0])[1], np.shape(constraint_input[0])[2], max(length_of_sequences), dtype=torch.long)*(-100)

        mask = torch.zeros(len(batch_input), self.num_classes, max(length_of_sequences), dtype=torch.float)
        for i in range(len(batch_input)):
            batch_input_tensor[i, :, :np.shape(batch_input[i])[1]] = torch.from_numpy(batch_input[i])
            batch_target_tensor[i, :np.shape(batch_target[i])[0]] = torch.from_numpy(batch_target[i])
            constraint_input_tensor[i, ..., :np.shape(constraint_input[i])[0]] = torch.from_numpy(constraint_input[i].transpose(1,2,0))
            constraint_target_tensor[i, ..., :np.shape(constraint_target[i])[0]] = torch.from_numpy(constraint_target[i].transpose(1,2,0))
            # import pdb; pdb.set_trace()
            mask[i, :, :np.shape(batch_target[i])[0]] = torch.ones(self.num_classes, np.shape(batch_target[i])[0])

        # import pdb; pdb.set_trace()

        return batch_input_tensor, batch_target_tensor, mask, constraint_input_tensor, constraint_target_tensor
