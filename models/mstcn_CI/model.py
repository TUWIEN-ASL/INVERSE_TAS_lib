#!/usr/bin/python2.7

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import copy
import numpy as np


class MultiStageModel(nn.Module):
    def __init__(self, num_stages, num_layers, num_f_maps, dim, dim_q, num_classes):
        super(MultiStageModel, self).__init__()
        self.stage1 = SingleStageModel(num_layers, num_f_maps, dim, dim_q, num_classes)
        self.stages = nn.ModuleList([copy.deepcopy(SingleStageModel(num_layers, num_f_maps, num_classes, dim_q, num_classes)) for s in range(num_stages-1)])

    def forward(self, x, q, mask):
        out = self.stage1(x, q, mask)
        outputs = out.unsqueeze(0)
        for s in self.stages:
            out = s(F.softmax(out, dim=1) * mask[:, 0:1, :], q, mask)
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)
        return outputs


class SingleStageModel(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, dim_q, num_classes):
        super(SingleStageModel, self).__init__()
        self.conv_1x1 = nn.Conv1d(dim+dim_q, num_f_maps, 1)
        self.layers = nn.ModuleList([copy.deepcopy(DilatedResidualLayer(2 ** i, num_f_maps, num_f_maps)) for i in range(num_layers)])
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x, q, mask):
        if len(q.shape) == 4:
            B, N, D, T = q.shape
            q = q.view(B, N*D, T)
        x = torch.concat([x, q], dim=-2)
        # import pdb; pdb.set_trace()
        out = self.conv_1x1(x)
        for layer in self.layers:
            out = layer(out, mask)
        out = self.conv_out(out) * mask[:, 0:1, :]
        return out


class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation, in_channels, out_channels):
        super(DilatedResidualLayer, self).__init__()
        self.conv_dilated = nn.Conv1d(in_channels, out_channels, 3, padding=dilation, dilation=dilation)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()

    def forward(self, x, mask):
        out = F.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return (x + out) * mask[:, 0:1, :]


class Trainer:
    def __init__(self, num_blocks, num_layers, num_f_maps, dim, dim_q, num_classes):
        self.model = MultiStageModel(num_blocks, num_layers, num_f_maps, dim, dim_q, num_classes)
        self.ce = nn.CrossEntropyLoss(ignore_index=-100)
        self.mse = nn.MSELoss(reduction='none')
        self.num_classes = num_classes

    def train(self, save_dir, batch_gen, num_epochs, batch_size, learning_rate, device):
        self.model.train()
        self.model.to(device)
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        for epoch in range(num_epochs):
            epoch_loss = 0
            correct = 0
            total = 0
            i=0
            while batch_gen.has_next():
                i+=1
                # print(f"batch {i}")
                batch_input, batch_target, mask, constraint_input_tensor, constraint_target_tensor = batch_gen.next_batch(batch_size)
                batch_input, batch_target, mask, constraint_input_tensor, constraint_target_tensor = batch_input.to(device), batch_target.to(device), mask.to(device), constraint_input_tensor.to(device), constraint_target_tensor.to(device)
                optimizer.zero_grad()
                predictions = self.model(batch_input, constraint_input_tensor, mask)

                loss = 0
                for p in predictions:
                    loss += self.ce(p.transpose(2, 1).contiguous().view(-1, self.num_classes), batch_target.view(-1))
                    loss += 0.15*torch.mean(torch.clamp(self.mse(F.log_softmax(p[:, :, 1:], dim=1), F.log_softmax(p.detach()[:, :, :-1], dim=1)), min=0, max=16)*mask[:, :, 1:])

                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()

                _, predicted = torch.max(predictions[-1].data, 1)

                correct += ((predicted == batch_target).float()*mask[:, 0, :].squeeze(1)).sum().item()
                total += torch.sum(mask[:, 0, :]).item()

            # import pdb; pdb.set_trace()

            batch_gen.reset()
            torch.save(self.model.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + ".model")
            torch.save(optimizer.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + ".opt")
            print("[epoch %d]: epoch loss = %f,   acc = %f" % (epoch + 1, epoch_loss / len(batch_gen.list_of_examples),
                                                               float(correct)/total))

    def predict(self, model_dir, results_dir, features_path, prio_path, vid_list_file, epoch, actions_dict, device, sample_rate):
        self.model.eval()
        from scipy.signal import medfilt
        with torch.no_grad():
            self.model.to(device)
            self.model.load_state_dict(torch.load(model_dir + "/epoch-" + str(epoch) + ".model"))
            file_ptr = open(vid_list_file, 'r')
            list_of_vids = file_ptr.read().split('\n')[:-1]
            file_ptr.close()
            for vid in list_of_vids:
                features = np.load(features_path + vid.split('.')[0] + '.npy').T
                prio_data = np.load(prio_path + vid.split('.')[0] + '.npy', allow_pickle=True)[()]
                constraint_data = np.concatenate(prio_data["constraint_state"])
                
                # Apply median filter to constraint data
                kernel_size = 51  # Same as in next_batch
                filtered_data = np.zeros_like(constraint_data)
                for i in range(constraint_data.shape[1]):
                    filtered_data[:, i] = medfilt(constraint_data[:, i], kernel_size)
                constraint_gt = filtered_data.reshape(prio_data["constraint_state"].shape)
                # Use the constraint data directly without adding noise for prediction
                constraint_input = constraint_gt

                constraint_gt = constraint_gt[:, -1, :][:, np.newaxis, :]
                constraint_input = constraint_input[:, -1, :][:, np.newaxis, :]
                
                # Downsample according to sample_rate
                features = features[:, ::sample_rate]
                constraint_input = constraint_input[:, ::sample_rate]

                input_x = torch.tensor(features, dtype=torch.float)
                input_x.unsqueeze_(0)
                input_x = input_x.to(device)
                constraint_input_tensor = torch.tensor(constraint_input.transpose(1,2,0), dtype=torch.float).unsqueeze(0).to(device)

                predictions = self.model(input_x, constraint_input_tensor, torch.ones(input_x.size(), device=device))
                _, predicted = torch.max(predictions[-1].data, 1)
                predicted = predicted.squeeze()
                recognition = []
                # import pdb
                # pdb.set_trace()
                for i in range(len(predicted)):
                    recognition = np.concatenate((recognition, [list(actions_dict.keys())[list(actions_dict.values()).index(predicted[i].item())]]*sample_rate))
                f_name = vid.split('/')[-1].split('.')[0]
                f_ptr = open(results_dir + "/" + f_name, "w")
                f_ptr.write("### Frame level recognition: ###\n")
                f_ptr.write(' '.join(recognition))
                f_ptr.close()

class Segmentator:
    def __init__(self, model_path, num_blocks, num_layers, num_f_maps, dim, num_classes, device):
        self.model = MultiStageModel(num_blocks, num_layers, num_f_maps, dim, num_classes)
        self.model.load_state_dict(torch.load(model_path))
        self.model.to(device)
        self.model.eval()
        self.device = device
        
    def predict(self, features):
        input_x = torch.tensor(features, dtype=torch.float)
        input_x.unsqueeze_(0)
        input_x = input_x.to(self.device)
        predictions = self.model(input_x, torch.ones(input_x.size(), device=self.device))
        _, predicted = torch.max(predictions[-1].data, 1)
        predicted = predicted.squeeze()
        
        return predicted
        