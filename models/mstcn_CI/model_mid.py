#!/usr/bin/python2.7

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import copy
import numpy as np


class MultiStageModel(nn.Module):
    def __init__(self, num_stages, num_layers, num_f_maps, dim_x, dim_const, num_classes):
        super(MultiStageModel, self).__init__()
        self.stage1 = SingleStageModel(num_layers, num_f_maps, dim_x, dim_const, num_classes)
        self.stages = nn.ModuleList([copy.deepcopy(SingleStageModel(num_layers, num_f_maps, dim_x, dim_const, num_classes, refiner=True)) for s in range(num_stages-1)])

    def forward(self, x, q, mask):
        out_cls, out_const = self.stage1(x, q, mask)
        outputs_cls = out_cls.unsqueeze(0)
        outputs_const = out_const.unsqueeze(0)
        for s in self.stages:
            out_cls, out_const = s(x, F.sigmoid(out_const) * mask[:, 0:1, :], mask, F.softmax(out_cls, dim=1) * mask[:, 0:1, :])
            outputs_cls = torch.cat((outputs_cls, out_cls.unsqueeze(0)), dim=0)
            outputs_const = torch.cat((outputs_const, out_const.unsqueeze(0)), dim=0)
        return outputs_cls, outputs_const


class SingleStageModel(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim_x, dim_const, num_classes, refiner=False):
        super(SingleStageModel, self).__init__()
        # import pdb; pdb.set_trace()
        if not refiner:
            self.conv_1x1_cls = nn.Conv1d(dim_x+dim_const, num_f_maps, 1)    
        else:
            self.conv_1x1_cls = nn.Conv1d(num_classes+dim_const, num_f_maps, 1)    

        self.layers_cls = nn.ModuleList([copy.deepcopy(DilatedResidualLayer(2 ** i, num_f_maps, num_f_maps)) for i in range(num_layers)])
        
        self.conv_1x1_const = nn.Conv1d(dim_x+num_classes, num_f_maps, 1)
        self.layers_const = nn.ModuleList([copy.deepcopy(DilatedResidualLayer(2 ** i, num_f_maps, num_f_maps)) for i in range(num_layers)])

        self.conv_out_classes = nn.Conv1d(num_f_maps, num_classes, 1)
        self.conv_out_const = nn.Conv1d(num_f_maps, dim_const, 1)

    def forward(self, x, q, mask, c_pred=None):
        if len(q.shape) == 4:
            B, N, D, T = q.shape
            q = q.view(B, N*D, T)

        if c_pred is None:
            x_cls = torch.concatenate([x, q], dim=-2)
        else:
            # import pdb; pdb.set_trace()
            x_cls = torch.concatenate([c_pred, q], dim=-2)
        x_cls = self.conv_1x1_cls(x_cls)
        for layer in self.layers_cls:
            x_cls = layer(x_cls, mask)

        out_cls = self.conv_out_classes(x_cls) * mask[:, 0:1, :]

        x_const = torch.concatenate([x, out_cls], dim=-2)
        x_const = self.conv_1x1_const(x_const)

        for layer in self.layers_const:
            x_const = layer(x_const, mask)
        out_const = self.conv_out_const(x_const) * mask[:, 0:1, :]

        return out_cls, out_const


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
    def __init__(self, num_blocks, num_layers, num_f_maps, dim_x, dim_const, num_classes):
        self.model = MultiStageModel(num_blocks, num_layers, num_f_maps, dim_x, dim_const, num_classes)
        self.ce = nn.CrossEntropyLoss(ignore_index=-100)
        self.bce = nn.BCEWithLogitsLoss(reduce='none')
        self.mse = nn.MSELoss(reduction='none')
        self.num_classes = num_classes
        self.dim_const = dim_const

    def train(self, save_dir, batch_gen, num_epochs, batch_size, learning_rate, device):
        self.model.train()
        self.model.to(device)
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        for epoch in range(num_epochs):
            epoch_loss = 0
            epoch_loss_cls = 0
            epoch_loss_const = 0
            correct = 0
            total = 0
            i=0
            while batch_gen.has_next():
                i+=1
                # print(f"batch {i}")
                batch_input, batch_target, mask, constraint_input_tensor, constraint_target_tensor = batch_gen.next_batch(batch_size)
                batch_input, batch_target, mask, constraint_input_tensor, constraint_target_tensor = batch_input.to(device), batch_target.to(device), mask.to(device), constraint_input_tensor.to(device), constraint_target_tensor.to(device)
                # import pdb; pdb.set_trace()
                optimizer.zero_grad()
                pred_cls, pred_const = self.model(batch_input, constraint_input_tensor, mask)

                # Convert constraint targets to float for BCE loss
                constraint_target_tensor = constraint_target_tensor.to(float)
                
                loss = 0
                for p_cls, p_const in zip(pred_cls, pred_const):
                    # Classification loss - unchanged
                    cls_loss = self.ce(p_cls.transpose(2, 1).contiguous().view(-1, self.num_classes), batch_target.view(-1))
                    
                    # Constraint loss - using BCE for multi-label classification
                    # Reshape predictions: [B, dim_const, T] -> [B, T, dim_const]
                    p_const_reshaped = p_const.transpose(2, 1).contiguous()
                    
                    # Reshape target: [B, N, D, T] -> [B, T, N*D]
                    B, N, D, T = constraint_target_tensor.shape
                    target_reshaped = constraint_target_tensor.reshape(B, N*D, T).transpose(2, 1).contiguous()
                    
                    # Apply BCE loss with appropriate mask if needed
                    # If you have a mask for constraints, apply it here
                    const_loss = self.bce(p_const_reshaped, target_reshaped)
                    
                    # Apply mask if needed and take mean
                    const_loss = const_loss.mean()
                    
                    loss += cls_loss + const_loss
                    epoch_loss_cls += cls_loss.item()
                    epoch_loss_const += const_loss.item()
                    
                    # Temporal smoothness - unchanged
                    loss += 0.15*torch.mean(torch.clamp(self.mse(F.log_softmax(p_cls[:, :, 1:], dim=1), 
                                                   F.log_softmax(p_cls.detach()[:, :, :-1], dim=1)), 
                                        min=0, max=16)*mask[:, :, 1:])

                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()


                _, predicted = torch.max(pred_cls[-1].data, 1)

                # import pdb; pdb.set_trace()

                correct += ((predicted == batch_target).float()*mask[:, 0, :].squeeze(1)).sum().item()
                total += torch.sum(mask[:, 0, :]).item()

            # import pdb; pdb.set_trace()

            batch_gen.reset()
            torch.save(self.model.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + ".model")
            torch.save(optimizer.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + ".opt")
            print("[epoch %d]: epoch loss = %f, epoch loss cls = %f, epoch loss  const = %f,   acc = %f" % (epoch + 1, epoch_loss / len(batch_gen.list_of_examples), epoch_loss_cls / len(batch_gen.list_of_examples), epoch_loss_const / len(batch_gen.list_of_examples),
                                                               float(correct)/total))

    def predict(self, model_dir, results_dir, features_path, prio_path, vid_list_file, epoch, actions_dict, device, sample_rate):
        self.model.eval()
        with torch.no_grad():
            self.model.to(device)
            self.model.load_state_dict(torch.load(model_dir + "/epoch-" + str(epoch) + ".model"))
            file_ptr = open(vid_list_file, 'r')
            list_of_vids = file_ptr.read().split('\n')[:-1]
            file_ptr.close()
            
            from scipy.signal import medfilt
            
            for vid in list_of_vids:
                # Load features and constraint data similar to next_batch
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
                
                # constraint_gt = constraint_gt[:, -1, :][:, np.newaxis, :]
                # constraint_input = constraint_input[:, -1, :][:, np.newaxis, :]

                # Downsample according to sample_rate
                features = features[:, ::sample_rate]
                constraint_input = constraint_input[:, ::sample_rate]
                
                # Prepare tensors
                input_x = torch.tensor(features, dtype=torch.float).unsqueeze(0).to(device)
                constraint_input_tensor = torch.tensor(constraint_input.transpose(1,2,0), dtype=torch.float).unsqueeze(0).to(device)
                
                # Create mask
                mask = torch.ones((1, self.num_classes, input_x.size(2)), device=device)
                
                # Forward pass
                pred_cls, pred_const = self.model(input_x, constraint_input_tensor, mask)
                
                # Get class predictions
                _, predicted = torch.max(pred_cls[-1].data, 1)
                predicted = predicted.squeeze()
                
                # Get constraint predictions
                const_predicted = pred_const[-1].squeeze()
                
                # Process results
                recognition = []
                constraint_values = []
                
                for i in range(len(predicted)):
                    action_key = list(actions_dict.keys())[list(actions_dict.values()).index(predicted[i].item())]
                    recognition = np.concatenate((recognition, [action_key]*sample_rate))
                    
                    # Process constraint predictions
                    if const_predicted.dim() > 1:  # Check if we have constraint predictions
                        # Determine the shape of constraint_values based on const_predicted
                        if i < const_predicted.size(-1):  # Make sure we don't go out of bounds
                            const_values = const_predicted[..., i].cpu().numpy()
                            constraint_values.append(const_values)
                
                # Write results to file
                f_name = vid.split('/')[-1].split('.')[0]
                f_ptr = open(results_dir + "/" + f_name, "w")
                f_ptr.write("### Frame level recognition: ###\n")
                f_ptr.write(' '.join(recognition))
                
                # Write constraint predictions if available
                # if constraint_values:
                #     f_ptr.write("\n### Frame level constraints: ###\n")
                #     # Flatten and format constraint values appropriately
                #     flattened_constraints = np.array(constraint_values).reshape(-1)
                #     f_ptr.write(' '.join(map(str, flattened_constraints)))
                    
                #     # Optionally, write structured constraint data for better analysis
                #     f_ptr.write("\n### Structured constraints: ###\n")
                #     np.save(results_dir + "/" + f_name + "_constraints.npy", np.array(constraint_values))
                #     f_ptr.write("Saved to " + f_name + "_constraints.npy")
                    
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
        