import os
import copy
import torch
import argparse
import numpy as np
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from scipy.ndimage import median_filter
from torch.utils.tensorboard import SummaryWriter
from dataset import restore_full_sequence
from dataset import get_data_dict
from dataset import VideoFeatureDataset
from model import ASDiffusionModel
from tqdm import tqdm
from utils import load_config_file, func_eval, set_random_seed, get_labels_start_end_time, plot_barcode
from utils import mode_filter

from configs.paths import project_path
import wandb

import random
import numpy as np
import torch
import os


def set_seed(seed=42):
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True  # Forces cuDNN to use deterministic algorithms
    torch.backends.cudnn.benchmark = False      # Disables cuDNN's auto-tuner (which is non-deterministic)
    torch.use_deterministic_algorithms(True, warn_only=True)  # Forces PyTorch operations to be deterministic
    
    # Set a fixed value for Python hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)

class Trainer:
    def __init__(self, encoder_params, decoder_params, diffusion_params, 
        event_list, sample_rate, temporal_aug, set_sampling_seed, postprocess, device):

        self.device = device
        self.num_classes = len(event_list)
        self.encoder_params = encoder_params
        self.decoder_params = decoder_params
        self.event_list = event_list
        self.sample_rate = sample_rate
        self.temporal_aug = temporal_aug
        self.set_sampling_seed = set_sampling_seed
        self.postprocess = postprocess

        self.model = ASDiffusionModel(encoder_params, decoder_params, diffusion_params, self.num_classes, self.device)
        print('Model Size: ', sum(p.numel() for p in self.model.parameters()))

    def train(self, train_train_dataset, train_test_dataset, test_test_dataset, loss_weights, class_weighting, soft_label,
              num_epochs, batch_size, learning_rate, weight_decay, label_dir, weights_dir, result_dir, log_freq, log_train_results=True, downsample=False, wandb_run=None):

        os.makedirs(weights_dir, exist_ok=True)
        os.makedirs(result_dir, exist_ok=True)

        device = self.device
        self.model.to(device)

        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        optimizer.zero_grad()

        restore_epoch = -1
        step = 1

        if os.path.exists(weights_dir):
            if 'latest.pt' in os.listdir(weights_dir):
                if os.path.getsize(os.path.join(weights_dir, 'latest.pt')) > 0:
                    saved_state = torch.load(os.path.join(weights_dir, 'latest.pt'))
                    self.model.load_state_dict(saved_state['model'])
                    optimizer.load_state_dict(saved_state['optimizer'])
                    restore_epoch = saved_state['epoch']
                    step = saved_state['step']

        if class_weighting:
            class_weights = train_train_dataset.get_class_weights()
            class_weights = torch.from_numpy(class_weights).float().to(device)
            ce_criterion = nn.CrossEntropyLoss(ignore_index=-100, weight=class_weights, reduction='none')
        else:
            ce_criterion = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')

        bce_criterion = nn.BCELoss(reduction='none')
        mse_criterion = nn.MSELoss(reduction='none')
        
        train_train_loader = torch.utils.data.DataLoader(
            train_train_dataset, batch_size=1, shuffle=True, num_workers=4)
        
        if result_dir:
            if not os.path.exists(result_dir):
                os.makedirs(result_dir)
            logger = SummaryWriter(result_dir)
        import time
        for epoch in range(restore_epoch+1, num_epochs):
            self.model.train()
            
            epoch_running_loss = 0
            
            for _id, data in enumerate(train_train_loader):
                # print(_id)
                s = time.process_time_ns()
                feature, label, boundary, video = data
                feature, label, boundary = feature.to(device), label.to(device), boundary.to(device)
                
                loss_dict = self.model.get_training_loss(feature, 
                    event_gt=F.one_hot(label.long(), num_classes=self.num_classes).permute(0, 2, 1),
                    boundary_gt=boundary,
                    encoder_ce_criterion=ce_criterion, 
                    encoder_mse_criterion=mse_criterion,
                    encoder_boundary_criterion=bce_criterion,
                    decoder_ce_criterion=ce_criterion,
                    decoder_mse_criterion=mse_criterion,
                    decoder_boundary_criterion=bce_criterion,
                    soft_label=soft_label
                )

                # ##############
                # # feature    torch.Size([1, F, T])
                # # label      torch.Size([1, T])
                # # boundary   torch.Size([1, 1, T])
                # # output    torch.Size([1, C, T]) 
                # ##################

                total_loss = 0

                for k,v in loss_dict.items():
                    total_loss += loss_weights[k] * v

                e = time.process_time_ns()
                # print("Forward time", (e-s)/10**9)

                if result_dir:
                    for k,v in loss_dict.items():
                        logger.add_scalar(f'Train-{k}', loss_weights[k] * v.item() / batch_size, step)
                    logger.add_scalar('Train-Total', total_loss.item() / batch_size, step)
                if wandb_run is not None:
                    log_dict = {f"train/{k}_loss": loss_weights[k] * v.item() / batch_size for k, v in loss_dict.items()}
                    log_dict["train/total_loss"] = total_loss.item() / batch_size
                    wandb_run.log(log_dict, step=step)

                total_loss /= batch_size
                total_loss.backward()
        
                epoch_running_loss += total_loss.item()
                
                if step % batch_size == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                step += 1
                
            epoch_running_loss /= len(train_train_dataset)

            print(f'Epoch {epoch} - Running Loss {epoch_running_loss}')
        
            if weights_dir:

                state = {
                    'model': self.model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'step': step
                }

            if epoch % log_freq == 0:
                if weights_dir:
                    torch.save(self.model.state_dict(), f'{weights_dir}/epoch-{epoch}.model')
                    torch.save(state, f'{weights_dir}/latest.pt')

            # evaluate every epoch
            for mode in ['decoder-agg']:
                test_result_dict = self.test(
                    test_test_dataset, mode, device, label_dir,
                    result_dir=result_dir, model_path=None, downsample=downsample, epoch=epoch)

                if result_dir and epoch % log_freq == 0:
                    for k, v in test_result_dict.items():
                        logger.add_scalar(f'Test-{mode}-{k}', v, epoch)
                    np.save(os.path.join(result_dir,
                        f'test_results_{mode}_epoch{epoch}.npy'), test_result_dict)

                for k, v in test_result_dict.items():
                    print(f'Epoch {epoch} - {mode}-Test-{k} {v}')

                if wandb_run is not None:
                    _key_map = {"Acc": "val/acc", "Edit": "val/edit",
                                "F1@10": "val/f1_10", "F1@25": "val/f1_25",
                                "F1@50": "val/f1_50", "DR": "val/dr"}
                    wandb_run.log({_key_map.get(k, f"val/{k}"): v
                                   for k, v in test_result_dict.items()}, step=epoch)

                if log_train_results and epoch % log_freq == 0:
                    train_result_dict = self.test(
                        train_test_dataset, mode, device, label_dir,
                        result_dir=result_dir, model_path=None, downsample=downsample, epoch=epoch)

                    if result_dir:
                        for k, v in train_result_dict.items():
                            logger.add_scalar(f'Train-{mode}-{k}', v, epoch)
                        np.save(os.path.join(result_dir,
                            f'train_results_{mode}_epoch{epoch}.npy'), train_result_dict)

                    for k, v in train_result_dict.items():
                        print(f'Epoch {epoch} - {mode}-Train-{k} {v}')

        if result_dir:
            logger.close()


    def test_single_video(self, video_idx, test_dataset, mode, device, model_path=None):  
        
        assert(test_dataset.mode == 'test')
        assert(mode in ['encoder', 'decoder-noagg', 'decoder-agg'])
        assert(self.postprocess['type'] in ['median', 'mode', 'purge', None])


        self.model.eval()
        self.model.to(device)

        if model_path:
            self.model.load_state_dict(torch.load(model_path))

        # if self.set_sampling_seed:
        #     seed = video_idx
        # else:
        seed = None
            
        with torch.no_grad():

            feature, label, _, video = test_dataset[video_idx]

            # feature:   [torch.Size([1, F, Sampled T])]
            # label:     torch.Size([1, Original T])
            # output: [torch.Size([1, C, Sampled T])]

            if mode == 'encoder':
                output = [self.model.encoder(feature[i].to(device)) 
                       for i in range(len(feature))] # output is a list of tuples
                output = [F.softmax(i, 1).cpu() for i in output]
                left_offset = self.sample_rate // 2
                right_offset = (self.sample_rate - 1) // 2

            if mode == 'decoder-agg':
                output = [self.model.ddim_sample(feature[i].to(device), seed) 
                           for i in range(len(feature))] # output is a list of tuples
                output = [i.cpu() for i in output]
                left_offset = self.sample_rate // 2
                right_offset = (self.sample_rate - 1) // 2

            if mode == 'decoder-noagg':  # temporal aug must be true
                output = [self.model.ddim_sample(feature[len(feature)//2].to(device), seed)] # output is a list of tuples
                output = [i.cpu() for i in output]
                left_offset = self.sample_rate // 2
                right_offset = 0

            assert(output[0].shape[0] == 1)

            min_len = min([i.shape[2] for i in output])
            output = [i[:,:,:min_len] for i in output]
            output = torch.cat(output, 0)  # torch.Size([sample_rate, C, T])
            output = output.mean(0).numpy()

            if self.postprocess['type'] == 'median': # before restoring full sequence
                smoothed_output = np.zeros_like(output)
                for c in range(output.shape[0]):
                    smoothed_output[c] = median_filter(output[c], size=self.postprocess['value'])
                output = smoothed_output / smoothed_output.sum(0, keepdims=True)

            output = np.argmax(output, 0)

            output = restore_full_sequence(output, 
                full_len=label.shape[-1], 
                left_offset=left_offset, 
                right_offset=right_offset, 
                sample_rate=self.sample_rate
            )

            if self.postprocess['type'] == 'mode': # after restoring full sequence
                output = mode_filter(output, self.postprocess['value'])

            if self.postprocess['type'] == 'purge':

                trans, starts, ends = get_labels_start_end_time(output)
                
                for e in range(0, len(trans)):
                    duration = ends[e] - starts[e]
                    if duration <= self.postprocess['value']:
                        
                        if e == 0:
                            output[starts[e]:ends[e]] = trans[e+1]
                        elif e == len(trans) - 1:
                            output[starts[e]:ends[e]] = trans[e-1]
                        else:
                            mid = starts[e] + duration // 2
                            output[starts[e]:mid] = trans[e-1]
                            output[mid:ends[e]] = trans[e+1]

            label = label.squeeze(0).cpu().numpy()

            assert(output.shape == label.shape)
            
            return video, output, label


    def test(self, test_dataset, mode, device, label_dir, result_dir=None, model_path=None, downsample=False,  epoch=None):
        
        assert(test_dataset.mode == 'test')

        self.model.eval()
        self.model.to(device)

        if model_path:
            self.model.load_state_dict(torch.load(model_path))
        
        plot_dir = os.path.join(result_dir, "plots", str(epoch))
        os.makedirs(plot_dir, exist_ok=True)

        with torch.no_grad():

            for video_idx in tqdm(range(len(test_dataset))):
                
                video, pred_np, label = self.test_single_video(
                    video_idx, test_dataset, mode, device, model_path)

                pred = [self.event_list[int(i)] for i in pred_np]
                
                if not os.path.exists(os.path.join(result_dir, 'prediction')):
                    os.makedirs(os.path.join(result_dir, 'prediction'))

                file_name = os.path.join(result_dir, 'prediction', f'{video}.txt')
                file_ptr = open(file_name, 'w')
                file_ptr.write('### Frame level recognition: ###\n')
                file_ptr.write(' '.join(pred))
                file_ptr.close()

                plot_barcode(test_dataset.class_num, label, pred_np, False, os.path.join(plot_dir, f"{video}.png"))

        acc, edit, f1s, f1_bound = func_eval(
            label_dir, os.path.join(result_dir, 'prediction'), test_dataset.video_list, downsample, 10)

        result_dict = {
            'Acc': acc,
            'Edit': edit,
            'F1@10': f1s[0],
            'F1@25': f1s[1],
            'F1@50': f1s[2],
            "DR": f1_bound
        }
        
        return result_dict


if __name__ == '__main__':

    set_seed(42)

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--config', type=str)
    parser.add_argument('--device', type=int)
    parser.add_argument('--f_type', type=str)
    parser.add_argument('--epoch', type=str)
    args = parser.parse_args()

    all_params = load_config_file(args.config)
    locals().update(all_params)

    weights_dir = os.path.join(project_path, "checkpoints","DiffAct")
    result_dir = os.path.join(project_path, "results","DiffAct", f"{all_params['dataset_name']}_{args.f_type}_{args.epoch}", f"split_{all_params['split_id']}")

    print(args.config)
    print(all_params)

    if args.device != -1:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
    
    if args.f_type != "BRPOAF":
        feature_dir = os.path.join(root_data_dir, dataset_name, args.f_type)
        oaf_feature_dir = None
    else:
        feature_dir = os.path.join(root_data_dir, dataset_name, "BRP_features")
        oaf_feature_dir = os.path.join(root_data_dir, dataset_name, "ipp_oaf_vvb_fixed_yolo", f"epoch-{args.epoch}")

    label_dir = os.path.join(root_data_dir, dataset_name, 'annotations')
    mapping_file = os.path.join(root_data_dir, dataset_name, 'mapping.txt')

    # if dataset_name != "50salads" and args.f_type == "I3D_features":
    #     downsample = True
    # else:
    #     downsample = False
    downsample = False

    event_list = np.loadtxt(mapping_file, dtype=str)
    event_list = [i[1] for i in event_list]
    num_classes = len(event_list)

    train_video_list = np.loadtxt(os.path.join(
        root_data_dir, dataset_name, 'splits', f'train.split{split_id}.bundle'), dtype=str)
    test_video_list = np.loadtxt(os.path.join(
        root_data_dir, dataset_name, 'splits', f'test.split{split_id}.bundle'), dtype=str)

    if train_video_list.shape == ():
        train_video_list = [str(train_video_list).replace(".txt", "")]
    else:
        train_video_list = [i.replace(".txt", "") for i in train_video_list]
    
    if test_video_list.shape == ():
        test_video_list =  [str(test_video_list).replace(".txt", "")]
    else:
        test_video_list =  [i.replace(".txt", "") for i in test_video_list]

    # import pdb; pdb.set_trace()

    train_data_dict = get_data_dict(
        feature_dir=feature_dir,
        oaf_feature_dir=oaf_feature_dir, 
        label_dir=label_dir, 
        video_list=train_video_list, 
        event_list=event_list, 
        sample_rate=sample_rate, 
        temporal_aug=temporal_aug,
        boundary_smooth=boundary_smooth,
        downsample=downsample
    )

    test_data_dict = get_data_dict(
        feature_dir=feature_dir, 
        oaf_feature_dir=oaf_feature_dir, 
        label_dir=label_dir, 
        video_list=test_video_list, 
        event_list=event_list, 
        sample_rate=sample_rate, 
        temporal_aug=temporal_aug,
        boundary_smooth=boundary_smooth,
        downsample=downsample
    )
    
    train_train_dataset = VideoFeatureDataset(train_data_dict, num_classes, mode='train')
    train_test_dataset = VideoFeatureDataset(train_data_dict, num_classes, mode='test')
    test_test_dataset = VideoFeatureDataset(test_data_dict, num_classes, mode='test')

    # import pdb; pdb.set_trace()

    trainer = Trainer(dict(encoder_params), dict(decoder_params), dict(diffusion_params),
        event_list, sample_rate, temporal_aug, set_sampling_seed, postprocess,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    )

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    run = wandb.init(
        project="inverse_tas",
        name=f"DiffAct_{dataset_name}_split{split_id}_{args.f_type}",
        config=dict(model="DiffAct", dataset=dataset_name, split=split_id,
                    f_type=args.f_type, epoch=args.epoch,
                    num_epochs=num_epochs, batch_size=batch_size,
                    learning_rate=learning_rate, weight_decay=weight_decay),
    )

    trainer.train(train_train_dataset, train_test_dataset, test_test_dataset,
        loss_weights, class_weighting, soft_label,
        num_epochs, batch_size, learning_rate, weight_decay,
        label_dir=label_dir, result_dir=os.path.join(result_dir, f"{naming}_{args.f_type}_{args.epoch}"),
        weights_dir=os.path.join(weights_dir, f"{naming}_{args.f_type}_{args.epoch}"),
        log_freq=log_freq, log_train_results=log_train_results, downsample=downsample,
        wandb_run=run,
    )

    run.finish()
