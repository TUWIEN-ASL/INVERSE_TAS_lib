from configs.paths import project_path
import os
import subprocess
import glob
import numpy as np
import torch

from models.video_features.models.i3d.extract_i3d import ExtractI3D
from models.BridgePrompt.extractor import BRP_Extractor
from types import SimpleNamespace


class Predictor:
    def __init__(self,
                 feature_extractor,
                 segmentation_model,
                 device,
                 stack_size,
                 step_size,
                 brp_config_path,
                 segmentation_model_config,
                 segmentation_model_weights,
                 num_classes,
                 mapping_path,
                 ):
        self.device = device
        self.stack_size = stack_size
        self.step_size = step_size
        
        file_ptr = open(mapping_path, 'r')
        actions = file_ptr.read().split('\n')[:-1]
        file_ptr.close()
        self.actions_dict = dict()
        for a in actions:
            self.actions_dict[a.split()[1]] = int(a.split()[0])

        if feature_extractor == "I3D":
            i3d_args = SimpleNamespace()
            i3d_args.feature_type = "i3d"
            i3d_args.on_extraction = "print"
            i3d_args.tmp_path = "."
            i3d_args.output_path = "."
            i3d_args.keep_tmp_files = False
            i3d_args.device = "cuda:0"
            i3d_args.flow_type = "raft"
            i3d_args.extraction_fps = 15
            i3d_args.step_size = step_size
            i3d_args.stack_size = stack_size
            i3d_args.show_pred = False
            i3d_args.streams = None
            self.i3d_extractor = ExtractI3D(i3d_args)
            self.f_extractor = self.extract_features_I3D
            features_dim = 2048
        elif feature_extractor == "BRP":
            self.brp_extractor = BRP_Extractor(brp_config_path)
            self.f_extractor = self.extract_features_BRPrompt
            features_dim = 768
        else:
            raise RuntimeError("unrecofnized feature extracotr")
        
        if segmentation_model == "MSTCN":
            from models.mstcn.model import Segmentator
            num_stages = 4
            num_layers = 10
            num_f_maps = 64
                
            self.seg_model = Segmentator(segmentation_model_weights, num_stages, num_layers, num_f_maps, features_dim, num_classes, device)
        elif segmentation_model == "ASFormer":
            from models.ASFormer.model import Segmentator
            
            num_layers = 10
            num_f_maps = 64
            channel_mask_rate = 0.3
            
            self.seg_model = Segmentator(segmentation_model_weights, num_layers, 2, 2, num_f_maps, features_dim, num_classes, channel_mask_rate, device)
        elif segmentation_model == "DiffAct":
            from models.DiffAct.model import Segmentator
            from models.DiffAct.utils import load_config_file
            
            all_params = load_config_file(segmentation_model_config)
            self.seg_model = Segmentator(segmentation_model_weights, all_params["encoder_params"], all_params["decoder_params"], all_params["diffusion_params"], num_classes, all_params["sample_rate"], all_params["temporal_aug"], all_params["set_sampling_seed"], all_params["postprocess"], device)
        elif segmentation_model == "ASRF":
            from models.asrf.predictor import Segmentator
            from models.asrf.libs.config import get_config
            
            all_params = get_config(segmentation_model_config)
            self.seg_model = Segmentator(segmentation_model_weights, all_params, num_classes, device, "refinement_with_boundary")


    def extract_features_I3D(self, path):
        features = self.i3d_extractor.extract(path)
        stacked = np.concatenate([features["rgb"], features["flow"]], axis=-1).T
        stacked = torch.tensor(stacked)
        
        return stacked

    def extract_features_BRPrompt(self, path):
        features = self.brp_extractor.extract(path).squeeze().T
        return features

    def predict(self, path, save=False):
        features = self.f_extractor(path)
        preds = self.seg_model.predict(features)
        
        if save:
            path_save, _ = os.path.splitext(path)
            path_save += ".txt"
            
            self.save_pred(preds, path_save)
        
        return preds
    
    def save_pred(self, preds, path):        
        recognition = []
        for i in range(len(preds)):
            recognition = np.concatenate((recognition, [list(self.actions_dict.keys())[list(self.actions_dict.values()).index(preds[i].item())]]))
        
        f_ptr = open(path, "w")
        f_ptr.write(' '.join(recognition))
        f_ptr.close()

if __name__ == "__main__":
    
    pred = Predictor("I3D", "MSTCN", "cuda:0", 21, 750, "/home/dsliwowski/Projects/Code_Inverse/configs/BRP/gears/gears_exfm.yaml", None, "/home/dsliwowski/Projects/Code_Inverse/catkin/src/inverse_tas/checkpoints/mstcn/test/split_0/epoch-1.model", 13, "/home/dsliwowski/Projects/Code_Inverse/catkin/src/inverse_tas/data/test/mapping.txt")
    # pred = Predictor("BRP", "ASFormer", "cuda:0", 21, 500, "/home/dsliwowski/Projects/Code_Inverse/configs/BRP/gears/gears_exfm.yaml", None, "/home/dsliwowski/Projects/Code_Inverse/checkpoints/ASFormer/NIST_gears/split_0/epoch-50.model", 13)
    # pred = Predictor("I3D", "DiffAct", "cuda:0", 21, 1000, "/home/dsliwowski/Projects/Code_Inverse/configs/BRP/gears/gears_exfm.yaml", "/home/dsliwowski/Projects/Code_Inverse/configs/DiffAct/GEARS-Trained-S0.json", "/home/dsliwowski/Projects/Code_Inverse/checkpoints/DiffAct/GEARS-Trained-S0/epoch-0.model", 13)
    # pred = Predictor("BRP", "ASRF", "cuda:0", 21, 1000, "/home/dsliwowski/Projects/Code_Inverse/configs/BRP/gears/gears_exfm.yaml", "/home/dsliwowski/Projects/Code_Inverse/configs/ASRF/in_channel-768_dataset-NIST_gears/config.yaml", "/home/dsliwowski/Projects/Code_Inverse/checkpoints/ASRF/final_model.prm", 13)
    # pred.extract_features_I3D("/home/dsliwowski/Projects/Code_Inverse/data/deployed/videos/2024-09-24-15-24-05_camera.mp4")
    res= pred.predict("/home/dsliwowski/Projects/Code_Inverse/catkin/src/inverse_tas/data/test/videos/2024-09-24-15-24-05_camera.mp4", True)
    print(res)