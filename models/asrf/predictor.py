import argparse
import os

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from models.asrf.libs import models
from models.asrf.libs.class_id_map import get_n_classes
from models.asrf.libs.config import get_config
from models.asrf.libs.dataset import ActionSegmentationDataset, collate_fn
from models.asrf.libs.helper import evaluate
from models.asrf.libs.transformer import TempDownSamp, ToTensor
from models.asrf.libs.postprocess import PostProcessor


class Segmentator():
    def __init__(self, model_path, config, n_classes, device, refinement_method):
        self.model = models.ActionSegmentRefinementFramework(
            in_channel=config.in_channel,
            n_features=config.n_features,
            n_classes=n_classes,
            n_stages=config.n_stages,
            n_layers=config.n_layers,
            n_stages_asb=config.n_stages_asb,
            n_stages_brb=config.n_stages_brb,
        )

        # load the state dict of the model
        state_dict = torch.load(model_path)
        self.model.load_state_dict(state_dict)
        self.model.to(device)
        self.model.eval()
        self.device = device
        
        self.postprocessor = PostProcessor(refinement_method, config.boundary_th)
        
    def predict(self, features):
        features.unsqueeze_(0)
        x = features.float()
        mask = torch.ones([1, 1, x.shape[-1]], dtype=bool)

        x = x.to(self.device)
        mask = mask.to(self.device)

        # compute output and loss
        output_cls, output_bound = self.model(x)

        # calcualte accuracy and f1 score
        output_cls = output_cls.to("cpu").data.numpy()
        output_bound = output_bound.to("cpu").data.numpy()

        x = x.to("cpu").data.numpy()
        mask = mask.to("cpu").data.numpy()

        refined_output_cls = self.postprocessor(output_cls, boundaries=output_bound, masks=mask)
        
        return refined_output_cls
