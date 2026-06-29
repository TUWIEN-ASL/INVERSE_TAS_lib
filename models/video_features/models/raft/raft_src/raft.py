"""
    Reference: https://github.com/princeton-vl/RAFT/tree/25eb2ac723c36865c636c9d1f497af8023981868
    Modified by Vladimir Iashin for github.com/v-iashin/video_features
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.video_features.models.raft.raft_src.update import BasicUpdateBlock, SmallUpdateBlock
from models.video_features.models.raft.raft_src.extractor import BasicEncoder, SmallEncoder
from models.video_features.models.raft.raft_src.corr import CorrBlock, AlternateCorrBlock
from models.video_features.models.raft.raft_src.utils.utils import bilinear_sampler, coords_grid, upflow8, coords_grid_gpu
from models.video_features.models.raft.raft_src.corr_fast import CorrBlockFast


try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass

        def __enter__(self):
            pass

        def __exit__(self, *args):
            pass


class InputPadder:
    """ Pads images such that dimensions are divisible by 8"""

    def __init__(self, dims, mode='sintel'):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // 8) + 1) * 8 - self.ht) % 8
        pad_wd = (((self.wd // 8) + 1) * 8 - self.wd) % 8
        if mode == 'sintel':
            self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, pad_ht // 2, pad_ht - pad_ht // 2]
        else:
            self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, 0, pad_ht]

    def pad(self, input):
        return F.pad(input, self._pad, mode='replicate')

    def unpad(self, x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht - self._pad[3], self._pad[0], wd - self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]


class RAFT(nn.Module):

    # (v-iashin) def __init__(self, model_is_small):
    def __init__(self):
        super(RAFT, self).__init__()
        self.dropout = 0
        self.alternate_corr = False
        self.model_is_small = False
        self.mixed_precision = False

        if self.model_is_small:
            self.corr_levels = 4
            self.corr_radius = 3
            self.hidden_dim = 96
            self.context_dim = 64
            self.cnet_out_dim = self.hidden_dim + self.context_dim

        else:
            self.corr_levels = 4
            self.corr_radius = 4
            self.hidden_dim = 128
            self.context_dim = 128
            self.cnet_out_dim = self.hidden_dim + self.context_dim

        # feature network, context network, and update block
        if self.model_is_small:
            self.fnet = SmallEncoder(output_dim=128, norm_fn='instance', dropout=self.dropout)
            self.cnet = SmallEncoder(output_dim=self.cnet_out_dim, norm_fn='none', dropout=self.dropout)
            self.update_block = SmallUpdateBlock(
                self.corr_levels, self.corr_radius, hidden_dim=self.hidden_dim
            )

        else:
            self.fnet = BasicEncoder(output_dim=256, norm_fn='instance', dropout=self.dropout)
            self.cnet = BasicEncoder(output_dim=self.cnet_out_dim, norm_fn='batch', dropout=self.dropout)
            self.update_block = BasicUpdateBlock(
                self.corr_levels, self.corr_radius, hidden_dim=self.hidden_dim
            )

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape

        coords0 = coords_grid_gpu(N, H // 8, W // 8, img.device)
        coords1 = coords_grid_gpu(N, H // 8, W // 8, img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8 * H, 8 * W)

    # (v-iashin) def forward(self, image1, image2, iters=12, flow_init=None, upsample=True, test_mode=False):
    def forward(self, image1, image2, iters=10, flow_init=None, upsample=True, test_mode=True):
        """ Estimate optical flow between pair of frames """
        # import time

        # pre_s = time.process_time_ns()
        image1 = 2 * (image1 / 255.0) - 1.0
        image2 = 2 * (image2 / 255.0) - 1.0

        image1 = image1.contiguous()
        image2 = image2.contiguous()

        hdim = self.hidden_dim
        cdim = self.context_dim
        # pre_e = time.process_time_ns()
        # print("\t\t preprocess", (pre_e-pre_s)/10**9)
        # run the feature network
        # fmap_s = time.process_time_ns()
        with autocast(enabled=self.mixed_precision):
            fmap1, fmap2 = self.fnet([image1, image2])

        fmap1 = fmap1.float()
        fmap2 = fmap2.float()
        # fmap_e = time.process_time_ns()
        # print("\t\t Fmap ", (fmap_e-fmap_s)/10**9)

        # corr_s = time.process_time_ns()
        if self.alternate_corr:
            corr_fn = AlternateCorrBlock(fmap1, fmap2, radius=self.corr_radius)
        else:
            # corr_fn = CorrBlock(fmap1, fmap2, radius=self.corr_radius)
            corr_fn_fast = CorrBlockFast(fmap1, fmap2, radius=self.corr_radius)
        # corr_e = time.process_time_ns()
        # print("\t\t Corr ", (corr_e-corr_s)/10**9)

        # run the context network
        # cont_s = time.process_time_ns()
        with autocast(enabled=self.mixed_precision):
            cnet = self.cnet(image1)
            net, inp = torch.split(cnet, [hdim, cdim], dim=1)
            net = torch.tanh(net)
            inp = torch.relu(inp)
        # cont_e = time.process_time_ns()
        # print("\t\t cont ", (cont_e-cont_s)/10**9)

        # init_s = time.process_time_ns()
        coords0, coords1 = self.initialize_flow(image1)
        # init_e = time.process_time_ns()
        # print("\t\t Init ", (init_e-init_s)/10**9)

        if flow_init is not None:
            coords1 = coords1 + flow_init

        # flow_pred_s = time.process_time_ns()
        flow_predictions = []
        for itr in range(iters):
            # s_corel = time.process_time_ns()
            # coords1 = coords1.detach()
            # corr = corr_fn(coords1)  # index correlation volume
            # e_corel = time.process_time_ns()
            # print("\t\t\t corelation ", (e_corel-s_corel)/10**9)

            # s_corel_fast = time.process_time_ns()
            coords1 = coords1.detach()
            corr = corr_fn_fast(coords1)
            # e_corel_fast = time.process_time_ns()
            # print("\t\t\t corelation ", (e_corel_fast-s_corel_fast)/10**9)

            # if (corr - corr_f).norm().item() != 0:
            #     import pdb
            #     pdb.set_trace()

            # s_flow_comp = time.process_time_ns()
            flow = coords1 - coords0
            with autocast(enabled=self.mixed_precision):
                net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow
            # e_flow_comp = time.process_time_ns()
            # print("\t\t\t flow comp delta ", (e_flow_comp-s_flow_comp)/10**9)

            # s_ups = time.process_time_ns()
            # upsample predictions
            if up_mask is None:
                flow_up = upflow8(coords1 - coords0)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)
            # e_ups = time.process_time_ns()
            # print("\t\t\t upsample", (e_ups-s_ups)/10**9)

            flow_predictions.append(flow_up)

        # flow_pred_e = time.process_time_ns()
        # print("\t\t flow pred", (flow_pred_e-flow_pred_s)/10**9)

        # import pdb
        # pdb.set_trace()

        if test_mode:
            # (v-iashin) return coords1 - coords0, flow_up
            return flow_up

        return flow_predictions
