import os
import torch


def epoch_saving(epoch, model, fusion_model, optimizer, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'fusion_model_state_dict': fusion_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, filename)


def epoch_saving_seg(epoch, model, fusion_model, frame_fusion_model, optimizer, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'fusion_model_state_dict': fusion_model.state_dict(),
        'frame_fusion_model_state_dict': frame_fusion_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, filename)


def best_saving(working_dir, epoch, model, fusion_model, optimizer):
    os.makedirs(working_dir, exist_ok=True)
    best_name = '{}/model_best.pt'.format(working_dir)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'fusion_model_state_dict': fusion_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, best_name)
