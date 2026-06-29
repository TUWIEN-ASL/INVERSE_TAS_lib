import os
import json
import copy
from configs.paths import project_path

import argparse

params_gtea = {
   "naming":"default",
   "root_data_dir":"./datasets",
   "dataset_name":"gtea",
   "split_id":1,
   "sample_rate":1,
   "temporal_aug":True,
   "encoder_params":{
      "use_instance_norm":False, 
      "num_layers":10,
      "num_f_maps":64,
      "input_dim":2048,
      "kernel_size":5,
      "normal_dropout_rate":0.5,
      "channel_dropout_rate":0.5,
      "temporal_dropout_rate":0.5,
      "feature_layer_indices":[
         5,
         7,
         9
      ]
   },
   "decoder_params":{
      "num_layers":8,
      "num_f_maps":24,
      "time_emb_dim":512,
      "kernel_size":5,
      "dropout_rate":0.1,
   },
   "diffusion_params":{
      "timesteps":1000,
      "sampling_timesteps":25,
      "ddim_sampling_eta":1.0,
      "snr_scale":0.5,
      "cond_types":  ['full', 'zero', 'boundary03-', 'segment=1', 'segment=1'],
     "detach_decoder": False,
   },
   "loss_weights":{
      "encoder_ce_loss":0.5,
      "encoder_mse_loss":0.1,
      "encoder_boundary_loss":0.0,
      "decoder_ce_loss":0.5,
      "decoder_mse_loss":0.1,
      "decoder_boundary_loss":0.1
   },
   "batch_size":4,
   "learning_rate":0.0005,
   "weight_decay":1e-6,
   "num_epochs":10001,
   "log_freq":100,
   "class_weighting":True,
   "set_sampling_seed":True,
   "boundary_smooth":1,
   "soft_label": 1.4,
   "log_train_results":False,
   "postprocess":{
      "type":"purge",
      "value":3
   },
}

params_50salads = {
   "naming":"default",
   "root_data_dir":"./datasets",
   "dataset_name":"50salads",
   "split_id":1,
   "sample_rate":8,
   "temporal_aug":True,
   "encoder_params":{
      "use_instance_norm":False,
      "num_layers":10,
      "num_f_maps":64,
      "input_dim":2048,
      "kernel_size":5,
      "normal_dropout_rate":0.5,
      "channel_dropout_rate":0.5,
      "temporal_dropout_rate":0.5,
      "feature_layer_indices":[
         5,
         7,
         9
      ]
   },
   "decoder_params":{
      "num_layers":8,
      "num_f_maps":24,
      "time_emb_dim":512,
      "kernel_size":7,
      "dropout_rate":0.1,
   },
   "diffusion_params":{
      "timesteps":1000,
      "sampling_timesteps":25,
      "ddim_sampling_eta":1.0,
      "snr_scale":1.0,
      "cond_types":[
         "full",
         "zero",
         "boundary05-",
         "segment=2",
         "segment=2"
      ],
     "detach_decoder": False,
   },
   "loss_weights":{
      "encoder_ce_loss":0.5,
      "encoder_mse_loss":0.1,
      "encoder_boundary_loss":0.0,
      "decoder_ce_loss":0.5,
      "decoder_mse_loss":0.1,
      "decoder_boundary_loss":0.1
   },
   "batch_size":4,
   "learning_rate":0.0005,
   "weight_decay":0,
   "num_epochs":5001,
   "log_freq":100,
   "class_weighting":True,
   "set_sampling_seed":True,
   "boundary_smooth":20,
   "soft_label": None,
   "log_train_results":False,
   "postprocess":{
      "type":"median", # W
      "value":30 # W
   },
}

params_GEARS = {
   "naming":"default",
   "root_data_dir":f"{project_path}/data",
   "dataset_name":"NIST_gears",
   "split_id":0,
   "sample_rate":8,
   "temporal_aug":True,
   "encoder_params":{
      "use_instance_norm":False,
      "num_layers":10,
      "num_f_maps":64,
      "input_dim":2048,
      "kernel_size":5,
      "normal_dropout_rate":0.5,
      "channel_dropout_rate":0.5,
      "temporal_dropout_rate":0.5,
      "feature_layer_indices":[
         5,
         7,
         9
      ]
   },
   "decoder_params":{
      "num_layers":8,
      "num_f_maps":24,
      "time_emb_dim":512,
      "kernel_size":7,
      "dropout_rate":0.1,
   },
   "diffusion_params":{
      "timesteps":1000,
      "sampling_timesteps":25,
      "ddim_sampling_eta":1.0,
      "snr_scale":0.5,
      "cond_types":[
         "full",
         "zero",
         "boundary05-",
         "segment=2",
         "segment=2"
      ],
     "detach_decoder": False,
   },
   "loss_weights":{
      "encoder_ce_loss":0.5,
      "encoder_mse_loss":0.1,
      "encoder_boundary_loss":0.0,
      "decoder_ce_loss":0.5,
      "decoder_mse_loss":0.1,
      "decoder_boundary_loss":0.1
   },
   "batch_size":4,
   "learning_rate":0.0005,
   "weight_decay":1e-6,
   "num_epochs":5001,
   "log_freq":100,
   "class_weighting":True,
   "set_sampling_seed":True,
   "boundary_smooth":1,
   "soft_label": 1.4,
   "log_train_results":False,
   "postprocess":{
      "type":"median", # W
      "value":30 # W
   },
}

params_breakfast = {
   "naming":"default",
   "root_data_dir":"./datasets",
   "dataset_name":"breakfast",
   "split_id":1,
   "sample_rate":1,
   "temporal_aug":True,
   "encoder_params":{
      "use_instance_norm":False,
      "num_layers":12,
      "num_f_maps":256,
      "input_dim":2048,
      "kernel_size":5,
      "normal_dropout_rate":0.5,
      "channel_dropout_rate":0.1,
      "temporal_dropout_rate":0.1,
      "feature_layer_indices":[
         7,
         8,
         9
      ]
   },
   "decoder_params":{
      "num_layers":8,
      "num_f_maps":128,
      "time_emb_dim":512,
      "kernel_size":5,
      "dropout_rate":0.1
   },
   "diffusion_params":{
      "timesteps":1000,
      "sampling_timesteps":25,
      "ddim_sampling_eta":1.0,
      "snr_scale":0.5,
      "cond_types":[
         "full",
         "zero",
         "boundary03-",
         "segment=1",
         "segment=1"
      ],
      "detach_decoder":False,
   },
   "loss_weights":{
      "encoder_ce_loss":0.5,
      "encoder_mse_loss":0.025,
      "encoder_boundary_loss":0.0,
      "decoder_ce_loss":0.5,
      "decoder_mse_loss":0.025,
      "decoder_boundary_loss":0.1
   },
   "batch_size":4,
   "learning_rate":0.0001,
   "weight_decay":0,
   "num_epochs":1001,
   "log_freq":20,
   "class_weighting":True,
   "set_sampling_seed":True,
   "boundary_smooth":3,
   "soft_label":4,
   "log_train_results":False,
   "postprocess":{
      "type":"median",
      "value":15
   },
}
################################################ REASSEMBLE M2R2
params_M2R2 = {
   "naming":"default",
   "root_data_dir":f"{project_path}/data",
   "dataset_name":"REASSEMBLEmm",
   "split_id":1,
   "sample_rate":1,
   "temporal_aug":True,
   "encoder_params":{
      "use_instance_norm":False,
      "num_layers":10,
      "num_f_maps":64,
      "input_dim":512,
      "kernel_size":5,
      "normal_dropout_rate":0.5,
      "channel_dropout_rate":0.5,
      "temporal_dropout_rate":0.5,
      "feature_layer_indices":[
         5,
         7,
         9
      ]
   },
   "decoder_params":{
      "num_layers":8,
      "num_f_maps":24,
      "time_emb_dim":512,
      "kernel_size":7,
      "dropout_rate":0.1,
   },
   "diffusion_params":{
      "timesteps":1000,
      "sampling_timesteps":25,
      "ddim_sampling_eta":1.0,
      "snr_scale":0.5,
      "cond_types":[
         "full",
         "zero",
         "boundary05-",
         "segment=2",
         "segment=2"
      ],
     "detach_decoder": False,
   },
   "loss_weights":{
      "encoder_ce_loss":0.5,
      "encoder_mse_loss":0.1,
      "encoder_boundary_loss":0.0,
      "decoder_ce_loss":0.5,
      "decoder_mse_loss":0.1,
      "decoder_boundary_loss":0.1
   },
   "batch_size":4,
   "learning_rate":0.0005,
   "weight_decay":1e-6,
   "num_epochs":501,
   "log_freq":100,
   "class_weighting":True,
   "set_sampling_seed":True,
   "boundary_smooth":1,
   "soft_label": 1.4,
   "log_train_results":False,
   "postprocess":{
      "type":"median", # W
      "value":30 # W
   },
}

parser = argparse.ArgumentParser()
parser.add_argument("--dataset",help="dataset name")
parser.add_argument("--split",help="split number")
parser.add_argument("--f_type",help="feature type")
args = parser.parse_args()

###################### REASSEMBLE mm #######################
    
params = copy.deepcopy(params_M2R2)

split_id = int(args.split)
params['split_id'] = split_id
params['naming'] = f'{args.dataset}-Trained-S{args.split}'
params['dataset_name'] = args.dataset

if args.f_type == "I3D_features":
   params['encoder_params']["input_dim"] = 2048
elif args.f_type == "BRP_features":
   params['encoder_params']["input_dim"] = 768
elif args.f_type == "BRPOAF":
   params['encoder_params']["input_dim"] = 2*768
elif "M2R2" in args.f_type:
   params['encoder_params']["input_dim"] = 512
else:
   raise RuntimeError("Unrecognized feature type")

if not os.path.exists('configs'):
   os.makedirs('configs')

file_name = os.path.join(project_path, 'configs', "DiffAct", f'{params["naming"]}.json')

with open(file_name, 'w') as outfile:
   json.dump(params, outfile, ensure_ascii=False)


###################### Custom #######################

# params = copy.deepcopy(params_GEARS)

# params['split_id'] = int(args.split)
# params['naming'] = f'{args.dataset}-Trained-S{args.split}'
# params['dataset_name'] = args.dataset

# if args.f_type == "I3D_features":
#    params['encoder_params']["input_dim"] = 2048
# elif args.f_type == "BRP_features":
#    params['encoder_params']["input_dim"] = 768
# else:
#    raise RuntimeError("Unrecognized feature type")

# if not os.path.exists('configs'):
#    os.makedirs('configs')

# file_name = os.path.join(project_path, 'configs', "DiffAct", f'{params["naming"]}.json')

# with open(file_name, 'w') as outfile:
#    json.dump(params, outfile, ensure_ascii=False)


###################### GTEA #######################

# split_num = 4

# for split_id in range(1, split_num+1):
    
#     params = copy.deepcopy(params_gtea)

#     params['split_id'] = split_id
#     params['naming'] = f'GTEA-Trained-S{split_id}'

#     if not os.path.exists('configs'):
#         os.makedirs('configs')
     
#     file_name = os.path.join(project_path, 'configs', "DiffAct", f'{params["naming"]}.json')

#     with open(file_name, 'w') as outfile:
#         json.dump(params, outfile, ensure_ascii=False)


# ###################### 50salads #######################

# split_num = 5

# for split_id in range(1, split_num+1):
    
#     params = copy.deepcopy(params_50salads)

#     params['split_id'] = split_id
#     params['naming'] = f'50salads-Trained-S{split_id}'

#     if not os.path.exists('configs'):
#         os.makedirs('configs')
     
#     file_name = os.path.join(project_path, 'configs', "DiffAct", f'{params["naming"]}.json')

#     with open(file_name, 'w') as outfile:
#         json.dump(params, outfile, ensure_ascii=False)


# ###################### Breakfast #######################

# split_num = 4

# for split_id in range(1, split_num+1):
    
#     params = copy.deepcopy(params_breakfast)

#     params['split_id'] = split_id
#     params['naming'] = f'Breakfast-Trained-S{split_id}'

#     if not os.path.exists('configs'):
#         os.makedirs('configs')
     
#     file_name = os.path.join(project_path, 'configs', "DiffAct", f'{params["naming"]}.json')

#     with open(file_name, 'w') as outfile:
#         json.dump(params, outfile, ensure_ascii=False)
