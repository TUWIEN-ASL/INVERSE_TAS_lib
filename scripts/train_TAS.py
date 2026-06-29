import argparse
from configs.paths import project_path
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default="ImPerfectPour")
parser.add_argument('--split', default='1')
parser.add_argument('--f_type', default='BRP')
parser.add_argument('--model', default='mstcn')
parser.add_argument('--oaf_type', default='')
parser.add_argument('--hasr_backbone', default='mstcn', choices=['mstcn', 'ASFormer', 'onlinetas'],
                    help='Backbone for HASR refiner (only used when --model HASR)')
parser.add_argument('--hasr_refiner', default='gru', choices=['transformer', 'gru'],
                    help='Refiner architecture: transformer or original GRU (default)')
parser.add_argument('--hasr_online', action='store_true',
                    help='Use online training strategy (requires --hasr_backbone onlinetas)')
parser.add_argument('--hasr_prefix_strategy', default='clip_boundaries',
                    choices=['random', 'clip_boundaries'],
                    help='Prefix-sampling strategy for online HASR training')

args = parser.parse_args()

if args.f_type == "I3D":
    ftype = "I3D_features"
    fsize = 2048
elif args.f_type == "BRP":
    ftype = "BRP_features"
    fsize = 768
elif "M2R2" in args.f_type:
    ftype = args.f_type
    fsize = 512
elif "BRPOAF" in args.f_type:
    ftype = "BRPOAF"
    fsize = 768 + 768

elif "UM2R" in args.f_type:
    ftype = args.f_type
    fsize = 512

if args.oaf_type != "" and "UM2R" not in args.f_type:
    fsize += 768


if args.model == "mstcn":
    # for i in ["1", "2", "3", "4"]:
    main_path = f"{project_path}/models/mstcn/main.py"
    eval_path = f"{project_path}/models/mstcn/eval.py"
    
    train_cmd = ["python", main_path, "--dataset", args.dataset, "--split", args.split, "--f_type", ftype, "--action", "train"]
    pred_cmd = ["python", main_path, "--dataset", args.dataset, "--split", args.split, "--f_type", ftype, "--action", "predict"]
    eval_cmd = ["python", eval_path, "--dataset", args.dataset, "--split", args.split, "--f_type", ftype]
    
    # Execute the command
    try:
        for cmd in [train_cmd, pred_cmd, eval_cmd]:
            subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while executing the script: {e}")

elif args.model == "ASFormer":
    main_path = f"{project_path}/models/ASFormer/main.py"
    eval_path = f"{project_path}/models/ASFormer/eval.py"
    
    train_cmd = ["python", main_path, "--dataset", args.dataset, "--split", args.split, "--f_type", ftype, "--action", "train"]
    pred_cmd = ["python", main_path, "--dataset", args.dataset, "--split", args.split, "--f_type", ftype, "--action", "predict"]
    eval_cmd = ["python", eval_path, "--dataset", args.dataset, "--split", args.split, "--f_type", ftype]

    # Execute the command
    try:
        for cmd in [train_cmd, pred_cmd, eval_cmd]:
            subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while executing the script: {e}")
        
elif args.model == "DiffAct":
    for i in ["1"]:
        config_create_path = f"{project_path}/models/DiffAct/default_configs.py"
        main_path = f"{project_path}/models/DiffAct/main.py"

        config_path = f"{project_path}/configs/DiffAct/{args.dataset}-Trained-S{i}.json"
        
        config_cmd = ["python", config_create_path, "--dataset", args.dataset, "--split", i, "--f_type", ftype]


        for epoch in range(100, 0, -10):    
            train_cmd = ["python", main_path, "--device", "0", "--config", config_path, "--f_type", ftype, "--epoch", str(epoch)]
            
            # Execute the command
            try:
                for cmd in [config_cmd, train_cmd]:
                    subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error occurred while executing the script: {e}")
        
elif args.model == "asrf":
    for epoch in range(100, 0, -10):
        print("Running for epoch: ", epoch, flush=True)
        generate_gt_path = f"{project_path}/models/asrf/utils/generate_gt_array.py"
        boundary_create_path = f"{project_path}/models/asrf/utils/generate_boundary_array.py"
        csv_create_path = f"{project_path}/models/asrf/utils/make_csv_files.py"
        config_create_path = f"{project_path}/models/asrf/utils/make_configs.py"
        main_path = f"{project_path}/models/asrf/train.py"
        eval_path = f"{project_path}/models/asrf/evaluate.py"
        
        oaf_type = args.oaf_type + f"/epoch-{epoch}"

        config_path = f"{project_path}/configs/ASRF/in_channel-{fsize}_dataset-{args.dataset}_split-{args.split}/config.yaml"
        # print(config_path)
        gt_cmd = ['python', generate_gt_path, '--dataset', args.dataset, '--dataset_dir', f"{project_path}/data"]
        boundary_cmd = ['python', boundary_create_path, '--dataset', args.dataset, '--dataset_dir', f"{project_path}/data"]
        cvs_cmd = ['python', csv_create_path, '--dataset', args.dataset, '--dataset_dir', f"{project_path}/data", '--f_type', ftype, "--oaf_type", oaf_type]
        config_cmd =['python', config_create_path, '--root_dir', f'{project_path}/configs/ASRF', '--in_channel', str(fsize), '--dataset', args.dataset, '--split', args.split]
        train_cmd = ["python", main_path, config_path, "--epoch", str(epoch)]
        eval_cmd  = ['python', eval_path, config_path, '--model', f"{project_path}/checkpoints/ASRF/{epoch}/best_f150_model.prm"]
        
        # Execute the command
        try:
            for cmd in [gt_cmd,boundary_cmd,cvs_cmd,config_cmd, eval_cmd]:
            # for cmd in [eval_cmd]:
                subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error occurred while executing the script: {e}")
elif args.model == "BaFormer":
    for i in ["1", "2", "3", "4"]:
        main_path = f"{project_path}/models/BaFormer/main.py"
        eval_path = f"{project_path}/models/BaFormer/Inference.py"
        train_cmd = [
            "python",
            main_path,
            "--config",
            "/home/dsliwowski/Projects/Code_Inverse/catkin/src/inverse_tas/models/BaFormer/configs/framed_en_de.yaml",
            "dataset.name", "50Salads",
            "dataset.split", i,
            "dataset.dir", 
            "/home/dsliwowski/Projects/Code_Inverse/catkin/src/inverse_tas/data/50Salads"]
        eval_cmd  = [
            "python",
            eval_path,
            "--config",
            "--checkpoint",
            "experiment/checkpoints/model_best.pth"
            "/home/dsliwowski/Projects/Code_Inverse/catkin/src/inverse_tas/models/BaFormer/configs/framed_en_de.yaml",
            "dataset.name", "50Salads",
            "dataset.split", i,
            "dataset.dir", 
            "/home/dsliwowski/Projects/Code_Inverse/catkin/src/inverse_tas/data/50Salads"]
        try:
            for cmd in [train_cmd, eval_cmd]:
                print("Runnig:", " ".join(cmd))
            # for cmd in [eval_cmd]:
                subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error occurred while executing the script: {e}")

elif args.model == "onlinetas":
    main_path = f"{project_path}/models/onlinetas/main.py"
    eval_path = f"{project_path}/models/onlinetas/eval.py"

    train_cmd = ["python", main_path, "--dataset", args.dataset, "--split", args.split,
                 "--f_type", ftype, "--action", "train"]
    pred_cmd  = ["python", main_path, "--dataset", args.dataset, "--split", args.split,
                 "--f_type", ftype, "--action", "predict"]
    eval_cmd  = ["python", eval_path, "--dataset", args.dataset, "--split", args.split,
                 "--f_type", ftype]

    try:
        for cmd in [pred_cmd, eval_cmd]:
            subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while executing the script: {e}")

elif args.model == "HASR":
    main_path = f"{project_path}/models/HASR/main.py"
    eval_path = f"{project_path}/models/HASR/eval.py"

    _common = [
        "--dataset",  args.dataset,
        "--split",    args.split,
        "--f_type",   ftype,
        "--backbone", args.hasr_backbone,
        "--refiner",  args.hasr_refiner,
        "--backbone_epoch", "42",
    ]

    if args.hasr_online:
        train_cmd = ["python", main_path, "--action", "online_train",
                     "--prefix_strategy", args.hasr_prefix_strategy] + _common
        pred_cmd  = ["python", main_path, "--action", "online_infer"] + _common
    else:
        train_cmd = ["python", main_path, "--action", "train"]    + _common
        pred_cmd  = ["python", main_path, "--action", "predict"]  + _common

    eval_cmd = [
        "python", eval_path,
        "--backbone",   args.hasr_backbone,
        "--refiner",    args.hasr_refiner,
        "--dataset",    args.dataset,
        "--split",      args.split,
        "--f_type",     ftype,
        "--num_epochs", "50",
    ] + (["--online"] if args.hasr_online else [])

    try:
        for cmd in [train_cmd, pred_cmd, eval_cmd]:
            subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while executing the script: {e}")

else:
    raise RuntimeError("Unrecognized TAS model")
