import argparse
import subprocess
from configs.paths import project_path
import os

import numpy as np
import os
import glob
import pandas as pd
import json

def combine(dataset, base_path):
    ds_path = os.path.join(base_path, dataset)
    feat_root = f"{ds_path}/brp_raw" #'./data/'+dataset+'/features_dir/'+feat_name
    final_root = f"{ds_path}/BRP_features"
    os.makedirs(final_root, exist_ok=True)

    frames_dir = os.path.join(ds_path, "frames")
    vids = os.listdir(frames_dir)

    v_vlen = {}

    for v in vids:
        v_vlen[v] = len(os.listdir(os.path.join(frames_dir, v)))


    feats = glob.glob(feat_root + '/*')
    feats = [x for x in feats if x.endswith('.npy')]
    if not os.path.exists(final_root):
        os.mkdir(final_root)

    df = pd.DataFrame(feats, columns=['paths'])
    df['vid'] = [d.rsplit('/', 1)[1].rsplit('_', 1)[0] for d in df.paths]
    df['ind'] = [d.rsplit('_', 1)[1][:-4] for d in df.paths]
    df['ind'] = df['ind'].astype(int)
    for name, group in df.groupby('vid'):
        group.sort_values('ind', inplace=True)
        vlen = v_vlen[name]
        result = np.zeros((vlen, 768))
        for index, row in group.iterrows():
            tfeat = np.load(row.paths)
            if index == group.index[-1]:
                diff = vlen - row.ind
                result[row.ind:, :] = tfeat[:diff, :]
            else:
                result[row.ind:row.ind + 32, :] = tfeat
        np.save(os.path.join(final_root, name + '.npy'), result)
        print(name + ' is combined.')

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run extract_frame_features.py with specified config and dataset.")
    parser.add_argument(
        "--dataset_name", type=str, required=True, help="Name of the dataset."
    )
    parser.add_argument(
        "--config_path", type=str, required=True, help="Path to the config file."
    )
    parser.add_argument(
        "--base_path", type=str, required=True, help="Path to dataset root"
    )

    # Parse arguments
    args = parser.parse_args()

    # Construct the command
    # command = [
    #     "python", os.path.join(project_path, "models", "BridgePrompt", "extract_frame_features.py"), 
    #     "--config", args.config_path,
    #     "--dataset", args.dataset_name
    # ]

    # # Execute the command
    # try:
    #     subprocess.run(command, check=True)
    # except subprocess.CalledProcessError as e:
    #     print(f"Error occurred while executing the script: {e}")

    combine(args.dataset_name, args.base_path)

if __name__ == "__main__":
    main()