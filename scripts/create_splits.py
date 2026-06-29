import glob
import os.path as osp
import os
import argparse
from sklearn.model_selection import StratifiedKFold, train_test_split

# Setup argument parsing
parser = argparse.ArgumentParser(description="Generate splits and annotations paths.")
parser.add_argument('base_dir', type=str, help='Base directory path')
parser.add_argument('n_splits', type=int, help='Base directory path')
args = parser.parse_args()

# Define paths based on base_dir
splits_path = osp.join(args.base_dir, "splits")
os.makedirs(splits_path, exist_ok=True)

annot_path = osp.join(args.base_dir, "annotations")
annot_paths = glob.glob(osp.join(annot_path, "*.txt"))

annot_names = [path.split("/")[-1].split(".")[0] for path in annot_paths if "hand" in path]

train, test = train_test_split(annot_names)

train_path = osp.join(splits_path, f"train.split1.bundle")
test_path = osp.join(splits_path, f"test.split1.bundle")

# Write train and test splits to respective files
with open(train_path, "w") as file:
    for item in train:
        file.write(item + ".txt" + "\n")

with open(test_path, "w") as file:
    for item in test:
        file.write(item + ".txt" + "\n")
