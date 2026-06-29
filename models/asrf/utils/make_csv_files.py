import argparse
import glob
import os

import pandas as pd

from configs.paths import project_path

def get_arguments() -> argparse.Namespace:
    """
    parse all the arguments from command line inteface
    return a list of parsed arguments
    """

    parser = argparse.ArgumentParser(
        description="make csv files for training and testing."
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="./dataset",
        help="path to a dataset directory (default: ./dataset)",
    )
    parser.add_argument(
        "--f_type",
        type=str,
        help="Type of used feature",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="./dataset",
        help="path to a dataset directory (default: ./dataset)",
    )
    parser.add_argument(
        "--oaf_type",
        type=str,
        help="Type of used feature",
    )

    return parser.parse_args()


def main() -> None:
    args = get_arguments()

    # create csv directory
    csv_dir = f"{project_path}/models/asrf/csv"
    if not os.path.exists(csv_dir):
        os.mkdir(csv_dir)

    feature_type = args.f_type
    oaf_feature_type = args.oaf_type
    datasets = [args.dataset]

    for dataset in datasets:
        # make directory for saving csv files
        save_dir = os.path.join(csv_dir, dataset)

        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        train_splits_paths = glob.glob(
            os.path.join(args.dataset_dir, dataset, "splits", "train.split1.bundle")
        )
        test_splits_paths = glob.glob(
            os.path.join(args.dataset_dir, dataset, "splits", "test.split1.bundle")
        )

        train_splits_paths = sorted(train_splits_paths)
        test_splits_paths = sorted(test_splits_paths)

        for i in range(len(train_splits_paths)):
            with open(train_splits_paths[i], "r") as f:
                train_ids = f.read().split("\n")[:-1]

            # remove .txt from file name of train_ids
            train_ids = [train_id for train_id in train_ids]

            train_feature_paths = []
            train_oaf_paths = []
            train_label_paths = []
            train_boundary_paths = []
            val_feature_paths = []
            val_oaf_paths = []
            val_label_paths = []
            val_boundary_paths = []

            # split train and val
            for j in range(len(train_ids)):
                if j % 10 == 9:
                    val_feature_paths.append(
                        os.path.join(
                            args.dataset_dir, dataset, feature_type, train_ids[j] + ".npy"
                        )
                    )
                    val_label_paths.append(
                        os.path.join(
                            args.dataset_dir, dataset, "gt_arr", train_ids[j] + ".npy"
                        )
                    )
                    val_boundary_paths.append(
                        os.path.join(
                            args.dataset_dir,
                            dataset,
                            "gt_boundary_arr",
                            train_ids[j] + ".npy",
                        )
                    )
                    val_oaf_paths.append(
                        os.path.join(
                            args.dataset_dir, dataset, oaf_feature_type, train_ids[j] + ".npy"
                        )
                    )
                else:
                    train_feature_paths.append(
                        os.path.join(
                            args.dataset_dir, dataset, feature_type, train_ids[j] + ".npy"
                        )
                    )
                    train_label_paths.append(
                        os.path.join(
                            args.dataset_dir, dataset, "gt_arr", train_ids[j] + ".npy"
                        )
                    )
                    train_boundary_paths.append(
                        os.path.join(
                            args.dataset_dir,
                            dataset,
                            "gt_boundary_arr",
                            train_ids[j] + ".npy",
                        )
                    )
                    train_oaf_paths.append(
                        os.path.join(
                            args.dataset_dir, dataset, oaf_feature_type, train_ids[j] + ".npy"
                        )
                    )

            # test data list
            with open(test_splits_paths[i], "r") as f:
                test_ids = f.read().split("\n")[:-1]

            # remove .txt from file name of test_ids
            test_ids = [test_id for test_id in test_ids]

            test_feature_paths = [
                os.path.join(args.dataset_dir, dataset, feature_type, test_id + ".npy")
                for test_id in test_ids
            ]
            test_label_paths = [
                os.path.join(args.dataset_dir, dataset, "gt_arr", test_id + ".npy")
                for test_id in test_ids
            ]
            test_boundary_paths = [
                os.path.join(
                    args.dataset_dir, dataset, "gt_boundary_arr", test_id + ".npy"
                )
                for test_id in test_ids
            ]
            test_oaf_paths = [
                os.path.join(
                    args.dataset_dir, dataset, oaf_feature_type, test_id + ".npy"
                )
                for test_id in test_ids
            ]

            # make dataframe to save csv files
            train_df = pd.DataFrame(
                {
                    "feature": train_feature_paths,
                    "label": train_label_paths,
                    "boundary": train_boundary_paths,
                    "oaf": train_oaf_paths if oaf_feature_type != "" else None
                },
                columns=["feature", "label", "boundary", "oaf"],
            )

            val_df = pd.DataFrame(
                {
                    "feature": val_feature_paths,
                    "label": val_label_paths,
                    "boundary": val_boundary_paths,
                    "oaf": val_oaf_paths if oaf_feature_type != "" else None
                },
                columns=["feature", "label", "boundary", "oaf"],
            )

            test_df = pd.DataFrame(
                {
                    "feature": test_feature_paths,
                    "label": test_label_paths,
                    "boundary": test_boundary_paths,
                    "oaf": test_oaf_paths if oaf_feature_type != "" else None
                },
                columns=["feature", "label", "boundary", "oaf"],
            )

            train_df.to_csv(
                os.path.join(save_dir, "train{}.csv".format(i+1)), index=None
            )
            val_df.to_csv(os.path.join(save_dir, "val{}.csv".format(i+1)), index=None)
            test_df.to_csv(
                os.path.join(save_dir, "test{}.csv".format(i+1)), index=None
            )

    print("Done")


if __name__ == "__main__":
    main()
