import matplotlib.pyplot as plt
import os
import numpy as np
import glob
import os

from models.DiffAct.utils import plot_barcode, read_file, edit_score, f_score

def txt_to_int(event, event_list):
    frame_num = len(event)
    event_seq_raw = np.zeros((frame_num,))
    for i in range(frame_num):
        if event[i] in event_list:
            event_seq_raw[i] = event_list.index(event[i])
        else:
            event_seq_raw[i] = -100  # background
    return event_seq_raw

base_path = "/home/dsliwowski/Projects/Code_Inverse/catkin/src/inverse_tas/results/mstcn/IRIS_M2R2/split_1"
ds_dir = "/home/dsliwowski/Projects/Code_Inverse/catkin/src/inverse_tas/data/IRIS"
gt_dir = ds_dir + "/annotations"

mapping_file = ds_dir + "/mapping.txt"
event_list = np.loadtxt(mapping_file, dtype=str)
event_list = [i[1] for i in event_list]

predictions_path = base_path
all_predictions = [os.path.join(predictions_path, "subject3-5_world_cam")] #glob.glob(os.path.join(predictions_path, "*.txt"))
plots_path = base_path + "/plots"
os.makedirs(plots_path, exist_ok=True)

num_classes = 69

overlap = [.1, .25, .5]
tp, fp, fn = np.zeros(3), np.zeros(3), np.zeros(3)

correct = 0
total = 0
edit = 0

for pred_path in all_predictions:
    name = os.path.split(pred_path)[-1]
    gt_path = os.path.join(gt_dir, name+".txt")

    # gt_file = os.path.join(label_dir, f'{vid}.txt')
    # gt_content = read_file(gt_path).split('\n')[0:-1]
    # gt_content = np.array(gt_content)
    gt_content = read_file(gt_path).split('\n')[0:-1]
    # gt_content_new = [int(x) for x in gt_content]
    gt_content = np.array(gt_content)
    gt_content = txt_to_int(gt_content, event_list)

    pred_content = read_file(pred_path).split('\n')[1].split()
    pred_content = np.array(pred_content)
    pred_content = txt_to_int(pred_content, event_list)

    # import pdb; pdb.set_trace()

    missing = len(gt_content) - len(pred_content)
    assert missing == 1
    gt_content = gt_content[missing:]

    assert len(gt_content) == len(pred_content)

    for i in range(len(gt_content)):
        total += 1
        if gt_content[i] == pred_content[i]:
            correct += 1

    edit += edit_score(pred_content, gt_content)

    for s in range(len(overlap)):
        tp1, fp1, fn1 = f_score(pred_content, gt_content, overlap[s])
        tp[s] += tp1
        fp[s] += fp1
        fn[s] += fn1

    # import pdb
    # pdb.set_trace()

    # import pdb
    # pdb.set_trace()
    save_path = os.path.join(plots_path, name.replace("txt", "png"))
    plot_barcode(num_classes, gt_content, pred_content, show=True)

# acc = 100 * float(correct) / total
# edit = (1.0 * edit) / len(all_predictions)
# f1s = np.array([0, 0 ,0], dtype=float)
# for s in range(len(overlap)):
#     precision = tp[s] / float(tp[s] + fp[s])
#     recall = tp[s] / float(tp[s] + fn[s])

#     f1 = 2.0 * (precision * recall) / (precision + recall)

#     f1 = np.nan_to_num(f1) * 100
#     f1s[s] = f1

# print("Acc:", acc)
# print("Edit:", edit)
# print("F1@10:", f1s[0])
# print("F1@25:", f1s[1])
# print("F1@50:", f1s[2])