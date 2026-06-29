import numpy as np
import json
import os

def convert_numpy_to_json(input_npy_path, output_json_path):
    """
    Convert a NumPy file with annotations to a JSON file.
    
    Parameters:
    input_npy_path (str): Path to the input NumPy (.npy) file
    output_json_path (str): Path where the output JSON file will be saved
    
    Returns:
    bool: True if conversion was successful, False otherwise
    """
    try:
        # Load the NumPy file with annotations
        data = np.load(input_npy_path).astype(int)
        # Find where adjacent elements differ
        boundaries = np.nonzero(np.diff(data))[0]
        segment_count = len(boundaries) + 1

        transformed_array = np.zeros_like(data)
        current_id = 0
        prev_value = None

        for i, value in enumerate(data):
            # If this is a new value or the first element
            if i == 0 or value != prev_value:
                current_id = current_id + (0 if i == 0 else 1)
            
            transformed_array[i] = current_id
            prev_value = value

        # import pdb; pdb.set_trace()

        # Convert NumPy array to a list format suitable for JSON
        json_data = [transformed_array.tolist()]
        
        # Save to JSON file
        with open(output_json_path, 'w') as f:
            json.dump(json_data, f)
        
        print(f"Successfully converted {input_npy_path} to {output_json_path}")
        return True
    
    except Exception as e:
        print(f"Error during conversion: {str(e)}")
        return False

# Example usage
if __name__ == "__main__":
    # convert_numpy_to_json('your_annotations.npy', 'gt_action_labels.json')
    val_split = "/home/dsliwowski/Projects/Code_Inverse/catkin/src/inverse_tas/data/REASSEMBLEmm/splits/test.split1.bundle"
    test_video_list = np.loadtxt(val_split, dtype=str)
    annot_base = "/home/dsliwowski/Projects/Code_Inverse/catkin/src/inverse_tas/data/REASSEMBLEmm/annotations"
    save_path = "/home/dsliwowski/Projects/Code_Inverse/catkin/src/inverse_tas/data/REASSEMBLEmm/Chirst_annot"
    # import pdb; pdb.set_trace()

    i = 0
    for path in test_video_list:
        path = path.replace(".txt", "")
        annot_path = os.path.join(annot_base, f"{path}.npy")
        out_path = os.path.join(save_path, path)
        os.makedirs(out_path, exist_ok=True)
        out_path = os.path.join(out_path, "assignments.json")
        convert_numpy_to_json(annot_path, out_path)
        # import pdb; pdb.set_trace()