import yaml
import json
import numpy as np
import argparse
import os

home = os.path.expanduser("~")

# --- Parse command line arguments ---
parser = argparse.ArgumentParser(description="Update JSON camera poses from YAML calibration")
parser.add_argument("--yaml", type=str, default='./build/my_output/calibrated_cameras_data.yml', help="Path to YAML calibration file")
parser.add_argument("--json", type=str, default=home+'/smpl_ros/zed_calib3.json', help="Path to input JSON configuration")
parser.add_argument("--output", type=str, default=home+"/smpl_ros/zed_calib3.json", help="Path to output updated JSON")
args = parser.parse_args()

print(f"YAML: {args.yaml}")
print(f"JSON: {args.json}")

# check if the files exist
if not os.path.isfile(args.yaml):
    raise FileNotFoundError(f"YAML file not found: {args.yaml}")
if not os.path.isfile(args.json):
    raise FileNotFoundError(f"JSON file not found: {args.json}")

# --- Define OpenCV matrix constructor ---
def opencv_matrix_constructor(loader, node):
    mapping = loader.construct_mapping(node, deep=True)
    data = mapping['data']
    rows = mapping['rows']
    cols = mapping['cols']
    return np.array(data).reshape((rows, cols))

yaml.add_constructor(u'!opencv-matrix', opencv_matrix_constructor)
yaml.add_constructor(u'!!opencv-matrix', opencv_matrix_constructor)  # some files use double !!

# --- Load YAML ---
# --- Load YAML (with cleaning) ---
with open(args.yaml, "r") as f:
    lines = f.readlines()

# remove header lines like %YAML:1.0 and ---
cleaned_lines = [l for l in lines if not l.strip().startswith("%YAML") and not l.strip() == "---"]

# replace !!opencv-matrix with !opencv-matrix
cleaned_text = "".join(cleaned_lines).replace("!!opencv-matrix", "")

calib = yaml.safe_load(cleaned_text)

# --- Load JSON ---
with open(args.json, "r") as f:
    fusion_cfg = json.load(f)

# --- Iterate over cameras ---
nb_cam = calib['nb_camera']
for i in range(nb_cam):
    cam_key = f"camera_{i}"
    T = np.array(calib[cam_key]['camera_pose_matrix']['data'])  # already a np.array from constructor
    T = T.reshape((4, 4))
    print(T)
    # Get serial from fusion_cfg (matching order)
    serial = list(fusion_cfg.keys())[i]

    # Flatten and convert to space-separated string
    T_str = " ".join(map(str, T.flatten().tolist()))

    fusion_cfg[serial]['FusionConfiguration']['pose'] = T_str

# --- Save updated JSON ---
with open(args.output, "w") as f:
    json.dump(fusion_cfg, f, indent=4)

print(f"Updated JSON saved to {args.output}")
