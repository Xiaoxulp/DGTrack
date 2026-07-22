import os
from functools import partial
from typing import List
from pathlib import Path
root = '/home/st3/DGTrack/data'
save_path = '/home/st3/DGTrack/datasets/data_path/data_path.txt'
dataset_list = ['DanceTrack', 'MOT17', 'SportsMOT']
def solve_MOT_train(root, dataset_name):
    """
    Generic solver for MOT-style datasets.
    Automatically supports 6-digit / 8-digit frame indices
    by reading actual filenames from img1/.
    """
    dataset_path = f'{dataset_name}/images/train'
    data_root =  os.path.join(root,dataset_path)
    video_paths = []
    for video_name in os.listdir(data_root):
        if dataset_name == 'MOT17':
            if 'FRCNN' not in video_name:
                continue
        video_paths.append(video_name)
    frames = []
    for video_name in video_paths:
        img_dir = os.path.join(data_root,video_name,'img1')
        if not os.path.isdir(img_dir):
            continue
        files = sorted(os.listdir(img_dir))
        for fname in files:
            if not fname.endswith('.jpg'):
                continue
            frame_path = Path(dataset_path) / video_name / 'img1' / fname
            frames.append(str(frame_path).replace('\\', '/'))
    return frames

# Dataset catalog
dataset_catalog = {
    'DanceTrack': partial(solve_MOT_train, dataset_name='Domain1'),
    'MOT17': partial(solve_MOT_train, dataset_name='Domain2'),
    'SportsMOT': partial(solve_MOT_train, dataset_name='Domain3'),
}
# Main solve
def solve(dataset_list: List[str], root, save_path):
    all_frames = []
    for dataset_name in dataset_list:
        dataset_frames = dataset_catalog[dataset_name](root)
        print(f"Solved {len(dataset_frames)} frames from dataset: {dataset_name}")
        all_frames.extend(dataset_frames)
    print(f"Totally {len(all_frames)} frames are solved.")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        for u in all_frames:
            f.write(u + '\n')
# main
if __name__ == '__main__':
    solve(dataset_list,root,save_path)