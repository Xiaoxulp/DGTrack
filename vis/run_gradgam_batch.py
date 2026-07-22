import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import cv2
import glob
import torch
import numpy as np
import math
import argparse
from PIL import Image
from torchvision.transforms import ToTensor
import torchvision.transforms as T
from tqdm import tqdm
from reid.models.rga_model import resnet50_rga
from vis.gradgam.utils import GradCAM, show_cam_on_image

TEXT_HEIGHT = 40  
FONT_SCALE = 0.8  
FONT_THICKNESS = 2  
FONT_COLOR = (0, 0, 0)  

to_tensor = ToTensor()
normalizer = T.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

def parse_args():
    
    parser = argparse.ArgumentParser(description='Train MOT Reid')
    # base config
    parser.add_argument('-d', '--dataset', type=str, default='E:/Domain/Cross_mot/UnifyTrack/UnifyTrack/data/') # DATA_ROOT
    parser.add_argument('-p', '--output', type=str, default='E:/Domain/Cross_mot/UnifyTrack/UnifyTrack/vis/vis_output/')
    parser.add_argument("--num_classes",type=int,default=419) 
    parser.add_argument('--subset', type=str, default='train') 
    parser.add_argument('--resume', type=str, default='E:/Domain/Cross_mot/UnifyTrack/UnifyTrack/Train/logs/mot17_200.pth.tar', metavar='PATH')
    parser.add_argument('--height', type=int, default=384)
    parser.add_argument('--width', type=int, default=128)
    parser.add_argument('--features', type=int, default=2048)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--use_cuda', type=bool, default=True)
    return parser.parse_args()

def load_reid_model(args):
    
    print(f"Loading model from {args.resume}")
    model = resnet50_rga(
        pretrained=args.features,
        num_feat=args.features,
        height=args.height,
        width=args.width,
        dropout=args.dropout,
        num_classes=args.num_classes
    )
    checkpoint = torch.load(args.resume)
    state_dict = checkpoint['state_dict']  # 'teacher_state_dict' 'state_dict'
    model.load_state_dict(state_dict)
    if args.use_cuda:
        model = model.cuda()
    model.eval()
    return model


def process_frame(args,model, cam, img_path, label_path, save_path):
    if not os.path.exists(label_path):
        return
    try:
        raw = np.loadtxt(label_path, dtype=np.float32).reshape(-1, 7)
    except:
        return
    try:
        pil_image = Image.open(img_path).convert('RGB')
        img_np = np.array(pil_image)
        img_h, img_w, _ = img_np.shape
    except:
        return
    input_tensor_list = []
    target_category_list = []
    crop_img_list = []
    if len(raw) > 0:
        xywh = raw[:, 2:6]
        tlxy = xywh.copy()
        tlxy[:, 2] = xywh[:, 0] + xywh[:, 2]
        tlxy[:, 3] = xywh[:, 1] + xywh[:, 3]
        for i in range(len(raw)):
            x1, y1, x2, y2 = tlxy[i].astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img_w, x2), min(img_h, y2)
            if x2 <= x1 or y2 <= y1: continue
            crop = img_np[y1:y2, x1:x2]
            crop_resized = cv2.resize(crop, (args.width, args.height), interpolation=cv2.INTER_LINEAR)
            crop_vis = crop_resized.astype(np.float32) / 255.
            crop_img_list.append(crop_vis)
            tensor = to_tensor(crop_resized)
            tensor = normalizer(tensor)
            input_tensor_list.append(tensor)
            target_id = int(raw[i][6])
            target_category_list.append(target_id)
    if not input_tensor_list: return

   
    MAX_BATCH_SIZE = 32 

    total_samples = len(input_tensor_list)
    grayscale_cam_results = [] 

    
    for i in range(0, total_samples, MAX_BATCH_SIZE):
        
        sub_input_list = input_tensor_list[i: i + MAX_BATCH_SIZE]
        sub_target_list = target_category_list[i: i + MAX_BATCH_SIZE]

        
        sub_batch = torch.stack(sub_input_list)
        if args.use_cuda:
            sub_batch = sub_batch.cuda()

       
        sub_cam_out = cam(input_tensor=sub_batch, target_category=sub_target_list)

        
        grayscale_cam_results.append(sub_cam_out)

        
        del sub_batch
        if args.use_cuda:
            torch.cuda.empty_cache()

   
    grayscale_cam_batch = np.concatenate(grayscale_cam_results, axis=0)

    
    num_imgs = len(crop_img_list)
    cols = 6
    rows = math.ceil(num_imgs / cols)
   
    GRID_GAP_X = 30  
    GRID_GAP_Y = 30  
    cell_w = args.width * 2  
    cell_h = TEXT_HEIGHT + args.height  
    
    canvas_w = cols * cell_w + (cols + 1) * GRID_GAP_X
    canvas_h = rows * cell_h + (rows + 1) * GRID_GAP_Y
    
    grid_img = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8) + 255
    for i in range(num_imgs):
        heatmap = grayscale_cam_batch[i, :]
        original = crop_img_list[i]
        vis_cam = show_cam_on_image(original, heatmap, use_rgb=True)
        vis_orig = (original * 255).astype(np.uint8)
        img_combined = np.hstack((vis_orig, vis_cam))
        header = np.zeros((TEXT_HEIGHT, cell_w, 3), dtype=np.uint8) + 255
        label_text = f"ID: {target_category_list[i]}"
        cv2.putText(header, label_text, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, FONT_COLOR, FONT_THICKNESS)
        final_cell = np.vstack((header, img_combined))
        
        r_idx = i // cols
        c_idx = i % cols
        
        x_start = c_idx * cell_w + (c_idx + 1) * GRID_GAP_X
        y_start = r_idx * cell_h + (r_idx + 1) * GRID_GAP_Y
        x_end = x_start + cell_w
        y_end = y_start + cell_h
        
        grid_img[y_start:y_end, x_start:x_end, :] = final_cell

    
    grid_img_bgr = cv2.cvtColor(grid_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, grid_img_bgr)



def main(args):
    model = load_reid_model(args)
    target_layers = model.backbone.layer4
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=args.use_cuda)

    domain_dirs = glob.glob(os.path.join(args.dataset, "Domain*"))

    for domain_path in domain_dirs:
        domain_name = os.path.basename(domain_path)
        print(f"\nProcessing {domain_name}...")
        seq_search_path = os.path.join(domain_path, "images", args.subset, "*")
        seq_dirs = glob.glob(seq_search_path)

        for seq_dir in seq_dirs:
            seq_name = os.path.basename(seq_dir)
            label_seq_dir = os.path.join(domain_path, "labels_with_ids", args.subset, seq_name, "img1")
            img_seq_dir = os.path.join(seq_dir, "img1")
            if not os.path.exists(img_seq_dir) or not os.path.exists(label_seq_dir):
                continue
            print(f"  > Sequence: {seq_name}")
            save_dir = os.path.join(args.output, domain_name, seq_name)
            os.makedirs(save_dir, exist_ok=True)

            img_files = glob.glob(os.path.join(img_seq_dir, "*.jpg"))
            img_files.sort()
            for img_path in tqdm(img_files, desc=f"    {seq_name}", ncols=100):
                file_name = os.path.basename(img_path)
                base_name = os.path.splitext(file_name)[0]
                label_path = os.path.join(label_seq_dir, base_name + ".txt")
                save_path = os.path.join(save_dir, f"{base_name}_vis.jpg")
                process_frame(args,model, cam, img_path, label_path, save_path)
    print("\nFinished.The Result is in:", os.path.abspath(args.output))

if __name__ == "__main__":
    args = parse_args()
    main(args)