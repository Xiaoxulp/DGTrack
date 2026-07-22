import os
import sys
import json
import cv2
import glob as gb
import numpy as np
import colorsys



def create_unique_color(tag, hue_step=0.37):

    h, v = (tag * hue_step) % 1, 0.9
    r, g, b = colorsys.hsv_to_rgb(h, 0.85, v)
    return int(255 * r), int(255 * g), int(255 * b)


def draw_rounded_rectangle(img, top_left, bottom_right, color, thickness=2, radius=8):

    x1, y1 = top_left
    x2, y2 = bottom_right
    cv2.line(img, (x1 + radius, y1), (x2 - radius, y1), color, thickness)
    cv2.line(img, (x1 + radius, y2), (x2 - radius, y2), color, thickness)
    cv2.line(img, (x1, y1 + radius), (x1, y2 - radius), color, thickness)
    cv2.line(img, (x2, y1 + radius), (x2, y2 - radius), color, thickness)
    cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness)
    cv2.ellipse(img, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness)
    cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness)
    cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness)


def draw_label_with_bg(img, text, pos, color, font_scale=0.7, thickness=2):
   
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = pos
   
    cv2.rectangle(img, (x - 2, y - text_h - 6), (x + text_w + 4, y), color, -1)
    
    cv2.putText(img, text, (x, y - 3), font, font_scale, (255, 255, 255), thickness, lineType=cv2.LINE_AA)



def draw_frame_number(img, frame_id):
    h, w = img.shape[:2]  
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = f"Frame: {frame_id}"
    font_scale = 0.9
    thickness = 2
    text_color = (255, 255, 255)  
    bg_color = (0, 0, 0)  

    
    (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
    
    x = w - text_w - 25
    y = h - 15

    
    cv2.rectangle(img, (x - 8, y - text_h - 8), (x + text_w + 8, y + 8), bg_color, -1)
    
    cv2.putText(img, text, (x, y), font, font_scale, text_color, thickness, lineType=cv2.LINE_AA)

def colormap(rgb=False):
    
    color_list = np.array(
        [
            0.000, 0.447, 0.741,
            0.850, 0.325, 0.098,
            0.929, 0.694, 0.125,
            0.494, 0.184, 0.556,
            0.466, 0.674, 0.188,
            0.301, 0.745, 0.933,
        ]
    ).astype(np.float32)
    color_list = color_list.reshape((-1, 3)) * 255
    if not rgb:
        color_list = color_list[:, ::-1]
    return color_list


def txt2img(visual_path="visual_val_gt"):
    print("Starting txt2img")

    valid_labels = {1}
    ignore_labels = {2, 7, 8, 12}

    if not os.path.exists(visual_path):
        os.makedirs(visual_path)

    gt_json_path = '/home/st3/DGTrack/data/MOT20/annotations/test.json'
    img_path = '/home/st3/DGTrack/data/MOT20/test/'
    show_video_names = ['MOT20-08']

    test_json_path = '/home/st3/DGTrack/data/MOT20/annotations/test.json'
    test_img_path = '/home/st3/DGTrack/data/MOT20/test/'
    test_show_video_names = ['MOT20-08']

    if visual_path == "visual_test_predict":
        show_video_names = test_show_video_names
        img_path = test_img_path
        gt_json_path = test_json_path

    for show_video_name in show_video_names:
        img_dict = dict()
        if visual_path == "visual_val_gt":
            txt_path = 'datasets/mot/train/' + show_video_name + '/gt/gt_val_half.txt'
        elif visual_path == "visual_yolox_x":
            txt_path = 'YOLOX_outputs/yolox_x_mix_det/track_results_deepsort/' + show_video_name + '.txt'
        elif visual_path == "visual_test_predict":
            txt_path = '/home/st3/DGTrack/tools/result/test/mot20/' + show_video_name + '.txt'
        else:
            raise NotImplementedError

        with open(gt_json_path, 'r') as f:
            gt_json = json.load(f)

        for ann in gt_json["images"]:
            file_name = ann['file_name']
            video_name = file_name.split('/')[0]
            if video_name == show_video_name:
                img_dict[ann['frame_id']] = img_path + file_name

        txt_dict = dict()
        with open(txt_path, 'r') as f:
            for line in f.readlines():
                linelist = line.split(',')

                mark = int(float(linelist[6]))
                label = int(float(linelist[7]))
                vis_ratio = float(linelist[8])

                if visual_path == "visual_val_gt":
                    if mark == 0 or label not in valid_labels or label in ignore_labels or vis_ratio <= 0:
                        continue

                img_id = linelist[0]
                obj_id = linelist[1]
                bbox = [float(linelist[2]), float(linelist[3]),
                        float(linelist[2]) + float(linelist[4]),
                        float(linelist[3]) + float(linelist[5]), int(obj_id)]
                if int(img_id) in txt_dict:
                    txt_dict[int(img_id)].append(bbox)
                else:
                    txt_dict[int(img_id)] = list()
                    txt_dict[int(img_id)].append(bbox)

        for img_id in sorted(txt_dict.keys()):
            img = cv2.imread(img_dict[img_id])
            for bbox in txt_dict[img_id]:
                x1, y1, x2, y2, track_id = bbox
                
                color = create_unique_color(track_id)
                
                draw_rounded_rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness=2)
                
                draw_label_with_bg(img, f"ID:{track_id}", (int(x1) + 5, int(y1) - 5), color)
            
            draw_frame_number(img, img_id)
            
            cv2.imwrite(visual_path + "/" + show_video_name + "{:0>6d}.png".format(img_id), img)
        print(show_video_name, "Done")
    print("txt2img Done")


def img2video(visual_path="visual_val_gt"):
    print("Starting img2video")
    img_paths = gb.glob(visual_path + "/*.png")
    fps = 16
    size = (1920, 1080)
    videowriter = cv2.VideoWriter(visual_path + "_video.mp4", cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, size)

    for img_path in sorted(img_paths):
        img = cv2.imread(img_path)
        img = cv2.resize(img, size)
        videowriter.write(img)

    videowriter.release()
    print("img2video Done")


if __name__ == '__main__':
    visual_path = "visual_test_predict"
    if len(sys.argv) > 1:
        visual_path = sys.argv[1]
    txt2img(visual_path)
    # img2video(visual_path)