# YOLOX产生跟踪数据集的预处理结果

- 处理产生json文件

> cd .\tools\
> 
> python  convert_mot17_to_coco.py


- Run产生对应的预处理的json文件

# For MOT17 validation
```
python detect.py -f "exps/yolox_x_mot17_val.py" -c "weights/mot17_half.pth.tar" --nms 0.80 -n "../outputs/1. det/mot17_val_0.80.pickle" -b 1 -d 1 --fp16 --fuse
python detect.py -f "exps/yolox_x_mot17_val.py" -c "weights/mot17_half.pth.tar" --nms 0.95 -n "../outputs/1. det/mot17_val_0.95.pickle" -b 1 -d 1 --fp16 --fuse
```
```
# For MOT17 test
python detect.py -f "E:/Paper/CrossMOT/CrossTracker/CrossMOT/external/YOLOX/exps/yolox_x_mot17_test.py" -c "E:/Paper/CrossMOT/CrossTracker/CrossMOT/external/YOLOX/pretrained/mot17.pth.tar" --nms 0.80 -n "E:/Paper/CrossMOT/CrossTracker/CrossMOT/data/MOT17/detections_yolox_x/test" -b 1 -d 1 --fp16 --fuse
python detect.py -f "E:/Paper/CrossMOT/CrossTracker/CrossMOT/external/YOLOX/exps/yolox_x_mot17_test.py" -c "E:/Paper/CrossMOT/CrossTracker/CrossMOT/external/YOLOX/pretrained/mot17.pth.tar" --nms 0.7 -n "E:/Paper/CrossMOT/CrossTracker/CrossMOT/data/MOT17/detections_yolox_x/test" -b 1 -d 1 --fp16 --fuse
```

这里我产生的文件和之前的区别在于我这里的编号不是顺序产生的每个序列是一个开始的序列编号

