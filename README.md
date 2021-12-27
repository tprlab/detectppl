# People detectors comparison
Contenders:
- YOLOv4 
- Tiny YOLO
- MobileSSDv1
- HOG
- Viola-Jones

All the runs made within OpenCV framework.

## Requirements
- Python3 v64
- opencv_python

## Steps
- Get the repo
- Download [YOLO](https://github.com/AlexeyAB/darknet) and ssd_mobilenet_v1_coco_11_06_2017
- Run: python3 detect_main.py soccer.jpg

## Results
```
yolo found 12 persons in 1.97 seconds avg 

tiny_yolo found 8 persons in 0.22 seconds avg 

ssd found 14 persons in 0.1 seconds avg 

hog found 2 persons in 0.18 seconds avg 

haar found 0 persons in 0.07 seconds avg
```

