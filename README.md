# YOLOX_deepsort_tracker

Using YOLOX as detector, and deepsort as tracker.


## :tada: How to use Detector and Tracker

### &#8627; Detect example

```python
from detector import Detector

detector = Detector(model='yolox-s', ckpt='yolo_s.pth') # instantiate Detector

img = cv2.imread('dog.jpg') 	# load image
result = detector.detect(img) 	# detect targets

img_visual = result['visual'] 	 # visualized image
cv2.imshow('detect', img_visual) # imshow
```

Detector uses yolo-x family models to detect targets. 

You can also get more information like *raw_img/boudingbox/score/class_id* from the result of detector.

### &#8627; Track example

```python
from tracker import Tracker

tracker = Tracker(model='yolox-s', ckpt='yolo_s.pth') # instantiate Tracker

cap = cv2.VideoCapture('test.mp4')	# load video

while True:
    _, frame = cap.read()	# get new frame
    if frame is None:
       break
    result = tracker.update(frame)	# detect and track targets
    
    cv2.imshow('demo', result['visual'])	# imshow visualized frame
    cv2.waitKey(1)
```

Tracker uses detector to get each frame's boundingbox, and use deepsort to get every bbox's ID. 

## :art: Install

1. Clone the repository recursively:

   `git clone --recurse-submodules https://github.com/pmj110119/YOLOX_deepsort_tracker.git`

   If you already cloned and forgot to use `--recurse-submodules` you can run `git submodule update --init`(clone最新的YOLOX仓库)

2. Make sure that you fulfill all the requirements: Python 3.8 or later with all [requirements.txt](https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch/blob/master/requirements.txt) dependencies installed, including torch>=1.7. To install, run:

   `pip install -r requirements.txt`


## :zap: Select a YOLOX family model

1. train your own model or just download pretrained models from https://github.com/Megvii-BaseDetection/YOLOX

2. select the model and checkpoint when using Detector and Tracker

   for example:

   ```python
   """
   YOLO family: yolox-s, yolox-m, yolox-l, yolox-x, yolox-tiny, yolox-nano, yolov3
   """
   # yolox-s example
   detector = Detector(model='yolox-s', ckpt='./yolox_s.pth')
   # yolox-m example
   detector = Detector(model='yolox-m', ckpt='./yolox_m.pth')
   ```

## :clap: Run demo

- Detect on image

  ```python
  python .\demo.py --mode=detect --file=dog.jpg
  ```

- Track on video

  ```python
  python .\demo.py --mode=track --file=test.mp4
  ```

## Filter tracked classes

coming soon...



## Train your own model

coming soon...