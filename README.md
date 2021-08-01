# YOLOX_deepsort_tracker

yolox+deepsort实现目标跟踪

最新的yolox尝尝鲜~~（yolox正处在频繁更新阶段，因此直接链接yolox仓库作为子模块）

## How to use Detector and Tracker

- Detect image

  ```python
  from detector import Detector
  # Instantiate Detector and select model and ckpt
  detector = Detector(model='yolox-s', ckpt='yolo_s.pth')
  # load image
  img = cv2.imread('dog.jpg')
  # inference
  result = detector.detect(img)
  # imshow
  img_visual = result['visual']	
  cv2.imshow('detect', img_visual)
  ```

  Detector uses yolox model to detect targets. 

  You can also get more information like *raw_img/boudingbox/score/class_id* from the result of detector.

- Track video (or camera)

  ```python
  from tracker import Tracker
  # Instantiate Detector and select model and ckpt
  tracker = Tracker(model='yolox-s', ckpt='yolo_s.pth')
  # load video
  cap = cv2.VideoCapture('test.mp4')
  # start tracking...
  while True:
      _, frame = cap.read()
      if frame is None:
         break
      result = tracker.update(frame)
      cv2.imshow('demo', result['visual'])
      cv2.waitKey(1)
  ```

  Tracker uses detector to get each frame's boundingbox, and use deepsort to get every bbox's ID. 

## Install

1. Clone the repository recursively:

   `git clone --recurse-submodules https://github.com/pmj110119/YOLOX_deepsort_tracker.git`

   If you already cloned and forgot to use `--recurse-submodules` you can run `git submodule update --init`(clone最新的YOLOX仓库)

2. Make sure that you fulfill all the requirements: Python 3.8 or later with all [requirements.txt](https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch/blob/master/requirements.txt) dependencies installed, including torch>=1.7. To install, run:

   `pip install -r requirements.txt`


## Select a YOLOX family model

1. train your own model or just download pretrained models from https://github.com/Megvii-BaseDetection/YOLOX

2. update the type and path of model in **detector.py**

   for example:

   ```python
   class Detector(BaseDetector):
   	""" 
   	YOLO family: yolox-s, yolox-m, yolox-l, yolox-x, yolox-tiny, yolox-nano, yolov3
   	"""
       def init_model(self):
           self.yolox_name = 'yolox-m' 
           self.weights = 'weights/yolox_m.pth'
           
       """ """
   ```

## Run demo

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