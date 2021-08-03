# YOLOX_deepsort_tracker

<div align="center">
<p>
<img src="utils/img2.gif" width="400"/> <img src="utils/img1.gif" width="400"/> 
</p>
<br>
<div>

</div>

</div>

## :tada: How to use

### &#8627; Easily use tracker in your project!

```python
from tracker import Tracker

tracker = Tracker()  # track specific category

result = tracker.update(img)
```

Detector uses yolo-x family models to detect targets. 

You can also get more information like *boudingbox/score/class_id/track_id* from the result of track.

### &#8627; Track specific category

You can also set some target categories, tracker will ignore other bboxes.

```python
tracker = Tracker(filter_classes=['car','person']) 
```

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
   detector = Tracker(model='yolox-s', ckpt='./yolox_s.pth')
   # yolox-m example
   detector = Tracker(model='yolox-m', ckpt='./yolox_m.pth')
   ```

## :clap: Run demo

```python
python .\demo.py --path=test.mp4
```

