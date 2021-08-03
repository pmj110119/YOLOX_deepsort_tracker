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

### &#8627; Easy use

```python
from tracker import Tracker

tracker = Tracker()

result = tracker.update(img)	# feed one frame and get bbox with track-id
```

Detector uses yolo-x family models to detect targets. 

You can also get more information like *boudingbox/score/class_id/track_id* from the result of track.

### &#8627; Select specific category

If you just  want to track only some specific categories, you can set by param *filter_classes*.

For example:

```python
tracker = Tracker(filter_classes=['car','person']) 
```

## &#8627; Run demo

```python
python demo.py --path=test.mp4
```

## :art: Install

1. Clone the repository recursively:

   `git clone --recurse-submodules https://github.com/pmj110119/YOLOX_deepsort_tracker.git`

   If you already cloned and forgot to use `--recurse-submodules` you can run `git submodule update --init`(clone最新的YOLOX仓库)

2. Make sure that you fulfill all the requirements: Python 3.8 or later with all [requirements.txt](https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch/blob/master/requirements.txt) dependencies installed, including torch>=1.7. To install, run:

   `pip install -r requirements.txt`


## :zap: Select a YOLOX family model

1. train your own model or just download pretrained models from https://github.com/Megvii-BaseDetection/YOLOX

   | Model                                       | size | mAP<sup>test<br>0.5:0.95 | Speed V100<br>(ms) | Params<br>(M) | FLOPs<br>(G) |                           weights                            |
   | ------------------------------------------- | :--: | :----------------------: | :----------------: | :-----------: | :----------: | :----------------------------------------------------------: |
   | [YOLOX-s](./exps/default/yolox_s.py)        | 640  |           39.6           |        9.8         |      9.0      |     26.8     | [onedrive](https://megvii-my.sharepoint.cn/:u:/g/personal/gezheng_megvii_com/EW62gmO2vnNNs5npxjzunVwB9p307qqygaCkXdTO88BLUg?e=NMTQYw)/[github](https://github.com/Megvii-BaseDetection/storage/releases/download/0.0.1/yolox_s.pth) |
   | [YOLOX-m](./exps/default/yolox_m.py)        | 640  |           46.4           |        12.3        |     25.3      |     73.8     | [onedrive](https://megvii-my.sharepoint.cn/:u:/g/personal/gezheng_megvii_com/ERMTP7VFqrVBrXKMU7Vl4TcBQs0SUeCT7kvc-JdIbej4tQ?e=1MDo9y)/[github](https://github.com/Megvii-BaseDetection/storage/releases/download/0.0.1/yolox_m.pth) |
   | [YOLOX-l](./exps/default/yolox_l.py)        | 640  |           50.0           |        14.5        |     54.2      |    155.6     | [onedrive](https://megvii-my.sharepoint.cn/:u:/g/personal/gezheng_megvii_com/EWA8w_IEOzBKvuueBqfaZh0BeoG5sVzR-XYbOJO4YlOkRw?e=wHWOBE)/[github](https://github.com/Megvii-BaseDetection/storage/releases/download/0.0.1/yolox_l.pth) |
   | [YOLOX-x](./exps/default/yolox_x.py)        | 640  |         **51.2**         |        17.3        |     99.1      |    281.9     | [onedrive](https://megvii-my.sharepoint.cn/:u:/g/personal/gezheng_megvii_com/EdgVPHBziOVBtGAXHfeHI5kBza0q9yyueMGdT0wXZfI1rQ?e=tABO5u)/[github](https://github.com/Megvii-BaseDetection/storage/releases/download/0.0.1/yolox_x.pth) |
   | [YOLOX-Darknet53](./exps/default/yolov3.py) | 640  |           47.4           |        11.1        |     63.7      |    185.3     | [onedrive](https://megvii-my.sharepoint.cn/:u:/g/personal/gezheng_megvii_com/EZ-MV1r_fMFPkPrNjvbJEMoBLOLAnXH-XKEB77w8LhXL6Q?e=mf6wOc)/[github](https://github.com/Megvii-BaseDetection/storage/releases/download/0.0.1/yolox_darknet53.pth) |

   You can download yolox_s.pth to the folder **weights** , which is the default model path of **Tracker**.

2. You can also use other yolox models, for example:

   ```python
   """
   YOLO family: yolox-s, yolox-m, yolox-l, yolox-x, yolox-tiny, yolox-nano, yolov3
   """
   # yolox-s example
   detector = Tracker(model='yolox-s', ckpt='./yolox_s.pth')
   # yolox-m example
   detector = Tracker(model='yolox-m', ckpt='./yolox_m.pth')
   ```

