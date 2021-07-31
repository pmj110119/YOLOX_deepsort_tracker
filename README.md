# YOLOX_deepsort_tracker

yolox+deepsort实现目标跟踪

最新的yolox尝尝鲜~~（yolox正处在频繁更新阶段，因此直接链接yolox仓库作为子模块）

## Install

1. Clone the repository recursively:

   `git clone --recurse-submodules https://github.com/pmj110119/YOLOX_deepsort_tracker.git`

   If you already cloned and forgot to use `--recurse-submodules` you can run `git submodule update --init`(clone最新的YOLOX仓库)

2. Make sure that you fulfill all the requirements: Python 3.8 or later with all [requirements.txt](https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch/blob/master/requirements.txt) dependencies installed, including torch>=1.7. To install, run:

   `pip install -r requirements.txt`


## Select a YOLOX family model

1. train your own model or just download the model from https://github.com/Megvii-BaseDetection/YOLOX

2. update the type and path of model in **detector.py**

   ```python
   class Detector(BaseDetector):
   	""" """
       def init_model(self):
           self.yolox_name = 'yolox-m'
           self.weights = 'weights/yolox_m.pth'
           
       """ """
   ```

## Run demo

```python
python demo.py
```



## Filter tracked classes

coming soon...



## Train your own model

