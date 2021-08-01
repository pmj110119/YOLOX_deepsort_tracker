import sys
sys.path.insert(0, './YOLOX')
import torch
import numpy as np
import cv2

from YOLOX.yolox.data.data_augment import preproc
from YOLOX.yolox.data.datasets import COCO_CLASSES
from YOLOX.yolox.exp import get_exp_by_name
from YOLOX.yolox.utils import fuse_model, get_model_info, postprocess, vis

#from tracker import update_tracker



COCO_MEAN = (0.485, 0.456, 0.406)
COCO_STD = (0.229, 0.224, 0.225)




class BaseDetector(object):
    def __init__(self):

        self.img_size = 640
        self.threshold = 0.3
        self.stride = 1

    def build_config(self):

        self.faceTracker = {}
        self.faceClasses = {}
        self.faceLocation1 = {}
        self.faceLocation2 = {}
        self.frameCounter = 0
        self.currentCarID = 0
        self.recorded = []

        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def feedCap(self, img):

        retDict = {
            'frame': None,
            'faces': None,
            'list_of_ids': None,
            'face_bboxes': []
        }
        # self.frameCounter += 1

        # img, faces, face_bboxes = update_tracker(self, img)

        # retDict['frame'] = img
        # retDict['faces'] = faces
        # retDict['face_bboxes'] = face_bboxes

        return retDict

    def init_model(self):
        raise EOFError("Undefined model type.")

    def preprocess(self):
        raise EOFError("Undefined model type.")

    def detect(self):
        raise EOFError("Undefined model type.")




class detect_info():
    def __call__(self):
        res = {
            'raw_img':None,
            'img':None,
            'boxes':None,
            'scores':None,
            'class_ids':None,
            'visual':None
        }
        return res


class Detector(BaseDetector):
    """ 图片检测器 """
    def __init__(self):
        super(Detector, self).__init__()
        self.init_model()
        self.build_config()

    def init_model(self):
        self.yolox_name = 'yolox-m'
        self.weights = 'weights/yolox_m.pth'
        

        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

        self.exp = get_exp_by_name(self.yolox_name)
        self.test_size = self.exp.test_size  # TODO: 改成图片自适应大小
        self.model = self.exp.get_model()
        self.model.to(self.device)
        self.model.eval()
        ckpt = torch.load(self.weights, map_location="cpu")
        self.model.load_state_dict(ckpt["model"])



    def detect(self, raw_img, visual=True, conf=0.5):
        info = {}
        img, ratio = preproc(raw_img, self.test_size, COCO_MEAN, COCO_STD)
        info['raw_img'] = raw_img
        info['img'] = img

        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(self.device)

        with torch.no_grad():
            outputs = self.model(img)
            outputs = postprocess(
                outputs, self.exp.num_classes, self.exp.test_conf, self.exp.nmsthre  # TODO:用户可更改
            )[0].cpu().numpy()

        info['boxes'] = outputs[:, 0:4]/ratio
        info['scores'] = outputs[:, 4] * outputs[:, 5]
        info['class_ids'] = outputs[:, 6]
        info['box_nums'] = outputs.shape[0]
        # 可视化绘图
        if visual:
            info['visual'] = vis(info['raw_img'], info['boxes'], info['scores'], info['class_ids'], conf, COCO_CLASSES)
        return info






if __name__=='__main__':
    detector = Detector()
    img = cv2.imread('dog.jpg')
    img_,out = detector.detect(img)
    print(out)
