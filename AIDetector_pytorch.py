import sys
sys.path.insert(0, './YOLOX')
import torch
import numpy as np
from YOLOX.yolox.data.data_augment import preproc
from YOLOX.yolox.data.datasets import COCO_CLASSES
from YOLOX.yolox.exp import get_exp_by_name
from YOLOX.yolox.utils import fuse_model, get_model_info, postprocess, vis

from utils.general import non_max_suppression, scale_coords, letterbox
from utils.torch_utils import select_device
from utils.BaseDetector import baseDet

import cv2


COCO_MEAN = (0.485, 0.456, 0.406)
COCO_STD = (0.229, 0.224, 0.225)





class Detector(baseDet):

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



    def detect(self, img):

        img, ratio = preproc(img, self.test_size, COCO_MEAN, COCO_STD)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(self.device)

        with torch.no_grad():
            outputs = self.model(img)
            outputs = postprocess(
                outputs, self.exp.num_classes, self.exp.test_conf, self.exp.nmsthre  # TODO:用户可更改
            )
        pred_boxes = []
        for output in outputs[0]:
            bbox = output[0:4]

            # preprocessing: resize
            bbox /= ratio
            class_idx = output[6]
            score = output[4] * output[5]
            if score<0.3:       # TODO:用户可更改
                continue
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            label = COCO_CLASSES[int(class_idx)]
    
            pred_boxes.append(
                (x1, y1, x2, y2, label, score))
        return img, pred_boxes


    def visual(self, output, img_info, cls_conf=0.35):
        ratio = img_info["ratio"]
        img = img_info["raw_img"]
        if output is None:
            return img
        output = output.cpu()

        bboxes = output[:, 0:4]

        # preprocessing: resize
        bboxes /= ratio

        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]

        vis_res = vis(img, bboxes, scores, cls, cls_conf, self.cls_names)
        return vis_res



if __name__=='__main__':
    detector = Detector()
    img = cv2.imread('dog.jpg')
    img_,out = detector.detect(img)
    print(out)
# class Detector(baseDet):

#     def __init__(self):
#         super(Detector, self).__init__()
#         self.init_model()
#         self.build_config()

#     def init_model(self):

#         self.weights = 'weights/yolov5m.pt'
#         self.device = '0' if torch.cuda.is_available() else 'cpu'
#         self.device = select_device(self.device)
        

#         # model = exp.get_model()

#         # model.to(device)
#         # model.eval()



#         model = attempt_load(self.weights, map_location=self.device)
#         model.to(self.device).eval()
#         model.half()
#         # torch.save(model, 'test.pt')
#         self.m = model
#         self.names = model.module.names if hasattr(
#             model, 'module') else model.names

#     def preprocess(self, img):

#         img0 = img.copy()
#         img = letterbox(img, new_shape=self.img_size)[0]
#         img = img[:, :, ::-1].transpose(2, 0, 1)
#         img = np.ascontiguousarray(img)
#         img = torch.from_numpy(img).to(self.device)
#         img = img.half()  # 半精度
#         img /= 255.0  # 图像归一化
#         if img.ndimension() == 3:
#             img = img.unsqueeze(0)

#         return img0, img

#     def detect(self, im):

#         im0, img = self.preprocess(im)

#         pred = self.m(img, augment=False)[0]
#         pred = pred.float()
#         pred = non_max_suppression(pred, self.threshold, 0.4)

#         pred_boxes = []
#         for det in pred:

#             if det is not None and len(det):
#                 det[:, :4] = scale_coords(
#                     img.shape[2:], det[:, :4], im0.shape).round()

#                 for *x, conf, cls_id in det:
#                     lbl = self.names[int(cls_id)]
#                     if not lbl in ['person', 'car', 'truck']:
#                         continue
#                     x1, y1 = int(x[0]), int(x[1])
#                     x2, y2 = int(x[2]), int(x[3])
#                     pred_boxes.append(
#                         (x1, y1, x2, y2, lbl, conf))

#         return im, pred_boxes

