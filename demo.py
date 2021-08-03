from tracker import Tracker
from detector import Detector
import imutils, argparse, cv2
import os
from glob import glob



def track_images(img_dir):
    tracker = Tracker(model='yolox-s', ckpt='weights/yolox_s.pth.tar',filter_class=['truck','person','car'])
    imgs = glob(os.path.join(img_dir,'*.png')) + glob(os.path.join(img_dir,'*.jpg')) + glob(os.path.join(img_dir,'*.jpeg'))
    for path in imgs:
        im = cv2.imread(path)
        im = imutils.resize(im, height=400)
        image,_ = tracker.update(im)
        image = imutils.resize(image, height=500)

        cv2.imshow('demo', image)
        cv2.waitKey(1)
        if cv2.getWindowProperty('demo', cv2.WND_PROP_AUTOSIZE) < 1:
            break
    cv2.destroyAllWindows()



def track_cap(file):
    cap = cv2.VideoCapture(file)
    tracker = Tracker()
    while True:
        _, im = cap.read()
        if im is None:
            break
        im = imutils.resize(im, height=500)
        image,_ = tracker.update(im)
       
 
        cv2.imshow('demo', image)
        cv2.waitKey(1)
        if cv2.getWindowProperty('demo', cv2.WND_PROP_AUTOSIZE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser("YOLOX-Tracker Demo!")
    parser.add_argument('-p', "--path", type=str, help="choose a video")
    args = parser.parse_args()

    if os.path.isfile(args.path):
        track_cap(args.path)
    else:
        track_images(args.path)