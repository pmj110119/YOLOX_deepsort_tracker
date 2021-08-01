from tracker import Tracker
from detector import Detector
import imutils, argparse, cv2



def detect_img(file):
    img = cv2.imread(file)
    print(file)
    detector = Detector(model='yolox-m', ckpt='weights/yolox_m.pth')
    info = detector.detect(img)
    result = info['visual']

    cv2.imwrite('result.png', result)
    cv2.imshow('demo', result)
    cv2.waitKey(0)
    





def track_cap(file):
    cap = cv2.VideoCapture(file)
    tracker = Tracker(model='yolox-m', ckpt='weights/yolox_m.pth')

    videoWriter = None
    while True:
        _, im = cap.read()
        if im is None:
            break
        result = tracker.update(im)
        result = imutils.resize(result, height=500)
        if videoWriter is None:
            fourcc = cv2.VideoWriter_fourcc(
                'm', 'p', '4', 'v')  # opencv3.0
            videoWriter = cv2.VideoWriter(
                'result.mp4', fourcc, int(cap.get(5)), (result.shape[1], result.shape[0]))
        videoWriter.write(result)
        cv2.imshow('demo', result)
        cv2.waitKey(1)
        if cv2.getWindowProperty('demo', cv2.WND_PROP_AUTOSIZE) < 1:
            break


    cap.release()
    videoWriter.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser("YOLOX-Tracker Demo!")
    parser.add_argument("--mode", type=str, default='detect', choices=['detect', 'track'])
    parser.add_argument('-f', "--file", type=str, help="choose an image or video file")
    args = parser.parse_args()

    if not args.file:
        print('>>>> Please select a file!')
        exit()
    if args.mode=='detect':
        detect_img(args.file)
    elif args.mode=='track':
        track_cap(args.file)