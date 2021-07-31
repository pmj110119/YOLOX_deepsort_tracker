from detector import Detector
import imutils
import cv2


def detect_img(raw_img):
    detector = Detector()
    img, pred_boxes, ratio = detector.detect(raw_img)
    result = detector.visual(pred_boxes, raw_img, ratio)
    cv2.imwrite('result.png', result)
    cv2.imshow('demo', result)
    cv2.waitKey(0)
    





def track_cap(cap):
    func_status = {}
    func_status['headpose'] = None


    detector = Detector()
    

    fps = int(cap.get(5))

    videoWriter = None
    while True:

        _, im = cap.read()
        if im is None:
            break
        
        result = detector.feedCap(im, func_status)
        result = result['frame']
        result = imutils.resize(result, height=500)
        if videoWriter is None:
            fourcc = cv2.VideoWriter_fourcc(
                'm', 'p', '4', 'v')  # opencv3.0
            videoWriter = cv2.VideoWriter(
                'result.mp4', fourcc, fps, (result.shape[1], result.shape[0]))
        videoWriter.write(result)
        cv2.imshow('demo', result)
        cv2.waitKey(1)
        if cv2.getWindowProperty('demo', cv2.WND_PROP_AUTOSIZE) < 1:
            break


    cap.release()
    videoWriter.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # file = 'dog.jpg'
    # img = cv2.imread(file)
    # detect_img(img)
    # exit()
    cap = cv2.VideoCapture('./test.mp4')
    track_cap(cap)