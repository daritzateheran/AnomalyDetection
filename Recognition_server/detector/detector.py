import cv2
import face_detection

def draw_faces(im, bboxes):
    for bbox in bboxes:
        x0, y0, x1, y1 = [int(_) for _ in bbox]
        cv2.rectangle(im, (x0, y0), (x1, y1), (0, 0, 255), 2)

detector = face_detection.build_detector(
  "DSFDDetector", confidence_threshold=.5, nms_iou_threshold=.3)

img = cv2.VideoCapture(0)
while True:
    ret, im = img.read()
    if ret == False:
        break
    #im = cv2.imread("D:\Descargas\jolie.jpg")
    detections = detector.detect(im[:, :, ::-1])[:, :4]
    print(detections)
    draw_faces(im, detections)
    #cv2.imwrite(f"D:\Descargas\cara{i}.jpg",im)
    #i = i+1cv2.imshow("plate", img)  
    cv2.imshow("plate", img)  
    cv2.waitKey(0)

#cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),3)
