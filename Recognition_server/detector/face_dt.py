import cv2
import face_detection

detector = face_detection.build_detector("DSFDDetector", confidence_threshold=.5, nms_iou_threshold=.3)

def detect_face():
    img = cv2.VideoCapture(0)
    while True: 
        #Lectura de una imagen de un fotograma
        ret, frame = img.read()
        if ret == False:
           break
        
        detections = detector.detect(frame[:, :, ::-1])[:, :4]
        for d in detections:
            x0, y0, x1, y1 = [int(_) for _ in d]
        face = frame[y0:y1,x0:x1] 
        #cv2.rectangle(frame, (x0,y0), (x1,y1),(255,0,0), 2)        
        # cv2.imshow("plate", frame)
        cv2.imshow("plate", face)

        t = cv2.waitKey(1) 
        if t == 27:
             break
            
    img.release()
    cv2.destroyAllWindows()

        

detect_face()

    