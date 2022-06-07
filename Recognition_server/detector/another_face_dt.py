# face detection for the 5 Celebrity Faces Dataset
import cv2
from os import listdir
from os.path import isdir
from PIL import Image
from matplotlib import pyplot
import numpy as np
from numpy import savez_compressed
from mtcnn.mtcnn import MTCNN


from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

# Load the model
model = load_model('keras_model.h5')


i=0
detector = MTCNN()
img = cv2.VideoCapture(0)
while True: 
    ret, pixels = img.read()
    if ret == False:
           break
    results = detector.detect_faces(pixels)
    if results:
        x1, y1, width, height = results[0]['box']
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height
        face = pixels[y1:y2, x1:x2]
        cv2.imshow("plate", face)
        cv2.imwrite(f"D:\Descargas\detector\Daritza\Daritza_{i}.jpg",face)
        i=i+1
        # cv2.rectangle(pixels, (x1,y1), (x2,y2),(255,0,0), 2)        
        # cv2.imshow("plate", pixels)
    t = cv2.waitKey(1) 




# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1.
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
# Replace this with the path to your image
image = Image.open('<IMAGE_PATH>')
#resize the image to a 224x224 with the same strategy as in TM2:
#resizing the image to be at least 224x224 and then cropping from the center
size = (224, 224)
image = ImageOps.fit(image, size, Image.ANTIALIAS)

#turn the image into a numpy array
image_array = np.asarray(image)
# Normalize the image
normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
# Load the image into the array
data[0] = normalized_image_array

# run the inference
prediction = model.predict(data)
print(prediction)




# # load image from file
# image = Image.open("D:\Descargas\jolie.jpg")
# # convert to RGB, if needed
# image = image.convert('RGB')
# # convert to array
# pixels = np.asarray(image)


# detect faces in the image




# def detect_face():
#     img = cv2.VideoCapture(0)
#     while True: 
#         #Lectura de una imagen de un fotograma
#         ret, frame = img.read()
#         if ret == False:
#            break
        
#         detections = detector.detect(frame[:, :, ::-1])[:, :4]
#         for d in detections:
#             x0, y0, x1, y1 = [int(_) for _ in d]
#         face = frame[y0:y1,x0:x1] 
#         #cv2.rectangle(frame, (x0,y0), (x1,y1),(255,0,0), 2)        
#         # cv2.imshow("plate", frame)
#         cv2.imshow("plate", face)

#         t = cv2.waitKey(1) 
#         if t == 27:
#              break
            
#     img.release()
#     cv2.destroyAllWindows()

        

# detect_face()

    