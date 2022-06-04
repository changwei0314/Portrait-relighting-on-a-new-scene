# taken from https://machinelearningmastery.com/how-to-perform-face-detection-with-classical-and-deep-learning-methods-in-python-with-keras/
# plot photo with detected faces using opencv cascade classifier
import cv2
from cv2 import imread, resize
from cv2 import imshow
from cv2 import waitKey
from cv2 import destroyAllWindows
from cv2 import CascadeClassifier
from cv2 import rectangle

def cropFace(img):

    # load the photograph
    # pixels = img
    pixels = img

    # load the pre-trained model
    classifier = CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')

    # perform face detection
    bboxes = classifier.detectMultiScale(pixels)

    if len(bboxes) == 0:
        print("ERROR: No faces found.")

    # extract
    x, y, width, height = bboxes[0]
    x2, y2 = x + width, y + height
    
    BUFFER = int(width * 0.25)

    # show the image
    image = pixels[max(y - BUFFER, 0):min(y2 + BUFFER, pixels.shape[0]), max(x - BUFFER, 0):min(x2 + BUFFER, pixels.shape[1])]
    imshow('face', image)
    waitKey(0)
    return image

def cropFaceWithPos(img):

    # load the photograph
    # pixels = img
    pixels = img

    # load the pre-trained model
    classifier = CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')

    # perform face detection
    bboxes = classifier.detectMultiScale(pixels)

    if len(bboxes) == 0:
        print("ERROR: No faces found.")

    # extract
    x, y, width, height = bboxes[0]
    x2, y2 = x + width, y + height
    
    BUFFER = int(width * 0.25)

    # show the image
    image = pixels[max(y - BUFFER, 0):min(y2 + BUFFER, pixels.shape[0]), max(x - BUFFER, 0):min(x2 + BUFFER, pixels.shape[1])]
    # imshow('face', image)
    # waitKey(0)
    return image,x,y,width, height
