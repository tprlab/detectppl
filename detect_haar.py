import cv2 as cv
import numpy as np
import detect_core



CASCADE_PATH = "haarcascade_fullbody.xml"

def init():
  return cv.CascadeClassifier(CASCADE_PATH)



def detect(haar, img, min_area_k=0.001, min_score=0.1):
  pic, r  = detect_core.resize_pic(img, 800)
  out = haar.detectMultiScale(pic, scaleFactor=1.3, minNeighbors=5)
  return out.astype(np.float32) * r if len(out) > 0 else out


