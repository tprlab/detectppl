import cv2 as cv
import numpy as np
import detect_core

def init():
  hog = cv.HOGDescriptor()
  hog.setSVMDetector(cv.HOGDescriptor_getDefaultPeopleDetector())
  return hog


def detect(hog, img, min_area_k=0.001, min_score=0.1):
  pic, r  = detect_core.resize_pic(img, 800)
  boxes, weights = hog.detectMultiScale(pic, winStride=(8,8))
  fboxes = boxes.astype(np.float32) * r if len(boxes) > 0 else boxes
  return fboxes.astype(np.uint)






