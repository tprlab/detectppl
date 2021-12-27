import cv2 as cv
import numpy as np
import time
import sys, os



def load_yolo():
  return cv.dnn.readNetFromDarknet("../yolo/yolov4.cfg", "../yolo/yolov4.weights")

def load_tiny_yolo():
  return cv.dnn.readNetFromDarknet("../yolo/yolov4-tiny.cfg", "../yolo/yolov4-tiny.weights")


def yolo_detect_pic(ynn, img, min_area_k=0.001, min_score=0.1):
  rows = img.shape[0]
  cols = img.shape[1]
  r = np.array([cols, rows, cols, rows])

  ynn.setInput(cv.dnn.blobFromImage(img, 1.0/255.0, (416, 416), swapRB=True, crop=False))

  ln = ynn.getLayerNames()

  ln = [ln[i[0] - 1] for i in ynn.getUnconnectedOutLayers()]

  yout = ynn.forward(ln)

  boxes = []
  scores = []
  for out in yout:
    for detection in out:
      dts = detection[5:]
      c = np.argmax(dts)
      if c != 0:
        continue
      score = dts[c]
      if score < min_score:
        continue
      area_k = detection[2] * detection[3]
      if area_k < min_area_k:
        continue
      box = detection[0:4] * r

      boxes.append(box.astype("int"))
      scores.append(float(score))

  ret = []
  dxs = cv.dnn.NMSBoxes(boxes, scores, 0.3, 0.1)
  if dxs is None:
    return ret
  try:
    for i in dxs.flatten():
      xc, yc, w, h = boxes[i]


      #if w * h < min_area:
      #  continue
      w2 = int(w/2)
      h2 = int(h/2)
      ret.append([xc - w2, yc - h2, w, h])
  except:
    pass
  return ret

