import cv2 as cv
import numpy as np
import time
import sys
import logging

def load_ssd():
  return cv.dnn.readNetFromTensorflow("frozen_inference_graph.pb", "ssd_mobilenet_v1_coco.pbtxt")


def detect_ssd(ssd, img, min_area_k = 0.001, thr = 0.3):
  ssd.setInput(cv.dnn.blobFromImage(img, 1.0/127.5, (300, 300), (127.5, 127.5, 127.5), swapRB=True, crop=False))
  out = ssd.forward()
  rows = img.shape[0]
  cols = img.shape[1]
  r = np.array([cols, rows, cols, rows])

  ret = []
  for d in out[0,0,:,:]:
    score = float(d[2])
    cls = int(d[1])
    if cls != 1:
      continue
    if score < thr:
      continue

    area_k = (d[5] - d[3]) * (d[6] - d[4])
    if area_k < min_area_k:
      continue

    box = d[3:7] * r
    box[2] -= box[0]
    box[3] -= box[1]
    ret.append(box.astype("int"))
  return ret


