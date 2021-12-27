import cv2 as cv
import numpy as np
import time
import sys, os
import logging




def draw_detects_simple(img, d, outpath = None):
  if d is None:
    return img
  for p in d:
    clr = (255, 0, 0)
    cv.rectangle(img, (p[0], p[1]), (p[0] + p[2], p[1] + p[3]), clr, thickness=2)

  if outpath is not None:
    cv.imwrite(outpath, img)
  return img

def draw_detects(img, d, outpath = None):
  #return tint_detects(img, d, outpath)
  return draw_detects_simple(img, d, outpath)

def tint_detects(img, d, outpath = None):
  if d is None:
    return img
  r = img.copy()
  clr = (50, 50, 150)
  for a in d:
    p = a[2]
    cv.rectangle(r, (p[0], p[1]), (p[0] + p[2], p[1] + p[3]), clr, thickness=-1)
  k = 0.5
  ret = cv.addWeighted(img, 1 - k, r, k, 0)

  if outpath is not None:
    cv.imwrite(outpath, ret)
  return ret



def resize_pic(frame, wsize):
  h, w = frame.shape[:2]
  r = w / wsize
  hsize = h / r
  return cv.resize(frame, (wsize, int(hsize))), r

