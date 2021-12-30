import sys
import logging
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format='%(asctime)s.%(msecs)03d %(threadName)s %(levelname)-8s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

import time
import cv2 as cv
import detect_core
import detect_yolo
import detect_ssd
import detect_hog
import detect_haar
import detect_yolo5


def run_detect(args):
  if len(args) < 2:
    print("Usage: detect_main <file>")
    return

  f = sys.argv[1]

  IMPLS = []


  #IMPLS.append([detect_yolo5.load_yolo5(), detect_yolo5.yolo_detect_pic, "yolo5"])
  IMPLS.append([detect_yolo.load_yolo(), detect_yolo.yolo_detect_pic, "yolo4"])
  IMPLS.append([detect_yolo.load_tiny_yolo(), detect_yolo.yolo_detect_pic, "tiny_yolo"])
  IMPLS.append([detect_ssd.load_ssd(), detect_ssd.detect_ssd, "ssd"])
  IMPLS.append([detect_hog.init(), detect_hog.detect, "hog"])
  IMPLS.append([detect_haar.init(), detect_haar.detect, "haar"])

  n = 5


  for I in IMPLS:
    T = 0
    D = None
    nn = I[0]
    inf = I[1]
    name = I[2]
    for i in range(n):
      img = cv.imread(f)
      t= time.time()    
      D = inf(nn, img)
      t = time.time() - t
      T += t
    T /= n
    logging.info("Detector %s found %s persons in %s seconds avg", name, len(D), round(T, 2))
    detect_core.draw_detects(img, D, "{0}.jpg".format(name))


if __name__ == '__main__':
  run_detect(sys.argv)
