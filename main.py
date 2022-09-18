import datetime
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2

from predictor import Predictor
from helpers.tf_utils import setup_gpu

def write_on_image(text, image, org = (25, 25)):
  img_tmp = image.copy()
  cv2.putText(img_tmp,
              text,
              org=org,
              fontFace=cv2.FONT_HERSHEY_SIMPLEX,
              fontScale=1.0,
              color=(255, 0, 0), #BGR
              thickness=2,
              lineType=cv2.LINE_AA)
  return img_tmp

def main():
  cam = cv2.VideoCapture('/dev/video0')
  starttime = None

  while True:
    # tick = time.time()
    success, frame = cam.read()
    assert success, 'Error while reading frame from camera.'
    h, w, _ = frame.shape

    pred = Predictor.predict(frame)
    # tock = time.time()
    # fps = 1.0 / (tock - tick)

    now = datetime.datetime.now()

    if pred == 1:
      if starttime == None:
        starttime = now
      dtime = now - starttime
      str_dtime = str(dtime)
      text = f'COUNT TIME {str_dtime}'
      frame = write_on_image(text, frame, org=(w//5, h//2))

    curr_time_text = 'CLOCK {}'.format(now.strftime('%H:%M:%S'))
    frame = write_on_image(curr_time_text, frame, org=(w//5, h//3))

    cv2.imshow('Personal Easy Clocking v0.1.0-alpha', frame)
    key = cv2.waitKey(1)
    if key in [27, ord('q'), ord('Q')]:
      break

if __name__=='__main__':
    setup_gpu(0)
    main()