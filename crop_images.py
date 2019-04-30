import numpy as np
import os
import cv2
import time

if __name__ == "__main__":

  data_dir = "ycb-data"
  new_dir = "/Users/Massi/Downloads/ycb-data-cropped"
  for object_dir in os.listdir(data_dir):
      if not object_dir.startswith("."):
          for img in os.listdir(os.path.join(data_dir, object_dir)):
              if not img.startswith(".") and img.endswith(".jpg"):
                image = cv2.imread(os.path.join(data_dir, object_dir, img))
                cropped = image[262:762, 390:890]
                # print(str(cropped.shape) + " " + img)
                # print(os.path.join(new_dir, object_dir, img))
                cv2.imwrite(os.path.join(new_dir, object_dir, img), cropped)
                cv2.imshow("cropped", cropped)
                cv2.waitKey(1)
                # time.sleep(0.5)
                # input("Press enter...")
