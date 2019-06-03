import cv2
import numpy
import time

seq = [1158, 1159, 1160, 1161, 1162, 1163, 1164, 1165, 1166, 1167, 1168, 1169]

def get_image():
    for i in seq:
        file_name = "DSC_%d.JPG" % i
        yield(cv2.imread(file_name))

def histogram_equalization(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_image[:,:,2] = cv2.equalizeHist(hsv_image[:,:,2])
    return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

def main():
    idx = 0
    for image in get_image():
        image = histogram_equalization(image)
        cv2.imshow("test", image)
        idx += 1
        key = cv2.waitKey(20) & 0xFF
        if key is 27:
            return
        time.sleep(10)
        

if __name__ == "__main__":
    main()