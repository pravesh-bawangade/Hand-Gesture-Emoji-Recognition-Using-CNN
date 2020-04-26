"""
Project: Hand sign emoji Recognition for Video Calling using CNN
@Author: Pravesh Bawangade
"""
import cv2
import numpy as np
import time
import os


def data_collection(label = "", sec = 20) -> None:
    """
    This function collects image data and for specified time(in Sec) and save it in folder "Data"
    :param label: Label of data being collected. example:Ok, peace,shaka
    :param sec: Time span for which data is collected.
    :return: None
    """

    if not os.path.exists("Data/"+label):
        os.mkdir("Data/"+label)
    count = 0
    cap = cv2.VideoCapture(0)
    time.sleep(5)
    start = time.time()
    while True:
        # (x_s, y_s) = (320,240)
        x_s, y_s = (640, 480)
        width_start, width_size = int(x_s / 6.4), int((x_s / 3.2) + (x_s / 6.4))

        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (x_s, y_s))
        kernel = np.ones((3, 3), np.uint8)

        # define region of interest
        roi = frame[width_start:width_size, width_start:width_size]

        cv2.rectangle(frame, (width_start, width_start), (width_size, width_size), (0, 255, 0), 0)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # define range of skin color in HSV
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)

        # extract skin colur imagw
        mask = cv2.inRange(hsv, lower_skin, upper_skin)

        # extrapolate the hand to fill dark spots within
        mask = cv2.dilate(mask, kernel, iterations=4)

        # blur the image
        mask = cv2.GaussianBlur(mask, (5, 5), 100)

        # write file in directory
        cv2.imwrite("Data/"+label+"/"+label+str(count)+".png",mask)
        count += 1

        # show the windows
        cv2.imshow('mask', mask)
        cv2.imshow(label, frame)

        stop = time.time()
        k = cv2.waitKey(5) & 0xFF
        diff = abs(stop-start)
        if (k == ord("q")) or (diff >= sec):
            break
    cv2.destroyAllWindows()
    cap.release()


if __name__ == "__main__":
    data_collection(label="thumbsUp", sec=15)