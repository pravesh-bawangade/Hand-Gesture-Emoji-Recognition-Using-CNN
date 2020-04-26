"""
Project: Hand sign emoji Recognition for Video Calling using CNN
@Author: Pravesh Bawangade
"""

import cv2
import numpy as np
from keras.models import load_model
from utils import overlay_emoji as oe


def main() -> None:
    """
    Main function to Run.
    :return: None
    """
    emoji_path = {0: "images/ok.png", 1: "images/peace.png", 2: "images/shaka.png", 3: "images/Thumbs_Up.png"}
    output = ['ok', 'peace', 'shaka', 'thumbsUp']
    model = load_model("model.h5")
    area = 0
    cap = cv2.VideoCapture(0)
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
        try:
            # find contours
            contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cnt = max(contours, key=lambda x: cv2.contourArea(x))
            # make convex hull around hand
            hull = cv2.convexHull(cnt)
            # define area of hull and area of hand
            area = cv2.contourArea(hull)
            if area >= 1800:
                mask1 = mask.reshape([1, 200, 200, 1])
                out = model.predict(mask1)
                ind = int(np.argmax(out))
                cv2.putText(frame, output[ind], (330, 330), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)

                # Overlay emoji
                overlay = cv2.imread(emoji_path[ind])
                overlay = cv2.resize(overlay, (80, 80))

                added_image = oe.overlay_emoji(frame, overlay, y=30, x=30)
                cv2.imshow('frame', added_image)
            else:
                cv2.imshow('frame', frame)
        except:
            cv2.imshow('frame', frame)

        k = cv2.waitKey(5) & 0xFF
        if k == ord("q"):
            break

    cv2.destroyAllWindows()
    cap.release()


if __name__ == "__main__":
    main()
