from __future__ import print_function
import cv2 as cv
import numpy as np
from collections import deque

def main():
    CANNY_LOW = 50
    CANNY_HIGH = 150
    MIN_CONTOUR_AREA = 1000
    MAX_TRAIL_LENGTH = 64

    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        exit(0)

    backSub = cv.createBackgroundSubtractorMOG2(
        history=300,
        varThreshold=40,
        detectShadows=False
    )

    kernel_open = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    kernel_close = cv.getStructuringElement(cv.MORPH_ELLIPSE, (11, 11))

    trajectory = deque(maxlen=MAX_TRAIL_LENGTH)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv.flip(frame, 1)
        output = frame.copy()

        fgMask = backSub.apply(frame)
        fgMask = cv.morphologyEx(fgMask, cv.MORPH_OPEN, kernel_open)
        fgMask = cv.morphologyEx(fgMask, cv.MORPH_CLOSE, kernel_close)
        _, fgMask = cv.threshold(fgMask, 200, 255, cv.THRESH_BINARY)

        # Edge Detection
        edges = cv.Canny(fgMask, CANNY_LOW, CANNY_HIGH)

        # Contour Detection
        contours, _ = cv.findContours(
            fgMask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
        )

        if contours:
            cnt = max(contours, key=cv.contourArea)

            if cv.contourArea(cnt) > MIN_CONTOUR_AREA:

                # Contour
                cv.drawContours(output, [cnt], -1, (0, 255, 0), 2)

                # Bounding Box
                x, y, w, h = cv.boundingRect(cnt)
                cv.rectangle(output, (x, y), (x + w, y + h), (255, 0, 0), 2)

                # Convex Hull
                hull = cv.convexHull(cnt)
                cv.drawContours(output, [hull], -1, (0, 0, 255), 2)

                # Centroid
                M = cv.moments(cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    center = (cx, cy)

                    trajectory.appendleft(center)
                    cv.circle(output, center, 5, (255, 255, 255), -1)

        # 5. Trajectory
        for i in range(1, len(trajectory)):
            cv.line(output, trajectory[i - 1], trajectory[i],
                    (0, 255, 255), 2)

        cv.imshow("Object Boundary Tracker", output)
        cv.imshow("Foreground Mask", fgMask)
        cv.imshow("Edges", edges)

        if cv.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()
