import cv2
import mediapipe as mp
import pyautogui
import math
import time

def main():
    cam = cv2.VideoCapture(0)
    face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True) # attention to eyes and lips

    pyautogui.FAILSAFE = False
    blink_frame_counter = 0
    blink_count = 0
    last_blink_time = 0
    last_click_time = 0
    gaze_threshold = 3  # pixels
    gaze_time = 3.0  # seconds
    gaze_start_time = None
    neutral_x = None
    neutral_y = None

    RIGHT_EYELID = {
        "h1": 33, "h2": 133,
        "v1_top": 159, "v1_bot": 145,
        "v2_top": 386, "v2_bot": 374
    }

    LEFT_EYELID = {
        "h1": 263, "h2": 362,
        "v1_top": 386, "v1_bot": 374,
        "v2_top": 159, "v2_bot": 145
    }

    def dist(p1, p2):
        return math.hypot(p1.x - p2.x, p1.y - p2.y)

    def eye_ear(landmarks, eye):
        v1 = dist(landmarks[eye["v1_top"]], landmarks[eye["v1_bot"]])
        v2 = dist(landmarks[eye["v2_top"]], landmarks[eye["v2_bot"]])
        h  = dist(landmarks[eye["h1"]], landmarks[eye["h2"]])
        return (v1 + v2) / (2.0 * h)

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        output = face_mesh.process(rgb_frame)

        frame_height, frame_width = frame.shape[:2]

        if output.multi_face_landmarks:
            landmarks = output.multi_face_landmarks[0].landmark

            iris = landmarks[475]  # right iris center
            iris_x = iris.x * frame_width # gives in percentage of frame width so multiply
            iris_y = iris.y * frame_height

            cv2.circle(frame, (int(iris_x), int(iris_y)), 3, (0, 255, 0), -1)

            if neutral_x is None:
                neutral_x = iris_x
                neutral_y = iris_y

            dx = iris_x - neutral_x
            dy = iris_y - neutral_y

            jitter_allowance = 4
            sensitivity = 0.5

            speed_x = 0
            speed_y = 0

            if abs(dx) > jitter_allowance:
                speed_x = dx * sensitivity
            if abs(dy) > jitter_allowance:
                speed_y = dy * sensitivity

            # Clamp speed
            speed_x = max(-20, min(20, speed_x))
            speed_y = max(-20, min(20, speed_y))

            pyautogui.moveRel(speed_x, speed_y)

            # gaze click detection

            movement_magnitude = abs(speed_x) + abs(speed_y)

            if movement_magnitude < gaze_threshold:
                if gaze_start_time is None:
                    gaze_start_time = time.time()
                elif time.time() - gaze_start_time >= gaze_time:
                    pyautogui.click(button='left')
                    print("LEFT CLICK due to steady gaze")
                    last_click_time = time.time()
                    gaze_start_time = None
            else:
                gaze_start_time = None

            # BLINK DETECTION

            ear = (eye_ear(landmarks, RIGHT_EYELID) +
                eye_ear(landmarks, LEFT_EYELID)) / 2.0

            if ear < 0.30:
                blink_frame_counter += 1
            else:
                if blink_frame_counter >= 1:
                    blink_count += 1
                    last_blink_time = time.time()
                    print("Blink detected:", blink_count)
                blink_frame_counter = 0

            curr = time.time()

            if (
                blink_count > 0 and
                curr - last_blink_time > 0.6 and
                curr - last_click_time > 0.6
            ):
                if blink_count == 2:
                    pyautogui.click(button='left')
                    print("LEFT CLICK")
                    last_click_time = curr
                    blink_count = 0

                elif blink_count == 3:
                    pyautogui.click(button='right')
                    print("RIGHT CLICK")
                    last_click_time = curr
                    blink_count = 0

                elif blink_count >= 4:
                    blink_count = 0

        cv2.imshow("Eye Controlled Mouse", frame)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        if key == ord('c'):
            neutral_x = iris_x
            neutral_y = iris_y
            print("Recalibrated neutral gaze")

    cam.release()
    cv2.destroyAllWindows()
