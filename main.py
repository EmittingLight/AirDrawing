import cv2
import mediapipe as mp
import math
import numpy as np
import random

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

canvas = None
prev_x, prev_y = None, None

smooth_x, smooth_y = None, None
SMOOTHING = 0.2

particles = []

WINDOW_NAME = "Hand Drawing"

# Счётчик для жеста очистки
clear_gesture_frames = 0
CLEAR_HOLD_FRAMES = 15  # сколько кадров держать открытую ладонь


def is_finger_up(landmarks, tip_id, pip_id):
    return landmarks[tip_id].y < landmarks[pip_id].y


def calc_angle(a, b, c):
    ab = (a[0] - b[0], a[1] - b[1])
    cb = (c[0] - b[0], c[1] - b[1])

    dot = ab[0] * cb[0] + ab[1] * cb[1]
    ab_len = math.hypot(ab[0], ab[1])
    cb_len = math.hypot(cb[0], cb[1])

    if ab_len == 0 or cb_len == 0:
        return 0

    cos_angle = dot / (ab_len * cb_len)
    cos_angle = max(-1.0, min(1.0, cos_angle))
    return math.degrees(math.acos(cos_angle))


def thumb_debug_values(landmarks):
    p2 = (landmarks[2].x, landmarks[2].y)
    p3 = (landmarks[3].x, landmarks[3].y)
    p4 = (landmarks[4].x, landmarks[4].y)
    p5 = (landmarks[5].x, landmarks[5].y)

    angle = calc_angle(p2, p3, p4)
    dist_tip_to_index_base = math.hypot(p4[0] - p5[0], p4[1] - p5[1])
    dist_thumb_base_to_index_base = math.hypot(p2[0] - p5[0], p2[1] - p5[1])

    return angle, dist_tip_to_index_base, dist_thumb_base_to_index_base


def is_thumb_up(landmarks):
    angle, d_tip, d_base = thumb_debug_values(landmarks)
    return angle > 160 and d_tip < d_base


def count_fingers(hand_landmarks):
    landmarks = hand_landmarks.landmark
    fingers_up = 0

    if is_thumb_up(landmarks):
        fingers_up += 1

    finger_tips = [8, 12, 16, 20]
    finger_pips = [6, 10, 14, 18]

    for tip_id, pip_id in zip(finger_tips, finger_pips):
        if is_finger_up(landmarks, tip_id, pip_id):
            fingers_up += 1

    return fingers_up


def only_index_finger_up(landmarks):
    index_up = is_finger_up(landmarks, 8, 6)
    middle_up = is_finger_up(landmarks, 12, 10)
    ring_up = is_finger_up(landmarks, 16, 14)
    pinky_up = is_finger_up(landmarks, 20, 18)

    # Большой палец игнорируем, чтобы режим рисования
    # работал стабильнее для обеих рук
    return index_up and not middle_up and not ring_up and not pinky_up


def is_open_palm(hand_landmarks):
    """
    Жест открытой ладони:
    4 или 5 пальцев подняты.
    Это надёжнее, чем требовать идеальный большой палец.
    """
    fingers = count_fingers(hand_landmarks)
    return fingers >= 4


def draw_neon_line(canvas, x1, y1, x2, y2):
    # Внешнее свечение
    cv2.line(canvas, (x1, y1), (x2, y2), (255, 200, 0), 10)

    # Средний слой
    cv2.line(canvas, (x1, y1), (x2, y2), (255, 255, 0), 5)

    # Яркое ядро
    cv2.line(canvas, (x1, y1), (x2, y2), (255, 255, 150), 2)


def add_sparks(x, y):
    for _ in range(5):
        particles.append({
            "x": x,
            "y": y,
            "dx": random.uniform(-3, 3),
            "dy": random.uniform(-3, 3),
            "life": random.randint(8, 16)
        })


def update_particles(frame):
    for p in particles[:]:
        p["x"] += p["dx"]
        p["y"] += p["dy"]
        p["life"] -= 1

        cv2.circle(frame, (int(p["x"]), int(p["y"])), 2, (255, 255, 180), -1)

        if p["life"] <= 0:
            particles.remove(p)


def clear_canvas():
    global canvas, particles, prev_x, prev_y, smooth_x, smooth_y
    if canvas is not None:
        canvas[:] = 0
    particles.clear()
    prev_x, prev_y = None, None
    smooth_x, smooth_y = None, None


cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, 960, 720)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Ошибка камеры")
        break

    frame = cv2.flip(frame, 1)

    if canvas is None:
        canvas = np.zeros_like(frame)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    fingers_count = 0
    drawing_mode = False
    open_palm_detected = False

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmarks = hand_landmarks.landmark
            fingers_count = count_fingers(hand_landmarks)

            # Проверяем жест открытой ладони
            if is_open_palm(hand_landmarks):
                open_palm_detected = True

            # Режим рисования
            if only_index_finger_up(landmarks):
                drawing_mode = True

                h, w, _ = frame.shape
                x = int(landmarks[8].x * w)
                y = int(landmarks[8].y * h)

                if smooth_x is None or smooth_y is None:
                    smooth_x, smooth_y = x, y
                else:
                    smooth_x = int(smooth_x + (x - smooth_x) * SMOOTHING)
                    smooth_y = int(smooth_y + (y - smooth_y) * SMOOTHING)

                cv2.circle(frame, (smooth_x, smooth_y), 10, (0, 255, 255), -1)

                if prev_x is None or prev_y is None:
                    prev_x, prev_y = smooth_x, smooth_y

                draw_neon_line(canvas, prev_x, prev_y, smooth_x, smooth_y)
                add_sparks(smooth_x, smooth_y)

                prev_x, prev_y = smooth_x, smooth_y
            else:
                prev_x, prev_y = None, None
                smooth_x, smooth_y = None, None
    else:
        prev_x, prev_y = None, None
        smooth_x, smooth_y = None, None

    # Логика очистки жестом
    if open_palm_detected and not drawing_mode:
        clear_gesture_frames += 1
    else:
        clear_gesture_frames = 0

    if clear_gesture_frames >= CLEAR_HOLD_FRAMES:
        clear_canvas()
        clear_gesture_frames = 0

    # Мягкое свечение
    glow = cv2.GaussianBlur(canvas, (0, 0), 6)

    # Сначала glow
    frame = cv2.addWeighted(frame, 1.0, glow, 0.3, 0)

    # Потом сам холст мягко
    frame = cv2.addWeighted(frame, 1.0, canvas, 0.7, 0)

    # Искры поверх
    update_particles(frame)

    cv2.putText(
        frame,
        f"Fingers: {fingers_count}",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    if drawing_mode:
        cv2.putText(
            frame,
            "DRAWING",
            (20, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2
        )

    # Подсказка про очистку жестом
    if open_palm_detected and not drawing_mode:
        progress = int((clear_gesture_frames / CLEAR_HOLD_FRAMES) * 100)
        cv2.putText(
            frame,
            f"CLEARING... {progress}%",
            (20, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 100, 255),
            2
        )

    _, _, window_w, window_h = cv2.getWindowImageRect(WINDOW_NAME)

    if window_w > 0 and window_h > 0:
        display_frame = cv2.resize(frame, (window_w, window_h))
    else:
        display_frame = frame

    cv2.imshow(WINDOW_NAME, display_frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    if key == ord('c'):
        clear_canvas()
        clear_gesture_frames = 0

cap.release()
cv2.destroyAllWindows()