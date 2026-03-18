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
win_particles = []

WINDOW_NAME = "Air TicTacToe"

clear_gesture_frames = 0
CLEAR_HOLD_FRAMES = 15

stroke_points = []
detected_shape = ""
detected_shape_timer = 0
status_message = ""
status_timer = 0

board = [["" for _ in range(3)] for _ in range(3)]

selected_row = 1
selected_col = 1

game_over = False
winner_text = ""
winning_cells = []

# Анимация победной линии
win_line_progress = 0.0
WIN_LINE_SPEED = 0.04
frame_counter = 0


def is_finger_up(landmarks, tip_id, pip_id):
    return landmarks[tip_id].y < landmarks[pip_id].y


def calc_angle(a, b, c):
    ab = (a[0] - b[0], a[1] - b[1])
    cb = (c[0] - b[0], c[1] - b[1])

    dot = ab[0] * cb[0] + ab[1] * cb[1]
    ab_len = math.hypot(ab[0] - 0, ab[1] - 0)
    cb_len = math.hypot(cb[0] - 0, cb[1] - 0)

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
    return index_up and not middle_up and not ring_up and not pinky_up


def is_open_palm(hand_landmarks):
    return count_fingers(hand_landmarks) >= 4


def draw_neon_line(canvas, x1, y1, x2, y2):
    cv2.line(canvas, (x1, y1), (x2, y2), (255, 200, 0), 10)
    cv2.line(canvas, (x1, y1), (x2, y2), (255, 255, 0), 5)
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


def add_win_particles(x, y):
    for _ in range(8):
        win_particles.append({
            "x": x,
            "y": y,
            "dx": random.uniform(-4, 4),
            "dy": random.uniform(-4, 4),
            "life": random.randint(10, 20)
        })


def update_win_particles(frame):
    for p in win_particles[:]:
        p["x"] += p["dx"]
        p["y"] += p["dy"]
        p["life"] -= 1

        size = 3 if p["life"] > 8 else 2
        cv2.circle(frame, (int(p["x"]), int(p["y"])), size, (0, 255, 255), -1)

        if p["life"] <= 0:
            win_particles.remove(p)


def clear_canvas():
    global canvas, particles, prev_x, prev_y, smooth_x, smooth_y, stroke_points, win_particles
    if canvas is not None:
        canvas[:] = 0
    particles.clear()
    win_particles.clear()
    prev_x, prev_y = None, None
    smooth_x, smooth_y = None, None
    stroke_points = []


def reset_board():
    global board, game_over, winner_text, winning_cells, win_line_progress, win_particles
    board = [["" for _ in range(3)] for _ in range(3)]
    game_over = False
    winner_text = ""
    winning_cells = []
    win_line_progress = 0.0
    win_particles.clear()


def clear_selected_cell(frame_w, frame_h):
    global status_message, status_timer, canvas, particles

    if game_over:
        status_message = "Game over. Press C for new game"
        status_timer = 90
        return

    cell_w = frame_w // 3
    cell_h = frame_h // 3

    x1 = selected_col * cell_w
    y1 = selected_row * cell_h
    x2 = x1 + cell_w
    y2 = y1 + cell_h

    was_empty = (board[selected_row][selected_col] == "")
    board[selected_row][selected_col] = ""

    if canvas is not None:
        canvas[y1:y2, x1:x2] = 0

    particles[:] = [
        p for p in particles
        if not (x1 <= int(p["x"]) < x2 and y1 <= int(p["y"]) < y2)
    ]

    if was_empty:
        status_message = f"Cell {selected_row + 1},{selected_col + 1} already empty"
    else:
        status_message = f"Cleared cell {selected_row + 1},{selected_col + 1}"

    status_timer = 90


def distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def simplify_points(points, min_dist=8):
    if not points:
        return []

    simplified = [points[0]]
    for p in points[1:]:
        if distance(p, simplified[-1]) >= min_dist:
            simplified.append(p)

    if len(simplified) == 1 and points:
        simplified.append(points[-1])

    return simplified


def path_length(points):
    total = 0.0
    for i in range(1, len(points)):
        total += distance(points[i - 1], points[i])
    return total


def detect_shape(points):
    if len(points) < 25:
        return "?"

    pts = simplify_points(points, min_dist=6)
    if len(pts) < 12:
        return "?"

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    width = max_x - min_x
    height = max_y - min_y

    if width < 40 or height < 40:
        return "?"

    bbox_ratio = width / height if height != 0 else 999
    diag = math.hypot(width, height)

    start_point = pts[0]
    end_point = pts[-1]
    end_distance = distance(start_point, end_point)
    close_ratio = end_distance / diag if diag != 0 else 999

    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2

    center_hits = 0
    center_rx = width * 0.18
    center_ry = height * 0.18
    for x, y in pts:
        if abs(x - center_x) <= center_rx and abs(y - center_y) <= center_ry:
            center_hits += 1
    center_ratio = center_hits / len(pts)

    quadrants = set()
    for x, y in pts:
        qx = 0 if x < center_x else 1
        qy = 0 if y < center_y else 1
        quadrants.add((qx, qy))

    radii = [math.hypot(x - center_x, y - center_y) for x, y in pts]
    mean_radius = sum(radii) / len(radii)
    if mean_radius == 0:
        return "?"
    radius_std = (sum((r - mean_radius) ** 2 for r in radii) / len(radii)) ** 0.5
    radius_std_ratio = radius_std / mean_radius

    diag1_hits = 0
    diag2_hits = 0
    useful_segments = 0

    for i in range(1, len(pts)):
        dx = pts[i][0] - pts[i - 1][0]
        dy = pts[i][1] - pts[i - 1][1]
        seg_len = math.hypot(dx, dy)

        if seg_len < 5:
            continue

        useful_segments += 1

        angle = abs(math.degrees(math.atan2(dy, dx)))
        angle = angle % 180

        if 20 <= angle <= 70:
            diag1_hits += 1
        elif 110 <= angle <= 160:
            diag2_hits += 1

    if useful_segments == 0:
        return "?"

    diag1_ratio = diag1_hits / useful_segments
    diag2_ratio = diag2_hits / useful_segments

    total_len = path_length(pts)
    perimeter_like = 2 * (width + height)
    len_box_ratio = total_len / perimeter_like if perimeter_like != 0 else 999

    if (
        0.55 <= bbox_ratio <= 1.6
        and close_ratio < 0.38
        and radius_std_ratio < 0.55
        and center_ratio < 0.30
        and 0.45 <= len_box_ratio <= 2.2
    ):
        return "O"

    if (
        0.45 <= bbox_ratio <= 1.9
        and close_ratio > 0.22
        and center_ratio > 0.12
        and len(quadrants) == 4
        and diag1_ratio > 0.22
        and diag2_ratio > 0.22
    ):
        return "X"

    return "?"


def get_stroke_bounds(points):
    if not points:
        return None

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    return min(xs), min(ys), max(xs), max(ys)


def get_stroke_cell(points, frame_w, frame_h):
    bounds = get_stroke_bounds(points)
    if bounds is None:
        return None

    min_x, min_y, max_x, max_y = bounds

    cell_w = frame_w // 3
    cell_h = frame_h // 3

    best_row, best_col = None, None
    best_overlap = 0

    for row in range(3):
        for col in range(3):
            cell_x1 = col * cell_w
            cell_y1 = row * cell_h
            cell_x2 = cell_x1 + cell_w
            cell_y2 = cell_y1 + cell_h

            overlap_x1 = max(min_x, cell_x1)
            overlap_y1 = max(min_y, cell_y1)
            overlap_x2 = min(max_x, cell_x2)
            overlap_y2 = min(max_y, cell_y2)

            overlap_w = max(0, overlap_x2 - overlap_x1)
            overlap_h = max(0, overlap_y2 - overlap_y1)
            overlap_area = overlap_w * overlap_h

            if overlap_area > best_overlap:
                best_overlap = overlap_area
                best_row, best_col = row, col

    if best_row is None:
        return None

    return best_row, best_col


def check_winner():
    for row in range(3):
        if board[row][0] != "" and board[row][0] == board[row][1] == board[row][2]:
            return board[row][0], [(row, 0), (row, 1), (row, 2)]

    for col in range(3):
        if board[0][col] != "" and board[0][col] == board[1][col] == board[2][col]:
            return board[0][col], [(0, col), (1, col), (2, col)]

    if board[0][0] != "" and board[0][0] == board[1][1] == board[2][2]:
        return board[0][0], [(0, 0), (1, 1), (2, 2)]

    if board[0][2] != "" and board[0][2] == board[1][1] == board[2][0]:
        return board[0][2], [(0, 2), (1, 1), (2, 0)]

    return None, []


def is_draw():
    for row in range(3):
        for col in range(3):
            if board[row][col] == "":
                return False
    return True


def update_game_state():
    global game_over, winner_text, status_message, status_timer, winning_cells, win_line_progress

    winner, cells = check_winner()
    if winner:
        game_over = True
        winner_text = f"{winner} WINS!"
        winning_cells = cells
        win_line_progress = 0.0
        status_message = "Press C for new game"
        status_timer = 120
        return

    if is_draw():
        game_over = True
        winner_text = "DRAW!"
        winning_cells = []
        status_message = "Press C for new game"
        status_timer = 120


def place_symbol(points, symbol, frame_w, frame_h):
    global status_message, status_timer

    if game_over:
        status_message = "Game over. Press C for new game"
        status_timer = 90
        return

    cell = get_stroke_cell(points, frame_w, frame_h)
    bounds = get_stroke_bounds(points)

    if cell is None or bounds is None:
        status_message = "No cell detected"
        status_timer = 90
        return

    row, col = cell
    min_x, min_y, max_x, max_y = bounds

    cell_w = frame_w // 3
    cell_h = frame_h // 3

    cell_x1 = col * cell_w
    cell_y1 = row * cell_h
    cell_x2 = cell_x1 + cell_w
    cell_y2 = cell_y1 + cell_h

    overlap_x1 = max(min_x, cell_x1)
    overlap_y1 = max(min_y, cell_y1)
    overlap_x2 = min(max_x, cell_x2)
    overlap_y2 = min(max_y, cell_y2)

    overlap_w = max(0, overlap_x2 - overlap_x1)
    overlap_h = max(0, overlap_y2 - overlap_y1)
    overlap_area = overlap_w * overlap_h

    shape_area = max(1, (max_x - min_x) * (max_y - min_y))
    overlap_ratio = overlap_area / shape_area

    if overlap_ratio < 0.25:
        status_message = "Draw more inside one cell"
        status_timer = 90
        return

    if board[row][col] != "":
        status_message = "Cell already occupied"
        status_timer = 90
        return

    board[row][col] = symbol
    status_message = f"Placed {symbol} at row {row + 1}, col {col + 1}"
    status_timer = 90

    update_game_state()


def get_cell_center(row, col, frame_w, frame_h):
    cell_w = frame_w // 3
    cell_h = frame_h // 3
    cx = col * cell_w + cell_w // 2
    cy = row * cell_h + cell_h // 2
    return cx, cy


def draw_board(frame):
    global selected_row, selected_col, frame_counter, win_line_progress

    h, w, _ = frame.shape
    cell_w = w // 3
    cell_h = h // 3

    line_color = (0, 255, 255)
    thickness = 2

    cv2.line(frame, (cell_w, 0), (cell_w, h), line_color, thickness)
    cv2.line(frame, (cell_w * 2, 0), (cell_w * 2, h), line_color, thickness)

    cv2.line(frame, (0, cell_h), (w, cell_h), line_color, thickness)
    cv2.line(frame, (0, cell_h * 2), (w, cell_h * 2), line_color, thickness)

    # Подсветка выбранной ячейки
    x1 = selected_col * cell_w
    y1 = selected_row * cell_h
    x2 = x1 + cell_w
    y2 = y1 + cell_h
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

    # Пульсация выигрышных клеток
    pulse_thickness = 2 + int((math.sin(frame_counter * 0.25) + 1) * 2)

    for row in range(3):
        for col in range(3):
            symbol = board[row][col]
            if symbol == "":
                continue

            x1 = col * cell_w
            y1 = row * cell_h
            x2 = x1 + cell_w
            y2 = y1 + cell_h

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # Подсветка выигрышных ячеек
            if (row, col) in winning_cells:
                cv2.rectangle(frame, (x1 + 4, y1 + 4), (x2 - 4, y2 - 4), (0, 0, 255), pulse_thickness)

            if symbol == "X":
                offset_x = cell_w // 4
                offset_y = cell_h // 4
                color = (0, 255, 0)
                thick = 5 if (row, col) in winning_cells else 4
                cv2.line(frame, (cx - offset_x, cy - offset_y), (cx + offset_x, cy + offset_y), color, thick)
                cv2.line(frame, (cx + offset_x, cy - offset_y), (cx - offset_x, cy + offset_y), color, thick)

            elif symbol == "O":
                radius = min(cell_w, cell_h) // 4
                color = (255, 255, 0)
                thick = 5 if (row, col) in winning_cells else 4
                cv2.circle(frame, (cx, cy), radius, color, thick)

    # Анимированная победная линия
    winner, cells = check_winner()
    if winner and len(cells) == 3:
        start_row, start_col = cells[0]
        end_row, end_col = cells[2]

        start_x, start_y = get_cell_center(start_row, start_col, w, h)
        end_x, end_y = get_cell_center(end_row, end_col, w, h)

        if win_line_progress < 1.0:
            win_line_progress += WIN_LINE_SPEED
            win_line_progress = min(1.0, win_line_progress)

        current_x = int(start_x + (end_x - start_x) * win_line_progress)
        current_y = int(start_y + (end_y - start_y) * win_line_progress)

        # Неоновая линия победы
        cv2.line(frame, (start_x, start_y), (current_x, current_y), (0, 0, 255), 8)
        cv2.line(frame, (start_x, start_y), (current_x, current_y), (0, 255, 255), 3)

        # Искры на кончике линии
        if frame_counter % 2 == 0:
            add_win_particles(current_x, current_y)


cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, 1100, 800)

was_drawing_last_frame = False

while True:
    frame_counter += 1

    ret, frame = cap.read()
    if not ret:
        print("Ошибка камеры")
        break

    frame = cv2.flip(frame, 1)

    if canvas is None:
        canvas = np.zeros_like(frame)

    frame_h, frame_w, _ = frame.shape
    cell_w = frame_w // 3
    cell_h = frame_h // 3

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

            if is_open_palm(hand_landmarks):
                open_palm_detected = True

            if only_index_finger_up(landmarks):
                drawing_mode = True

                x = int(landmarks[8].x * frame_w)
                y = int(landmarks[8].y * frame_h)

                selected_col = min(2, max(0, x // cell_w))
                selected_row = min(2, max(0, y // cell_h))

                if smooth_x is None or smooth_y is None:
                    smooth_x, smooth_y = x, y
                else:
                    smooth_x = int(smooth_x + (x - smooth_x) * SMOOTHING)
                    smooth_y = int(smooth_y + (y - smooth_y) * SMOOTHING)

                cv2.circle(frame, (smooth_x, smooth_y), 10, (0, 255, 255), -1)

                if prev_x is None or prev_y is None:
                    prev_x, prev_y = smooth_x, smooth_y

                if not game_over:
                    draw_neon_line(canvas, prev_x, prev_y, smooth_x, smooth_y)
                    add_sparks(smooth_x, smooth_y)
                    stroke_points.append((smooth_x, smooth_y))

                prev_x, prev_y = smooth_x, smooth_y
            else:
                prev_x, prev_y = None, None
                smooth_x, smooth_y = None, None
    else:
        prev_x, prev_y = None, None
        smooth_x, smooth_y = None, None

    if was_drawing_last_frame and not drawing_mode:
        if len(stroke_points) >= 25:
            detected_shape = detect_shape(stroke_points)
            detected_shape_timer = 90

            if detected_shape in ["X", "O"]:
                place_symbol(stroke_points, detected_shape, frame_w, frame_h)

        stroke_points = []

    was_drawing_last_frame = drawing_mode

    if open_palm_detected and not drawing_mode:
        clear_gesture_frames += 1
    else:
        clear_gesture_frames = 0

    if clear_gesture_frames >= CLEAR_HOLD_FRAMES:
        clear_selected_cell(frame_w, frame_h)
        clear_gesture_frames = 0
        detected_shape = ""
        detected_shape_timer = 0

    glow = cv2.GaussianBlur(canvas, (0, 0), 6)
    frame = cv2.addWeighted(frame, 1.0, glow, 0.3, 0)
    frame = cv2.addWeighted(frame, 1.0, canvas, 0.7, 0)

    update_particles(frame)
    update_win_particles(frame)
    draw_board(frame)

    cv2.putText(
        frame,
        f"Fingers: {fingers_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    if drawing_mode and not game_over:
        cv2.putText(
            frame,
            "DRAWING",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2
        )

    if open_palm_detected and not drawing_mode and not game_over:
        progress = int((clear_gesture_frames / CLEAR_HOLD_FRAMES) * 100)
        cv2.putText(
            frame,
            f"CELL CLEAR... {progress}%",
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 100, 255),
            2
        )

    if detected_shape_timer > 0 and detected_shape:
        color = (255, 255, 0) if detected_shape != "?" else (0, 100, 255)
        cv2.putText(
            frame,
            f"Detected: {detected_shape}",
            (20, 160),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            color,
            2
        )
        detected_shape_timer -= 1

    if status_timer > 0 and status_message:
        cv2.putText(
            frame,
            status_message,
            (20, 200),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2
        )
        status_timer -= 1

    if game_over and winner_text:
        cv2.putText(
            frame,
            winner_text,
            (20, 255),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.3,
            (0, 0, 255),
            3
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
        reset_board()
        clear_gesture_frames = 0
        detected_shape = ""
        detected_shape_timer = 0
        status_message = "New game started"
        status_timer = 90

cap.release()
cv2.destroyAllWindows()