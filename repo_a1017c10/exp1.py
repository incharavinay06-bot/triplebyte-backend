import cv2
import mediapipe as mp
import numpy as np
import time
import sys

# ---------------- CONFIG ----------------
WINDOW_W, WINDOW_H = 960, 720
BACKGROUND_COLOR = (0, 0, 0)
HOLD_TIME = 0.35  # seconds required to hover to select
BLINK_INTERVAL = 0.25  # seconds blink interval
POST_SELECT_BLINK = 0.6  # seconds of post-selection blink

# Top tools (minimalistic icons)
TOP_TOOLS = ["Free", "Line", "Rect", "Circle", "Eraser", "Exit"]

# Left colors
LEFT_COLORS = [("Blue", (255, 0, 0)), ("Green", (0, 255, 0)), ("Red", (0, 0, 255))]

# UI geometry
TOP_H = 80
LEFT_W = 140
PANEL_MARGIN = 12

# drawing defaults
current_tool = "Free"
current_color = (255, 255, 255)
brush_size = 8
eraser_size = 40

# canvas and drawing state
canvas = np.zeros((WINDOW_H, WINDOW_W, 3), dtype=np.uint8)
canvas[:] = BACKGROUND_COLOR
xp, yp = 0, 0  # previous point for smooth free drawing

drawing = False
start_pt = None  # for shapes
preview_mode = False

# selection state
hold_start_time = None
hold_target = None  # ("top", idx) or ("left", idx)
last_blink_time = time.time()
pointer_visible = True
post_select_until = 0.0

# mediapipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# helper functions: draw minimal icons in top row
def draw_top_bar(img):
    h, w = img.shape[:2]
    cv2.rectangle(img, (0, 0), (w, TOP_H), (40, 40, 40), -1)
    box_w = w // len(TOP_TOOLS)
    for i, name in enumerate(TOP_TOOLS):
        x1 = i * box_w
        x2 = x1 + box_w
        cx = x1 + box_w // 2
        cy = TOP_H // 2
        cv2.rectangle(img, (x1, 0), (x2, TOP_H), (100, 100, 100), 2)
        # minimal icons
        if name == "Free":
            # small pencil line
            cv2.line(img, (cx - 24, cy + 14), (cx + 10, cy - 10), (255, 255, 255), 3)
            cv2.circle(img, (cx + 12, cy - 12), 4, (255,255,255), -1)
        elif name == "Line":
            cv2.line(img, (cx - 24, cy + 12), (cx + 24, cy - 12), (255, 255, 255), 4)
        elif name == "Rect":
            cv2.rectangle(img, (cx - 24, cy - 12), (cx + 24, cy + 12), (255, 255, 255), 4)
        elif name == "Circle":
            cv2.circle(img, (cx, cy), 20, (255, 255, 255), 4)
        elif name == "Eraser":
            pts = np.array([[cx-20, cy+12], [cx-5, cy-18], [cx+28, cy+12], [cx+10, cy+28]], np.int32)
            cv2.fillPoly(img, [pts], (200,200,200))
            cv2.polylines(img, [pts], True, (80,80,80), 3)
        elif name == "Exit":
            cv2.putText(img, "EXIT", (x1+10, cy+10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 3)
        # label small
        cv2.putText(img, name, (x1 + 6, TOP_H - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,200), 1)

def draw_left_colors(img):
    h, w = img.shape[:2]
    panel_y1 = TOP_H + PANEL_MARGIN
    panel_y2 = h - PANEL_MARGIN
    cv2.rectangle(img, (0, panel_y1), (LEFT_W, panel_y2), (30,30,30), -1)
    item_h = (panel_y2 - panel_y1) // len(LEFT_COLORS)
    for i, (name, col) in enumerate(LEFT_COLORS):
        y1 = panel_y1 + i * item_h
        y2 = y1 + item_h
        cv2.rectangle(img, (10, y1+8), (LEFT_W-10, y2-8), col, -1)
        # label with contrasting color
        text_col = (0,0,0) if sum(col) > 380 else (255,255,255)
        cv2.putText(img, name, (18, y1 + item_h//2 + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_col, 2)

def which_top_index(px, py, frame_w):
    if 0 <= py < TOP_H:
        box_w = frame_w // len(TOP_TOOLS)
        idx = int(px // box_w)
        idx = min(max(idx, 0), len(TOP_TOOLS)-1)
        return idx
    return None

def which_left_index(px, py):
    panel_y1 = TOP_H + PANEL_MARGIN
    panel_y2 = WINDOW_H - PANEL_MARGIN
    if not (0 <= px < LEFT_W and panel_y1 <= py < panel_y2):
        return None
    item_h = (panel_y2 - panel_y1) // len(LEFT_COLORS)
    idx = int((py - panel_y1) // item_h)
    idx = min(max(idx, 0), len(LEFT_COLORS)-1)
    return idx

def landmarks_to_px(landmarks, w, h):
    pts = []
    for lm in landmarks.landmark:
        pts.append((int(lm.x * w), int(lm.y * h)))
    return pts

# camera & mediapipe
cap = cv2.VideoCapture(0)
cv2.namedWindow("Gesture Paint", cv2.WINDOW_NORMAL)

with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.65, min_tracking_confidence=0.6) as hands:
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # Mirror view so left/right match user
            frame = cv2.flip(frame, 1)
            frame = cv2.resize(frame, (WINDOW_W, WINDOW_H))

            display = frame.copy()
            draw_top_bar(display)
            draw_left_colors(display)

            # process hand
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands.process(img_rgb)

            hand_present = False
            ix = iy = mx = my = None
            index_up = False
            middle_up = False

            if res.multi_hand_landmarks:
                hand_present = True
                lm_list = landmarks_to_px(res.multi_hand_landmarks[0], WINDOW_W, WINDOW_H)
                mp_draw.draw_landmarks(display, res.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)

                # simple finger-up detection (index tip vs pip; middle tip vs pip)
                index_up = lm_list[8][1] < lm_list[6][1]
                middle_up = lm_list[12][1] < lm_list[10][1]
                ix, iy = lm_list[8]
                mx, my = lm_list[12]

                # pointer blink timing
                now = time.time()
                if now - last_blink_time > BLINK_INTERVAL:
                    pointer_visible = not pointer_visible
                    last_blink_time = now

                # Determine dot color: green when index-only (ready to select), red when drawing (both up), blue otherwise
                if index_up and not middle_up:
                    dot_col = (0,255,0)
                elif index_up and middle_up:
                    dot_col = (0,0,255)
                else:
                    dot_col = (255,150,0)

                # SELECTION (index only) -> hover with hold
                hover_target = None
                if index_up and not middle_up:
                    top_idx = which_top_index(ix, iy, WINDOW_W)
                    left_idx = which_left_index(ix, iy)
                    if left_idx is not None:
                        hover_target = ("left", left_idx)
                    elif top_idx is not None:
                        hover_target = ("top", top_idx)
                    else:
                        hover_target = None

                    # if new hover target, start hold timer
                    if hover_target != hold_target:
                        hold_target = hover_target
                        hold_start_time = time.time() if hover_target is not None else None
                    else:
                        if hold_target is not None and (time.time() - hold_start_time) >= HOLD_TIME:
                            # perform selection for hold_target
                            kind, idx = hold_target
                            if kind == "left":
                                name, col = LEFT_COLORS[idx]
                                current_tool = "Free"
                                current_color = col
                            elif kind == "top":
                                sel = TOP_TOOLS[idx]
                                if sel == "Exit":
                                    # cleanup and exit
                                    cap.release()
                                    cv2.destroyAllWindows()
                                    sys.exit(0)
                                elif sel == "Eraser":
                                    current_tool = "Eraser"
                                else:
                                    current_tool = sel
                                # If selecting Free, ensure brush color remains current_color
                            # post-selection blink feedback
                            post_select_until = time.time() + POST_SELECT_BLINK
                            # reset hold_target so it won't re-trigger immediately
                            hold_target = None
                            hold_start_time = None
                else:
                    # not hovering
                    hold_target = None
                    hold_start_time = None

                # draw pointer (dot) with blink behavior:
                show_dot = False
                now = time.time()
                if (hold_target is not None and pointer_visible) or (now < post_select_until and pointer_visible):
                    # during hold or immediate post-select blink: larger dot
                    cv2.circle(display, (ix, iy), 12, dot_col, -1)
                else:
                    cv2.circle(display, (ix, iy), 8, dot_col, -1)

                # DRAWING when both index+middle are up
                if index_up and middle_up:
                    if not drawing:
                        drawing = True
                        start_pt = (ix, iy)
                        xp, yp = ix, iy  # reset previous point for smooth drawing
                    if current_tool == "Free":
                        # draw a smooth line from previous point to current
                        cv2.line(canvas, (xp, yp), (ix, iy), current_color, brush_size)
                        xp, yp = ix, iy
                    elif current_tool == "Eraser":
                        cv2.line(canvas, (xp, yp), (ix, iy), BACKGROUND_COLOR, eraser_size)
                        xp, yp = ix, iy
                    else:
                        # shape preview: show on display, commit when fingers lowered
                        preview = display.copy()
                        if current_tool == "Line":
                            cv2.line(preview, start_pt, (ix, iy), current_color, brush_size)
                        elif current_tool == "Rect":
                            cv2.rectangle(preview, start_pt, (ix, iy), current_color, brush_size)
                        elif current_tool == "Circle":
                            r = int(np.hypot(ix - start_pt[0], iy - start_pt[1]))
                            cv2.circle(preview, start_pt, r, current_color, brush_size)
                        combined_preview = cv2.add(preview, canvas)
                        cv2.putText(combined_preview, f"Tool: {current_tool}", (10, WINDOW_H - 12),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 2)
                        cv2.imshow("Gesture Paint", combined_preview)
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q'):
                            break
                        elif key == ord('f'):
                            prop = cv2.getWindowProperty("Gesture Paint", cv2.WND_PROP_FULLSCREEN)
                            if prop == cv2.WINDOW_FULLSCREEN:
                                cv2.setWindowProperty("Gesture Paint", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                            else:
                                cv2.setWindowProperty("Gesture Paint", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                        continue  # skip usual merging/display (we already showed preview)
                else:
                    # finalize shape when fingers lowered (if we were in drawing)
                    if drawing:
                        # if the current tool was a shape, commit it
                        if current_tool in ["Line", "Rect", "Circle"]:
                            end_pt = (ix, iy) if (ix is not None and iy is not None) else start_pt
                            if current_tool == "Line":
                                cv2.line(canvas, start_pt, end_pt, current_color, brush_size)
                            elif current_tool == "Rect":
                                cv2.rectangle(canvas, start_pt, end_pt, current_color, brush_size)
                            elif current_tool == "Circle":
                                r = int(np.hypot(end_pt[0] - start_pt[0], end_pt[1] - start_pt[1]))
                                cv2.circle(canvas, start_pt, r, current_color, brush_size)
                        # free/eraser drawing already committed as lines
                    drawing = False
                    xp, yp = 0, 0
                    start_pt = None

            else:
                # no hand - reset hold and drawing state
                hold_target = None
                hold_start_time = None
                if drawing:
                    drawing = False
                    xp, yp = 0, 0
                    start_pt = None

            # end of hand_present block

            # merge canvas & display
            merged = cv2.add(display, canvas)
            cv2.putText(merged, f"Tool: {current_tool}  Color: {current_color}  Brush: {brush_size}",
                        (10, WINDOW_H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 2)

            cv2.imshow("Gesture Paint", merged)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('f'):
                prop = cv2.getWindowProperty("Gesture Paint", cv2.WND_PROP_FULLSCREEN)
                if prop == cv2.WINDOW_FULLSCREEN:
                    cv2.setWindowProperty("Gesture Paint", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                else:
                    cv2.setWindowProperty("Gesture Paint", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    finally:
        cap.release()
        cv2.destroyAllWindows()
