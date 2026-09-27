# Import necessary modules
import pickle  # Module for serializing and deserializing Python objects
import math
import cv2  # OpenCV for video capture and image processing
import mediapipe as mp  # MediaPipe for hand detection and landmark processing
import numpy as np  # NumPy for array and numerical operations

# Load the pre-trained model from a pickle file
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Initialize video capture with DirectShow fallback on Windows
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    cap = cv2.VideoCapture(0)

# Initialize MediaPipe's hand detection and drawing utilities
mp_hands = mp.solutions.hands  # Hands solution from MediaPipe
mp_drawing = mp.solutions.drawing_utils  # Drawing utilities for visualization
mp_drawing_styles = mp.solutions.drawing_styles  # Predefined drawing styles for landmarks

def is_hand_pointing_up(hand_landmarks):
    lm = hand_landmarks.landmark
    idx_tip, idx_pip, idx_mcp, wrist = lm[8], lm[6], lm[5], lm[0]
    if not (idx_tip.y < idx_pip.y and idx_tip.y < idx_mcp.y and idx_tip.y < wrist.y - 0.04):
        return False
    dy = abs(idx_mcp.y - idx_tip.y)
    dx = abs(idx_mcp.x - idx_tip.x)
    if dy < dx * 0.40:
        return False
    mid_curled = (lm[12].y > idx_tip.y + 0.035) or (lm[12].y > lm[10].y)
    ring_curled = (lm[16].y > idx_tip.y + 0.035) or (lm[16].y > lm[14].y)
    pinky_curled = (lm[20].y > idx_tip.y + 0.035) or (lm[20].y > lm[18].y)
    if not (mid_curled and ring_curled and pinky_curled):
        return False
    if idx_tip.y > lm[12].y or idx_tip.y > lm[16].y or idx_tip.y > lm[20].y:
        return False
    return True


def mirror_features(data_aux):
    """
    Computes scale-invariant canonicalized features for left-handed gesture recognition.
    Reflects x-coordinates horizontally and re-normalizes to ensure dual-chirality parity.
    """
    xs = np.array(data_aux[0::2])
    ys = np.array(data_aux[1::2])
    xs_m = 1.0 - xs
    min_x, max_x = np.min(xs_m), np.max(xs_m)
    min_y, max_y = np.min(ys), np.max(ys)
    scale = max(max_x - min_x, max_y - min_y)
    if scale < 1e-4:
        scale = 1.0
    out = []
    for x, y in zip((xs_m - min_x) / scale, (ys - min_y) / scale):
        out.extend([float(x), float(y)])
    return out


def targeted_disambiguation(class_id, confidence, data_aux):
    """
    Carefully resolves specific confusions requested by the user,
    leaving ALL other alphabets and words 100% untouched:
      1. R (17) vs D (3): D has ONLY index finger up, middle curled. R has BOTH index and middle up with twist/cross.
         If two fingers are pointing upwards, D is strictly forbidden.
      2. D (3) vs Z (25): D points straight up (lean >= -0.20), Z leans to the left (lean < -0.22).
      3. R (17) vs U (20)/V (21): R has crossed index and middle fingers.
      4. J (9) vs I (8): J has hooked/tilted pinky or elevated thumb, I is straight vertical pinky.
    """
    if len(data_aux) != 42:
        return class_id, confidence

    xs = data_aux[0::2]
    ys = data_aux[1::2]

    # Finger extension states (smaller y means higher in the image frame)
    idx_up = ys[8] < ys[6]
    mid_up = ys[12] < ys[10]

    # --- 1. Prevent D when two fingers are pointing upwards (R vs D vs U vs V) ---
    if class_id == 3 and mid_up:
        knuckle_dx = xs[5] - xs[9]
        tip_dx = xs[8] - xs[12]
        cross_metric = knuckle_dx * tip_dx
        tip_dist = abs(xs[8] - xs[12])
        # Strict twist / cross test: cross_metric must be strictly negative!
        if cross_metric < -0.0001:
            return 17, max(confidence, 0.92)  # R: crossed / twisted fingers
        else:
            return (21 if tip_dist >= 0.14 else 20), max(confidence, 0.90)  # U: straight together, V: straight spread

    # --- 2. D (3) vs Z (25) ONLY (only valid when middle finger is NOT up) ---
    if class_id in [3, 25]:
        if mid_up:
            knuckle_dx = xs[5] - xs[9]
            tip_dx = xs[8] - xs[12]
            cross_metric = knuckle_dx * tip_dx
            tip_dist = abs(xs[8] - xs[12])
            if cross_metric < -0.0001:
                return 17, max(confidence, 0.90)
            else:
                return (21 if tip_dist >= 0.14 else 20), max(confidence, 0.90)
        dx = xs[8] - xs[5]
        dy = ys[5] - ys[8]
        if dy > 0.05:
            lean = dx / dy
            if lean < -0.22:
                return 25, max(confidence, 0.90)  # Z: leaning to left
            elif lean >= -0.20:
                return 3, max(confidence, 0.90)   # D: pointing straight up

    # --- 3. R (17) vs U (20)/V (21) STRICT DISAMBIGUATION ---
    # R: index and middle fingers MUST BE TWISTED / CROSSED (cross_metric < -0.0001).
    # U: fingers MUST BE STRAIGHT & PARALLEL TOGETHER (cross_metric >= -0.0001, tip_dist < 0.14).
    # V: fingers MUST BE STRAIGHT & SPREAD APART (cross_metric >= -0.0001, tip_dist >= 0.14).
    elif class_id in [17, 20, 21]:
        knuckle_dx = xs[5] - xs[9]
        tip_dx = xs[8] - xs[12]
        cross_metric = knuckle_dx * tip_dx
        tip_dist = abs(xs[8] - xs[12])

        # If index and middle fingers are twisted / crossed:
        if cross_metric < -0.0001:
            return 17, max(confidence, 0.92)  # R: twisted/crossed fingers
        else:
            # Fingers are uncrossed and straight!
            if tip_dist >= 0.14:
                return 21, max(confidence, 0.92)  # V: straight and spread
            else:
                return 20, max(confidence, 0.92)  # U: straight and together

    # --- 4. J (9) vs I (8) vs Y (24) STRICT THUMB DISAMBIGUATION ---
    # In Y: BOTH thumb and pinky MUST be extended (th_pinky_dist >= 0.82 and th_idx_dist >= 0.34).
    # In J and I: thumb is NOT extended (th_pinky_dist < 0.80).
    # If thumb is not extended, the sign can NEVER be Y!
    elif class_id in [8, 9, 24]:
        pinky_up = ys[20] < ys[18]
        other_curled = (ys[8] > ys[5] - 0.05) and (ys[12] > ys[9] - 0.05) and (ys[16] > ys[13] - 0.05)
        if pinky_up and other_curled:
            th_pinky_dist = np.sqrt((xs[4] - xs[20])**2 + (ys[4] - ys[20])**2)
            th_idx_dist = np.sqrt((xs[4] - xs[5])**2 + (ys[4] - ys[5])**2)
            th_is_extended = (th_pinky_dist >= 0.82) and (th_idx_dist >= 0.34)

            # If class is 24 (Y) but thumb is NOT extended: user is signing J (or I)!
            if not th_is_extended:
                pinky_dx = xs[20] - xs[17]
                thumb_higher = ys[4] < ys[20] + 0.05
                if abs(pinky_dx) > 0.16 or thumb_higher or class_id in [9, 24]:
                    return 9, max(confidence, 0.92)  # J: hooked/tilted pinky, thumb NOT extended
                else:
                    return 8, max(confidence, 0.92)  # I: straight upright pinky
            else:
                # Thumb IS extended outwards: Y (24)
                return 24, max(confidence, 0.92)  # Y: both thumb and pinky extended

    # --- 5. G (6) vs H (7) stabilization ---
    elif class_id in [6, 7]:
        idx_horiz_ext = abs(xs[8] - xs[5])
        mid_horiz_ext = abs(xs[12] - xs[9])
        if mid_horiz_ext >= 0.40 and abs(ys[8] - ys[12]) < 0.15:
            return 7, max(confidence, 0.92)  # H: two horizontal fingers
        elif idx_horiz_ext >= 0.35 and mid_horiz_ext < 0.35:
            return 6, max(confidence, 0.92)  # G: one horizontal finger

    # --- 6. T (19) vs Fist family (A=0, S=18, N=13, M=12) stabilization ---
    elif class_id in [19, 18, 13, 12, 0]:
        fist_curled = (ys[8] > ys[5] - 0.05) and (ys[12] > ys[9] - 0.05) and (ys[16] > ys[13] - 0.05) and (ys[20] > ys[17] - 0.05)
        if fist_curled:
            th_x = xs[4]
            th_y = ys[4]
            idx_mcp_x = xs[5]
            mid_mcp_x = xs[9]
            if class_id in [19, 18, 13]:
                if th_x > mid_mcp_x + 0.02 and th_x < idx_mcp_x + 0.02 and th_y < 0.22:
                    return 19, max(confidence, 0.92)  # T: thumb between index and middle knuckles
                elif th_x <= mid_mcp_x - 0.02 and class_id == 19:
                    return 18, max(confidence, 0.88)  # S: thumb over middle/ring

    # ALL OTHER ALPHABETS AND WORDS: 100% UNTOUCHED
    return class_id, confidence


from collections import deque

# History for temporal probability smoothing across consecutive frames
prediction_history = deque(maxlen=4)

# Configure MediaPipe Hands for single & dual hand recognition (Messi celebration)
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    model_complexity=0,
    min_detection_confidence=0.40,
    min_tracking_confidence=0.40
)

# Define a dictionary for mapping model output to sign labels
labels_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 
    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 
    19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: 'Hello', 
    27: 'Done', 28: 'Thank You', 29: 'I Love you', 30: 'Sorry', 31: 'Please', 32: 'You are welcome.',
    33: 'lional messi-the goat🐐'
}

print("Running Real-Time Sign Language Interpreter... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        continue

    # Mirror frame for natural interaction
    frame = cv2.flip(frame, 1)
    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to detect hand landmarks safely
    try:
        results = hands.process(frame_rgb)
    except Exception:
        continue
    valid_hands = []
    if results.multi_hand_landmarks:
        for i, hl in enumerate(results.multi_hand_landmarks):
            h_score = 1.0
            h_label = 'Right'
            if results.multi_handedness and i < len(results.multi_handedness):
                classification = results.multi_handedness[i].classification[0]
                h_score = classification.score
                h_label = classification.label

            hx = [lm.x for lm in hl.landmark]
            hy = [lm.y for lm in hl.landmark]
            min_x, max_x = min(hx), max(hx)
            min_y, max_y = min(hy), max(hy)
            box_w = max_x - min_x
            box_h = max_y - min_y
            area = box_w * box_h

            # Filter out wall textures, shadows, and low-confidence phantom hands
            if h_score >= 0.58 and area >= 0.012 and 0.20 <= (box_w / (box_h + 1e-4)) <= 5.0:
                valid_hands.append({
                    'landmarks': hl,
                    'score': h_score,
                    'label': h_label,
                    'area': area,
                    'min_x': min_x, 'max_x': max_x,
                    'min_y': min_y, 'max_y': max_y
                })

    if valid_hands:
        # Check for Lionel Messi celebration: BOTH hands pointing upward to the sky
        is_messi = False
        if len(valid_hands) >= 2:
            h1 = valid_hands[0]['landmarks']
            h2 = valid_hands[1]['landmarks']

            w1_x = h1.landmark[0].x
            w2_x = h2.landmark[0].x
            t1_x = h1.landmark[8].x
            t2_x = h2.landmark[8].x

            hands_separated = (abs(w1_x - w2_x) >= 0.18) or (abs(t1_x - t2_x) >= 0.16)

            if hands_separated and is_hand_pointing_up(h1) and is_hand_pointing_up(h2):
                is_messi = True

        if is_messi:
            prediction_history.clear()
            gold_color = (0, 215, 255) # BGR: Golden Yellow
            for vh in valid_hands[:2]:
                hl = vh['landmarks']
                mp_drawing.draw_landmarks(
                    frame,
                    hl,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
                hx = [lm.x for lm in hl.landmark]
                hy = [lm.y for lm in hl.landmark]
                bx1 = max(0, int(min(hx) * W) - 15)
                by1 = max(0, int(min(hy) * H) - 15)
                bx2 = min(W, int(max(hx) * W) + 15)
                by2 = min(H, int(max(hy) * H) + 15)
                cv2.rectangle(frame, (bx1, by1), (bx2, by2), gold_color, 3)
                cv2.putText(frame, "Lional Messi - The GOAT (99.0%)", (bx1, max(30, by1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, gold_color, 2, cv2.LINE_AA)

            banner_w = 420
            bx_start = max(0, (W - banner_w) // 2)
            cv2.rectangle(frame, (bx_start, 10), (bx_start + banner_w, 55), (0, 140, 255), -1)
            cv2.putText(frame, "LIONAL MESSI - THE GOAT", (bx_start + 25, 43),
                        cv2.FONT_HERSHEY_DUPLEX, 0.90, (255, 255, 255), 2, cv2.LINE_AA)
        else:
            # Single-hand signs evaluated on primary hand only
            primary_vh = max(valid_hands, key=lambda item: (item['area'], item['score']))
            primary_hl = primary_vh['landmarks']

            # Draw landmarks ONLY for real primary hand
            mp_drawing.draw_landmarks(
                frame,
                primary_hl,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            min_x, max_x = primary_vh['min_x'], primary_vh['max_x']
            min_y, max_y = primary_vh['min_y'], primary_vh['max_y']
            scale = max(max_x - min_x, max_y - min_y)
            if scale < 1e-4:
                scale = 1.0

            # Normalize 21 landmarks into 42 scale-invariant relative coordinates
            data_aux = []
            for lm in primary_hl.landmark:
                data_aux.append((lm.x - min_x) / scale)
                data_aux.append((lm.y - min_y) / scale)

            # Calculate safe bounding box coordinates
            x1 = max(0, int(min_x * W) - 15)
            y1 = max(0, int(min_y * H) - 15)
            x2 = min(W, int(max_x * W) + 15)
            y2 = min(H, int(max_y * H) + 15)

            if len(data_aux) == 42:
                try:
                    data_aux_mirr = mirror_features(data_aux)
                    feat_orig = np.asarray(data_aux, dtype=np.float32).reshape(1, -1)
                    feat_mirr = np.asarray(data_aux_mirr, dtype=np.float32).reshape(1, -1)

                    prob_orig = model.predict_proba(feat_orig)[0]
                    prob_mirr = model.predict_proba(feat_mirr)[0]

                    if np.max(prob_mirr) > np.max(prob_orig):
                        active_feat = data_aux_mirr
                    else:
                        active_feat = data_aux

                    combined_proba = np.maximum(prob_orig, prob_mirr)
                    prediction_history.append(combined_proba)

                    smoothed_proba = np.mean(prediction_history, axis=0)
                    best_idx = int(np.argmax(smoothed_proba))
                    class_id = int(str(model.classes_[best_idx]))
                    confidence = float(smoothed_proba[best_idx])
                    class_id, confidence = targeted_disambiguation(class_id, confidence, active_feat)
                    predicted_character = labels_dict.get(class_id, f"Sign {class_id}")

                    # Draw bounding box and label
                    box_color = (0, 255, 0) if confidence >= 0.55 else (255, 165, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                    label_text = f"{predicted_character} ({confidence * 100:.1f}%)"
                    cv2.putText(
                        frame,
                        label_text,
                        (x1, max(25, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75,
                        box_color,
                        2,
                        cv2.LINE_AA
                    )
                except Exception as e:
                    pass
    else:
        prediction_history.clear()

    # Display the frame in a window
    cv2.imshow('Sign Language Interpreter', frame)

    # Check if the window was closed by the user (pressing 'q' key)
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
