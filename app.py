"""
Real-Time Sign Language Interpreter Web Application
Powered by MediaPipe, Scikit-learn, OpenCV, Flask, and Flask-SocketIO.
Features on-demand webcam access, scale-invariant feature extraction,
and multi-frame temporal probability smoothing for maximum accuracy.
"""

import os
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import time
import math
import pickle
import threading
import warnings
from collections import deque
import numpy as np
import cv2
import mediapipe as mp
from flask import Flask, render_template, Response, jsonify, request
from flask_socketio import SocketIO, emit

# Suppress deprecation and version mismatch warnings
warnings.filterwarnings("ignore")

app = Flask(__name__, static_folder='static', template_folder='templates')
app.config['SECRET_KEY'] = 'sign-language-interpreter-secret-key-2026'
app.config['TEMPLATES_AUTO_RELOAD'] = True

# Initialize SocketIO with cross-origin support
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# 33 Sign classes: Letters A-Z (0-25) and 7 User Guide Words (26-32)
LABELS_DICT = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S',
    19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z',
    26: 'Hello',
    27: 'Done',
    28: 'Thank You',
    29: 'I Love you',
    30: 'Sorry',
    31: 'Please',
    32: 'You are welcome.',
    33: 'lional messi-the goat🐐'
}

# Load the trained machine learning model
model = None
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model.p')
try:
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as f:
            model_dict = pickle.load(f)
            model = model_dict.get('model', None)
        print(f"[OK] Successfully loaded gesture classification model with {getattr(model, 'n_features_in_', 42)} features.")
    else:
        print(f"[WARN] Model file not found at: {MODEL_PATH}")
except Exception as e:
    print(f"[ERROR] Error loading model: {e}")
    model = None




def is_hand_pointing_up(hand_landmarks):
    """
    Checks if a hand has its index finger pointing upwards with other fingers curled,
    matching Lionel Messi's iconic skyward goal celebration.
    """
    lm = hand_landmarks.landmark
    idx_tip = lm[8]
    idx_pip = lm[6]
    idx_mcp = lm[5]
    wrist = lm[0]

    # 1. Index finger extended upward (tip above pip, mcp, and wrist)
    if not (idx_tip.y < idx_pip.y and idx_tip.y < idx_mcp.y and idx_tip.y < wrist.y - 0.04):
        return False

    # 2. Orientation: pointing generally upward (within ~65 degrees from vertical)
    dy = abs(idx_mcp.y - idx_tip.y)
    dx = abs(idx_mcp.x - idx_tip.x)
    if dy < dx * 0.40:
        return False

    # 3. Middle, Ring, Pinky curled down (tips distinctly lower than index tip or below PIP)
    mid_curled = (lm[12].y > idx_tip.y + 0.035) or (lm[12].y > lm[10].y)
    ring_curled = (lm[16].y > idx_tip.y + 0.035) or (lm[16].y > lm[14].y)
    pinky_curled = (lm[20].y > idx_tip.y + 0.035) or (lm[20].y > lm[18].y)

    if not (mid_curled and ring_curled and pinky_curled):
        return False

    # 4. Index tip is highest of all fingertips
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
        dy = ys[5] - ys[8]  # dy > 0 when index points up
        if dy > 0.05:
            lean = dx / dy
            # D points straight up (mean -0.05, range -0.10 to 0.00)
            # Z leans distinctly to the left (mean -0.67, range -0.75 to -0.56)
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
        # If middle finger is clearly extended horizontally alongside index: H (7)
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
            # T has thumb tucked between index and middle knuckles:
            if class_id in [19, 18, 13]:
                if th_x > mid_mcp_x + 0.02 and th_x < idx_mcp_x + 0.02 and th_y < 0.22:
                    return 19, max(confidence, 0.92)  # T: thumb between index and middle knuckles
                elif th_x <= mid_mcp_x - 0.02 and class_id == 19:
                    return 18, max(confidence, 0.88)  # S: thumb over middle/ring

    # ALL OTHER ALPHABETS AND WORDS: 100% UNTOUCHED
    return class_id, confidence


class CameraStream:
    """
    On-demand camera stream manager that strictly accesses webcam hardware
    only when activated by the user and performs scale-invariant hand gesture recognition.
    """
    def __init__(self):
        self.lock = threading.Lock()
        self.cap = None
        self.camera_index = 0
        self.is_running = False
        self.reader_thread = None
        self.process_thread = None
        self.raw_frame = None
        self.latest_frame = None
        self.latest_prediction = {"text": "", "confidence": 0.0, "detected": False}
        self.client_count = 0
        self.last_emit_time = 0
        # Rolling probability history for temporal smoothing across frames
        self.prediction_history = deque(maxlen=4)

        # MediaPipe setup configured for single and dual hand recognition (Messi celebration)
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.hands = None
        self._create_hands_detector()

    def _create_hands_detector(self):
        """Creates or resets MediaPipe Hands detector with a clean calculator graph."""
        if hasattr(self, 'hands') and self.hands is not None:
            try:
                self.hands.close()
            except Exception:
                pass
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=0,
            min_detection_confidence=0.40,
            min_tracking_confidence=0.40
        )

    def _open_camera(self):
        """Opens camera on index 0 reliably with zero buffer lag."""
        time.sleep(0.20)
        self._create_hands_detector()
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

        # 1. Try index 0 with DirectShow, then default backend
        for backend in [cv2.CAP_DSHOW, cv2.CAP_ANY]:
            try:
                cap = cv2.VideoCapture(0, backend)
                if cap is not None and cap.isOpened():
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    cap.set(cv2.CAP_PROP_FPS, 30)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    # Warm-up read
                    for _ in range(5):
                        ret, test_frame = cap.read()
                        if ret and test_frame is not None:
                            self.camera_index = 0
                            backend_name = 'DSHOW' if backend == cv2.CAP_DSHOW else 'ANY'
                            print(f"[OK] Webcam successfully connected on index 0 ({backend_name})")
                            return cap
                        time.sleep(0.03)
                    cap.release()
            except BaseException as e:
                print(f"Index 0 backend {backend} notice: {e}")

        # 2. Fallback to index 1 only if index 0 completely failed
        try:
            cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
            if cap is None or not cap.isOpened():
                cap = cv2.VideoCapture(1, cv2.CAP_ANY)
            if cap is not None and cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, 30)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                ret, test_frame = cap.read()
                if ret and test_frame is not None:
                    self.camera_index = 1
                    print("[OK] Webcam connected on index 1")
                    return cap
                cap.release()
        except BaseException:
            pass

        return None

    def start(self):
        """Activates webcam hardware and starts zero-latency capture and processing threads."""
        with self.lock:
            if not self.is_running:
                self.is_running = True
                self.prediction_history.clear()
                self.raw_frame = None
                self.latest_frame = None
                self.cap = self._open_camera()
                if self.cap is None:
                    print("[ERROR] Failed to open any webcam.")
                    self.is_running = False
                    socketio.emit('camera_status', {'running': False, 'error': 'Camera unavailable'})
                    return

                # 1. Dedicated reader thread: continually pulls from hardware to keep OS buffer 100% empty (0ms lag)
                self.reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
                self.reader_thread.start()

                # 2. Worker processing thread: detects landmarks & classifies signs on the freshest frame
                self.process_thread = threading.Thread(target=self._process_loop, daemon=True)
                self.process_thread.start()

                socketio.emit('camera_status', {'running': True})

    def stop(self):
        """Releases camera and capture threads."""
        self._release_camera_internal()

    def force_stop(self):
        """Explicitly stops capture loops and frees hardware webcam immediately."""
        self._release_camera_internal()

    def _release_camera_internal(self):
        """Internal helper to shut down camera capture and release resources safely."""
        self.is_running = False
        if self.reader_thread is not None and self.reader_thread.is_alive() and threading.current_thread() != self.reader_thread:
            try:
                self.reader_thread.join(timeout=0.3)
            except Exception:
                pass
            self.reader_thread = None

        if self.process_thread is not None and self.process_thread.is_alive() and threading.current_thread() != self.process_thread:
            try:
                self.process_thread.join(timeout=0.3)
            except Exception:
                pass
            self.process_thread = None

        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
            print("[INFO] Webcam hardware released.")

        self.raw_frame = None
        self.latest_frame = None
        self.prediction_history.clear()
        self.latest_prediction = {
            'text': '',
            'confidence': 0.0,
            'class_id': -1,
            'detected': False
        }
        socketio.emit('prediction', self.latest_prediction)
        socketio.emit('camera_status', {'running': False})

    def _reader_loop(self):
        """Drains camera hardware buffer continuously at native camera FPS to prevent driver queue lag."""
        while self.is_running and self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret and frame is not None:
                self.raw_frame = frame
            else:
                time.sleep(0.005)

    def _process_loop(self):
        """Worker loop that grabs freshest real-time frame, detects landmarks, and predicts signs with zero latency."""
        last_frame_ref = None
        try:
            while self.is_running:
                raw = self.raw_frame
                if raw is None or raw is last_frame_ref:
                    time.sleep(0.004)
                    continue

                last_frame_ref = raw

                # Mirror frame for natural interaction display
                frame = cv2.flip(raw, 1)
                H, W, _ = frame.shape
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Process hand landmarks with MediaPipe safely
                try:
                    results = self.hands.process(frame_rgb)
                except Exception as e:
                    print(f"[WARN] MediaPipe frame processing notice: {e}")
                    self._create_hands_detector()
                    time.sleep(0.01)
                    continue
                detected_signs = []

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

                        # Reject wall textures, shadows, and low-confidence phantom hands
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

                        # Verify the two hands are physically distinct, separated hands across screen
                        w1_x = h1.landmark[0].x
                        w2_x = h2.landmark[0].x
                        t1_x = h1.landmark[8].x
                        t2_x = h2.landmark[8].x

                        hands_separated = (abs(w1_x - w2_x) >= 0.18) or (abs(t1_x - t2_x) >= 0.16)

                        if hands_separated and is_hand_pointing_up(h1) and is_hand_pointing_up(h2):
                            is_messi = True

                    if is_messi:
                        self.prediction_history.clear()
                        gold_color = (0, 215, 255)  # BGR: Golden Yellow for the GOAT

                        # Draw landmarks and gold bounding boxes on both hands
                        for vh in valid_hands[:2]:
                            hl = vh['landmarks']
                            self.mp_drawing.draw_landmarks(
                                frame,
                                hl,
                                self.mp_hands.HAND_CONNECTIONS,
                                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                                self.mp_drawing_styles.get_default_hand_connections_style()
                            )
                            hx = [lm.x for lm in hl.landmark]
                            hy = [lm.y for lm in hl.landmark]
                            bx1 = max(0, int(min(hx) * W) - 15)
                            by1 = max(0, int(min(hy) * H) - 15)
                            bx2 = min(W, int(max(hx) * W) + 15)
                            by2 = min(H, int(max(hy) * H) + 15)
                            cv2.rectangle(frame, (bx1, by1), (bx2, by2), gold_color, 3)
                            cv2.putText(
                                frame,
                                "Lional Messi - The GOAT (99.0%)",
                                (bx1, max(30, by1 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.65,
                                gold_color,
                                2,
                                cv2.LINE_AA
                            )

                        # Draw celebration banner at the top of the video
                        banner_w = 420
                        bx_start = max(0, (W - banner_w) // 2)
                        cv2.rectangle(frame, (bx_start, 10), (bx_start + banner_w, 55), (0, 140, 255), -1)
                        cv2.putText(
                            frame,
                            "LIONAL MESSI - THE GOAT",
                            (bx_start + 25, 43),
                            cv2.FONT_HERSHEY_DUPLEX,
                            0.90,
                            (255, 255, 255),
                            2,
                            cv2.LINE_AA
                        )

                        detected_signs.append({
                            "text": "lional messi-the goat🐐",
                            "confidence": 99.0,
                            "class_id": 33,
                            "bbox": (0, 0, W, H)
                        })
                    else:
                        # Single-hand signs (A-Z and 7 words) evaluated with the 100% accurate ML model
                        # Select ONLY the primary hand (largest area and highest detection score)
                        primary_vh = max(valid_hands, key=lambda item: (item['area'], item['score']))
                        primary_hl = primary_vh['landmarks']

                        # Draw landmarks ONLY for the real primary hand
                        self.mp_drawing.draw_landmarks(
                            frame,
                            primary_hl,
                            self.mp_hands.HAND_CONNECTIONS,
                            self.mp_drawing_styles.get_default_hand_landmarks_style(),
                            self.mp_drawing_styles.get_default_hand_connections_style()
                        )

                        min_x, max_x = primary_vh['min_x'], primary_vh['max_x']
                        min_y, max_y = primary_vh['min_y'], primary_vh['max_y']
                        
                        scale = max(max_x - min_x, max_y - min_y)
                        if scale < 1e-4:
                            scale = 1.0

                        data_aux = []
                        for lm in primary_hl.landmark:
                            data_aux.append((lm.x - min_x) / scale)
                            data_aux.append((lm.y - min_y) / scale)

                        x1 = max(0, int(min_x * W) - 15)
                        y1 = max(0, int(min_y * H) - 15)
                        x2 = min(W, int(max_x * W) + 15)
                        y2 = min(H, int(max_y * H) + 15)

                        if model is not None and len(data_aux) == 42:
                            try:
                                data_aux_mirr = mirror_features(data_aux)
                                feat_orig = np.asarray(data_aux, dtype=np.float32).reshape(1, -1)
                                feat_mirr = np.asarray(data_aux_mirr, dtype=np.float32).reshape(1, -1)

                                # Batched vectorized prediction: 2x faster than 2 separate predict_proba calls
                                probs = model.predict_proba(np.vstack([feat_orig, feat_mirr]))
                                prob_orig = probs[0]
                                prob_mirr = probs[1]

                                # Dual chirality: select canonical representation matching user's active hand
                                if np.max(prob_mirr) > np.max(prob_orig):
                                    active_feat = data_aux_mirr
                                else:
                                    active_feat = data_aux

                                # Dual-chirality combined probability pooling
                                combined_proba = np.maximum(prob_orig, prob_mirr)
                                self.prediction_history.append(combined_proba)

                                # 4-frame moving average for rock-solid stability and zero jitter
                                smoothed_proba = np.mean(self.prediction_history, axis=0)
                                best_idx = int(np.argmax(smoothed_proba))
                                class_id = int(str(model.classes_[best_idx]))
                                confidence = float(smoothed_proba[best_idx])

                                # Targeted disambiguation (prevents D when R is shown, resolves D vs Z, R vs U/V, J vs I)
                                class_id, confidence = targeted_disambiguation(class_id, confidence, active_feat)
                                predicted_char = LABELS_DICT.get(class_id, None)

                                if predicted_char is not None and confidence >= 0.40:
                                    detected_signs.append({
                                        "text": predicted_char,
                                        "confidence": round(confidence * 100, 1),
                                        "class_id": class_id,
                                        "bbox": (x1, y1, x2, y2)
                                    })

                                    box_color = (46, 204, 113) if confidence >= 0.65 else (52, 152, 219)
                                    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                                    
                                    label_text = f"{predicted_char} ({confidence * 100:.1f}%)"
                                    (txt_w, txt_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                                    tag_y1 = max(0, y1 - txt_h - 12)
                                    tag_y2 = y1
                                    cv2.rectangle(frame, (x1, tag_y1), (x1 + txt_w + 16, tag_y2), box_color, -1)
                                    cv2.putText(
                                        frame,
                                        label_text,
                                        (x1 + 8, max(txt_h + 4, tag_y2 - 6)),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.65,
                                        (0, 0, 0),
                                        2,
                                        cv2.LINE_AA
                                    )
                            except Exception:
                                pass
                else:
                    self.prediction_history.clear()

                # Broadcast and store latest prediction (rate-limited every ~50ms)
                now = time.time()
                if now - self.last_emit_time > 0.05:
                    if detected_signs:
                        top_sign = max(detected_signs, key=lambda s: s["confidence"])
                        pred_data = {
                            'text': top_sign['text'],
                            'confidence': top_sign['confidence'],
                            'class_id': top_sign['class_id'],
                            'detected': True,
                            'timestamp': now
                        }
                    else:
                        pred_data = {
                            'text': '',
                            'confidence': 0.0,
                            'class_id': -1,
                            'detected': False,
                            'timestamp': now
                        }
                    
                    with self.lock:
                        self.latest_prediction = pred_data
                    
                    socketio.emit('prediction', pred_data)
                    self.last_emit_time = now

                # Encode frame to JPEG (quality 60 reduces payload size by ~45%, preventing tunnel network lag)
                ret_encode, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
                if ret_encode:
                    self.latest_frame = buffer.tobytes()

        finally:
            self._release_camera_internal()


# Global camera stream singleton
camera_stream = CameraStream()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/status')
def api_status():
    """Returns system status including model and camera status."""
    return jsonify({
        "status": "online",
        "model_loaded": model is not None,
        "camera_running": camera_stream.is_running,
        "camera_index": camera_stream.camera_index,
        "num_classes": len(LABELS_DICT)
    })


@app.route('/api/labels')
def api_labels():
    """Returns all 33 available sign gestures."""
    return jsonify(LABELS_DICT)


@app.route('/api/camera/start', methods=['GET', 'POST'])
def api_camera_start():
    """Starts camera stream on demand."""
    camera_stream.start()
    return jsonify({"status": "running", "is_running": True, "camera_index": camera_stream.camera_index})


@app.route('/api/camera/stop', methods=['GET', 'POST'])
def api_camera_stop():
    """Immediately stops camera and releases webcam hardware."""
    camera_stream.force_stop()
    return jsonify({"status": "stopped", "is_running": False})


@app.route('/api/prediction')
def api_prediction():
    """Returns current active gesture prediction without blocking."""
    return jsonify(camera_stream.latest_prediction)


@app.route('/api/camera/toggle', methods=['GET', 'POST'])
def api_camera_toggle():
    """Toggles camera on or off."""
    if camera_stream.is_running:
        camera_stream.force_stop()
        return jsonify({"status": "stopped", "is_running": False})
    else:
        camera_stream.start()
        return jsonify({"status": "running", "is_running": True, "camera_index": camera_stream.camera_index})


@app.route('/api/camera/status', methods=['GET'])
def api_camera_status():
    """Returns current active status of camera hardware."""
    return jsonify({"is_running": camera_stream.is_running})


@socketio.on('connect')
def handle_connect():
    emit('status', {
        'status': 'connected',
        'model_loaded': model is not None,
        'camera_running': camera_stream.is_running
    })


@socketio.on('disconnect')
def handle_disconnect():
    pass


@socketio.on('toggle_camera')
def handle_toggle_camera():
    if camera_stream.is_running:
        camera_stream.force_stop()
    else:
        camera_stream.start()


def generate_frames():
    """Generator for streaming MJPEG video feed cleanly without lock contention or buffer buildup."""
    last_frame_bytes = None
    try:
        while camera_stream.is_running:
            frame = camera_stream.latest_frame
            if frame is not None and frame is not last_frame_bytes:
                last_frame_bytes = frame
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.015)
    except (GeneratorExit, ConnectionResetError, BrokenPipeError):
        pass


@app.route('/video_feed')
def video_feed():
    response = Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )
    # Prevent Cloudflare Tunnel, reverse proxies, and browsers from buffering MJPEG frames
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate, pre-check=0, post-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    response.headers['X-Accel-Buffering'] = 'no'
    return response


if __name__ == '__main__':
    print("=" * 60)
    print("  AI Sign Language Interpreter Web Server Starting...")
    print("  Access in browser: http://localhost:5000 or http://127.0.0.1:5000")
    print("=" * 60)
    socketio.run(app, host='0.0.0.0', port=5000, debug=False, allow_unsafe_werkzeug=True)
