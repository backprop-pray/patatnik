#!/home/yasen/.venv/bin/python3
"""
Stable-Baselines3 PPO rover controller.

- PPO via stable-baselines3 (PyTorch backend, far more efficient than numpy)
- Gymnasium custom environment wrapping rover hardware
- Deterministic safety / hazard / GreenNavigator overrides inside env.step()
- Per-step YOLO logging to console
- Live stream on port 5000 with YOLO bounding boxes drawn on frames
- Fixed reward: alignment terms only fire when green is actually visible
"""

import json
import os
import socket
import struct
import sys
import time
import signal
import logging
import threading

import cv2
import numpy as np
import gymnasium
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from api import RoverAPI
from vision import VisionPipeline, VisionFeatures, ZERO_VISION

# ── CONSTANTS ──────────────────────────────────────────────────────────────────
SENSOR_MAX_CM   = 30.0
SAFETY_CM       = 15.0
FRONT_SAFETY_CM = 25.0
TIMESTEP_S      = 0.05
NOISE_STD       = 0.02
MIN_FWD_DUTY    = 50.0
MAX_FWD_DUTY    = 70.0

# ── CAPTURE APPROACH CONSTANTS ─────────────────────────────────────────────
PROX_TRIGGER_CM    = 55.0   # front sensor cm that starts an approach
APPROACH_STOP_CM   = 40.0   # stop inside 30-50 cm window (centre = 40 cm)
SPIN_180_S         = 1.8    # seconds to spin ~180° after capture (tune if needed)
CAPTURE_COOLDOWN_S = 15.0   # minimum gap between captures (seconds)

OBS_DIM   = 16
N_ACTIONS = 9

STREAM_PORT      = 5000
JPEG_QUALITY     = 70
FRAME_W, FRAME_H = 640, 480

YOLO_MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'yolo11n.pt')
CHECKPOINT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ppo_sb3')

# ── ACTION TABLE ───────────────────────────────────────────────────────────────
ACTIONS = [
    (-1, -1),  # 0  both backward
    (-1,  0),  # 1  left back, right stop
    (-1,  1),  # 2  spin left
    ( 0, -1),  # 3  left stop, right back
    ( 0,  0),  # 4  full stop
    ( 0,  1),  # 5  gentle right turn
    ( 1, -1),  # 6  spin right
    ( 1,  0),  # 7  gentle left turn
    ( 1,  1),  # 8  both forward
]

ACT_BACKWARD     = 0
ACT_SPIN_LEFT    = 2
ACT_SPIN_RIGHT   = 6
ACT_GENTLE_RIGHT = 5
ACT_GENTLE_LEFT  = 7
ACT_FORWARD      = 8

_CMD = {-1: 'backward', 0: 'stop', 1: 'forward'}

# ── SENSOR HELPERS ─────────────────────────────────────────────────────────────
def _norm(dist_cm):
    if dist_cm is None:
        return 1.0
    return float(min(dist_cm, SENSOR_MAX_CM) / SENSOR_MAX_CM)

def read_sensors(rover):
    raw = rover.get_ultrasonic()
    L = raw.get(1) if raw.get(1) is not None else SENSOR_MAX_CM
    F = raw.get(3) if raw.get(3) is not None else SENSOR_MAX_CM
    R = raw.get(2) if raw.get(2) is not None else SENSOR_MAX_CM
    return (L, F, R), np.array([_norm(L), _norm(F), _norm(R)], dtype=np.float32)

def fwd_duty(free_path_C):
    return MIN_FWD_DUTY + (MAX_FWD_DUTY - MIN_FWD_DUTY) * float(free_path_C)

def action_to_vx(action_idx, duty=None):
    if duty is None:
        duty = MIN_FWD_DUTY
    lc, rc = ACTIONS[action_idx]
    return duty / 100.0 if (lc == 1 and rc == 1) else 0.0

# ── OBSERVATION BUILDER ────────────────────────────────────────────────────────
def build_obs(sensor_obs, vf, vx_capped):
    return np.array([
        sensor_obs[0], sensor_obs[1], sensor_obs[2],
        vf.plant_L,    vf.plant_C,    vf.plant_R,
        vf.tree_L,     vf.tree_C,     vf.tree_R,
        vf.hazard_L,   vf.hazard_C,   vf.hazard_R,
        vf.crop_L,     vf.crop_R,
        vf.free_path_C,
        vx_capped,
    ], dtype=np.float32)

# ── SAFETY OVERRIDE ────────────────────────────────────────────────────────────
def safety_override(raw_L, raw_F, raw_R):
    if raw_F < FRONT_SAFETY_CM:
        return ACT_BACKWARD, True
    if raw_L < SAFETY_CM:
        return ACT_SPIN_RIGHT, True
    if raw_R < SAFETY_CM:
        return ACT_SPIN_LEFT, True
    return None, False

# ── HAZARD STEER OVERRIDE ──────────────────────────────────────────────────────
HAZARD_STEER_THR  = 0.25
GREEN_SEEK_THR    = 0.15
GREEN_ORBIT_THR   = 0.20
GREEN_SIDE_BIAS   = 0.12
ORBIT_SPIN_STEPS  = 5

def hazard_steer_override(vf):
    if vf.hazard_C < HAZARD_STEER_THR:
        return None, False
    if vf.hazard_L <= vf.hazard_R:
        return ACT_SPIN_LEFT, True
    else:
        return ACT_SPIN_RIGHT, True

# ── GREEN NAVIGATOR ────────────────────────────────────────────────────────────
GN_DETECT_THR   = 0.06   # lowered: catch smaller/partial plant detections
GN_CENTER_THR   = 0.18
GN_ORBIT_THR    = 0.25
GN_SCAN_SPIN    = 10
GN_REPOSE_STEPS = 8
GN_APPROACH_MAX = 15

_GN_SCAN, _GN_SEEK, _GN_APPROACH, _GN_ORBIT, _GN_REPOSE = 0, 1, 2, 3, 4
_GN_NAMES = ['scan', 'seek', 'approach', 'orbit', 'repose']


class GreenNavigator:
    def __init__(self):
        self._state      = _GN_SCAN
        self._scan_tick  = 0
        self._scan_dir   = ACT_SPIN_LEFT
        self._repose_n   = 0
        self._orbit_dir  = ACT_GENTLE_LEFT
        self._approach_n = 0
        self._was_found  = False

    @property
    def state_name(self):
        return _GN_NAMES[self._state]

    def step(self, vf, announce=None):
        left_g   = vf.plant_L + vf.tree_L
        center_g = vf.plant_C + vf.tree_C
        right_g  = vf.plant_R + vf.tree_R
        total_g  = left_g + center_g + right_g
        found    = total_g >= GN_DETECT_THR

        if found and not self._was_found and announce:
            announce(
                f'{"=" * 52}\n'
                f'  PLANT / TREE DETECTED\n'
                f'  L={left_g:.2f}  C={center_g:.2f}  R={right_g:.2f}  total={total_g:.2f}\n'
                f'{"=" * 52}'
            )
        self._was_found = found

        if self._state == _GN_SCAN:
            if found:
                self._state     = _GN_SEEK
                self._scan_tick = 0
        elif self._state == _GN_SEEK:
            if not found:
                self._state    = _GN_REPOSE
                self._repose_n = GN_REPOSE_STEPS
            elif center_g >= GN_CENTER_THR:
                self._state      = _GN_APPROACH
                self._approach_n = 0
        elif self._state == _GN_APPROACH:
            if not found:
                self._state    = _GN_REPOSE
                self._repose_n = GN_REPOSE_STEPS
            elif center_g >= GN_ORBIT_THR or self._approach_n >= GN_APPROACH_MAX:
                self._orbit_dir = (ACT_GENTLE_LEFT if right_g >= left_g else ACT_GENTLE_RIGHT)
                self._state = _GN_ORBIT
            else:
                self._approach_n += 1
        elif self._state == _GN_ORBIT:
            if not found:
                self._state    = _GN_REPOSE
                self._repose_n = GN_REPOSE_STEPS
        elif self._state == _GN_REPOSE:
            self._repose_n -= 1
            if self._repose_n <= 0:
                self._state = _GN_SCAN

        if self._state == _GN_SCAN:
            self._scan_tick += 1
            phase = self._scan_tick % (GN_SCAN_SPIN * 2)
            if phase == 0:
                self._scan_dir = (ACT_SPIN_RIGHT if self._scan_dir == ACT_SPIN_LEFT else ACT_SPIN_LEFT)
            return (self._scan_dir if phase < GN_SCAN_SPIN else ACT_FORWARD), True
        if self._state == _GN_SEEK:
            if left_g > right_g + 0.05:
                return ACT_GENTLE_LEFT, True
            if right_g > left_g + 0.05:
                return ACT_GENTLE_RIGHT, True
            return ACT_FORWARD, True
        if self._state == _GN_APPROACH:
            return ACT_FORWARD, True
        if self._state == _GN_ORBIT:
            if center_g < 0.08:
                return (ACT_GENTLE_LEFT if left_g >= right_g else ACT_GENTLE_RIGHT), True
            return self._orbit_dir, True
        if self._state == _GN_REPOSE:
            return ACT_BACKWARD, True
        return ACT_FORWARD, True


# ── REWARD ─────────────────────────────────────────────────────────────────────
def compute_reward(raw_sensors, vf, action_idx, safety_triggered,
                   prev_center=0.0, prev_action=None):
    raw_L, raw_F, raw_R = raw_sensors
    left_cmd, right_cmd = ACTIONS[action_idx]
    r = 0.0

    green_total   = (vf.plant_L + vf.plant_C + vf.plant_R
                     + vf.tree_L + vf.tree_C + vf.tree_R)
    crop_presence = vf.crop_L + vf.crop_R

    # 1. Forward motion
    if left_cmd == 1 and right_cmd == 1:
        r += 0.20

    # 2. Temporal progress toward green
    current_center = vf.plant_C + vf.tree_C
    r += 0.30 * (current_center - prev_center)

    # 3. Green centering — ONLY fires when green is actually visible
    if green_total > 0.05:
        r += 0.25 * (1.0 - abs(vf.plant_L - vf.plant_R))
        r += 0.25 * (1.0 - abs(vf.tree_L  - vf.tree_R))
        r += 0.30 * (vf.plant_C + vf.tree_C)
        r += 0.10 * (vf.plant_L + vf.plant_R + vf.tree_L + vf.tree_R)
        # Steer toward green side
        lc_sign   = 1 if left_cmd == 1 else (-1 if left_cmd == -1 else 0)
        rc_sign   = 1 if right_cmd == 1 else (-1 if right_cmd == -1 else 0)
        turn_dir  = (rc_sign - lc_sign) / 2.0
        r += 0.20 * ((vf.plant_R + vf.tree_R) - (vf.plant_L + vf.tree_L)) * turn_dir

    # 4. Crop-row following — ONLY fires when crops visible
    if crop_presence > 0.05:
        r += 0.15 * (1.0 - abs(vf.crop_L - vf.crop_R))
        r += 0.08 * crop_presence

    # 5. Free path bonus — only when moving forward
    if left_cmd == 1 and right_cmd == 1:
        r += 0.15 * vf.free_path_C

    # 6. No-green penalty (stronger: was -0.40)
    if green_total < 0.05 and crop_presence < 0.05:
        r -= 0.60

    # 7. Smoothness
    if prev_action is not None and action_idx != prev_action:
        r -= 0.05

    # 8. Hazard penalty
    r -= 1.0 * min(1.0, vf.hazard_L + vf.hazard_C + vf.hazard_R)

    # 9. Backward discouragement
    if left_cmd == -1 and right_cmd == -1 and not safety_triggered:
        r -= 0.08

    # 10. Collision shaping
    r -= 1.0 * (1.0 - _norm(raw_F))

    # 11. Same-direction spin penalty
    if (prev_action is not None
            and action_idx in {ACT_SPIN_LEFT, ACT_SPIN_RIGHT}
            and action_idx == prev_action):
        r -= 0.30

    return r


# ── STREAM SERVER ──────────────────────────────────────────────────────────────
_stream_state = {'frame_jpeg': None, 'meta': b'{}', 'lock': None}


def _annotate_frame(frame_bgr, detections, src_w, src_h):
    """Draw YOLO bounding boxes on an already-resized BGR frame."""
    dh, dw = frame_bgr.shape[:2]
    sx, sy = dw / src_w, dh / src_h
    _plant_kws = (
        'plant', 'flower', 'shrub', 'potted', 'vase',
        'banana', 'apple', 'orange', 'broccoli', 'carrot',
        'fruit', 'vegetable', 'crop', 'vine', 'leaf', 'grass',
        'bush', 'weed', 'herb', 'tomato', 'lettuce', 'cabbage',
        'celery', 'cucumber', 'pepper', 'pumpkin', 'squash',
        'berry', 'cherry', 'grape', 'lemon', 'mango', 'melon',
        'pear', 'pineapple', 'strawberry', 'watermelon',
    )
    for name, conf, x1, y1, x2, y2 in detections:
        bx1, by1 = int(x1 * sx), int(y1 * sy)
        bx2, by2 = int(x2 * sx), int(y2 * sy)
        if any(kw in name for kw in _plant_kws):
            color = (0, 200, 0)    # green
        elif 'tree' in name:
            color = (0, 150, 50)   # dark green
        else:
            color = (0, 0, 220)    # red for hazard
        cv2.rectangle(frame_bgr, (bx1, by1), (bx2, by2), color, 2)
        cv2.putText(frame_bgr, f'{name} {conf:.2f}',
                    (bx1, max(by1 - 5, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    return frame_bgr


def _serve_client(conn):
    try:
        while True:
            with _stream_state['lock']:
                jpeg = _stream_state['frame_jpeg']
                meta = _stream_state['meta']
            if jpeg is None:
                time.sleep(0.01)
                continue
            conn.sendall(struct.pack('>I', len(meta)))
            conn.sendall(meta)
            conn.sendall(struct.pack('>I', len(jpeg)))
            conn.sendall(jpeg)
    except (BrokenPipeError, ConnectionResetError, OSError):
        pass
    finally:
        conn.close()


def _stream_server_thread():
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(('0.0.0.0', STREAM_PORT))
    srv.listen(4)
    log.info(f'Stream server on :{STREAM_PORT}')
    while True:
        try:
            conn, addr = srv.accept()
            log.info(f'Viewer connected from {addr}')
            threading.Thread(target=_serve_client, args=(conn,), daemon=True).start()
        except Exception:
            pass


# ── GLOBALS ────────────────────────────────────────────────────────────────────
plant_found           = False
_rover                = None          # set by main(); orchestrator reads GPS from here
_capture_event        = threading.Event()   # set after approach+capture; orchestrator waits on this
_captured_image_path  = None          # kept for compatibility
_captured_image_bytes = None          # in-memory JPEG bytes; orchestrator reads this

# YOLO classes that count as a plant/veg/fruit detection worth sending
_PLANT_SEND_KEYWORDS = frozenset({
    'plant', 'potted', 'vase',
    'banana', 'apple', 'orange', 'broccoli', 'carrot', 'tomato',
    'lettuce', 'cabbage', 'celery', 'cucumber', 'pepper', 'pumpkin',
    'squash', 'berry', 'cherry', 'grape', 'lemon', 'mango', 'melon',
    'pear', 'pineapple', 'strawberry', 'watermelon',
    'fruit', 'vegetable', 'crop', 'vine', 'leaf', 'grass',
    'bush', 'weed', 'herb', 'flower', 'shrub', 'tree',
})
_PLANT_SEND_CONF = 0.20   # minimum YOLO confidence to trigger approach+capture

logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(message)s', datefmt='%H:%M:%S')
log = logging.getLogger('ppo_rover')


# ── GYMNASIUM ENVIRONMENT ──────────────────────────────────────────────────────
class RoverEnv(gymnasium.Env):
    metadata = {}

    def __init__(self, rover, vision):
        super().__init__()
        self.observation_space = gymnasium.spaces.Box(0.0, 1.0, shape=(OBS_DIM,), dtype=np.float32)
        self.action_space      = gymnasium.spaces.Discrete(N_ACTIONS)
        self.rover  = rover
        self.vision = vision
        self._navigator           = GreenNavigator()
        self._prev_center         = 0.0
        self._prev_action         = None
        self._prev_plant_detected = False
        self._prev_prox_triggered = False
        self._last_capture_ts     = 0.0
        self._raw_sensors         = (SENSOR_MAX_CM, SENSOR_MAX_CM, SENSOR_MAX_CM)
        self._vf                  = ZERO_VISION
        self._t0                  = time.monotonic()
        self._step_count          = 0

    # ── Capture sensors + vision, update stream, return obs ───────────────────
    def _capture(self):
        (raw_L, raw_F, raw_R), sensor_obs = read_sensors(self.rover)
        self._raw_sensors = (raw_L, raw_F, raw_R)
        try:
            raw_frame = np.array(self.rover.getframe(), dtype=np.uint8)
            # Rotate 180° (camera is mounted upside-down) then convert RGB→BGR
            raw_frame = cv2.flip(raw_frame, -1)
            raw_frame = cv2.cvtColor(raw_frame, cv2.COLOR_RGB2BGR)
            self._vf  = self.vision.update(raw_frame)
            # Annotate and push to stream
            if _stream_state['lock'] is not None:
                dets, src_w, src_h = self.vision.get_detections()
                bgr = cv2.resize(raw_frame, (FRAME_W, FRAME_H))
                bgr = _annotate_frame(bgr, dets, src_w, src_h)
                _, jpeg = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                meta = json.dumps({
                    'L': round(raw_L, 1), 'F': round(raw_F, 1), 'R': round(raw_R, 1)
                }).encode()
                with _stream_state['lock']:
                    _stream_state['frame_jpeg'] = jpeg.tobytes()
                    _stream_state['meta']       = meta
        except Exception as _cam_exc:
            log.warning('Camera frame failed: %s', _cam_exc)
            self._vf = ZERO_VISION

        vf  = self._vf
        vx  = action_to_vx(self._prev_action or ACT_FORWARD, fwd_duty(vf.free_path_C))
        obs = build_obs(sensor_obs, vf, vx)
        return np.clip(
            obs + np.random.normal(0, NOISE_STD, OBS_DIM).astype(np.float32),
            0.0, 1.0,
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.rover.stop_motors()
        self._navigator           = GreenNavigator()
        self._prev_center         = 0.0
        self._prev_action         = None
        self._prev_plant_detected = False
        self._prev_prox_triggered = False
        self._t0                  = time.monotonic()
        obs = self._capture()
        return obs, {}

    def step(self, action):
        global plant_found
        action    = int(action)
        raw_L, raw_F, raw_R = self._raw_sensors
        vf        = self._vf

        # ── Deterministic overrides (same priority as before) ──────────────
        safe_act, triggered = safety_override(raw_L, raw_F, raw_R)
        if triggered:
            final_action = safe_act
        else:
            hazard_act, hazard_trig = hazard_steer_override(vf)
            if hazard_trig:
                final_action = hazard_act
            else:
                nav_act, nav_on = self._navigator.step(vf, announce=log.info)
                final_action = nav_act if nav_on else action

        # ── Drive ─────────────────────────────────────────────────────────
        lc, rc = ACTIONS[final_action]
        duty   = fwd_duty(vf.free_path_C)
        self.rover.drive(_CMD[lc], _CMD[rc], left_speed=duty, right_speed=duty)

        # ── Pace ──────────────────────────────────────────────────────────
        elapsed = time.monotonic() - self._t0
        slack   = TIMESTEP_S - elapsed
        if slack > 0:
            time.sleep(slack)
        self._t0 = time.monotonic()

        # ── New observation ────────────────────────────────────────────────
        new_obs        = self._capture()
        vf_new         = self._vf
        raw_new        = self._raw_sensors

        # ── Reward ────────────────────────────────────────────────────────
        reward = compute_reward(
            raw_new, vf_new, final_action, triggered,
            self._prev_center, self._prev_action,
        )
        self._prev_center = vf_new.plant_C + vf_new.tree_C
        self._prev_action = final_action
        self._step_count += 1

        # ── Per-step YOLO log ──────────────────────────────────────────────
        dets, _, _ = self.vision.get_detections()
        det_str    = ', '.join(f'{d[0]}:{d[1]:.2f}' for d in dets) if dets else 'none'
        log.info(
            f'step={self._step_count:6d}  '
            f'yolo=[{det_str}]  '
            f'plant=({vf_new.plant_L:.2f},{vf_new.plant_C:.2f},{vf_new.plant_R:.2f})  '
            f'us=({raw_new[0]:.1f},{raw_new[1]:.1f},{raw_new[2]:.1f})cm  '
            f'rew={reward:+.3f}  '
            f'cmd=({_CMD[lc][:3]},{_CMD[rc][:3]})  '
            f'nav={self._navigator.state_name}  '
            f'plant_found={plant_found}'
        )

        # ── Capture trigger: proximity-based (primary) + YOLO nav-lock (fallback) ──
        # Fires on the rising edge entering the 30-50 cm capture window,
        # or when GreenNavigator locks onto a detected plant/object,
        # subject to a per-capture cooldown so we don't re-trigger immediately.
        from vision import PLANT_BBOX_THR
        bbox_area  = self.vision.get_max_plant_bbox_area()
        nav_locked = self._navigator.state_name in ('seek', 'approach', 'orbit')
        raw_f      = self._raw_sensors[1]   # front sensor (cm)

        # Primary trigger: any object enters the 30-55 cm capture zone
        prox_triggered = 0.0 < raw_f <= PROX_TRIGGER_CM

        cooldown_ok = (time.monotonic() - self._last_capture_ts) >= CAPTURE_COOLDOWN_S
        rising_edge = (prox_triggered and not self._prev_prox_triggered) or                       (nav_locked and not self._prev_plant_detected)

        if rising_edge and cooldown_ok:
            self._approach_and_capture()
            self._last_capture_ts = time.monotonic()

        self._prev_prox_triggered = prox_triggered
        self._prev_plant_detected = nav_locked

        return new_obs, float(reward), False, False, {}

    def _approach_and_capture(self):
        """
        Approach the nearest object, stop at 30-50 cm, burst-capture the
        sharpest frame, then spin 180° and resume exploration.
        """
        global plant_found, _captured_image_path
        APPROACH_TIMEOUT = 120     # max iterations (~6 s at 20 Hz)
        STEER_CONF_THR   = 0.10   # any YOLO detection helps steer
        FWD_SPEED        = 45
        STEER_SPEED      = 42
        CAPTURE_DIR      = os.path.dirname(os.path.abspath(__file__))
        N_BURST          = 7
        BURST_INTERVAL   = 0.12

        log.info('[approach] Object in range — approaching to %gcm', APPROACH_STOP_CM)
        self.rover.stop_motors()
        time.sleep(0.15)

        # ── Phase 1: close the gap to APPROACH_STOP_CM ───────────────────
        for _ in range(APPROACH_TIMEOUT):
            sensors  = self.rover.get_ultrasonic()
            front_cm = sensors.get(3) if sensors.get(3) is not None else 999.0

            if front_cm <= APPROACH_STOP_CM:
                log.info('[approach] In capture window (front=%.1f cm)', front_cm)
                break

            # Steer toward highest-confidence detection (any class)
            dets, w, _ = self.vision.get_detections()
            if dets:
                best = max(dets, key=lambda d: d[1])   # highest conf
                name, conf, x1, y1, x2, y2 = best
                cx_norm = ((x1 + x2) / 2.0) / max(w, 1)
                if conf >= STEER_CONF_THR:
                    if cx_norm < 0.37:
                        self.rover.drive('backward', 'forward', left_speed=STEER_SPEED, right_speed=STEER_SPEED)
                    elif cx_norm > 0.63:
                        self.rover.drive('forward', 'backward', left_speed=STEER_SPEED, right_speed=STEER_SPEED)
                    else:
                        self.rover.drive('forward', 'forward', left_speed=FWD_SPEED, right_speed=FWD_SPEED)
                else:
                    self.rover.drive('forward', 'forward', left_speed=FWD_SPEED, right_speed=FWD_SPEED)
            else:
                self.rover.drive('forward', 'forward', left_speed=FWD_SPEED, right_speed=FWD_SPEED)

            time.sleep(0.05)

        self.rover.stop_motors()
        time.sleep(0.4)   # let vibration settle before burst

        # ── Phase 2: burst capture — pick sharpest frame, keep in memory ────
        log.info('[approach] Burst capturing %d frames', N_BURST)
        frames, scores = [], []
        try:
            for _ in range(N_BURST):
                raw = np.array(self.rover.getframe(), dtype=np.uint8)
                raw = cv2.flip(raw, -1)
                raw = cv2.cvtColor(raw, cv2.COLOR_RGB2BGR)
                gray  = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
                score = cv2.Laplacian(gray, cv2.CV_64F).var()
                frames.append(raw)
                scores.append(score)
                time.sleep(BURST_INTERVAL)

            best = frames[int(np.argmax(scores))]
            log.info('[approach] Sharpness: %s — best=%d',
                     [f'{s:.0f}' for s in scores], int(np.argmax(scores)))

            # CLAHE contrast normalisation
            lab = cv2.cvtColor(best, cv2.COLOR_BGR2LAB)
            l_ch, a_ch, b_ch = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            best  = cv2.cvtColor(cv2.merge([clahe.apply(l_ch), a_ch, b_ch]),
                                 cv2.COLOR_LAB2BGR)

            best = cv2.flip(best, -1)  # 180 rotate for app display
            _, buf = cv2.imencode('.jpg', best, [cv2.IMWRITE_JPEG_QUALITY, 92])
            global _captured_image_bytes
            _captured_image_bytes = buf.tobytes()
            log.info('[approach] Captured %d bytes in memory', len(_captured_image_bytes))
        except Exception as exc:
            log.error('[approach] Burst capture failed: %s', exc)
            _captured_image_bytes = None

        # ── Phase 3: turn ~180° and resume ───────────────────────────────
        log.info('[approach] Spinning 180° (%.1fs)', SPIN_180_S)
        self.rover.drive('forward', 'backward', left_speed=50, right_speed=50)
        time.sleep(SPIN_180_S)
        self.rover.stop_motors()
        log.info('[approach] Resuming exploration')

        plant_found = True
        _capture_event.set()


# ── STOP CALLBACK ──────────────────────────────────────────────────────────────
class _StopOnEvent(BaseCallback):
    def __init__(self, event):
        super().__init__()
        self._event = event

    def _on_step(self) -> bool:
        return not self._event.is_set()


# ── MAIN ───────────────────────────────────────────────────────────────────────
def main():
    global _rover
    rover  = RoverAPI()
    _rover = rover
    vision = VisionPipeline(model_path=YOLO_MODEL)
    env    = RoverEnv(rover, vision)

    checkpoint_zip = CHECKPOINT + '.zip'
    if os.path.exists(checkpoint_zip):
        model = PPO.load(CHECKPOINT, env=env)
        log.info(f'Resumed SB3 checkpoint: {checkpoint_zip}')
    else:
        model = PPO(
            'MlpPolicy', env,
            n_steps       = 64,
            batch_size    = 16,
            n_epochs      = 4,
            gamma         = 0.99,
            gae_lambda    = 0.95,
            clip_range    = 0.20,
            ent_coef      = 0.05,
            vf_coef       = 0.50,
            learning_rate = 3e-4,
            policy_kwargs = dict(net_arch=[64]),
            verbose       = 0,
        )
        log.info('No checkpoint — training from scratch.')

    # Stream server
    _stream_state['lock'] = threading.Lock()
    threading.Thread(target=_stream_server_thread, daemon=True).start()

    # Graceful shutdown
    stop_event = threading.Event()
    def _stop(sig, frame):
        stop_event.set()
    signal.signal(signal.SIGINT,  _stop)
    signal.signal(signal.SIGTERM, _stop)
    log.info('PPO rover online (SB3). Ctrl-C to stop.')

    ckpt_dir = os.path.dirname(CHECKPOINT)
    callbacks = [
        CheckpointCallback(save_freq=512, save_path=ckpt_dir, name_prefix='ppo_sb3'),
        _StopOnEvent(stop_event),
    ]

    try:
        model.learn(
            total_timesteps     = 10_000_000,
            callback            = callbacks,
            reset_num_timesteps = False,
            progress_bar        = False,
        )
    finally:
        rover.stop_motors()
        rover.close()
        model.save(CHECKPOINT)
        log.info('Shutdown. Model saved.')


if __name__ == '__main__':
    raise SystemExit(main())
