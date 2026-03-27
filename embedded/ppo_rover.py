#!/usr/bin/env python3
"""
Online PPO self-driving rover controller (pure numpy).

Observation (16-dim, all [0, 1]):
  [rf_L, rf_F, rf_R,
   plant_L, plant_C, plant_R,
   tree_L,  tree_C,  tree_R,
   hazard_L, hazard_C, hazard_R,
   crop_L,  crop_R,
   free_path_C,
   vx_capped]

Actions : 9 discrete (left_motor x right_motor) combinations
Safety  : hard deterministic override before every motor command
Speed   : forward PWM scales with free_path_C in [MIN_FWD_DUTY, MAX_FWD_DUTY] = [30, 80] %
"""

import json
import os
import socket
import struct
import sys
import time
import signal
import logging

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from api import RoverAPI
from vision import VisionPipeline, VisionFeatures, ZERO_VISION

# ── CONSTANTS ─────────────────────────────────────────────────────────────────
SENSOR_MAX_CM  = 30.0
SAFETY_CM      = 15.0    # left/right spin threshold (cm)
FRONT_SAFETY_CM = 25.0   # front backward threshold — larger for more reaction time
TIMESTEP_S     = 0.05
STREAM_PORT    = 5000
JPEG_QUALITY   = 70
FRAME_W, FRAME_H = 640, 480
NOISE_STD      = 0.02
MIN_FWD_DUTY   = 35.0    # minimum forward PWM (%)
MAX_FWD_DUTY   = 55.0    # maximum forward PWM (%)

OBS_DIM    = 16
HIDDEN_DIM = 64
N_ACTIONS  = 9

LR           = 3e-4
GAMMA        = 0.99
LAM          = 0.95
CLIP_EPS     = 0.20
ENTROPY_COEF = 0.05
VALUE_COEF   = 0.50
BUFFER_SIZE  = 64
PPO_EPOCHS   = 4
MINIBATCH    = 16

YOLO_MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'yolov8n.pt')

# ── ACTION TABLE ──────────────────────────────────────────────────────────────
ACTIONS = [
    (-1, -1),   # 0  both backward
    (-1,  0),   # 1  left back, right stop
    (-1,  1),   # 2  spin left
    ( 0, -1),   # 3  left stop, right back
    ( 0,  0),   # 4  full stop
    ( 0,  1),   # 5  gentle right turn
    ( 1, -1),   # 6  spin right
    ( 1,  0),   # 7  gentle left turn
    ( 1,  1),   # 8  both forward
]

ACT_BACKWARD     = 0
ACT_SPIN_LEFT    = 2
ACT_SPIN_RIGHT   = 6
ACT_GENTLE_RIGHT = 5   # left=stop, right=fwd 2192 arcs left
ACT_GENTLE_LEFT  = 7   # left=fwd, right=stop 2192 arcs right
ACT_FORWARD      = 8

# Actions that move backward voluntarily (blocked unless safety triggers)
_VOLUNTARY_BACKWARD = {0, 1, 3}  # both-back, left-back, right-back

_CMD = {-1: 'backward', 0: 'stop', 1: 'forward'}


# ── ACTOR-CRITIC ──────────────────────────────────────────────────────────────
class ActorCritic:
    """Linear(16->64)->ReLU->{ Actor: Linear(64->9), Critic: Linear(64->1) }"""

    def __init__(self, lr=LR):
        rng   = np.random.default_rng(0)
        scale = np.sqrt(2.0 / OBS_DIM)

        self.W1 = rng.standard_normal((OBS_DIM,    HIDDEN_DIM)) * scale
        self.b1 = np.zeros(HIDDEN_DIM)
        self.Wa = rng.standard_normal((HIDDEN_DIM, N_ACTIONS))  * 0.01
        self.ba = np.zeros(N_ACTIONS)
        self.Wv = rng.standard_normal((HIDDEN_DIM, 1))          * 1.0
        self.bv = np.zeros(1)

        self._params = [self.W1, self.b1, self.Wa, self.ba, self.Wv, self.bv]
        self._m = [np.zeros_like(p) for p in self._params]
        self._v = [np.zeros_like(p) for p in self._params]
        self._t = 0
        self.lr = lr

    def _fwd(self, x):
        h_pre  = x @ self.W1 + self.b1
        h      = np.maximum(0.0, h_pre)
        logits = h @ self.Wa + self.ba
        values = (h @ self.Wv + self.bv).ravel()
        return logits, values, h, h_pre

    @staticmethod
    def _softmax(logits):
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    def act(self, obs):
        x = obs[None].astype(np.float32)
        logits, values, _, _ = self._fwd(x)
        probs      = self._softmax(logits)[0]
        action_idx = int(np.random.choice(N_ACTIONS, p=probs))
        log_prob   = float(np.log(probs[action_idx] + 1e-8))
        return action_idx, log_prob, float(values[0])

    def update(self, obs_b, act_b, logp_old_b, ret_b, adv_b):
        B = len(act_b)
        x   = obs_b.astype(np.float32)
        ret = ret_b.astype(np.float32)
        adv = adv_b.astype(np.float32)

        logits, values, h, h_pre = self._fwd(x)
        probs     = self._softmax(logits)
        log_probs = np.log(probs + 1e-8)
        logp_new  = log_probs[np.arange(B), act_b]

        ratio         = np.exp(logp_new - logp_old_b)
        clipped_ratio = np.clip(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS)
        is_clipped    = (ratio * adv) > (clipped_ratio * adv)
        entropy       = -(probs * log_probs).sum(axis=1)

        d_val      = VALUE_COEF * (values - ret) / B
        d_logp     = np.where(is_clipped, 0.0, -adv * ratio) / B
        d_logits_e = (ENTROPY_COEF / B) * probs * (log_probs + entropy[:, None])

        one_hot          = np.zeros((B, N_ACTIONS), dtype=np.float32)
        one_hot[np.arange(B), act_b] = 1.0
        d_logits         = d_logp[:, None] * (one_hot - probs) + d_logits_e

        dWa  = h.T @ d_logits
        dba  = d_logits.sum(axis=0)
        d_h  = d_logits @ self.Wa.T
        dWv  = h.T @ d_val[:, None]
        dbv  = np.array([d_val.sum()])
        d_h += d_val[:, None] * self.Wv.T

        d_hp = d_h * (h_pre > 0).astype(np.float32)
        dW1  = x.T @ d_hp
        db1  = d_hp.sum(axis=0)

        self._adam([dW1, db1, dWa, dba, dWv, dbv])

    def _adam(self, grads, beta1=0.9, beta2=0.999, eps=1e-8):
        self._t += 1
        bc1 = 1.0 - beta1 ** self._t
        bc2 = 1.0 - beta2 ** self._t
        for i, (p, g) in enumerate(zip(self._params, grads)):
            self._m[i] = beta1 * self._m[i] + (1 - beta1) * g
            self._v[i] = beta2 * self._v[i] + (1 - beta2) * g ** 2
            p -= self.lr * (self._m[i] / bc1) / (np.sqrt(self._v[i] / bc2) + eps)

    def save(self, path):
        np.savez(path, W1=self.W1, b1=self.b1,
                 Wa=self.Wa, ba=self.ba, Wv=self.Wv, bv=self.bv)

    def load(self, path):
        """Returns True on success, False on shape mismatch."""
        d = np.load(path)
        expected = {
            'W1': (OBS_DIM, HIDDEN_DIM), 'b1': (HIDDEN_DIM,),
            'Wa': (HIDDEN_DIM, N_ACTIONS), 'ba': (N_ACTIONS,),
            'Wv': (HIDDEN_DIM, 1), 'bv': (1,),
        }
        for name, shape in expected.items():
            if name not in d or d[name].shape != shape:
                return False
        for name in expected:
            getattr(self, name)[:] = d[name]
        return True


# ── SENSOR HELPERS ────────────────────────────────────────────────────────────
def _norm(dist_cm):
    if dist_cm is None:
        return 1.0  # timeout = nothing detected = max range
    return float(min(dist_cm, SENSOR_MAX_CM) / SENSOR_MAX_CM)


def read_sensors(rover):
    raw = rover.get_ultrasonic()
    # None (timeout) = nothing in range → treat as max range, not collision
    L = raw.get(1) if raw.get(1) is not None else SENSOR_MAX_CM
    F = raw.get(3) if raw.get(3) is not None else SENSOR_MAX_CM
    R = raw.get(2) if raw.get(2) is not None else SENSOR_MAX_CM
    return (L, F, R), np.array([_norm(L), _norm(F), _norm(R)], dtype=np.float32)


# ── OBSERVATION BUILDER ───────────────────────────────────────────────────────
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


# ── SAFETY OVERRIDE ───────────────────────────────────────────────────────────
def safety_override(raw_L, raw_F, raw_R):
    # Backward ONLY when front sensor detects danger
    if raw_F < FRONT_SAFETY_CM:
        return ACT_BACKWARD,   True
    # Spin away from left/right obstacles
    if raw_L < SAFETY_CM:
        return ACT_SPIN_RIGHT, True
    if raw_R < SAFETY_CM:
        return ACT_SPIN_LEFT,  True
    return None, False


# ── HAZARD STEER OVERRIDE ────────────────────────────────────────────────────
HAZARD_STEER_THR  = 0.25  # min hazard_C score to trigger steering
GREEN_SEEK_THR    = 0.15  # min total green score to activate seeking
GREEN_ORBIT_THR   = 0.20  # center green threshold to start orbit
GREEN_SIDE_BIAS   = 0.12  # L-R imbalance to steer toward a side
ORBIT_SPIN_STEPS  = 5     # spin steps per orbit cycle

def hazard_steer_override(vf):
    """
    When a hazard (person/chair/table) is detected ahead, spin around it
    instead of reversing. Steers toward the clearer side.
    Returns (action_idx | None, triggered: bool).
    """
    if vf.hazard_C < HAZARD_STEER_THR:
        return None, False
    # Spin toward the side with less hazard
    if vf.hazard_L <= vf.hazard_R:
        return ACT_SPIN_LEFT, True   # less hazard on left → spin left
    else:
        return ACT_SPIN_RIGHT, True  # less hazard on right → spin right


# ── GREEN NAVIGATOR ─────────────────────────────────────────────────────────────────────
GN_DETECT_THR   = 0.12   # total YOLO green to consider plant found
GN_CENTER_THR   = 0.18   # center green to start approaching
GN_ORBIT_THR    = 0.25   # center green to lock into orbit
GN_SCAN_SPIN    = 10     # steps to spin per scan phase
GN_REPOSE_STEPS = 8      # backward steps when plant is lost
GN_APPROACH_MAX = 15     # max forward steps before forcing orbit

_GN_SCAN     = 0
_GN_SEEK     = 1
_GN_APPROACH = 2
_GN_ORBIT    = 3
_GN_REPOSE   = 4
_GN_NAMES    = ['scan', 'seek', 'approach', 'orbit', 'repose']


class GreenNavigator:
    """
    Deterministic finite-state machine for plant-finding and orbiting.

    SCAN     : No green visible. Alternates spin + forward to survey.
    SEEK     : Green on one side — steers toward it.
    APPROACH : Green centred ahead — drives straight toward it.
    ORBIT    : Plant well-centred — arcs around it continuously.
    REPOSE   : Just lost the plant — backs up, then returns to SCAN.

    Always returns an action (triggered=True), so PPO is only active
    as a background learner while this controller drives.
    """

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

        # Print to console on first detection
        if found and not self._was_found and announce:
            announce(
                f'{"=" * 52}\n'
                f'  PLANT / TREE DETECTED\n'
                f'  L={left_g:.2f}  C={center_g:.2f}  R={right_g:.2f}  total={total_g:.2f}\n'
                f'{"=" * 52}'
            )
        self._was_found = found

        # ── Transitions ───────────────────────────────────────────────────
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
                self._orbit_dir = (ACT_GENTLE_LEFT if right_g >= left_g
                                   else ACT_GENTLE_RIGHT)
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

        # ── Action selection ──────────────────────────────────────────────
        if self._state == _GN_SCAN:
            self._scan_tick += 1
            phase = self._scan_tick % (GN_SCAN_SPIN * 2)
            if phase == 0:
                self._scan_dir = (ACT_SPIN_RIGHT if self._scan_dir == ACT_SPIN_LEFT
                                  else ACT_SPIN_LEFT)
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
                return (ACT_GENTLE_LEFT if left_g >= right_g
                        else ACT_GENTLE_RIGHT), True
            return self._orbit_dir, True

        if self._state == _GN_REPOSE:
            return ACT_BACKWARD, True

        return ACT_FORWARD, True




# ── REWARD ────────────────────────────────────────────────────────────────────
def compute_reward(raw_sensors, vf, action_idx, safety_triggered,
                   prev_center=0.0, prev_action=None):
    raw_L, raw_F, raw_R = raw_sensors
    left_cmd, right_cmd = ACTIONS[action_idx]
    r = 0.0

    # 1. Survival bonus — stabilises training, encourages long safe runs
    r += 0.02

    # 2. Forward motion (0.10 -> 0.20: competes better against vision rewards)
    if left_cmd == 1 and right_cmd == 1:
        r += 0.20

    # 3. Temporal progress — reward approaching green, penalise drifting away
    #    This is the highest-impact term: agent must MOVE toward green, not just see it.
    current_center = vf.plant_C + vf.tree_C
    delta_center   = current_center - prev_center
    r += 0.30 * delta_center

    # 4. Static green centering (reduced from 0.45 — progress signal covers approach)
    r += 0.25 * (1.0 - abs(vf.plant_L - vf.plant_R))   # lateral alignment
    r += 0.25 * (1.0 - abs(vf.tree_L  - vf.tree_R))
    r += 0.30 * (vf.plant_C + vf.tree_C)                # centering bonus

    # 5. Side green presence (0.20 -> 0.10: reduce passive-sit reward)
    r += 0.10 * (vf.plant_L + vf.plant_R + vf.tree_L + vf.tree_R)

    # 6. Crop-row following (reduced: 0.15 -> 0.08 presence bonus)
    r += 0.15 * (1.0 - abs(vf.crop_L - vf.crop_R))
    r += 0.08 * (vf.crop_L + vf.crop_R)

    # 7. Free path ahead
    r += 0.15 * vf.free_path_C

    # 8. Proportional steering toward green (was binary +0.15)
    #    Scales continuously with how correct the turn direction is
    lc_sign     = 1 if left_cmd == 1 else (-1 if left_cmd == -1 else 0)
    rc_sign     = 1 if right_cmd == 1 else (-1 if right_cmd == -1 else 0)
    turn_dir    = (rc_sign - lc_sign) / 2.0   # +1 = right, -1 = left, 0 = straight
    left_green  = vf.plant_L + vf.tree_L
    right_green = vf.plant_R + vf.tree_R
    r += 0.20 * (right_green - left_green) * turn_dir

    # 9. No-green penalty
    green_total   = (vf.plant_L + vf.plant_C + vf.plant_R
                     + vf.tree_L + vf.tree_C + vf.tree_R)
    crop_presence = vf.crop_L + vf.crop_R
    if green_total < 0.10 and crop_presence < 0.05:
        r -= 0.40

    # 10. Smoothness — penalise jitter / rapid action switching
    if prev_action is not None and action_idx != prev_action:
        r -= 0.05

    # 11. Hazard penalty (smoothed + capped: was uncapped 0.5 * sum)
    r -= 1.0 * min(1.0, vf.hazard_L + vf.hazard_C + vf.hazard_R)

    # 12. Light backward discouragement (backward is now used for repositioning)
    if left_cmd == -1 and right_cmd == -1 and not safety_triggered:
        r -= 0.08

    # 13. Continuous collision shaping (was -1.0 hard threshold at 20%)
    #     "Closer = worse" learned continuously, not as sudden fear
    r -= 1.0 * (1.0 - _norm(raw_F))

    return r


# ── GAE ───────────────────────────────────────────────────────────────────────
def compute_gae(rewards, values, next_value):
    T   = len(rewards)
    adv = np.zeros(T, dtype=np.float32)
    gae = 0.0
    for t in reversed(range(T)):
        nv    = next_value if t == T - 1 else values[t + 1]
        delta = rewards[t] + GAMMA * nv - values[t]
        gae   = delta + GAMMA * LAM * gae
        adv[t] = gae
    returns = adv + np.array(values, dtype=np.float32)
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    return adv, returns


def fwd_duty(free_path_C):
    # Scale forward PWM with path clarity: 30% (blocked) -> 80% (clear)
    return MIN_FWD_DUTY + (MAX_FWD_DUTY - MIN_FWD_DUTY) * float(free_path_C)


def action_to_vx(action_idx, duty=None):
    if duty is None:
        duty = MIN_FWD_DUTY
    lc, rc = ACTIONS[action_idx]
    return duty / 100.0 if (lc == 1 and rc == 1) else 0.0



# ── STREAM SERVER ─────────────────────────────────────────────────────────────
_stream_state = {'frame_jpeg': None, 'meta': b'{}', 'lock': None}

def _serve_client(conn):
    import time as _t
    try:
        while True:
            with _stream_state['lock']:
                jpeg = _stream_state['frame_jpeg']
                meta = _stream_state['meta']
            if jpeg is None:
                _t.sleep(0.01)
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
    import threading as _th
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(('0.0.0.0', STREAM_PORT))
    srv.listen(4)
    log.info(f'Stream server listening on :{STREAM_PORT}')
    while True:
        try:
            conn, addr = srv.accept()
            log.info(f'Viewer connected from {addr}')
            _th.Thread(target=_serve_client, args=(conn,), daemon=True).start()
        except Exception:
            pass

# ── CONTROL LOOP ──────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(message)s',
                    datefmt='%H:%M:%S')
log = logging.getLogger('ppo_rover')

CHECKPOINT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ppo_weights.npz')


def main():
    rover  = RoverAPI()
    model  = ActorCritic()
    vision = VisionPipeline(model_path=YOLO_MODEL)

    if os.path.exists(CHECKPOINT):
        if model.load(CHECKPOINT):
            log.info(f'Resumed from checkpoint: {CHECKPOINT}')
        else:
            log.warning('Checkpoint shapes mismatch (obs_dim changed) — starting fresh.')
    else:
        log.info('No checkpoint — training from scratch.')

    obs_buf, act_buf, logp_buf, rew_buf, val_buf = [], [], [], [], []

    step         = 0
    ep_reward    = 0.0
    safety_count = 0
    green_count  = 0
    running      = True
    navigator    = GreenNavigator()
    prev_center  = 0.0   # for temporal progress reward

    # Watchdog: stops motors if main loop freezes for > 5 s
    import threading
    _last_tick = [time.monotonic()]
    WATCHDOG_S = 5.0
    def _watchdog():
        while running:
            time.sleep(1.0)
            if time.monotonic() - _last_tick[0] > WATCHDOG_S:
                log.error('Watchdog: main loop froze -- emergency stop')
                try:
                    rover.stop_motors()
                except Exception:
                    pass
    threading.Thread(target=_watchdog, daemon=True).start()

    _stream_state['lock'] = threading.Lock()
    threading.Thread(target=_stream_server_thread, daemon=True).start()

    def _stop(sig, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT,  _stop)
    signal.signal(signal.SIGTERM, _stop)
    log.info('PPO rover online. Ctrl-C to stop.')

    try:
        while running:
            t0 = time.monotonic()

            # 1. Ultrasonic sensors
            (raw_L, raw_F, raw_R), sensor_obs = read_sensors(rover)

            # 2. Camera + vision pipeline
            try:
                raw_frame = np.array(rover.getframe(), dtype=np.uint8)
                vf = vision.update(raw_frame)
                _frame_bgr = cv2.cvtColor(raw_frame, cv2.COLOR_RGB2BGR)
                _frame_bgr = cv2.resize(_frame_bgr, (FRAME_W, FRAME_H))
                _, _jpeg = cv2.imencode('.jpg', _frame_bgr,
                                        [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                _meta = json.dumps({'L': round(raw_L, 1),
                                    'F': round(raw_F, 1),
                                    'R': round(raw_R, 1)}).encode()
                with _stream_state['lock']:
                    _stream_state['frame_jpeg'] = _jpeg.tobytes()
                    _stream_state['meta'] = _meta
            except Exception:
                vf = ZERO_VISION

            # 3. Build 16-dim observation
            vx_capped = action_to_vx(act_buf[-1], fwd_duty(vf.free_path_C)) if act_buf else 0.0
            obs = build_obs(sensor_obs, vf, vx_capped)
            obs_n = np.clip(
                obs + np.random.normal(0, NOISE_STD, OBS_DIM).astype(np.float32),
                0.0, 1.0,
            )

            # 4. PPO action
            action_idx, log_prob, value = model.act(obs_n)

            # 5. Safety override (ultrasonic — highest priority)
            safe_act, triggered = safety_override(raw_L, raw_F, raw_R)
            if triggered:
                action_idx    = safe_act
                safety_count += 1
            else:
                # Hazard steer (second priority)
                hazard_act, hazard_triggered = hazard_steer_override(vf)
                if hazard_triggered:
                    action_idx = hazard_act
                else:
                    # GreenNavigator — primary deterministic controller
                    nav_act, nav_on = navigator.step(vf, announce=log.info)
                    if nav_on:
                        action_idx = nav_act
                        green_count += 1

            # 6. Drive: forward PWM scales 30->80% with path clarity
            lc, rc = ACTIONS[action_idx]
            duty = fwd_duty(vf.free_path_C)
            rover.drive(
                _CMD[lc], _CMD[rc],
                left_speed  = duty if lc == 1 else 100.0,
                right_speed = duty if rc == 1 else 100.0,
            )

            # 7. Reward
            prev_act   = act_buf[-1] if act_buf else None
            reward     = compute_reward(
                (raw_L, raw_F, raw_R), vf, action_idx, triggered,
                prev_center, prev_act,
            )
            prev_center = vf.plant_C + vf.tree_C
            ep_reward  += reward

            obs_buf.append(obs_n); act_buf.append(action_idx)
            logp_buf.append(log_prob); rew_buf.append(reward); val_buf.append(value)
            step += 1

            # 8. PPO update
            if len(obs_buf) >= BUFFER_SIZE:
                (_, nF, _), ns_obs = read_sensors(rover)
                next_obs = build_obs(ns_obs, ZERO_VISION, action_to_vx(act_buf[-1]))
                _, _, next_val = model.act(next_obs)  # value is 3rd element

                adv, returns = compute_gae(rew_buf, val_buf, next_val)
                obs_arr  = np.stack(obs_buf)
                act_arr  = np.array(act_buf,  dtype=np.int32)
                logp_arr = np.array(logp_buf, dtype=np.float32)

                for _ in range(PPO_EPOCHS):
                    idx = np.random.permutation(BUFFER_SIZE)
                    for s in range(0, BUFFER_SIZE, MINIBATCH):
                        mb = idx[s:s + MINIBATCH]
                        model.update(obs_arr[mb], act_arr[mb],
                                     logp_arr[mb], returns[mb], adv[mb])

                log.info(
                    f'step={step:6d}  rew={ep_reward:+7.2f}  '
                    f'safe={safety_count:2d}  green={green_count:2d}  nav={navigator.state_name:8s}  '
                    f'us=({raw_L:5.1f},{raw_F:5.1f},{raw_R:5.1f})cm  '
                    f'plant=({vf.plant_L:.2f},{vf.plant_C:.2f},{vf.plant_R:.2f})  '
                    f'tree=({vf.tree_L:.2f},{vf.tree_C:.2f},{vf.tree_R:.2f})  '
                    f'crop=({vf.crop_L:.2f},{vf.crop_R:.2f})  '
                    f'free={vf.free_path_C:.2f}  cmd=({_CMD[lc][:3]},{_CMD[rc][:3]})'
                )
                model.save(CHECKPOINT)
                obs_buf.clear(); act_buf.clear(); logp_buf.clear()
                rew_buf.clear();  val_buf.clear()
                ep_reward = 0.0;  safety_count = 0;  green_count = 0

            # 9. Pace loop
            _last_tick[0] = time.monotonic()
            elapsed = time.monotonic() - t0
            slack   = TIMESTEP_S - elapsed
            if slack > 0:
                time.sleep(slack)

    finally:
        rover.stop_motors()
        rover.close()
        model.save(CHECKPOINT)
        log.info(f'Shutdown after {step} steps. Weights saved -> {CHECKPOINT}')


if __name__ == '__main__':
    raise SystemExit(main())
