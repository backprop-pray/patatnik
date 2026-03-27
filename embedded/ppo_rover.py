#!/usr/bin/env python3
"""
Online PPO self-driving rover controller (pure numpy).

State  : [rf_L, rf_F, rf_R] — 3 normalised ultrasonic readings
Actions: 9 discrete (left_motor, right_motor) combinations
Safety : hard deterministic override that fires before every motor command
"""

import os
import sys
import time
import signal
import logging

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from api import RoverAPI

# ── CONSTANTS ─────────────────────────────────────────────────────────────────
SENSOR_MAX_CM = 30.0   # sensors saturate at this range (cm)
SAFETY_CM     = 15.0   # hard safety threshold (cm)
TIMESTEP_S    = 0.10   # control-loop period → 10 Hz
NOISE_STD     = 0.02   # gaussian noise added to obs during training

OBS_DIM    = 3
HIDDEN_DIM = 32
N_ACTIONS  = 9

LR            = 3e-4
GAMMA         = 0.99
LAM           = 0.95   # GAE lambda
CLIP_EPS      = 0.20
ENTROPY_COEF  = 0.01
VALUE_COEF    = 0.50
BUFFER_SIZE   = 32     # steps collected before each PPO update
PPO_EPOCHS    = 4
MINIBATCH     = 16

# ── ACTION TABLE ──────────────────────────────────────────────────────────────
# Each entry: (left_motor, right_motor)  -1=back  0=stop  +1=forward
ACTIONS = [
    (-1, -1),   # 0  both backward
    (-1,  0),   # 1  left back,  right stop
    (-1,  1),   # 2  spin left   (escape right obstacle)
    ( 0, -1),   # 3  left stop,  right back
    ( 0,  0),   # 4  full stop
    ( 0,  1),   # 5  left stop,  right forward
    ( 1, -1),   # 6  spin right  (escape left obstacle)
    ( 1,  0),   # 7  left forward, right stop
    ( 1,  1),   # 8  both forward  ← target behaviour
]

ACT_BACKWARD   = 0
ACT_SPIN_LEFT  = 2
ACT_SPIN_RIGHT = 6
ACT_FORWARD    = 8

_CMD = {-1: 'backward', 0: 'stop', 1: 'forward'}


# ── ACTOR-CRITIC (pure numpy) ─────────────────────────────────────────────────
class ActorCritic:
    """
    Shared MLP:  Linear(3→32) → ReLU → { Actor: Linear(32→9),
                                           Critic: Linear(32→1) }
    Trained with Adam + PPO clipped surrogate.
    """

    def __init__(self, lr: float = LR):
        rng = np.random.default_rng(0)

        # Weight init: orthogonal-style for body, tiny for heads
        self.W1 = rng.standard_normal((OBS_DIM, HIDDEN_DIM)) * np.sqrt(2.0 / OBS_DIM)
        self.b1 = np.zeros(HIDDEN_DIM)
        self.Wa = rng.standard_normal((HIDDEN_DIM, N_ACTIONS)) * 0.01
        self.ba = np.zeros(N_ACTIONS)
        self.Wv = rng.standard_normal((HIDDEN_DIM, 1)) * 1.0
        self.bv = np.zeros(1)

        self._params = [self.W1, self.b1, self.Wa, self.ba, self.Wv, self.bv]
        self._m = [np.zeros_like(p) for p in self._params]
        self._v = [np.zeros_like(p) for p in self._params]
        self._t = 0
        self.lr = lr

    # ── Forward ──────────────────────────────────────────────────────────────
    def _fwd(self, x: np.ndarray):
        """x: (B, 3).  Returns logits(B,9), values(B,), h(B,H), h_pre(B,H)."""
        h_pre  = x @ self.W1 + self.b1
        h      = np.maximum(0.0, h_pre)
        logits = h @ self.Wa + self.ba
        values = (h @ self.Wv + self.bv).ravel()
        return logits, values, h, h_pre

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    def act(self, obs: np.ndarray):
        """
        Single inference step.
        obs: (3,)  →  action_idx (int), log_prob (float), value (float)
        """
        x = obs[None].astype(np.float32)
        logits, values, _, _ = self._fwd(x)
        probs      = self._softmax(logits)[0]
        action_idx = int(np.random.choice(N_ACTIONS, p=probs))
        log_prob   = float(np.log(probs[action_idx] + 1e-8))
        return action_idx, log_prob, float(values[0])

    # ── PPO update ───────────────────────────────────────────────────────────
    def update(self, obs_b, act_b, logp_old_b, ret_b, adv_b):
        """One minibatch PPO update (all numpy arrays)."""
        B = len(act_b)
        x    = obs_b.astype(np.float32)
        ret  = ret_b.astype(np.float32)
        adv  = adv_b.astype(np.float32)

        # ── Forward pass ─────────────────────────────────────────────────────
        logits, values, h, h_pre = self._fwd(x)
        probs     = self._softmax(logits)               # (B, 9)
        log_probs = np.log(probs + 1e-8)                # (B, 9)
        logp_new  = log_probs[np.arange(B), act_b]     # (B,)

        ratio         = np.exp(logp_new - logp_old_b)
        clipped_ratio = np.clip(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS)
        surr1 = ratio * adv
        surr2 = clipped_ratio * adv
        is_clipped = surr1 > surr2                      # where clip is active

        entropy = -(probs * log_probs).sum(axis=1)      # (B,)

        # ── Gradients ────────────────────────────────────────────────────────

        # Critic: L_v = VALUE_COEF * 0.5*(v-r)^2
        d_val = VALUE_COEF * (values - ret) / B         # (B,)

        # Actor PPO surrogate (clipped):
        # dL/d(logp_new) = -adv * ratio  if not clipped, else 0
        d_logp = np.where(is_clipped, 0.0, -adv * ratio) / B   # (B,)

        # Entropy bonus: d(-ENTROPY_COEF*H)/d(logits_i) = ENTROPY_COEF * p_i*(log_p_i + H)
        d_logits_ent = (ENTROPY_COEF / B) * probs * (log_probs + entropy[:, None])

        # Actor gradient through log-softmax:
        # d(logp_k)/d(logits_i) = delta(i,k) - p_i
        one_hot = np.zeros((B, N_ACTIONS), dtype=np.float32)
        one_hot[np.arange(B), act_b] = 1.0
        d_logits = d_logp[:, None] * (one_hot - probs) + d_logits_ent   # (B, 9)

        # ── Backprop through layers ───────────────────────────────────────────

        # Actor head
        dWa = h.T @ d_logits                            # (H, 9)
        dba = d_logits.sum(axis=0)                      # (9,)
        d_h = d_logits @ self.Wa.T                      # (B, H)

        # Critic head
        dWv = h.T @ d_val[:, None]                      # (H, 1)
        dbv = np.array([d_val.sum()])                   # (1,)
        d_h += d_val[:, None] * self.Wv.T               # (B, H)

        # ReLU + input layer
        d_h_pre = d_h * (h_pre > 0).astype(np.float32)
        dW1 = x.T @ d_h_pre                            # (3, H)
        db1 = d_h_pre.sum(axis=0)                      # (H,)

        self._adam([dW1, db1, dWa, dba, dWv, dbv])

    def _adam(self, grads, beta1=0.9, beta2=0.999, eps=1e-8):
        self._t += 1
        bc1 = 1.0 - beta1 ** self._t
        bc2 = 1.0 - beta2 ** self._t
        for i, (p, g) in enumerate(zip(self._params, grads)):
            self._m[i] = beta1 * self._m[i] + (1 - beta1) * g
            self._v[i] = beta2 * self._v[i] + (1 - beta2) * g ** 2
            p -= self.lr * (self._m[i] / bc1) / (np.sqrt(self._v[i] / bc2) + eps)

    # ── Persistence ──────────────────────────────────────────────────────────
    def save(self, path: str):
        np.savez(path, W1=self.W1, b1=self.b1,
                 Wa=self.Wa, ba=self.ba, Wv=self.Wv, bv=self.bv)

    def load(self, path: str):
        d = np.load(path)
        for name in ('W1', 'b1', 'Wa', 'ba', 'Wv', 'bv'):
            getattr(self, name)[:] = d[name]


# ── SENSOR HELPERS ────────────────────────────────────────────────────────────
def _norm(dist_cm):
    """cm → [0,1].  None (timeout) treated as 0.0 (imminent collision)."""
    if dist_cm is None:
        return 0.0
    return float(min(dist_cm, SENSOR_MAX_CM) / SENSOR_MAX_CM)


def read_sensors(rover: RoverAPI):
    """
    Returns:
      raw  : (L_cm, F_cm, R_cm)
      obs  : np.ndarray shape (3,) normalised to [0,1]

    Sensor wiring (from DualUltrasonicArray._normalize_sensor_id):
      sensor 1 → left,  sensor 2 → right,  sensor 3 → center/front
    """
    raw = rover.get_ultrasonic()
    L = raw.get(1) or 0.0
    F = raw.get(3) or 0.0
    R = raw.get(2) or 0.0
    obs = np.array([_norm(L), _norm(F), _norm(R)], dtype=np.float32)
    return (L, F, R), obs


# ── SAFETY OVERRIDE ───────────────────────────────────────────────────────────
def safety_override(raw_L: float, raw_F: float, raw_R: float):
    """
    Returns (action_idx | None, triggered: bool).
    Priority order: front > left > right.
    """
    if raw_F < SAFETY_CM:
        return ACT_BACKWARD, True    # obstacle ahead  → reverse
    if raw_L < SAFETY_CM:
        return ACT_SPIN_RIGHT, True  # left obstacle   → spin right
    if raw_R < SAFETY_CM:
        return ACT_SPIN_LEFT, True   # right obstacle  → spin left
    return None, False


# ── REWARD ────────────────────────────────────────────────────────────────────
def compute_reward(raw_sensors, action_idx: int, safety_triggered: bool) -> float:
    raw_L, raw_F, raw_R = raw_sensors
    left_cmd, right_cmd = ACTIONS[action_idx]

    # Smooth proximity penalty: 0 at max range, -0.5 when touching
    valid = [d for d in (raw_L, raw_F, raw_R) if d > 0]
    min_dist = min(valid) if valid else 0.0
    prox = -max(0.0, (SENSOR_MAX_CM - min_dist) / SENSOR_MAX_CM) * 0.5

    # Motion reward
    if left_cmd == 1 and right_cmd == 1:
        motion = 1.0     # full forward
    elif left_cmd == 1 or right_cmd == 1:
        motion = 0.3     # partial forward
    elif left_cmd == 0 and right_cmd == 0:
        motion = -0.2    # idle
    elif left_cmd == -1 and right_cmd == -1:
        motion = -0.5    # full reverse
    else:
        motion = 0.0     # turning

    safety_penalty = -2.0 if safety_triggered else 0.0

    return motion + prox + safety_penalty


# ── GAE ───────────────────────────────────────────────────────────────────────
def compute_gae(rewards, values, next_value: float):
    T   = len(rewards)
    adv = np.zeros(T, dtype=np.float32)
    gae = 0.0
    for t in reversed(range(T)):
        nv    = next_value if t == T - 1 else values[t + 1]
        delta = rewards[t] + GAMMA * nv - values[t]
        gae   = delta + GAMMA * LAM * gae
        adv[t] = gae
    returns = adv + np.array(values, dtype=np.float32)
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)   # normalise
    return adv, returns


# ── CONTROL LOOP ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger('ppo_rover')

CHECKPOINT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ppo_weights.npz')


def main():
    rover = RoverAPI()
    model = ActorCritic()

    if os.path.exists(CHECKPOINT):
        model.load(CHECKPOINT)
        log.info(f'Resumed from checkpoint: {CHECKPOINT}')
    else:
        log.info('No checkpoint found — training from scratch')

    obs_buf, act_buf, logp_buf, rew_buf, val_buf = [], [], [], [], []

    step          = 0
    ep_reward     = 0.0
    safety_count  = 0
    running       = True

    def _stop(sig, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT,  _stop)
    signal.signal(signal.SIGTERM, _stop)

    log.info('PPO rover online. Ctrl-C to stop.')

    try:
        while running:
            t0 = time.monotonic()

            # ── 1. Read sensors ───────────────────────────────────────────────
            (raw_L, raw_F, raw_R), obs = read_sensors(rover)

            # ── 2. Noisy observation for policy ──────────────────────────────
            obs_n = np.clip(
                obs + np.random.normal(0, NOISE_STD, OBS_DIM).astype(np.float32),
                0.0, 1.0,
            )

            # ── 3. PPO action ─────────────────────────────────────────────────
            action_idx, log_prob, value = model.act(obs_n)

            # ── 4. Safety override ────────────────────────────────────────────
            safe_act, triggered = safety_override(raw_L, raw_F, raw_R)
            if triggered:
                action_idx = safe_act
                safety_count += 1

            # ── 5. Send motor command ─────────────────────────────────────────
            lc, rc = ACTIONS[action_idx]
            rover.drive(_CMD[lc], _CMD[rc])

            # ── 6. Reward ─────────────────────────────────────────────────────
            reward    = compute_reward((raw_L, raw_F, raw_R), action_idx, triggered)
            ep_reward += reward

            # ── 7. Buffer ─────────────────────────────────────────────────────
            obs_buf.append(obs_n)
            act_buf.append(action_idx)
            logp_buf.append(log_prob)
            rew_buf.append(reward)
            val_buf.append(value)
            step += 1

            # ── 8. PPO update every BUFFER_SIZE steps ─────────────────────────
            if len(obs_buf) >= BUFFER_SIZE:
                # Bootstrap value
                _, next_obs = read_sensors(rover)
                _, next_val, _ = model.act(next_obs)

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
                    f'step={step:6d}  '
                    f'rew={ep_reward:+7.2f}  '
                    f'safety={safety_count:3d}  '
                    f'sensors=({raw_L:5.1f},{raw_F:5.1f},{raw_R:5.1f})cm  '
                    f'cmd=({_CMD[lc][:3]},{_CMD[rc][:3]})'
                )

                model.save(CHECKPOINT)
                obs_buf.clear(); act_buf.clear(); logp_buf.clear()
                rew_buf.clear(); val_buf.clear()
                ep_reward = 0.0
                safety_count = 0

            # ── 9. Pace loop ──────────────────────────────────────────────────
            elapsed = time.monotonic() - t0
            slack   = TIMESTEP_S - elapsed
            if slack > 0:
                time.sleep(slack)

    finally:
        rover.stop_motors()
        rover.close()
        model.save(CHECKPOINT)
        log.info(f'Shutdown after {step} steps. Weights saved to {CHECKPOINT}')


if __name__ == '__main__':
    raise SystemExit(main())
