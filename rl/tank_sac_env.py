"""
TankSACEnv — Gymnasium environment for SAC-Discrete self-driving.

Tank with 2 track sides, each with 3 discrete actions (fwd / back / stop),
giving Discrete(9) total. 3 ultrasonic rangefinder sensors, continuous state.
"""

import os
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces


class TankSACEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}

    # ── Action table: (left_mult, right_mult) ──
    # -1 = backward, 0 = stop, +1 = forward
    ACTION_TABLE = [
        (-1, -1),  # 0: reverse
        (-1,  0),  # 1: pivot reverse-left
        (-1, +1),  # 2: spin left
        ( 0, -1),  # 3: pivot reverse-right
        ( 0,  0),  # 4: coast
        ( 0, +1),  # 5: pivot right
        (+1, -1),  # 6: spin right
        (+1,  0),  # 7: pivot left
        (+1, +1),  # 8: forward
    ]

    def __init__(self, render_mode=None,
                 base_xml_path="agricultural_tank_base.xml",
                 seed=None):
        super().__init__()
        self.render_mode = render_mode

        # Resolve XML path relative to this file
        self.base_xml_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), base_xml_path
        )
        if not os.path.exists(self.base_xml_path):
            raise FileNotFoundError(f"XML not found: {self.base_xml_path}")

        self.np_random = np.random.RandomState(seed)

        # ── Spaces ──
        self.action_space = spaces.Discrete(9)
        # obs: [rf_front, rf_left, rf_right, vx, vy, sin_yaw, cos_yaw]
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(7,), dtype=np.float32
        )

        # ── Physics tuning ──
        # Static motor control value.  Rover body ~0.15 m long,
        # corridor ~0.35 m wide, gear 0.1 → effective ~0.3 m/s.
        self.MOTOR_CTRL = 10.0
        self.MAX_STEPS = 1000
        self.SENSOR_CUTOFF = 5.0   # must match XML rangefinder cutoff
        self.SIM_SUBSTEPS = 10      # mj_step calls per env step

        # Will be set in reset()
        self.model = None
        self.data = None
        self.step_count = 0
        self.wave_freq = 0.0
        self.wave_amp = 0.0

        self.reset(seed=seed)

    # ──────────────────────────────────────────────────────────
    #  Procedural crop maze generation
    # ──────────────────────────────────────────────────────────
    def _spawn_crops(self, wb, row_spacing, n_plants, plant_spacing):
        """Create two sinusoidal walls of crops."""
        self.wave_freq = self.np_random.uniform(0.4, 0.8)
        self.wave_amp = self.np_random.uniform(1.5, 2.5)

        for i in range(n_plants):
            base_x = 0.5 + i * plant_spacing
            base_y = np.sin(base_x * self.wave_freq) * self.wave_amp

            for row_idx, sign in enumerate([-1, 1]):
                if self.np_random.random() < 0.05:  # occasional gap
                    continue

                x = base_x + self.np_random.uniform(-0.05, 0.05)
                y = (base_y
                     + sign * (row_spacing / 2.0)
                     + self.np_random.uniform(-0.02, 0.02))

                mat = self.np_random.choice(
                    ["crop_mat", "crop_ripe_mat"], p=[0.7, 0.3]
                )
                body = ET.Element(
                    "body", name=f"crop_{row_idx}_{i}", pos=f"{x} {y} 0"
                )

                if self.np_random.random() < 0.5:
                    r = self.np_random.uniform(0.04, 0.08)
                    h = self.np_random.uniform(0.15, 0.35)
                    ET.SubElement(body, "geom", type="capsule",
                                 size=f"{r} {h}", pos=f"0 0 {h}",
                                 material=mat, contype="1", conaffinity="1")
                else:
                    tr = self.np_random.uniform(0.02, 0.04)
                    th = self.np_random.uniform(0.05, 0.15)
                    cr = self.np_random.uniform(0.08, 0.12)
                    ET.SubElement(body, "geom", type="cylinder",
                                 size=f"{tr} {th}", pos=f"0 0 {th}",
                                 material="trunk_mat",
                                 contype="1", conaffinity="1")
                    ET.SubElement(body, "geom", type="sphere",
                                 size=f"{cr}",
                                 pos=f"0 0 {th*2+cr-0.05}",
                                 material=mat,
                                 contype="1", conaffinity="1")
                wb.append(body)

    def _spawn_background(self, wb, corridor_len):
        """Decorative weeds & trees outside the corridor."""
        for i in range(30):
            x = self.np_random.uniform(0, corridor_len)
            y = self.np_random.uniform(-4, 4)
            b = ET.Element("body", name=f"weed_{i}", pos=f"{x} {y} 0")
            ET.SubElement(b, "geom", type="sphere", size="0.08",
                         pos="0 0 0.04", material="weed_mat",
                         contype="0", conaffinity="0")
            wb.append(b)

        for i in range(15):
            x = self.np_random.uniform(-2, corridor_len + 5)
            y = (self.np_random.choice([-1, 1])
                 * self.np_random.uniform(2.5, 8.0))
            b = ET.Element("body", name=f"tree_{i}", pos=f"{x} {y} 0")
            tr = self.np_random.uniform(0.08, 0.15)
            th = self.np_random.uniform(0.4, 0.7)
            cr = self.np_random.uniform(0.5, 1.0)
            ET.SubElement(b, "geom", type="cylinder",
                         size=f"{tr} {th}", pos=f"0 0 {th}",
                         material="trunk_mat",
                         contype="1", conaffinity="1")
            ET.SubElement(b, "geom", type="sphere",
                         size=f"{cr}",
                         pos=f"0 0 {th*2+cr-0.2}",
                         material="canopy_mat",
                         contype="1", conaffinity="1")
            wb.append(b)

    def _build_xml(self):
        """Parse base XML, add procedural crops, return XML string."""
        tree = ET.parse(self.base_xml_path)
        root = tree.getroot()
        wb = root.find("worldbody")

        # Ground heightfield
        ET.SubElement(wb, "geom", name="ground", type="hfield",
                     hfield="terrain", material="grass_mat", pos="0 0 0")

        row_spacing = self.np_random.uniform(0.32, 0.42)
        plant_spacing = self.np_random.uniform(0.1, 0.2)
        n_plants = self.np_random.randint(60, 100)
        corridor_len = n_plants * plant_spacing

        self._spawn_crops(wb, row_spacing, n_plants, plant_spacing)
        self._spawn_background(wb, corridor_len)

        return ET.tostring(root, encoding="unicode")

    # ──────────────────────────────────────────────────────────
    #  Gymnasium API
    # ──────────────────────────────────────────────────────────
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.RandomState(seed)

        xml = self._build_xml()
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)

        # Populate heightfield
        if self.model.nhfield > 0:
            nr, nc = self.model.hfield_nrow[0], self.model.hfield_ncol[0]
            x = np.linspace(0, 10, nc)
            y = np.linspace(0, 10, nr)
            X, Y = np.meshgrid(x, y)
            Z = 0.5*np.sin(X*2.5)*np.cos(Y*2.5) + 0.2*self.np_random.normal(size=(nr, nc))
            Z -= Z.min()
            Z = Z / (Z.max() + 1e-6) * 0.15
            self.model.hfield_data[:] = Z.flatten()

        self.step_count = 0

        # Place rover at start of the curve
        y_start = np.sin(0 * self.wave_freq) * self.wave_amp
        slope = self.wave_freq * self.wave_amp * np.cos(0 * self.wave_freq)
        yaw = np.arctan(slope)

        self.data.qpos[0] = 0.0        # x
        self.data.qpos[1] = y_start    # y
        self.data.qpos[2] = 0.05       # z
        self.data.qpos[3] = np.cos(yaw / 2.0)  # qw
        self.data.qpos[6] = np.sin(yaw / 2.0)  # qz

        self.data.qvel[0] = 0.3  # small initial forward nudge
        mujoco.mj_forward(self.model, self.data)

        return self._get_obs(), {}

    def _get_obs(self):
        """Return 7D observation: [rf_front, rf_left, rf_right, vx, vy, sin_yaw, cos_yaw]"""
        obs = np.zeros(7, dtype=np.float32)

        # Rangefinders (normalised to [0, 1], 0=touching, 1=max range)
        for i, name in enumerate(["rf_front", "rf_left", "rf_right"]):
            sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, name)
            val = self.data.sensordata[self.model.sensor_adr[sid]]
            if val < 0:
                val = self.SENSOR_CUTOFF
            obs[i] = np.clip(val / self.SENSOR_CUTOFF, 0.0, 1.0)

        # Velocity (normalised, clamp to [-1, 1])
        obs[3] = np.clip(self.data.qvel[0] / 2.0, -1.0, 1.0)  # vx
        obs[4] = np.clip(self.data.qvel[1] / 2.0, -1.0, 1.0)  # vy

        # Heading as sin/cos of yaw (extracted from quaternion)
        qw, qx, qy, qz = self.data.qpos[3:7]
        yaw = np.arctan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz))
        obs[5] = np.sin(yaw)
        obs[6] = np.cos(yaw)

        return obs

    def step(self, action):
        self.step_count += 1
        action = int(action)
        left_mult, right_mult = self.ACTION_TABLE[action]

        # Apply static velocity to motors
        left_ctrl = left_mult * self.MOTOR_CTRL
        right_ctrl = right_mult * self.MOTOR_CTRL
        for i in range(min(len(self.data.ctrl), 4)):
            self.data.ctrl[i] = left_ctrl if i < 2 else right_ctrl

        # Record position before stepping
        pos_sid = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SENSOR, "rover_pos"
        )
        pos_adr = self.model.sensor_adr[pos_sid]
        prev_x = self.data.sensordata[pos_adr]

        # Simulate
        for _ in range(self.SIM_SUBSTEPS):
            mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        curr_x = self.data.sensordata[pos_adr]

        # ── Reward (scaled for SAC stability: ~[-5, 5] per step) ──
        reward = 0.0

        # 1. Alignment: dot(velocity, track_tangent)
        track_angle = np.arctan(
            self.wave_freq * self.wave_amp
            * np.cos(curr_x * self.wave_freq)
        )
        tangent = np.array([np.cos(track_angle), np.sin(track_angle)])
        vel = self.data.qvel[0:2]
        reward += np.clip(np.dot(vel, tangent) * 2.0, -3.0, 3.0)

        # 2. Spinning penalty
        yaw_rate = abs(self.data.qvel[5])
        reward -= min(yaw_rate * 1.0, 2.0)

        # 3. Wall proximity penalty (continuous)
        rf_front = obs[0]  # normalised [0,1]
        if rf_front < 0.15:
            reward -= 3.0 * (1.0 - rf_front / 0.15)

        # 4. Off-track penalty (after initial phase)
        if self.step_count > 50:
            rover_bid = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, "rover"
            )
            curr_y = self.data.xpos[rover_bid][1]
            ideal_y = np.sin(curr_x * self.wave_freq) * self.wave_amp
            deviation = abs(curr_y - ideal_y)
            reward -= min(deviation * 5.0, 3.0)

        # ── Termination ──
        terminated = False
        truncated = False

        # Win
        if curr_x > 12.0:
            terminated = True
            reward += 10.0

        # Collision (front sensor very close)
        if rf_front < 0.02:
            terminated = True
            reward -= 10.0

        # Max steps
        if self.step_count >= self.MAX_STEPS:
            truncated = True

        info = {
            "x_progress": float(curr_x),
            "step_reward": float(reward),
        }
        return obs, float(reward), terminated, truncated, info

    def render(self):
        pass
