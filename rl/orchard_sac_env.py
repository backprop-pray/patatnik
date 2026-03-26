"""
OrchardSACEnv — Gymnasium environment for SAC-Discrete self-driving.
A closed fenced field with sparsely separated trees for the agent to navigate around.
"""

import os
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces


class OrchardSACEnv(gym.Env):
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

    def __init__(self, render_mode=None, base_xml_path="agricultural_tank_base.xml", seed=None):
        super().__init__()
        self.render_mode = render_mode
        self.base_xml_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), base_xml_path
        )
        if not os.path.exists(self.base_xml_path):
            raise FileNotFoundError(f"XML not found: {self.base_xml_path}")

        self.np_random = np.random.RandomState(seed)

        self.action_space = spaces.Discrete(9)
        # obs: [rf_front, rf_left, rf_right, vx, vy, sin_yaw, cos_yaw]
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(7,), dtype=np.float32
        )

        self.MOTOR_CTRL = 15.0  # slightly more speed for wide orchard
        self.MAX_STEPS = 1000
        self.SENSOR_CUTOFF = 5.0
        self.SIM_SUBSTEPS = 20
        self.SAFETY_THRESHOLD = 0.22 
        self.enable_override = False
        self.last_safeguard_used = False

        self.model = None
        self.data = None
        self.step_count = 0
        self._prev_x = 0.0

        self.reset(seed=seed)

    def _build_xml(self):
        tree = ET.parse(self.base_xml_path)
        root = tree.getroot()
        wb = root.find("worldbody")

        # Floor
        ET.SubElement(wb, "geom", name="ground", type="plane",
                      size="25 25 0.1", rgba="0.3 0.45 0.2 1", material="grass_mat", pos="0 0 0")

        field_size = 10.0 # From -10 to +10

        # Boundary Fences
        for x in np.linspace(-field_size, field_size, 20):
            # Top and bottom fences
            ET.SubElement(wb, "geom", name=f"fence_T_{x}", type="box", size="0.5 0.1 0.5",
                          pos=f"{x} {field_size} 0.5", material="fence_mat")
            ET.SubElement(wb, "geom", name=f"fence_B_{x}", type="box", size="0.5 0.1 0.5",
                          pos=f"{x} {-field_size} 0.5", material="fence_mat")
        for y in np.linspace(-field_size, field_size, 20):
            # Left and right fences
            ET.SubElement(wb, "geom", name=f"fence_L_{y}", type="box", size="0.1 0.5 0.5",
                          pos=f"{-field_size} {y} 0.5", material="fence_mat")
            ET.SubElement(wb, "geom", name=f"fence_R_{y}", type="box", size="0.1 0.5 0.5",
                          pos=f"{field_size} {y} 0.5", material="fence_mat")

        # Dense Trees Randomly Scattered
        num_trees = self.np_random.randint(80, 120)
        for i in range(num_trees):
            # Keep a clear spawn area at X in [-9.5, -7.0] near Y=0
            while True:
                x = self.np_random.uniform(-field_size + 1.0, field_size - 1.0)
                y = self.np_random.uniform(-field_size + 1.0, field_size - 1.0)
                if x < -7.0 and abs(y) < 2.0:
                    continue  # Keep spawn clear
                break

            b = ET.Element("body", name=f"tree_{i}", pos=f"{x} {y} 0")
            tr = self.np_random.uniform(0.12, 0.25)
            th = self.np_random.uniform(0.6, 1.2)
            cr = self.np_random.uniform(0.8, 1.5)
            ET.SubElement(b, "geom", type="cylinder", size=f"{tr} {th}",
                          pos=f"0 0 {th}", material="trunk_mat", contype="1", conaffinity="1")
            ET.SubElement(b, "geom", type="sphere", size=f"{cr}",
                          pos=f"0 0 {th*2+cr-0.3}", material="canopy_mat", contype="1", conaffinity="1")
            wb.append(b)

        return ET.tostring(root, encoding="unicode")

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.RandomState(seed)

        # Only build the model once; afterwards just reset positions
        if self.model is None:
            xml = self._build_xml()
            self.model = mujoco.MjModel.from_xml_string(xml)
            self.data = mujoco.MjData(self.model)

        self.step_count = 0
        self.last_safeguard_used = False

        # Spawn rover at the far left, facing +x
        self._prev_x = -8.5
        self.data.qpos[0] = self._prev_x
        self.data.qpos[1] = 0.0
        self.data.qpos[2] = 0.05
        # facing +x: w=1, x=0, y=0, z=0
        self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        self.data.qvel[:] = 0
        self.data.qvel[0] = 2.0 # initial forward push
        mujoco.mj_forward(self.model, self.data)

        return self._get_obs(), {}

    def _get_raw_sensor_distances(self):
        """Return True distances (unnormalized) for safety checks."""
        dists = []
        for name in ["rf_front", "rf_left", "rf_right"]:
            sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, name)
            val = self.data.sensordata[self.model.sensor_adr[sid]]
            dists.append(val if val >= 0 else self.SENSOR_CUTOFF)
        return dists

    def _get_obs(self):
        obs = np.zeros(7, dtype=np.float32)

        raw_rf = self._get_raw_sensor_distances()
        obs[0] = np.clip(raw_rf[0] / self.SENSOR_CUTOFF, 0.0, 1.0)
        obs[1] = np.clip(raw_rf[1] / self.SENSOR_CUTOFF, 0.0, 1.0)
        obs[2] = np.clip(raw_rf[2] / self.SENSOR_CUTOFF, 0.0, 1.0)

        obs[3] = np.clip(self.data.qvel[0] / 2.0, -1.0, 1.0)
        obs[4] = np.clip(self.data.qvel[1] / 2.0, -1.0, 1.0)

        qw, qx, qy, qz = self.data.qpos[3:7]
        yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
        obs[5] = np.sin(yaw)
        obs[6] = np.cos(yaw)

        return obs

    def _deterministic_override(self, agent_action, rf_front, rf_left, rf_right):
        """Override RL actions if danger is imminent."""
        self.last_safeguard_used = False

        if rf_front < self.SAFETY_THRESHOLD:
            self.last_safeguard_used = True
            return 2 if rf_left > rf_right else 6

        if rf_left < self.SAFETY_THRESHOLD * 0.8:
            self.last_safeguard_used = True
            return 5

        if rf_right < self.SAFETY_THRESHOLD * 0.8:
            self.last_safeguard_used = True
            return 7

        return agent_action

    def step(self, action):
        self.step_count += 1
        action = int(action)

        # ── Deterministic Safety Override ──
        raw_rf = self._get_raw_sensor_distances()
        if getattr(self, "enable_override", False):
            safe_action = self._deterministic_override(action, *raw_rf)
        else:
            safe_action = action
            self.last_safeguard_used = False

        motor_actions = [
            (-1.0, -1.0), (-1.0, 0.0), (-1.0, 1.0),
            ( 0.0, -1.0), ( 0.0, 0.0), ( 0.0, 1.0),
            ( 1.0, -1.0), ( 1.0, 0.0), ( 1.0, 1.0),
        ]
        left_val, right_val = motor_actions[safe_action]
        speed_mult = self.MOTOR_CTRL

        for i in range(min(len(self.data.ctrl), 4)):
            self.data.ctrl[i] = (left_val if i < 2 else right_val) * speed_mult

        pos_sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "rover_pos")
        pos_adr = self.model.sensor_adr[pos_sid]
        
        for _ in range(self.SIM_SUBSTEPS):
            mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        curr_x = self.data.sensordata[pos_adr]

        # ── Reward ──
        reward = 0.0

        # Primary goal: Explore forward across the field
        delta_x = curr_x - self._prev_x
        self._prev_x = curr_x
        reward += delta_x * 5.0

        # Mild time bleed (encourages speed)
        reward -= 0.01

        # Spinning penalty
        yaw_rate = abs(self.data.qvel[5])
        reward -= yaw_rate * 0.5

        if self.last_safeguard_used:
            reward -= 2.0

        raw_rf_new = self._get_raw_sensor_distances()
        min_rf = min(raw_rf_new)

        terminated = False
        truncated = False

        # Collision (hard hit against tree or fence)
        if min_rf < 0.08:
            terminated = True
            reward -= 5.0

        # Success: Crossed the entire orchard
        if curr_x > 8.0:
            terminated = True
            reward += 10.0

        if self.step_count >= self.MAX_STEPS:
            truncated = True

        info = {
            "x_progress": float(curr_x),
            "step_reward": float(reward),
            "collisions": int(terminated)
        }
        return obs, float(reward), terminated, truncated, info

    def render(self):
        pass
