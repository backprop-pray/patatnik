"""
FieldSACEnv — Gymnasium environment for SAC-Discrete self-driving.
Open field with randomized obstacles and deterministic safety override.
"""

import os
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces


class FieldSACEnv(gym.Env):
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

        self.MOTOR_CTRL = 10.0
        self.MAX_STEPS = 1000
        self.SENSOR_CUTOFF = 5.0
        self.SIM_SUBSTEPS = 10
        self.SAFETY_THRESHOLD = 0.22 # distance in m to trigger deterministic override
        self.enable_override = False

        self.model = None
        self.data = None
        self.step_count = 0

        self.reset(seed=seed)

    def _spawn_obstacles(self, wb, num_obstacles, max_x, max_y):
        """Randomly place rocks, bushes, posts across the field."""
        for i in range(num_obstacles):
            # Keep clear area near start (x < 1.0)
            x = self.np_random.uniform(1.0, max_x)
            y = self.np_random.uniform(-max_y, max_y)

            # 0=Rock, 1=Bush, 2=Post
            obs_type = self.np_random.choice(3, p=[0.4, 0.4, 0.2])
            b = ET.Element("body", name=f"obstacle_{i}", pos=f"{x} {y} 0")

            if obs_type == 0:  # Rock (clumps of spheres)
                for j in range(self.np_random.randint(1, 4)):
                    r = self.np_random.uniform(0.05, 0.12)
                    ox = self.np_random.uniform(-0.1, 0.1)
                    oy = self.np_random.uniform(-0.1, 0.1)
                    ET.SubElement(b, "geom", type="sphere", size=f"{r}",
                                  pos=f"{ox} {oy} {r*0.6}", material="rock_mat",
                                  contype="1", conaffinity="1")
            elif obs_type == 1:  # Bush/Weed
                r = self.np_random.uniform(0.1, 0.2)
                h = self.np_random.uniform(0.1, 0.3)
                ET.SubElement(b, "geom", type="ellipsoid", size=f"{r} {r} {h}",
                              pos=f"0 0 {h}", material="weed_mat",
                              contype="1", conaffinity="1")
            else:  # Post
                r = self.np_random.uniform(0.03, 0.05)
                h = self.np_random.uniform(0.2, 0.4)
                ang = self.np_random.uniform(0, 180)
                ET.SubElement(b, "geom", type="cylinder", size=f"{r} {h}",
                              pos=f"0 0 {r}", euler=f"90 {ang} 0", material="fence_mat",
                              contype="1", conaffinity="1")
            wb.append(b)

    def _spawn_trees(self, wb, num_trees, max_x, max_y):
        for i in range(num_trees):
            x = self.np_random.uniform(-1.0, max_x + 2.0)
            y = self.np_random.choice([-1, 1]) * self.np_random.uniform(max_y, max_y + 3.0)
            b = ET.Element("body", name=f"tree_{i}", pos=f"{x} {y} 0")
            tr = self.np_random.uniform(0.1, 0.2)
            th = self.np_random.uniform(0.5, 0.9)
            cr = self.np_random.uniform(0.6, 1.2)
            ET.SubElement(b, "geom", type="cylinder", size=f"{tr} {th}",
                          pos=f"0 0 {th}", material="trunk_mat", contype="1", conaffinity="1")
            ET.SubElement(b, "geom", type="sphere", size=f"{cr}",
                          pos=f"0 0 {th*2+cr-0.2}", material="canopy_mat", contype="1", conaffinity="1")
            wb.append(b)

    def _build_xml(self):
        tree = ET.parse(self.base_xml_path)
        root = tree.getroot()
        wb = root.find("worldbody")

        ET.SubElement(wb, "geom", name="ground", type="hfield",
                      hfield="terrain", material="grass_mat", pos="0 0 0")

        self.field_len = 15.0
        self.field_width = 5.0
        num_obstacles = self.np_random.randint(40, 60)
        num_trees = self.np_random.randint(15, 25)

        self._spawn_obstacles(wb, num_obstacles, self.field_len, self.field_width)
        self._spawn_trees(wb, num_trees, self.field_len, self.field_width)

        return ET.tostring(root, encoding="unicode")

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.RandomState(seed)

        xml = self._build_xml()
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)

        if self.model.nhfield > 0:
            nr, nc = self.model.hfield_nrow[0], self.model.hfield_ncol[0]
            x = np.linspace(0, 10, nc)
            y = np.linspace(0, 10, nr)
            X, Y = np.meshgrid(x, y)
            # Create bumpy terrain
            Z = 0.4 * np.sin(X * 3.0) * np.cos(Y * 3.0) + 0.3 * self.np_random.normal(size=(nr, nc))
            Z -= Z.min()
            Z = (Z / (Z.max() + 1e-6)) * 0.12  # lower hills than track
            self.model.hfield_data[:] = Z.flatten()

        self.step_count = 0
        self.last_safeguard_used = False

        self.data.qpos[0:3] = [0.0, 0.0, 0.05]
        self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        self.data.qvel[0] = 0.2
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
            # Blocked front: spin left or right depending on space
            if rf_left > rf_right:
                return 2  # spin left
            else:
                return 6  # spin right

        if rf_left < self.SAFETY_THRESHOLD * 0.8:
            self.last_safeguard_used = True
            return 5  # pivot right

        if rf_right < self.SAFETY_THRESHOLD * 0.8:
            self.last_safeguard_used = True
            return 7  # pivot left

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

        left_mult, right_mult = self.ACTION_TABLE[safe_action]
        left_ctrl = left_mult * self.MOTOR_CTRL
        right_ctrl = right_mult * self.MOTOR_CTRL

        for i in range(min(len(self.data.ctrl), 4)):
            self.data.ctrl[i] = left_ctrl if i < 2 else right_ctrl

        pos_sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "rover_pos")
        pos_adr = self.model.sensor_adr[pos_sid]
        prev_x = self.data.sensordata[pos_adr]
        prev_pos = self.data.xpos[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "rover")].copy()

        for _ in range(self.SIM_SUBSTEPS):
            mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        curr_x = self.data.sensordata[pos_adr]
        curr_pos = self.data.xpos[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "rover")]

        # Distance moved forward (in local or global frame)
        # We reward global forward exploration
        dist_forward = curr_x - prev_x

        # ── Reward ──
        reward = 0.0

        # Primary goal: Explore forward
        reward += np.clip(dist_forward * 100.0, -2.0, 2.0)

        # Continuous base reward for surviving and moving quickly
        speed = np.linalg.norm(self.data.qvel[0:2])
        reward += np.clip(speed * 1.5, 0.0, 1.5)

        # Spinning penalty
        yaw_rate = abs(self.data.qvel[5])
        reward -= min(yaw_rate * 0.5, 1.0)

        # Penalty if autonomous system had to step in
        # (Teaches agent to avoid triggering the safeguard)
        if self.last_safeguard_used:
            reward -= 2.0

        raw_rf_new = self._get_raw_sensor_distances()
        terminated = False
        truncated = False

        # Collision (hard hit)
        if raw_rf_new[0] < 0.02 or raw_rf_new[1] < 0.02 or raw_rf_new[2] < 0.02:
            terminated = True
            reward -= 10.0

        # Success: Crossed the field
        if curr_x > self.field_len:
            terminated = True
            reward += 10.0

        if self.step_count >= self.MAX_STEPS:
            truncated = True

        info = {
            "x_progress": float(curr_x),
            "step_reward": float(reward),
            "safeguard_used": self.last_safeguard_used,
            "actual_action": safe_action
        }
        return obs, float(reward), terminated, truncated, info

    def render(self):
        pass
