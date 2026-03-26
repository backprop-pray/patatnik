import os
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces

class MazeSACEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}

    def __init__(self, render_mode=None, base_xml_path="agricultural_tank_base.xml", seed=None):
        super().__init__()
        self.render_mode = render_mode
        self.base_xml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), base_xml_path)
        
        if not os.path.exists(self.base_xml_path):
            raise FileNotFoundError(f"XML not found: {self.base_xml_path}")

        self.np_random = np.random.RandomState(seed)

        self.action_space = spaces.Discrete(9)
        # obs: [rf_front, rf_left, rf_right, vx, vy, sin_yaw, cos_yaw]
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)

        self.MOTOR_CTRL = 8.0  # Slower speed for finer turning
        self.MAX_STEPS = 2000 # Longer survival
        self.SENSOR_CUTOFF = 5.0
        self.SIM_SUBSTEPS = 20
        self.SAFETY_THRESHOLD = 0.22 
        self.enable_override = True
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
        ET.SubElement(wb, "geom", name="ground", type="plane", size="15 15 0.1", rgba="0.3 0.3 0.3 1")

        field_size = 15.0 # From -15 to +15 (30x30m area)

        # Outer Boundary Fences (no exits)
        ET.SubElement(wb, "geom", name="wall_T", type="box", size=f"{field_size} 0.5 1.0", pos=f"0 {field_size} 1.0", rgba="0.5 0.5 0.5 1")
        ET.SubElement(wb, "geom", name="wall_B", type="box", size=f"{field_size} 0.5 1.0", pos=f"0 {-field_size} 1.0", rgba="0.5 0.5 0.5 1")
        ET.SubElement(wb, "geom", name="wall_L", type="box", size=f"0.5 {field_size} 1.0", pos=f"{-field_size} 0 1.0", rgba="0.5 0.5 0.5 1")
        ET.SubElement(wb, "geom", name="wall_R", type="box", size=f"0.5 {field_size} 1.0", pos=f"{field_size} 0 1.0", rgba="0.5 0.5 0.5 1")

        # Internal Maze Blocks scaled up 1.5x
        blocks = [
            {"size": "3.0 3.0 1.0", "pos": "0 0 1.0"},
            {"size": "4.5 1.5 1.0", "pos": "-7.5 7.5 1.0"},
            {"size": "1.5 3.0 1.0", "pos": "-10.5 6.0 1.0"},
            {"size": "1.5 6.0 1.0", "pos": "7.5 6.0 1.0"},
            {"size": "3.0 1.5 1.0", "pos": "9.0 10.5 1.0"},
            {"size": "6.0 1.5 1.0", "pos": "6.0 -9.0 1.0"},
            {"size": "1.5 3.0 1.0", "pos": "10.5 -7.5 1.0"},
            {"size": "1.5 4.5 1.0", "pos": "-9.0 -7.5 1.0"},
            {"size": "4.5 1.5 1.0", "pos": "-6.0 -10.5 1.0"},
            {"size": "3.0 0.8 1.0", "pos": "-9.0 0 1.0"},
            {"size": "3.0 0.8 1.0", "pos": "9.0 0 1.0"},
            {"size": "0.8 3.0 1.0", "pos": "0 9.0 1.0"},
            {"size": "0.8 3.0 1.0", "pos": "0 -9.0 1.0"},
        ]

        for i, b in enumerate(blocks):
            ET.SubElement(wb, "geom", name=f"maze_block_{i}", type="box", size=b["size"], pos=b["pos"], rgba="0.6 0.6 0.6 1")

        return ET.tostring(root, encoding="unicode")

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.RandomState(seed)

        if self.model is None:
            xml = self._build_xml()
            self.model = mujoco.MjModel.from_xml_string(xml)
            self.data = mujoco.MjData(self.model)

        self.step_count = 0
        self.last_safeguard_used = False

        # Spawn rover in bottom left corner facing right (+x)
        self.data.qpos[0] = -13.0
        self.data.qpos[1] = -13.0
        self.data.qpos[2] = 0.05
        # facing +x: w=0.707, x=0, y=0, z=0.707 to face roughly +x or +y depending
        self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        self.data.qvel[:] = 0
        self.data.qvel[0] = 2.0 # initial forward push
        mujoco.mj_forward(self.model, self.data)

        return self._get_obs(), {}

    def _get_raw_sensor_distances(self):
        dists = []
        for name in ["rf_front", "rf_left", "rf_right"]:
            sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, name)
            val = self.data.sensordata[self.model.sensor_adr[sid]]
            if val < 0: val = 5.0
            dists.append(val)
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

        for _ in range(self.SIM_SUBSTEPS):
            mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        
        # ── Reward for endless maze survival ──
        reward = 0.0

        # Primary goal: Continuous forward movement
        # Since it's endless, we reward local linear velocity to encourage moving
        # qvel[0:2] are world-frame X and Y velocities. We must project them into local forward direction using yaw.
        qw, qx, qy, qz = self.data.qpos[3:7]
        moving_yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
        
        v_world_x = self.data.qvel[0]
        v_world_y = self.data.qvel[1]
        # Dot product with forward vector
        v_forward = v_world_x * np.cos(moving_yaw) + v_world_y * np.sin(moving_yaw)
        
        reward += np.clip(v_forward * 3.0, -2.0, 5.0)

        # Mild time bleed to discourage standing still
        reward -= 0.1

        # Spinning penalty
        yaw_rate = abs(self.data.qvel[5])
        reward -= yaw_rate * 0.5

        if self.last_safeguard_used:
            reward -= 2.0

        raw_rf_new = self._get_raw_sensor_distances()
        min_rf = min(raw_rf_new)

        terminated = False
        truncated = False

        # Collision (hard hit against wall)
        if min_rf < 0.08:
            reward -= 500.0
            
            # Extract yaw to calculate correct world-frame components for a local backward push
            qw, qx, qy, qz = self.data.qpos[3:7]
            crash_yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
            
            # Push backward slightly, simulate bouncing off the wall
            push_speed = 3.0
            self.data.qvel[0] = -push_speed * np.cos(crash_yaw)
            self.data.qvel[1] = -push_speed * np.sin(crash_yaw)
            self.data.qvel[2:] = 0.0  # kill other momentum
            
            mujoco.mj_forward(self.model, self.data)
            # We do NOT trigger termination so the agent learns to correct from the bump

        if self.step_count >= self.MAX_STEPS:
            truncated = True

        info = {
            "x_progress": 0.0,
            "step_reward": float(reward),
            "collisions": int(terminated)
        }
        return obs, float(reward), terminated, truncated, info

    def render(self):
        pass
