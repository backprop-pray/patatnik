import os
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces

SPAWN_X = 0.3
WHEAT_START_X = 1.2
BACK_FENCE_X  = 0.05


class WheatFarmEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 20}

    def __init__(self, base_xml_path="agricultural_tank_base.xml", seed=None):
        super().__init__()

        if not os.path.exists(base_xml_path):
            raise FileNotFoundError(base_xml_path)

        self.base_xml_path = base_xml_path
        self.np_random = np.random.RandomState(seed)

        # ✅ Continuous actions: [left_wheel, right_wheel]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # ✅ Added offset_y to observation
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32)

        self.MAX_STEPS = 1000
        self.SENSOR_CUTOFF = 5.0
        self.MOTOR_CTRL = 6.0
        self.SIM_SUBSTEPS = 10

        self.reset(seed=seed)

    # ─────────────────────────────
    # 🌾 World generation (unchanged)
    # ─────────────────────────────
    def spawn_wheat(self, worldbody_node, num_plants_x, num_rows):
        self.f1 = self.np_random.uniform(0.5, 1.2)
        self.f2 = self.np_random.uniform(1.5, 2.5)
        self.a1 = self.np_random.uniform(1.0, 2.0)
        self.a2 = self.np_random.uniform(0.5, 1.0)

        self.row_pitch = 0.45
        plant_pitch = 0.10

        gap_map = np.zeros((num_plants_x, num_rows), dtype=bool)
        for j in range(num_rows):
            i = 0
            while i < num_plants_x:
                if self.np_random.random() < 0.05:
                    gl = self.np_random.randint(10, 30)
                    gap_map[i:i+gl, j] = True
                    i += gl + 20
                else:
                    i += 1

        for i in range(num_plants_x):
            x = WHEAT_START_X + i * plant_pitch
            base_curve = np.sin(x * self.f1) * self.a1 + np.cos(x * self.f2) * self.a2

            for j in range(num_rows):
                if gap_map[i, j]:
                    continue
                y_offset = (j - num_rows // 2 + 0.5) * self.row_pitch
                y = y_offset + base_curve
                rx = x + self.np_random.uniform(-0.02, 0.02)
                ry = y + self.np_random.uniform(-0.02, 0.02)

                ET.SubElement(worldbody_node, "geom",
                              type="cylinder",
                              size="0.04 0.35",
                              pos=f"{rx} {ry} 0.35",
                              rgba="0.85 0.7 0.2 1")

    def generate_environment_xml(self):
        tree = ET.parse(self.base_xml_path)
        root = tree.getroot()
        worldbody = root.find("worldbody")

        worldbody.append(ET.Element("geom", type="plane",
                                   size="50 50 0.1",
                                   rgba="0.4 0.25 0.15 1"))

        self.spawn_wheat(worldbody, 250, 15)
        return ET.tostring(root, encoding="unicode")

    # ─────────────────────────────
    # 🔄 Reset
    # ─────────────────────────────
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        xml = self.generate_environment_xml()
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)

        self.step_count = 0
        self._prev_x = SPAWN_X

        self._spawn()

        return self._get_obs(), {}

    def _spawn(self):
        x = SPAWN_X
        y = np.sin(x * self.f1) * self.a1 + np.cos(x * self.f2) * self.a2
        slope = self.f1 * self.a1 * np.cos(x * self.f1) - self.f2 * self.a2 * np.sin(x * self.f2)
        yaw = np.arctan(slope)

        self.data.qpos[:3] = [x, y, 0.1]
        self.data.qpos[3:7] = [np.cos(yaw/2), 0, 0, np.sin(yaw/2)]
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)

    # ─────────────────────────────
    # 👁 Observations
    # ─────────────────────────────
    def _get_sensors(self):
        vals = []
        for name in ["rf_front", "rf_left", "rf_right"]:
            sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, name)
            adr = self.model.sensor_adr[sid]
            v = self.data.sensordata[adr]
            vals.append(v if v >= 0 else self.SENSOR_CUTOFF)
        return vals

    def _get_obs(self):
        obs = np.zeros(8, dtype=np.float32)

        rf = self._get_sensors()
        obs[0:3] = np.clip(np.array(rf) / self.SENSOR_CUTOFF, 0, 1)

        obs[3] = np.clip(self.data.qvel[0] / 2.0, -1, 1)
        obs[4] = np.clip(self.data.qvel[1] / 2.0, -1, 1)

        qw, qx, qy, qz = self.data.qpos[3:7]
        yaw = np.arctan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz))
        obs[5] = np.sin(yaw)
        obs[6] = np.cos(yaw)

        x, y = self.data.qpos[0], self.data.qpos[1]
        center_y = np.sin(x*self.f1)*self.a1 + np.cos(x*self.f2)*self.a2
        offset = y - center_y

        obs[7] = np.clip(offset / 2.0, -1, 1)

        return obs

    # ─────────────────────────────
    # ▶ Step
    # ─────────────────────────────
    def step(self, action):
        self.step_count += 1

        left, right = action
        self.data.ctrl[:] = np.array([left, left, right, right]) * self.MOTOR_CTRL

        for _ in range(self.SIM_SUBSTEPS):
            mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()

        x, y = self.data.qpos[0], self.data.qpos[1]
        delta_x = x - self._prev_x
        self._prev_x = x

        rf = self._get_sensors()
        min_rf = min(rf)

        center_y = np.sin(x*self.f1)*self.a1 + np.cos(x*self.f2)*self.a2
        offset = y - center_y

        # ── Reward ──
        reward = 0.0

        # forward progress
        reward += delta_x * 5.0

        # stay centered
        reward -= abs(offset) * 2.0

        # avoid obstacles (dense penalty)
        reward -= max(0, (0.25 - min_rf)) * 6.0

        # forward velocity bonus
        qw, qx, qy, qz = self.data.qpos[3:7]
        yaw = np.arctan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz))
        v_forward = self.data.qvel[0]*np.cos(yaw) + self.data.qvel[1]*np.sin(yaw)
        reward += np.clip(v_forward, -1, 2)

        # small time penalty
        reward -= 0.05

        # ── Termination ──
        terminated = False

        if min_rf < 0.08:
            reward -= 200
            terminated = True

        if abs(offset) > 3.0:
            reward -= 50
            terminated = True

        if x > 12.0:
            reward += 20
            terminated = True

        truncated = self.step_count >= self.MAX_STEPS

        return obs, float(reward), terminated, truncated, {}