import os
import cv2
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces

# Spawn constants — safe clear zone before the maze
SPAWN_X = 0.3
WHEAT_START_X = 1.2   # wheat starts here, giving rover 0.9m clear runway
BACK_FENCE_X  = 0.05  # wall behind spawn


class WheatFarmEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}

    # ── HSV ranges (used for OpenCV feature extraction) ─────────
    # Golden wheat / green crop
    CROP_LOWER  = np.array([15,  60,  60],  dtype=np.uint8)
    CROP_UPPER  = np.array([90, 255, 255],  dtype=np.uint8)
    # Brown soil / floor
    FLOOR_LOWER = np.array([5,  20,  30],  dtype=np.uint8)
    FLOOR_UPPER = np.array([30, 160, 180], dtype=np.uint8)
    # Dark obstacles (trunks, fences)
    OBS_LOWER   = np.array([0,   0,  0],   dtype=np.uint8)
    OBS_UPPER   = np.array([180, 60, 80],  dtype=np.uint8)

    def __init__(self, render_mode=None, base_xml_path="agricultural_tank_base.xml", seed=None, use_vision=True):

        self.render_mode = render_mode
        self.base_xml_path = base_xml_path
        self.use_vision = use_vision  # if False, vision features use geometry shortcut

        if not os.path.exists(self.base_xml_path):
            raise FileNotFoundError(f"Base XML not found: {self.base_xml_path}")

        self.model = None
        self.data  = None
        self.renderer = None
        self.np_random = np.random.RandomState(seed)

        self.action_space = spaces.Discrete(9)

        # 7-D observation vector (all normalized to [0, 1]):
        #  0: front_ultra       — forward clearance
        #  1: left_ultra        — left clearance
        #  2: right_ultra       — right clearance
        #  3: crop_left_density — crop pixels in left camera half
        #  4: crop_right_density— crop pixels in right camera half
        #  5: lane_error        — lateral offset of floor centroid from center
        #  6: vision_obstacle   — obstacle pixel density in front-center zone
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(7,), dtype=np.float32)

        self.reset(seed=seed)

    # ── World generation ─────────────────────────────────────────
    def spawn_wheat(self, worldbody_node, num_plants_x, num_rows):
        self.f1 = self.np_random.uniform(0.5, 1.2)
        self.f2 = self.np_random.uniform(1.5, 2.5)
        self.a1 = self.np_random.uniform(1.0, 2.0)
        self.a2 = self.np_random.uniform(0.5, 1.0)

        self.row_pitch = 0.45   # Wider corridors for easier navigation
        plant_pitch    = 0.10   # Slightly more spacing along rows

        # gap map (maze effect)
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
            x = WHEAT_START_X + i * plant_pitch          # ← safe runway gap
            base_curve = np.sin(x * self.f1) * self.a1 + np.cos(x * self.f2) * self.a2

            for j in range(num_rows):
                if gap_map[i, j]:
                    continue
                y_offset = (j - num_rows // 2 + 0.5) * self.row_pitch
                y = y_offset + base_curve
                rx = x + self.np_random.uniform(-0.02, 0.02)
                ry = y + self.np_random.uniform(-0.02, 0.02)
                ET.SubElement(worldbody_node, "geom",
                              name=f"wheat_{i}_{j}", type="cylinder",
                              size="0.04 0.35", pos=f"{rx} {ry} 0.35",
                              rgba="0.85 0.7 0.2 1", material="crop_mat")

        # Side fences (left + right boundary)
        for i in range(0, num_plants_x, 2):
            x = WHEAT_START_X + i * plant_pitch
            base_curve = np.sin(x * self.f1) * self.a1 + np.cos(x * self.f2) * self.a2
            y_left  = (-num_rows // 2 - 1.0) * self.row_pitch + base_curve
            y_right = ( num_rows // 2 + 1.0) * self.row_pitch + base_curve
            ET.SubElement(worldbody_node, "geom",
                          name=f"fence_L_{i}", type="box", size="0.1 0.05 0.5",
                          pos=f"{x} {y_left} 0.5", rgba="0.6 0.4 0.2 1", material="fence_mat")
            ET.SubElement(worldbody_node, "geom",
                          name=f"fence_R_{i}", type="box", size="0.1 0.05 0.5",
                          pos=f"{x} {y_right} 0.5", rgba="0.6 0.4 0.2 1", material="fence_mat")

        # Back fence — closes the starting box
        base_curve_start = np.sin(BACK_FENCE_X * self.f1) * self.a1 + np.cos(BACK_FENCE_X * self.f2) * self.a2
        fence_half_width = (num_rows // 2 + 2.0) * self.row_pitch
        for k in range(24):
            fy = base_curve_start - fence_half_width + k * (2 * fence_half_width / 24)
            ET.SubElement(worldbody_node, "geom",
                          name=f"fence_back_{k}", type="box", size="0.05 0.15 0.5",
                          pos=f"{BACK_FENCE_X} {fy} 0.5",
                          rgba="0.6 0.4 0.2 1", material="fence_mat")

    def generate_environment_xml(self):
        tree = ET.parse(self.base_xml_path)
        root = tree.getroot()
        worldbody = root.find("worldbody")
        worldbody.append(ET.Element("geom", name="ground", type="plane",
                                    size="50 50 0.1", rgba="0.4 0.25 0.15 1"))
        self.spawn_wheat(worldbody, 250, 15)
        return ET.tostring(root, encoding="unicode")

    # ── Reset ────────────────────────────────────────────────────
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.RandomState(seed)

        xml_string = self.generate_environment_xml()
        self.model = mujoco.MjModel.from_xml_string(xml_string)

        if self.renderer is not None:
            self.renderer.close()
        # Always create renderer — OpenCV is cheap
        self.renderer = mujoco.Renderer(self.model, 256, 256)

        self.data = mujoco.MjData(self.model)
        self.step_count = 0
        self._prev_x = SPAWN_X   # for delta-x reward tracking

        self._do_spawn(SPAWN_X)

        obs = self._get_obs()
        return obs, {}

    def _do_spawn(self, spawn_x):
        """Teleport rover to spawn_x facing along the curve."""
        y_s = np.sin(spawn_x * self.f1) * self.a1 + np.cos(spawn_x * self.f2) * self.a2
        slope = self.f1 * self.a1 * np.cos(spawn_x * self.f1) - self.f2 * self.a2 * np.sin(spawn_x * self.f2)
        yaw = np.arctan(slope)

        self.data.qpos[0] = spawn_x
        self.data.qpos[1] = y_s
        self.data.qpos[2] = 0.05
        self.data.qpos[3] = np.cos(yaw / 2.0)
        self.data.qpos[4] = 0.0
        self.data.qpos[5] = 0.0
        self.data.qpos[6] = np.sin(yaw / 2.0)
        self.data.qvel[:] = 0
        self.data.qvel[0] = 2.0   # Strong kickstart → rover clearly faces forward
        mujoco.mj_forward(self.model, self.data)

    # ── Observations ─────────────────────────────────────────────
    def _get_rangefinders(self):
        vals = []
        for sname in ["rf_front", "rf_left", "rf_right"]:
            sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, sname)
            adr = self.model.sensor_adr[sid]
            v   = self.data.sensordata[adr]
            if v < 0:
                v = 5.0
            vals.append(np.clip(v / 5.0, 0.0, 1.0))   # normalise → [0, 1]
        return vals   # [front, left, right]

    def _compute_vision_features(self, frame_bgr):
        """
        Extract 4 vision features from a BGR frame using OpenCV HSV heuristics.
        Returns: [crop_left, crop_right, lane_error, obstacle_score]  all in [0, 1]
        """
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        h, w = hsv.shape[:2]

        # — Crop density (yellow/green pixels in left vs right half) ——
        crop_mask = cv2.inRange(hsv, self.CROP_LOWER, self.CROP_UPPER)
        left_half  = crop_mask[:, :w//2]
        right_half = crop_mask[:, w//2:]
        max_pix    = (h * w // 2) or 1
        crop_left  = float(left_half.sum()  / 255) / max_pix
        crop_right = float(right_half.sum() / 255) / max_pix

        # — Lane error (floor centroid in bottom strip) ———————————
        bottom = hsv[int(h*0.65):, :]
        floor_mask = cv2.inRange(bottom, self.FLOOR_LOWER, self.FLOOR_UPPER)
        cols = np.where(floor_mask > 0)[1]
        if len(cols) > 0:
            centroid_x = float(np.mean(cols)) / w          # 0=left, 1=right
            lane_error = abs(centroid_x - 0.5) * 2.0       # 0=centered, 1=edge
        else:
            lane_error = 1.0                                # no floor visible

        # — Obstacle score (dark pixels in front-centre zone) ————
        cx, cy = w // 2, h // 2
        zone   = hsv[cy-40:cy+40, cx-40:cx+40]
        obs_mask  = cv2.inRange(zone, self.OBS_LOWER, self.OBS_UPPER)
        zone_pix  = obs_mask.size or 1
        vision_obstacle = float(obs_mask.sum() / 255) / zone_pix

        return [
            np.clip(crop_left, 0, 1),
            np.clip(crop_right, 0, 1),
            np.clip(lane_error, 0, 1),
            np.clip(vision_obstacle, 0, 1),
        ]

    def _simulated_vision_features(self, curr_x, curr_y):
        """
        Geometry-based shortcut replicating the OpenCV features without rendering.
        Used only if use_vision=False (rare; default is True).
        """
        look = 1.0
        tx   = curr_x + look
        ty   = np.sin(tx * self.f1) * self.a1 + np.cos(tx * self.f2) * self.a2
        lat_err = ty - curr_y

        # Simulate crop density based on proximity to row boundaries
        # (further from centre → more crop visible on that side)
        crop_left  = np.clip(0.5 - lat_err * 0.3, 0, 1)
        crop_right = np.clip(0.5 + lat_err * 0.3, 0, 1)
        lane_error = np.clip(abs(lat_err) / (self.row_pitch * 4), 0, 1)
        vision_obs = 0.0

        return [crop_left, crop_right, lane_error, vision_obs]

    def _get_obs(self):
        rf = self._get_rangefinders()   # 3 values

        rover_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "rover")
        curr_pos = self.data.xpos[rover_id]

        if self.use_vision:
            self.renderer.update_scene(self.data, camera="rgb_cam")
            frame_rgb = self.renderer.render()
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            vis = self._compute_vision_features(frame_bgr)
        else:
            vis = self._simulated_vision_features(curr_pos[0], curr_pos[1])

        return np.array(rf + vis, dtype=np.float32)   # 7 values

    # ── Step ─────────────────────────────────────────────────────
    def step(self, action):
        self.step_count += 1
        action_idx = int(action)
        motor_actions = [
            (-1.0, -1.0), (-1.0, 0.0), (-1.0, 1.0),
            ( 0.0, -1.0), ( 0.0, 0.0), ( 0.0, 1.0),
            ( 1.0, -1.0), ( 1.0, 0.0), ( 1.0, 1.0),
        ]
        left_val, right_val = motor_actions[action_idx]
        speed_mult = 15.0
        for i in range(min(len(self.data.ctrl), 4)):
            self.data.ctrl[i] = (left_val if i < 2 else right_val) * speed_mult

        robot_pos_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "rover_pos")
        pos_adr = self.model.sensor_adr[robot_pos_id]

        for _ in range(10):
            mujoco.mj_step(self.model, self.data)

        obs     = self._get_obs()
        curr_x  = self.data.sensordata[pos_adr]
        rover_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "rover")
        curr_y  = self.data.xpos[rover_id][1]

        # ── Rewards ──────────────────────────────────────────────
        # Core: reward raw x-progress (learnable from step 1)
        delta_x = curr_x - self._prev_x
        reward  = delta_x * 100.0          # +1 per cm of forward progress

        # Mild time bleed (encourages speed without destroying random exploration)
        reward -= 0.1

        # Anti-spin (keep turns purposeful)
        yaw_rate = abs(self.data.qvel[5])
        reward -= yaw_rate * 5.0

        # Rangefinder proximity penalty (soft)
        min_rf = min(obs[0], obs[1], obs[2]) * 5.0   # un-normalise
        if min_rf < 0.2:
            reward -= 10.0 * (1.0 - min_rf / 0.2)

        base_curve_y = np.sin(curr_x * self.f1) * self.a1 + np.cos(curr_x * self.f2) * self.a2
        offset_y = curr_y - base_curve_y

        self._prev_x = curr_x              # track for next step

        terminated = False
        truncated  = False

        # Win
        if curr_x > 12.0:
            terminated = True
            reward += 500.0

        # Physical collision — respawn
        ground_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "ground")
        has_collision = False
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if c.geom1 != ground_id and c.geom2 != ground_id:
                has_collision = True
                break

        needs_respawn = False
        if has_collision or min_rf < 0.08:
            reward -= 500.0
            needs_respawn = True

        if curr_x < -0.2 or abs(offset_y) > 2.0:
            reward -= 200.0
            needs_respawn = True

        if needs_respawn and not terminated:
            self._do_spawn(SPAWN_X)
            obs = self._get_obs()



        info = {
            "x_progress": float(curr_x),
            "step_reward": float(reward),
            "collisions":  int(needs_respawn),
        }
        return obs, float(reward), terminated, truncated, info

    def render(self):
        pass
