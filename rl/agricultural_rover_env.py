import os
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
import gymnasium as gym
import torch
from gymnasium import spaces
from ultralytics import FastSAM

class AgriculturalRoverEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 20
    }

    def __init__(self, render_mode=None, base_xml_path="agricultural_tank_base.xml", seed=None):
        self.render_mode = render_mode
        self.base_xml_path = base_xml_path
        
        if not os.path.exists(self.base_xml_path):
            raise FileNotFoundError(f"Base XML file not found at {self.base_xml_path}")
            
        self.model = None
        self.data = None
        
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0.0, high=5.0, shape=(5,), dtype=np.float32) # Expanded for Vision (X, Y)
        
        # Determine device (MPS for Mac Silicon, or CPU)
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"Vision running on device: {self.device}")
        
        # Load FastSAM Heavyweight Vision AI once
        self.vision_model = FastSAM("FastSAM-s.pt").to(self.device)
        self.renderer = None
        
        self.reset(seed=seed)

    def spawn_crops(self, worldbody_node, row_base_spacing, num_plants, plant_spacing):
        wave_freq = self.np_random.uniform(0.1, 0.4)
        wave_amp = self.np_random.uniform(0.1, 0.3)
        
        for row_idx, row_sign in enumerate([-1, 1]):
            x_start = 0.5
            for i in range(num_plants):
                if self.np_random.random() < 0.1:
                    continue
                    
                x_pos = x_start + i * plant_spacing + self.np_random.uniform(-0.05, 0.05)
                y_pos = row_sign * (row_base_spacing / 2.0) + np.sin(x_pos * wave_freq) * wave_amp + self.np_random.uniform(-0.05, 0.05)
                
                mat = self.np_random.choice(["crop_mat", "crop_ripe_mat"], p=[0.7, 0.3])
                
                plant_body = ET.Element("body", name=f"crop_{row_idx}_{i}", pos=f"{x_pos} {y_pos} 0")
                plant_type = self.np_random.choice(["capsule", "multi"])
                
                if plant_type == "capsule":
                    radius = self.np_random.uniform(0.04, 0.08)
                    height = self.np_random.uniform(0.15, 0.35)
                    ET.SubElement(plant_body, "geom", type="capsule", size=f"{radius} {height}", 
                                  pos=f"0 0 {height}", material=mat, contype="1", conaffinity="1")
                else:
                    trunk_r = self.np_random.uniform(0.02, 0.04)
                    trunk_h = self.np_random.uniform(0.05, 0.15)
                    canopy_r = self.np_random.uniform(0.08, 0.12)
                    ET.SubElement(plant_body, "geom", type="cylinder", size=f"{trunk_r} {trunk_h}", 
                                  pos=f"0 0 {trunk_h}", material="trunk_mat", contype="1", conaffinity="1")
                    ET.SubElement(plant_body, "geom", type="sphere", size=f"{canopy_r}", 
                                  pos=f"0 0 {trunk_h * 2 + canopy_r - 0.05}", material=mat, contype="1", conaffinity="1")
                                  
                worldbody_node.append(plant_body)

    def spawn_obstacles(self, worldbody_node, num_obstacles, row_spacing, max_x):
        for i in range(num_obstacles):
            x_pos = self.np_random.uniform(1.0, max_x - 1.0)
            y_pos = self.np_random.uniform(-row_spacing/2.0 + 0.15, row_spacing/2.0 - 0.15)
            
            obs_type = self.np_random.choice(["rock", "bush", "post"])
            obs_body = ET.Element("body", name=f"obstacle_{i}", pos=f"{x_pos} {y_pos} 0")
            
            if obs_type == "rock":
                for j in range(self.np_random.integers(1, 4)):
                    r = self.np_random.uniform(0.05, 0.12)
                    ox = self.np_random.uniform(-0.1, 0.1)
                    oy = self.np_random.uniform(-0.1, 0.1)
                    ET.SubElement(obs_body, "geom", type="sphere", size=f"{r}", 
                                  pos=f"{ox} {oy} {r}", material="rock_mat", contype="1", conaffinity="1")
            elif obs_type == "post":
                radius = self.np_random.uniform(0.03, 0.06)
                height = self.np_random.uniform(0.2, 0.3)
                angle = self.np_random.uniform(0, 180)
                ET.SubElement(obs_body, "geom", type="cylinder", size=f"{radius} {height}", 
                              pos=f"0 0 {radius}", euler=f"90 {angle} 0", material="fence_mat", contype="1", conaffinity="1")
            else:
                radius = self.np_random.uniform(0.1, 0.2)
                height = self.np_random.uniform(0.1, 0.3)
                ET.SubElement(obs_body, "geom", type="ellipsoid", size=f"{radius} {radius} {height}",
                              pos=f"0 0 {height}", material="weed_mat", contype="1", conaffinity="1")
            
            worldbody_node.append(obs_body)

    def spawn_background(self, worldbody_node, corridor_len):
        for i in range(40):
            x = self.np_random.uniform(0, corridor_len)
            y = self.np_random.uniform(-4, 4)
            w = ET.Element("body", name=f"weed_{i}", pos=f"{x} {y} 0")
            ET.SubElement(w, "geom", type="sphere", size="0.08", pos="0 0 0.04", material="weed_mat", contype="0", conaffinity="0")
            worldbody_node.append(w)
            
        for i in range(20):
            x = self.np_random.uniform(-2, corridor_len + 5)
            y = self.np_random.choice([-1, 1]) * self.np_random.uniform(2.5, 8.0)
            
            tree = ET.Element("body", name=f"tree_{i}", pos=f"{x} {y} 0")
            trunk_r = self.np_random.uniform(0.08, 0.15)
            trunk_h = self.np_random.uniform(0.4, 0.7)
            canopy_r = self.np_random.uniform(0.5, 1.0)
            
            ET.SubElement(tree, "geom", type="cylinder", size=f"{trunk_r} {trunk_h}", pos=f"0 0 {trunk_h}", material="trunk_mat", contype="1", conaffinity="1")
            ET.SubElement(tree, "geom", type="sphere", size=f"{canopy_r}", pos=f"0 0 {trunk_h * 2 + canopy_r - 0.2}", material="canopy_mat", contype="1", conaffinity="1")
            worldbody_node.append(tree)

    def generate_environment_xml(self):
        tree = ET.parse(self.base_xml_path)
        root = tree.getroot()
        worldbody = root.find("worldbody")
        
        ground = ET.Element("geom", name="ground", type="hfield", hfield="terrain", material="grass_mat", pos="0 0 0")
        worldbody.append(ground)
        
        row_spacing = self.np_random.uniform(1.3, 1.8)
        plant_spacing = self.np_random.uniform(0.3, 0.6)
        num_plants_per_row = self.np_random.integers(20, 30)
        num_obstacles = self.np_random.integers(3, 8)
        
        max_x = num_plants_per_row * plant_spacing
        
        self.spawn_crops(worldbody, row_spacing, num_plants_per_row, plant_spacing)
        self.spawn_obstacles(worldbody, num_obstacles, row_spacing, max_x)
        self.spawn_background(worldbody, max_x)
            
        return ET.tostring(root, encoding="unicode")

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.RandomState(seed)
            
        xml_string = self.generate_environment_xml()
        
        self.model = mujoco.MjModel.from_xml_string(xml_string)
        
        if self.renderer is not None:
            self.renderer.close()
        self.renderer = mujoco.Renderer(self.model, 256, 256)
        
        if self.model.nhfield > 0:
            nrow = self.model.hfield_nrow[0]
            ncol = self.model.hfield_ncol[0]
            
            x = np.linspace(0, 10, ncol)
            y = np.linspace(0, 10, nrow)
            X, Y = np.meshgrid(x, y)
            
            Z = 0.5 * np.sin(X * 2.5) * np.cos(Y * 2.5) + 0.2 * self.np_random.normal(size=(nrow, ncol))
            Z = Z - np.min(Z)
            Z = Z / (np.max(Z) + 1e-6)
            Z = Z * 0.15 
            
            self.model.hfield_data[:] = Z.flatten()
        
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        
        obs = self._get_obs()
        info = {}
        return obs, info
        
    def _get_obs(self):
        obs = []
        for sensor_name in ["rf_front", "rf_left", "rf_right"]:
            sensor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
            adr = self.model.sensor_adr[sensor_id]
            val = self.data.sensordata[adr]
            if val < 0:
                val = 5.0
            obs.append(val)
            
        # FastSAM Vision integration
        self.renderer.update_scene(self.data, camera="rgb_cam")
        frame = self.renderer.render()
        
        # Resize to 256 for YOLO/MPS stability
        results = self.vision_model(frame, verbose=False, imgsz=256, device=self.device)
        
        vision_x = 2.5 # Default centered (mapped to 0-5 scale)
        vision_y = 2.5
        
        if results[0].masks is not None:
            masks = results[0].masks.data.cpu().numpy()
            if len(masks) > 0:
                # Find largest mask
                largest_mask = masks[np.argmax([m.sum() for m in masks])]
                ys, xs = np.nonzero(largest_mask)
                if len(xs) > 0 and len(ys) > 0:
                    vision_x = (np.mean(xs) / 256.0) * 5.0
                    vision_y = (np.mean(ys) / 256.0) * 5.0
                    
        obs.append(vision_x)
        obs.append(vision_y)
        
        return np.array(obs, dtype=np.float32)
        
    def step(self, action):
        left_vel = action[0] * 5.0
        right_vel = action[1] * 5.0
        
        # If using our tank car's motors, the control input is essentially torque.
        # We allow the env to map these properly.
        # In this env, ctrl[0:4] corresponds to the 4 wheel motors.
        for i in range(min(len(self.data.ctrl), 4)):
            self.data.ctrl[i] = action[i % 2] * 5.0
        
        robot_pos_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "rover_pos")
        pos_adr = self.model.sensor_adr[robot_pos_id]
        prev_x = self.data.sensordata[pos_adr]
        
        for _ in range(10):
            mujoco.mj_step(self.model, self.data)
            
        obs = self._get_obs()
        curr_x = self.data.sensordata[pos_adr]
        
        # 1. Forward Progress Reward
        progress = curr_x - prev_x
        reward = progress * 100.0  # Massive multiplier to force driving!
        
        # 2. Time Penalty
        reward -= 0.05  # Bleed points for staying still to break 7-step plateau
        
        # 3. Collision / Obstacle Penalty
        # Penalty increases as we get very close to obstacles
        if obs[0] < 0.25:
            reward -= 2.0 * (1.0 - obs[0]/0.25)
            
        # 4. Off-track / Lateral Penalty
        # Encourage staying near the center (Y=0)
        rover_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "rover")
        curr_y = self.data.xpos[rover_body_id][1]
        reward -= abs(curr_y) * 0.5
            
        terminated = False
        truncated = False
        
        # Win condition
        if curr_x > 12.0:
            terminated = True
            reward += 100.0
            
        # Fail condition (Collision or stuck)
        if obs[0] < 0.1:
            terminated = True
            reward -= 50.0
            
        if progress < 0.0001 and obs[0] > 1.0: # Times out if just sitting still
             truncated = True
        
        info = {
            "x_progress": float(curr_x),
            "step_reward": float(reward)
        }
            
        return obs, float(reward), terminated, truncated, info

    def render(self):
        pass
