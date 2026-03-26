import time
import os
import glob
import cv2
import mujoco
import numpy as np
from stable_baselines3 import SAC
from wheat_farm import WheatFarmEnv

print("Looking for latest trained model...")
checkpoints = glob.glob("./sac_models/sac_wheat_fast_*.zip")

if not checkpoints:
    print("No checkpoints found! Let training run to at least 10,000 steps.")
    exit(1)

latest_checkpoint = max(checkpoints, key=os.path.getctime)
print(f"Loading {latest_checkpoint}...")

env = WheatFarmEnv()
model = SAC.load(latest_checkpoint, env=env)

obs, _ = env.reset()

print("\n--- 🌾 ROVER DEPLOYED IN WHEAT FIELD 🌾 ---\n")
print("Opening simulation rendering window...")

renderer = mujoco.Renderer(env.model, 480, 640)

step = 0
total_reward = 0
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    step += 1
    
    # ── RENDERING ──
    # Render from the camera attached to the rover "rgb_cam" or default free camera
    renderer.update_scene(env.data, camera="rgb_cam")
    img = renderer.render()
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # Overlay telemetry directly onto the video feed
    cv2.putText(img_bgr, f"Step: {step}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(img_bgr, f"Distance: {env.data.qpos[0]:.2f}m", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # ── RECORDING VIDEO ──
    if step == 1:
        # Initialize video writer on the first step
        height, width = img_bgr.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_out = cv2.VideoWriter("test_run.mp4", fourcc, 30.0, (width, height))
        
    video_out.write(img_bgr)

    if terminated or truncated:
        print(f"\n--- 🛑 RUN FINISHED at Step {step} 🛑 ---")
        print(f"Final Distance: {env.data.qpos[0]:.2f}m | Final Reward: {total_reward:.1f}\n")
        break

if 'video_out' in locals():
    video_out.release()
    print("🎬 Video saved to test_run.mp4 !")

