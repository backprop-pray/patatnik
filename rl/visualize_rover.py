import os
import sys
import time
import torch
import mujoco
import numpy as np
from mujoco import viewer
from stable_baselines3 import PPO
from wheat_farm_env import WheatFarmEnv

def get_latest_model(models_dir="./models/"):
    """Finds the most recent .zip model in the models/ folder."""
    if not os.path.exists(models_dir):
        return None
    all_zips = []
    for root, _, filz in os.walk(models_dir):
        for f in filz:
            if f.endswith(".zip"):
                all_zips.append(os.path.join(root, f))
    if not all_zips:
        return None
    all_zips.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return all_zips[0].replace(".zip", "")

def visualize():
    # Parse flags
    use_vision = "--vision" in sys.argv
    auto_reload = "--reload" in sys.argv
    
    print(f"=== Rover Visualizer ===")
    print(f"  Vision AI: {'ON (FastSAM)' if use_vision else 'OFF (Simulated)'}")
    print(f"  Auto-Reload: {'ON' if auto_reload else 'OFF'}")
    print(f"  Flags: --vision (enable camera AI)  --reload (hot-swap newest model)\n")
    
    model_path = get_latest_model()
    if not model_path:
        print("No model found. Start training first!")
        return
    
    print(f"Loading: {model_path}")
    env = WheatFarmEnv(use_vision=use_vision)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PPO.load(model_path, env=env, device=device)
    
    obs, info = env.reset()
    
    # Stats tracking
    ep_count = 0
    ep_reward = 0.0
    ep_steps = 0
    best_progress = 0.0
    all_progress = []
    last_reload_check = time.time()
    loaded_path = model_path
    
    print("Launching MuJoCo Viewer. Close window to stop.\n")
    print(f"{'Ep':>4} | {'Steps':>6} | {'Reward':>10} | {'Progress':>8} | {'Best':>6} | {'Avg(10)':>8} | Model")
    print("-" * 80)
    
    with viewer.launch_passive(env.model, env.data) as viewer_window:
        while viewer_window.is_running():
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            ep_steps += 1
            
            viewer_window.sync()
            time.sleep(0.02)
            
            if terminated or truncated:
                ep_count += 1
                prog = info["x_progress"]
                all_progress.append(prog)
                if prog > best_progress:
                    best_progress = prog
                
                avg_10 = np.mean(all_progress[-10:]) if all_progress else 0
                
                print(f"{ep_count:4d} | {ep_steps:6d} | {ep_reward:10.1f} | {prog:8.2f} | {best_progress:6.2f} | {avg_10:8.2f} | {os.path.basename(loaded_path)}")
                
                ep_reward = 0.0
                ep_steps = 0
                obs, info = env.reset()
                
                # Hot-reload newest model every 30 seconds
                if auto_reload and (time.time() - last_reload_check > 30):
                    last_reload_check = time.time()
                    newest = get_latest_model()
                    if newest and newest != loaded_path:
                        print(f"\n  >> Hot-reloading: {os.path.basename(newest)}\n")
                        model = PPO.load(newest, env=env, device=device)
                        loaded_path = newest

if __name__ == "__main__":
    visualize()
