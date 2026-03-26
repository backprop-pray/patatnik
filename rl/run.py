"""
Autonomous Wheat Maze — Train + Live Visualize
Usage:
    python run.py train            # Train only (8 parallel envs, 2M steps)
    python run.py viz              # Visualize latest checkpoint
    python run.py viz --vision     # Visualize with real FastSAM camera
    python run.py viz --reload     # Auto hot-swap newest model every 30s
    python run.py live             # Train + live visualize simultaneously
"""
import os
import sys
import time
import shutil
import threading
import torch
import numpy as np
import mujoco
from mujoco import viewer
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from wheat_farm_env import WheatFarmEnv

def get_latest_model(models_dir="./models/"):
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

# ── Training ──────────────────────────────────────────────────
def train():
    device = "cpu"
    print(f"Training on device: {device}")
    env = make_vec_env(WheatFarmEnv, n_envs=8, env_kwargs={"use_vision": False})
    model = PPO(
        "MlpPolicy", env, verbose=1,
        learning_rate=1e-3, n_steps=2048, batch_size=512,
        n_epochs=10, gamma=0.99, device=device,
        tensorboard_log="./ppo_farm_logs/"
    )
    os.makedirs("./models", exist_ok=True)
    cb = CheckpointCallback(save_freq=5000, save_path="./models/", name_prefix="ppo_rover_fast")
    print("Starting training (2,000,000 steps × 8 parallel envs)...")
    model.learn(total_timesteps=2000000, callback=cb)
    model.save("./models/ppo_agricultural_rover_final")
    print("Training complete!")
    env.close()

# ── Visualization ─────────────────────────────────────────────
def visualize(use_vision=False, auto_reload=False):
    print(f"=== Rover Visualizer ===")
    print(f"  Vision: {'FastSAM' if use_vision else 'Simulated'}  |  Reload: {'ON' if auto_reload else 'OFF'}\n")

    model_path = None
    while not model_path:
        model_path = get_latest_model()
        if not model_path:
            print("Waiting for first checkpoint...")
            time.sleep(5)

    print(f"Loading: {model_path}")
    env = WheatFarmEnv(use_vision=use_vision)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PPO.load(model_path, env=env, device=device)
    obs, info = env.reset()

    ep_count, ep_reward, ep_steps, ep_collisions = 0, 0.0, 0, 0
    best_progress = 0.0
    all_progress = []
    last_reload = time.time()
    loaded_path = model_path

    print(f"\n{'Ep':>4} | {'Steps':>5} | {'Reward':>9} | {'Prog':>6} | {'Best':>6} | {'Avg10':>6} | {'Crashes':>7} | Model")
    print("-" * 85)

    with viewer.launch_passive(env.model, env.data) as vw:
        while vw.is_running():
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            ep_steps += 1
            ep_collisions += info.get("collisions", 0)
            vw.sync()
            time.sleep(0.02)

            if terminated or truncated:
                ep_count += 1
                prog = info["x_progress"]
                all_progress.append(prog)
                best_progress = max(best_progress, prog)
                avg10 = np.mean(all_progress[-10:])
                print(f"{ep_count:4d} | {ep_steps:5d} | {ep_reward:9.1f} | {prog:6.2f} | {best_progress:6.2f} | {avg10:6.2f} | {ep_collisions:7d} | {os.path.basename(loaded_path)}")
                ep_reward, ep_steps, ep_collisions = 0.0, 0, 0
                obs, info = env.reset()

                if auto_reload and (time.time() - last_reload > 30):
                    last_reload = time.time()
                    newest = get_latest_model()
                    if newest and newest != loaded_path:
                        print(f"\n  >> Hot-reload: {os.path.basename(newest)}\n")
                        model = PPO.load(newest, env=env, device=device)
                        loaded_path = newest

# ── Live Mode (Train + Visualize) ─────────────────────────────
def live():
    print("=== LIVE MODE: Training + Visualization ===\n")
    # Start training in background thread
    train_thread = threading.Thread(target=train, daemon=True)
    train_thread.start()
    # Give training a head start to create first checkpoint
    print("Training started in background. Waiting for first checkpoint...\n")
    time.sleep(15)
    # Launch visualizer with auto-reload
    visualize(use_vision=False, auto_reload=True)

# ── Entry Point ───────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("train", "viz", "live"):
        print(__doc__)
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == "train":
        if os.path.exists("./models"):
            shutil.rmtree("./models")
        train()
    elif cmd == "viz":
        visualize(
            use_vision="--vision" in sys.argv,
            auto_reload="--reload" in sys.argv
        )
    elif cmd == "live":
        if os.path.exists("./models"):
            shutil.rmtree("./models")
        live()
