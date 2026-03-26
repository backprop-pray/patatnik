import os
import mujoco
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from wheat_farm_env import WheatFarmEnv

def train():
    # Parallel environments for speed
    n_envs = 8
    print(f"Creating {n_envs} parallel environments...")
    env = make_vec_env(
        WheatFarmEnv, 
        n_envs=n_envs, 
        env_kwargs={"use_vision": False}
    )
    
    # Hyperparameters for Hackathon speed
    # Learning rate 1e-3 for faster initial climb. 
    # Batch size 256 for stability with parallel envs.
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        learning_rate=1e-3,
        n_steps=2048,
        batch_size=512,
        n_epochs=10,
        gamma=0.99,
        device="cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"),
        tensorboard_log="./ppo_farm_logs/PPO_wheat"
    )

    print("Starting training (2,000,000 steps)...")
    checkpoint_callback = CheckpointCallback(
        save_freq=5000, 
        save_path="./models/",
        name_prefix="ppo_rover_fast"
    )

    model.learn(total_timesteps=2000000, callback=checkpoint_callback)
    
    model_path = "ppo_agricultural_rover_final"
    model.save(model_path)
    print(f"Training complete. Model saved to {model_path}")

def evaluate():
    import time
    from mujoco import viewer
    
    print("Evaluating trained model...")
    # Must use same vision setting as training!
    env = WheatFarmEnv(use_vision=False)
    model = PPO.load("ppo_agricultural_rover_final")
    
    obs, info = env.reset()
    
    with viewer.launch_passive(env.model, env.data) as viewer_window:
        while viewer_window.is_running():
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            viewer_window.sync()
            time.sleep(0.01)
            
            if terminated or truncated:
                print(f"Episode Done. Progress: {info['x_progress']:.2f}")
                obs, info = env.reset()

if __name__ == "__main__":
    # If model doesn't exist, train it. Otherwise, evaluate.
    if not os.path.exists("ppo_agricultural_rover_final.zip"):
        train()
    
    evaluate()
