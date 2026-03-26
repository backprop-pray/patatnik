import os
import mujoco
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from agricultural_rover_env import AgriculturalRoverEnv

def train():
    print("Initializing environment...")
    # Headless env for fast training
    env = AgriculturalRoverEnv(base_xml_path="agricultural_tank_base.xml")
    
    # Hyperparameters for PPO 
    # Learning rate 3e-4 is standard. 
    # Batch size 64 for small observation spaces.
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        # Use MPS for Mac Silicon, else CPU
        device="mps" if torch.backends.mps.is_available() else "cpu",
        tensorboard_log="./ppo_farm_logs/"
    )

    print("Starting training (300,000 steps)...")
    checkpoint_callback = CheckpointCallback(
        save_freq=50000, 
        save_path="./models/",
        name_prefix="ppo_rover"
    )

    model.learn(total_timesteps=300000, callback=checkpoint_callback)
    
    model_path = "ppo_agricultural_rover_final"
    model.save(model_path)
    print(f"Training complete. Model saved to {model_path}")

def evaluate():
    import time
    from mujoco import viewer
    
    print("Evaluating trained model...")
    env = AgriculturalRoverEnv(base_xml_path="agricultural_tank_base.xml")
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
