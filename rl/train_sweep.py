"""
Simple continuous training: 1 env, CUDA, 2M timesteps.
Continues from latest checkpoint if available.
Prints average reward every 10,000 timesteps.
"""
import os
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from wheat_farm_env import WheatFarmEnv


class RewardPrinterCallback(BaseCallback):
    """Prints average reward every `print_freq` timesteps."""

    def __init__(self, print_freq=10000, verbose=0):
        super().__init__(verbose)
        self.print_freq = print_freq
        self.rewards = []

    def _on_step(self) -> bool:
        # Collect rewards from infos
        infos = self.locals.get("infos", [])
        for info in infos:
            if "step_reward" in info:
                self.rewards.append(info["step_reward"])

        if self.num_timesteps % self.print_freq < self.training_env.num_envs:
            if self.rewards:
                avg = np.mean(self.rewards)
                best = max(self.rewards)
                print(f"  Step {self.num_timesteps:>9,d}  |  avg_reward: {avg:8.2f}  |  best_reward: {best:8.2f}")
                self.rewards = []
        return True


def train():
    device = "cpu"
    print(f"Device: {device}")

    env = WheatFarmEnv(use_vision=False)

    # Continue from latest checkpoint if one exists
    model_dir = "./models/"
    latest = None
    if os.path.exists(model_dir):
        zips = [f for f in os.listdir(model_dir) if f.endswith(".zip")]
        if zips:
            zips.sort(key=lambda f: os.path.getmtime(os.path.join(model_dir, f)), reverse=True)
            latest = os.path.join(model_dir, zips[0]).replace(".zip", "")

    if latest:
        print(f"Continuing from: {latest}")
        model = PPO.load(latest, env=env, device=device)
        model.learning_rate = 1e-3
    else:
        print("Starting fresh training.")
        model = PPO(
            "MlpPolicy", env, verbose=0,
            learning_rate=1e-3, n_steps=2048, batch_size=64,
            n_epochs=10, gamma=0.99, device=device,
            tensorboard_log="./ppo_farm_logs/"
        )

    os.makedirs(model_dir, exist_ok=True)

    print("Training for 2,000,000 timesteps...\n")
    model.learn(
        total_timesteps=2_000_000,
        callback=RewardPrinterCallback(print_freq=10000),
        reset_num_timesteps=False,
    )

    model.save(os.path.join(model_dir, "ppo_rover_final"))
    print("\nTraining complete! Saved to ./models/ppo_rover_final")
    env.close()


if __name__ == "__main__":
    train()
