import os
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from wheat_farm import WheatFarmEnv

if __name__ == "__main__":
    env = WheatFarmEnv()
    env = Monitor(env)

    # 🚀 SUPER FAST SAC CONFIG 🚀
    # Instead of backpropagating every single step (which is very slow), 
    # we gather 64 steps, then do 64 PyTorch updates back-to-back on the GPU. 
    # Large batch size (512) and tuned learning_rate massively accelerates convergence.

    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=1e-3,
        buffer_size=200000,
        batch_size=512,
        gamma=0.99,
        tau=0.01,
        train_freq=64,
        gradient_steps=64,
        ent_coef="auto",
        device="cuda",
        verbose=1
    )

    checkpoint = CheckpointCallback(save_freq=10000, save_path="./sac_models/", name_prefix="sac_wheat_fast")

    print("\n🚀 LAUNCHING ACCELERATED SAC TRAINING 🚀")
    model.learn(total_timesteps=1_000_000, log_interval=4, callback=checkpoint)