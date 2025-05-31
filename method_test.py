from stable_baselines3 import PPO
from DigitalEnv import DigitalEnv
import matplotlib.pyplot as plt
from stable_baselines3.common.monitor import Monitor
import pandas as pd
import numpy as np

env = Monitor(DigitalEnv())
model_50000 = PPO.load("ppo_needle_path_planning_true_50000", env=env)
model_100000 = PPO.load("ppo_needle_path_planning_true_100000", env=env)
model_150000 = PPO.load("ppo_needle_path_planning_true_150000", env=env)
model_200000 = PPO.load("ppo_needle_path_planning_true_200000", env=env)
model_250000 = PPO.load("ppo_needle_path_planning_true_250000", env=env)
model_300000 = PPO.load("ppo_needle_path_planning_true_300000", env=env)
model_350000 = PPO.load("ppo_needle_path_planning_true_350000", env=env)
model_400000 = PPO.load("ppo_needle_path_planning_true_400000", env=env)
model_450000 = PPO.load("ppo_needle_path_planning_true_450000", env=env)
model_500000 = PPO.load("ppo_needle_path_planning_true_500000", env=env)
model_550000 = PPO.load("ppo_needle_path_planning_true_550000", env=env)
model_600000 = PPO.load("ppo_needle_path_planning_true_600000", env=env)
model_650000 = PPO.load("ppo_needle_path_planning_true_650000", env=env)
model_700000 = PPO.load("ppo_needle_path_planning_true_700000", env=env)
model_750000 = PPO.load("ppo_needle_path_planning_true_750000", env=env)
model_800000 = PPO.load("ppo_needle_path_planning_true_800000", env=env)
model_850000 = PPO.load("ppo_needle_path_planning_true_850000", env=env)
model_900000 = PPO.load("ppo_needle_path_planning_true_900000", env=env)
model_list = [model_50000, model_100000, model_150000, model_200000, model_250000, model_300000, model_350000, model_400000, model_450000, model_500000, model_550000, model_600000, model_650000, model_700000, model_750000, model_800000, model_850000, model_900000]
step_list = [50000, 100000, 150000, 200000, 250000, 300000, 350000, 400000, 450000, 500000, 550000, 600000, 650000, 700000, 750000, 800000, 850000, 900000]
# 加载模型

all_actions = []

for model in model_list:
    obs, info = env.reset()
    action, _ = model.predict(obs, deterministic=True)
    all_actions.append(action)

all_actions = np.array(all_actions)
print(all_actions)  # (num_models, num_steps, action_dim)
dz_list = all_actions[:,0]  # 假设第一个动作是dz
theta_x_list = all_actions[:,1]  # 假设第二个动作是theta_x
theta_y_list = all_actions[:,2]  # 假设第三个动作是theta_y
theta_z_list = all_actions[:,3]  # 假设第四个动作是theta_z
print(dz_list, theta_x_list, theta_y_list, theta_z_list)

plt.figure(figsize=(10, 8))
plt.subplot(4, 1, 1)
plt.plot(step_list, dz_list, marker='o')
plt.ylabel('dz1')
plt.subplot(4, 1, 2)
plt.plot(step_list, theta_x_list, marker='o')
plt.ylabel('theta_x')
plt.subplot(4, 1, 3)
plt.plot(step_list, theta_y_list, marker='o')
plt.ylabel('theta_y')
plt.subplot(4, 1, 4)
plt.plot(step_list, theta_z_list, marker='o')
plt.ylabel('theta_z')
plt.xlabel('Step')
plt.tight_layout()
plt.show()

