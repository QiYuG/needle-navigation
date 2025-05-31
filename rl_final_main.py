from stable_baselines3 import PPO
from DigitalEnv import DigitalEnv
import matplotlib.pyplot as plt
from stable_baselines3.common.monitor import Monitor
import pandas as pd
import os
import pickle

turns = 100
max_steps = 1400000
episode_length = max_steps // turns

action_save_path = "saved_actions.pkl"

# 检查是否有保存的动作序列
if os.path.exists(action_save_path):
    # 加载已保存的动作序列
    with open(action_save_path, 'rb') as f:
        all_actions = pickle.load(f)
    print(f"Loaded {len(all_actions)} saved actions")

# all_actions = []

# env = Monitor(DigitalEnv())
test_evn = Monitor(DigitalEnv())
# model = PPO("MlpPolicy", env, verbose=1)
# model.save("ppo_needle_path_planning_only_model")
# for i in range(turns):
#     model = PPO.load("ppo_needle_path_planning_only_model", env=env)
#     model.learn(total_timesteps=episode_length)
#     model.save("ppo_needle_path_planning_only_model")
#     obs, info = test_evn.reset()
#     action, _ = model.predict(obs, deterministic=True)
#     all_actions.append(action)

test_evn.reset()
for i in range(len(all_actions)): 
    action = all_actions[i]
    obs, reward, done, truncated, info = test_evn.step(action)
    test_evn.render()

# with open(action_save_path, 'wb') as f:
#         pickle.dump(all_actions, f)







