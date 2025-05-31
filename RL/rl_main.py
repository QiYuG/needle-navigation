# from stable_baselines3 import PPO
# from Evn import DigitalEnv
# import matplotlib.pyplot as plt
# from stable_baselines3.common.monitor import Monitor
# import pandas as pd

# env = Monitor(DigitalEnv())
# model = PPO("MlpPolicy", env, verbose=1)
# # model = PPO.load("ppo_needle_path_planning", env=env)

# model.learn(total_timesteps=100000)
# model.save("ppo_needle_path_planning_true_1")

# # 渲染一条轨迹
# obs, info = env.reset()
# done = False
# rewards = []
# position_biases = []
# direction_biases = []
# while not done:
#     action, _ = model.predict(obs, deterministic=True)
#     obs, reward, done, truncated, info = env.step(action)
#     env.render()
#     rewards.append(reward)
#     position_biases.append(info["position_bias"])
#     direction_biases.append(info["direction_bias"])
#     if done or truncated:
#         break
    
    
# plt.figure()
# plt.subplot(3, 1, 1)
# plt.plot(rewards, label='Reward')
# plt.legend()
# plt.subplot(3, 1, 2)
# plt.plot(position_biases, label='Position Bias')
# plt.legend()
# plt.subplot(3, 1, 3)
# plt.plot(direction_biases, label='Direction Bias')
# plt.legend()
# plt.xlabel('Step')
# plt.tight_layout()
# plt.show()

# df = pd.read_csv("monitor.csv", skiprows=1)
# plt.plot(df["l"].values, df["r"].values)  # l: episode length, r: episode reward
# plt.xlabel("Episode")
# plt.ylabel("Episode Reward")
# plt.show()


from stable_baselines3 import PPO
from Evn import DigitalEnv
import matplotlib.pyplot as plt
from stable_baselines3.common.monitor import Monitor
import pandas as pd

# 创建环境
env = Monitor(DigitalEnv())

# 创建模型
model = PPO("MlpPolicy", env, verbose=1)
# 如果你有训练好的模型，可取消以下注释加载
# model = PPO.load("ppo_needle_path_planning_segmented", env=env)

# 训练
model.learn(total_timesteps=9000000)
model.save("ppo_needle_path_planning_segmented")

# 可视化一条轨迹
obs, info = env.reset()
done = False
rewards = []
position_biases = []
direction_biases = []
while not done:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    env.render()
    rewards.append(reward)
    position_biases.append(info["position_bias"])
    direction_biases.append(info["direction_bias"])
    if done or truncated:
        break

plt.figure()
plt.subplot(3, 1, 1)
plt.plot(rewards, label='Reward')
plt.legend()
plt.subplot(3, 1, 2)
plt.plot(position_biases, label='Position Bias')
plt.legend()
plt.subplot(3, 1, 3)
plt.plot(direction_biases, label='Direction Bias')
plt.legend()
plt.xlabel('Step')
plt.tight_layout()
plt.show()

# 显示训练记录
try:
    df = pd.read_csv("monitor.csv", skiprows=1)
    plt.plot(df["l"].values, df["r"].values)
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward")
    plt.show()
except:
    print("未找到 monitor.csv 文件，跳过训练记录可视化。")
