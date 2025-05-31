# import gymnasium as gym
# from gymnasium import spaces
# import torch
# import matplotlib.pyplot as plt
# import numpy as np

# from stl import mesh
# from mpl_toolkits.mplot3d.art3d import Poly3DCollection
# import utility
# from Needle import Needle

# '''
# 强化学习思路：
# 沿袭人工势场法
# 观测：[位置距离，方向差距，相对位置，最小距离]
# [torch.norm(end_point - destination), 
# torch.norm(end_direction - normal_vector), 
# utility.calculate_loss(obstacle_points, env.needle.catheter_points), 
# utility.calculate_min_dis(obstacle_points, env.needle.catheter_points)]
# 奖励：就是势场函数值
# 动作：[dz1, theta_x, theta_y, theta_z]
# '''


# class DigitalEnv(gym.Env):
#     metadata = {"render_modes": ["human"]}

#     def __init__(self, 
#                  filename_heart='./HeartModel/Heart_sim.STL',
#                  filename_valve='./HeartModel/Valve_sim.STL',
#                  device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
#         super(DigitalEnv, self).__init__()
#         self.device = device
#         # 心脏模型处理
#         stl_mesh = mesh.Mesh.from_file(filename_heart)
#         all_vertices = torch.from_numpy(np.vstack((stl_mesh.v0, stl_mesh.v1, stl_mesh.v2))).to(device)
#         # self.sampled_vertices = torch.as_tensor(
#         #     np.unique(all_vertices, axis=0),
#         #     dtype=torch.float32,
#         #     device=self.device
#         # )
#         self.sampled_vertices = torch.unique(
#             all_vertices,
#             dim=0,          # 按行去重（等价于np.unique的axis=0）
#             sorted=True     # 保持与NumPy一致的默认排序行为
#         ).to(dtype=torch.float32, device=self.device)
        
#         # 面片采样
#         self.sampled_faces = torch.as_tensor(
#             np.ascontiguousarray(stl_mesh.vectors),
#             dtype=torch.float32,
#             device=self.device
#         )

#         # 瓣膜模型
#         valve_mesh = mesh.Mesh.from_file(filename_valve)
#         valve_vertices = torch.from_numpy(np.vstack((valve_mesh.v0, valve_mesh.v1, valve_mesh.v2))).to(device)
#         self.valve_unique_vertices = torch.unique(valve_vertices, dim=0)
#         self.valve_faces = torch.as_tensor(
#             np.ascontiguousarray(valve_mesh.vectors),
#             dtype=torch.float32,
#             device=self.device
#         )

#         # 几何参数
#         self.centerline = torch.tensor([[37.1, -0.46, 91.97],
#                                       [-110.64, -62.12, 129.89]], 
#                                        dtype=torch.float32,
#                                        device=self.device)
#         self.centroid = (self.centerline[0] + self.centerline[1]) / 2
#         self.normal_vector = (self.centerline[1] - self.centroid)
#         self.normal_vector /= torch.norm(self.normal_vector)

#         # 沿法线生成点
#         t = torch.linspace(-50, 50, 10, device=self.device)
#         self.line_points = self.centroid + t[:, None] * self.normal_vector

#         # 空间定义
#         self.observation_space = spaces.Box(
#             low=np.array([0,0,0,0]),
#             high=np.array([1000,10,10,100]),
#             dtype=np.float32)
#         self.action_space = spaces.Box(
#             low=np.array([-1, -5, -5, -5]),
#             high=np.array([1, 5, 5, 5]),
#             dtype=np.float32)

#         self.obstacle_points = torch.cat([self.sampled_vertices, self.valve_unique_vertices])

#         # 导管实例
#         self.needle = Needle(device=device)
#         self.fig = None
#         self.destination = utility.get_destination(self.centerline[0], self.normal_vector, self.valve_faces)
#         print("destination: ", self.destination)
#         self.start = torch.tensor([0, 0, 0], dtype=torch.float32, device=self.device)

#     def reset(self, dz1=0, theta_z=0, theta_x=0, theta_y=0, dz2=20, seed=None, options=None):
#         self.needle.dz1 = torch.tensor(dz1, dtype=torch.float32, device=self.device, requires_grad=True) 
#         self.needle.theta_z = torch.tensor(theta_z, dtype=torch.float32, device=self.device, requires_grad=True)
#         self.needle.theta_x = torch.tensor(theta_x, dtype=torch.float32, device=self.device, requires_grad=True)
#         self.needle.r_x = torch.tensor(0, dtype=torch.float32, device=self.device, requires_grad=True) if theta_x == 0 else (62.89620622886024 * 180) / (self.needle.theta_x * torch.pi)
#         self.needle.theta_y = torch.tensor(theta_y, dtype=torch.float32, device=self.device, requires_grad=True)
#         self.needle.dz2 = torch.tensor(dz2, dtype=torch.float32, device=self.device, requires_grad=True)
#         self.needle.forward_kinematics()
#         self.needle.calculate_shape()
        
#         position_bias = torch.norm(self.needle.catheter_points[-1]-self.destination)
#         direction_bias = torch.norm(self.needle.T_12[0:3, 2]-self.normal_vector)
#         distance1 = utility.calculate_loss(self.obstacle_points, self.needle.catheter_points)
#         distance2,_ = utility.calculate_min_dis(self.obstacle_points, self.needle.catheter_points)
#         observation = torch.tensor([position_bias, direction_bias, distance1, distance2], dtype=torch.float32, device=self.device)
#         info = {}

#         return observation.cpu().numpy(), info

#     def step(self, action):
#         """
#         执行动作，更新导管状态，返回观测、奖励、终止标志和信息
#         :param action: [dz1, theta_x, theta_y, theta_z] (torch.Tensor or np.ndarray)
#         :return: observation, reward, terminated, truncated, info
#         """
#         # 1. 解析动作并更新导管参数
#         if isinstance(action, np.ndarray):
#             action = torch.tensor(action, dtype=torch.float32, device=self.device)
#         with torch.no_grad():
#             self.needle.dz1 += action[0]
#             self.needle.theta_x += action[1] 
#             self.needle.theta_y += action[2]
#             self.needle.theta_z += action[3]
#             # r_x 依赖于 theta_x
#             self.needle.r_x = torch.tensor(
#                 (62.89620622886024 * 180) / (self.needle.theta_x * torch.pi + 1e-6),
#                 dtype=torch.float32, device=self.device
#             )

#         # 2. 更新导管形状
#         self.needle.forward_kinematics()
#         self.needle.calculate_shape()

#         # 3. 计算观测
#         position_bias = torch.norm(self.needle.catheter_points[-1] - self.destination)
#         direction_bias = torch.norm(self.needle.T_12[0:3, 2] - self.normal_vector)
#         distance1 = utility.calculate_loss(self.obstacle_points, self.needle.catheter_points)
#         distance2, _ = utility.calculate_min_dis(self.obstacle_points, self.needle.catheter_points)
#         observation = torch.tensor([position_bias, direction_bias, distance1, distance2], dtype=torch.float32, device=self.device)

#         distances = distances = 1 * distance1 + 0.001 * distance2
        
#         # 4. 计算奖励（负势场或负距离等）
#         k_attract = 1.0
#         k_repel = 0.01 + 1.0 / (position_bias + 1e-3)
#         # 方向奖励权重随距离减小而增大
#         k_direction = 10 + 100.0 / (position_bias + 1e-3)  # 你可以调整0.1和1.0的比例
#         attract_potential, repel_potential, direction_potential = utility.calculate_potential(self.needle.catheter_points[-1], self.needle.T_12[0:3, 2], distances, self.destination, self.normal_vector, k_attract=k_attract, k_repel=k_repel, k_direction=k_direction, epsilon=1e-6)
#         reward = -attract_potential - repel_potential - direction_potential

#         # 5. 判断终止条件
#         terminated = bool((position_bias < 0.0001) and (direction_bias < 0.0001))
#         truncated = bool(distances < 0.0001)

#         # 6. info 字典
#         info = {
#             "position_bias": position_bias.item(),
#             "direction_bias": direction_bias.item(),
#             "distance1": distance1.item(),
#             "distance2": distance2.item()
#         }

#         return observation.cpu().numpy(), reward.item(), terminated, truncated, info

    
#     def render(self, mode='human'):
#         if self.fig is None:
#             self.fig = plt.figure()
#             self.ax = self.fig.add_subplot(111, projection='3d')

#         self.ax.cla()
        
#         # 心脏模型可视化（需转NumPy）
#         self.ax.add_collection3d(Poly3DCollection(
#             self.sampled_faces.cpu().numpy(), 
#             alpha=0.1, 
#             edgecolor='gray'
#         ))
        
#         # 瓣膜模型可视化
#         self.ax.add_collection3d(Poly3DCollection(
#             self.valve_faces.cpu().numpy(), 
#             alpha=0.1, 
#             edgecolor='blue'
#         ))
        
#         # 中心线可视化
#         self.ax.plot(
#             self.centerline[:, 0].cpu().numpy(), 
#             self.centerline[:, 1].cpu().numpy(), 
#             self.centerline[:, 2].cpu().numpy(), 
#             c='r', linewidth=2
#         )
        
#         # 导管可视化
#         catheter_np = self.needle.catheter_points.detach().cpu().numpy()
#         self.ax.scatter(
#             catheter_np[:, 0], 
#             catheter_np[:, 1], 
#             catheter_np[:, 2]
#         )
        
#         # 坐标设置
#         self.ax.set_xlim([-150, 150])
#         self.ax.set_ylim([-150, 150])
#         self.ax.set_zlim([0, 300])
#         plt.pause(0.01)
        
#     def close(self):
#         return super().close()




import gymnasium as gym
from gymnasium import spaces
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from stl import mesh
import utility
from needle import NeedleSegmented

class DigitalEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self,
                 filename_heart='./HeartModel/Heart_sim.STL',
                 filename_valve='./HeartModel/Valve_sim.STL',
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super(DigitalEnv, self).__init__()
        self.device = device

        # 载入模型
        stl_mesh = mesh.Mesh.from_file(filename_heart)
        all_vertices = torch.from_numpy(np.vstack((stl_mesh.v0, stl_mesh.v1, stl_mesh.v2))).to(device)
        self.sampled_vertices = torch.unique(all_vertices, dim=0).to(dtype=torch.float32, device=self.device)
        self.sampled_faces = torch.as_tensor(np.ascontiguousarray(stl_mesh.vectors), dtype=torch.float32, device=self.device)

        valve_mesh = mesh.Mesh.from_file(filename_valve)
        valve_vertices = torch.from_numpy(np.vstack((valve_mesh.v0, valve_mesh.v1, valve_mesh.v2))).to(device)
        self.valve_unique_vertices = torch.unique(valve_vertices, dim=0)
        self.valve_faces = torch.as_tensor(np.ascontiguousarray(valve_mesh.vectors), dtype=torch.float32, device=self.device)

        self.centerline = torch.tensor([[37.1, -0.46, 91.97], [-110.64, -62.12, 129.89]], dtype=torch.float32, device=self.device)
        self.centroid = (self.centerline[0] + self.centerline[1]) / 2
        self.normal_vector = (self.centerline[1] - self.centroid)
        self.normal_vector /= torch.norm(self.normal_vector)

        self.line_points = self.centroid + torch.linspace(-50, 50, 10, device=self.device)[:, None] * self.normal_vector

        self.obstacle_points = torch.cat([self.sampled_vertices, self.valve_unique_vertices])

        self.observation_space = spaces.Box(low=np.array([0, 0, 0, 0]), high=np.array([1000, 10, 10, 100]), dtype=np.float32)
        self.action_space = spaces.Box(low=np.array([0.0, -0.1, -0.1, -0.1]), high=np.array([2.0, 0.1, 0.1, 0.1]), dtype=np.float32)

        self.needle = NeedleSegmented(device=self.device)
        self.destination = utility.get_destination(self.centerline[0], self.normal_vector, self.valve_faces)
        self.fig = None

    def reset(self, seed=None, options=None):
        self.needle.reset()
        tip_pos, tip_dir = self.needle.get_tip()

        position_bias = torch.norm(tip_pos - self.destination)
        direction_bias = torch.norm(tip_dir - self.normal_vector)
        distance1 = utility.calculate_loss(self.obstacle_points, self.needle.get_points())
        distance2, _ = utility.calculate_min_dis(self.obstacle_points, self.needle.get_points())

        observation = torch.tensor([position_bias, direction_bias, distance1, distance2], dtype=torch.float32, device=self.device)
        return observation.cpu().numpy(), {}

    def step(self, action):
        if isinstance(action, np.ndarray):
            action = torch.tensor(action, dtype=torch.float32, device=self.device)

        self.needle.step(*action)
        tip_pos, tip_dir = self.needle.get_tip()

        position_bias = torch.norm(tip_pos - self.destination)
        direction_bias = torch.norm(tip_dir - self.normal_vector)
        distance1 = utility.calculate_loss(self.obstacle_points, self.needle.get_points())
        distance2, _ = utility.calculate_min_dis(self.obstacle_points, self.needle.get_points())
        observation = torch.tensor([position_bias, direction_bias, distance1, distance2], dtype=torch.float32, device=self.device)

        smoothness_penalty = 0.0
        if self.needle.num_steps() > 1:
            prev_dir = self.needle.directions[-2]
            curr_dir = self.needle.directions[-1]
            angle_diff = torch.acos(torch.clamp(torch.dot(prev_dir, curr_dir), -1.0, 1.0))
            smoothness_penalty = angle_diff**2  # 或直接 angle_diff
            
        collision_penalty = 0.0
        if distance2 < 3.0:  # 距离小于3mm判定为碰撞危险
            collision_penalty = 10.0 * (3.0 - distance2)**2
            
        length_penalty = 0.01 * self.needle.total_length
        
        distances = 1 * distance1 + 0.001 * distance2
        k_attract = 1.0
        k_repel = 0.01 + 1.0 / (position_bias + 1e-3)
        k_direction = 10 + 100.0 / (position_bias + 1e-3)

        attract_potential, repel_potential, direction_potential = utility.calculate_potential(
            tip_pos, tip_dir, distances, self.destination, self.normal_vector,
            k_attract=k_attract, k_repel=k_repel, k_direction=k_direction, epsilon=1e-6
        )
        reward = -attract_potential - repel_potential - direction_potential
        reward -= 0.1 * smoothness_penalty
        reward -= collision_penalty
        reward -= length_penalty
        
        terminated = bool((position_bias < 0.5) and (direction_bias < 0.2))
        truncated = bool(self.needle.num_steps() > 200)

        info = {
            "position_bias": position_bias.item(),
            "direction_bias": direction_bias.item(),
            "distance1": distance1.item(),
            "distance2": distance2.item()
        }

        return observation.cpu().numpy(), reward.item(), terminated, truncated, info

    def render(self, mode='human'):
        if self.fig is None:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(111, projection='3d')

        self.ax.cla()
        self.ax.add_collection3d(Poly3DCollection(self.sampled_faces.cpu().numpy(), alpha=0.1, edgecolor='gray'))
        self.ax.add_collection3d(Poly3DCollection(self.valve_faces.cpu().numpy(), alpha=0.1, edgecolor='blue'))
        self.ax.plot(self.centerline[:, 0].cpu().numpy(), self.centerline[:, 1].cpu().numpy(), self.centerline[:, 2].cpu().numpy(), c='r', linewidth=2)

        catheter_np = self.needle.get_points().detach().cpu().numpy()
        self.ax.plot(catheter_np[:, 0], catheter_np[:, 1], catheter_np[:, 2], color='green')

        self.ax.set_xlim([-150, 150])
        self.ax.set_ylim([-150, 150])
        self.ax.set_zlim([0, 300])
        plt.pause(0.01)

    def close(self):
        return super().close()
