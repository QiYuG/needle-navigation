# import gymnasium as gym
# from gymnasium import spaces    # 强化学习环境标准接口
# import numpy as np
# import matplotlib.pyplot as plt
# import torch

# from stl import mesh
# from mpl_toolkits.mplot3d.art3d import Poly3DCollection
# import utility
# from Needle import Needle


# # 强化学习环境类
# class DigitalEnv(gym.Env):
#     metadata = {"render_modes": ["human"]} # 可视化模型配置

#     def __init__(self, 
#                  filename_heart = './HeartModel/Heart_sim.STL',
#                  filename_valve = './HeartModel/Valve_sim.STL'):
#         super(DigitalEnv, self).__init__()
        
#         # load heart, adapted to every STL file
#         # 心脏模型加载与处理
#         stl_mesh = mesh.Mesh.from_file(filename_heart)
#         # tl_mesh = mesh.Mesh.from_file('./HeartModel/Heart1.STL')
#         all_vertices = np.vstack((stl_mesh.v0, stl_mesh.v1, stl_mesh.v2))
#         unique_vertices = np.unique(all_vertices, axis=0)

#         # sample
#         # 顶点采样配置，有问题，可能出现模型拓扑结构的破坏
#         '''
#         sample_ratio [0.0, 1.0]：采样比例，决定采样的顶点数量，比如当其为0.5时，随机保留50%的顶点
#         目的：降低计算复杂度，防止算法偏差，优化可视化性能
#         可能性：训练时不断增强采样率，开始时鼓励探索，后期确保精度
#         '''
#         sample_ratio = 1
#         num_samples = int(sample_ratio * len(unique_vertices))
#         self.sampled_vertices = unique_vertices[np.random.choice(len(unique_vertices), num_samples, replace=False)]
        
#         # 面片采样配置
#         face_ratio = 1
#         num_faces = int(len(stl_mesh.vectors) * face_ratio)
#         self.sampled_faces = stl_mesh.vectors[np.random.choice(len(stl_mesh.vectors), num_faces, replace=False)]

#         # 瓣膜模型加载
#         valve_mesh = mesh.Mesh.from_file(filename_valve)
#         # valve_mesh = mesh.Mesh.from_file('./HeartModel/Valve0.STL')
#         valve_vertices = np.vstack((valve_mesh.v0, valve_mesh.v1, valve_mesh.v2))
#         self.valve_unique_vertices = np.unique(valve_vertices, axis=0)
#         self.valve_faces = valve_mesh.vectors
#         # 瓣膜就不采样了
        

#         # find centroid of the valve
#         # 目标平面参数计算,这里是计算了心脏中心线的方向向量，用参数方程表示了距质心长度-50 到 50 之间的十个点的坐标
#         # 心脏中心线，希望导管沿着它走
#         self.centerline = np.array([[37.1,-0.46,91.97],
#                               [-110.64,-62.12,129.89]])
#         # self.centroid, self.normal_vector = utility.fit_plane_and_find_normal_line(self.valve_unique_vertices)
        
#         #心脏质心
#         self.centroid = (self.centerline[0]+self.centerline[1])/2
#         self.normal_vector = self.centerline[1]-self.centroid
#         self.normal_vector = self.normal_vector/np.linalg.norm(self.normal_vector)

#         # Compute points along the normal line using the parametric equation:：X = centroid + t * normal_vector
#         t = np.linspace(-50, 50, 10)  
#         self.line_x = self.centroid[0] + t * self.normal_vector[0]
#         self.line_y = self.centroid[1] + t * self.normal_vector[1]
#         self.line_z = self.centroid[2] + t * self.normal_vector[2]

#         # plot
#         # 可视化初始化
#         fig = plt.figure()
#         self.ax = fig.add_subplot(111, projection='3d') #构建3D坐标轴

#         # needle
#         # 造个实例
#         self.needle = Needle()
#         # 定义观测空间与动作空间，6个维度
#         self.observation_space = spaces.Box(low=np.array([1,-150,10,10,0,1]), 
#                                             high=np.array([50,150,50,100,30,50]))
        
#         self.action_space = spaces.Box(low=np.array([-3]*6), 
#                                        high=np.array([3]*6))


#     def reset(self, dz1=30, theta_z=0, theta_x=80, theta_y=0, dz2=20, seed=None, options=None):
#         self.needle.dz1 = dz1
#         self.needle.theta_z = theta_z
#         self.needle.theta_x = theta_x
#         self.needle.r_x = (62.89620622886024 * 180) / (self.needle.theta_x * np.pi)
#         self.needle.theta_y = theta_y
#         self.needle.dz2 = dz2
#         self.needle.forward_kinematics()
#         self.needle.calculate_shape()
#         # 完成这部分后调用needle.catheter_points即可得到导管的点云
#         '''
#         此处需要添加返回值，返回初始观测值与可选的字典信息
#         '''

#     def step(self, action=0):
#         """处理动作输入，更新导管状态"""
        
#         # -------此处待更改，这里theta_x，theta_z, dz1, theta_y都是根据经验求出的-------
        
#         # 计算导管与法线的角度关系
#         # find theta_x
#         z_axis = np.array([0, 0, 1]) 
#         angle_with_z = utility.angle_between_vectors(self.normal_vector, z_axis)
#         self.needle.theta_x = angle_with_z
#         # 这里意义不明，中心线与z轴夹角恒定不变，不用更新
#         # 此处的目的是希望导管的x轴与心脏中心线的方向一致，theta_x的含义还是
        
        
#         # find theta_z, dz
#         theta_z_opt, d_z_opt, t_val = utility.solve_parameters(self.centroid, -self.normal_vector,
#                                  self.needle.theta_x, self.needle.r_x)
#         self.needle.theta_z = theta_z_opt*180/np.pi
#         self.needle.dz1 = d_z_opt
#         self.needle.theta_y = 0
#         # update
#         self.needle.forward_kinematics()
#         self.needle.calculate_shape()
        
#         # find theta_y
#         angle_with_y = utility.angle_between_vectors(self.normal_vector, self.needle.T_12[0:3,2])
#         self.needle.theta_y = -angle_with_y
        
#         # -------改到这里-------
#         '''
#         这里我需要将观测输入Actor网络，得到theta_x, theta_y, dz1, theta_z，这里似乎和r_x没有关系
#         '''
        
#         # update
#         self.needle.forward_kinematics()
#         self.needle.calculate_shape()

#         '''
#         此处需要添加返回值：
#         observation: 当前的观测值。
#         reward: 当前的奖励值。
#         done: 一个布尔值，表示当前回合是否结束。
#         info: 一个字典，包含调试信息（可选）。
#         '''


#     '''
#     需要支持mode="human"，并在需要时显示环境的当前状态。
#     '''
#     def render(self):
#         # 3D渲染
#         # plot heart
#         self.ax.cla()   # 清空画布
#         # self.ax.scatter(self.sampled_vertices[:, 0], 
#         #                 self.sampled_vertices[:, 1], 
#         #                 self.sampled_vertices[:, 2], c='#4D5793', s=1)
        
#         # 绘制心脏
#         self.ax.add_collection3d(Poly3DCollection(self.sampled_faces, alpha=0.1, edgecolor='gray'))
        
#         # plot_valve
#         # self.ax.scatter(self.valve_unique_vertices[:, 0], 
#         #                 self.valve_unique_vertices[:, 1], 
#         #                 self.valve_unique_vertices[:, 2], c='r', s=1)
        
#         # 绘制瓣膜
#         self.ax.add_collection3d(Poly3DCollection(self.valve_faces, 
#                                                   alpha=0.1, edgecolor='r'))
      
#         # self.ax.plot(self.line_x, self.line_y, self.line_z, color='r', linewidth=2, label="Fitted Normal Line")
#         # centerline
#         # 中心线
#         self.ax.plot(self.centerline[:,0],self.centerline[:,1],self.centerline[:,2], c='r', linewidth=2)
        
#         # plot needle
#         # 导管
#         self.ax.scatter(self.needle.catheter_points[:,0],
#                         self.needle.catheter_points[:,1],
#                         self.needle.catheter_points[:,2])

        
#         # self.ax.set_title(title)

#         self.ax.set_xlabel('X(mm)')
#         self.ax.set_ylabel('Y(mm)')
#         self.ax.set_zlabel('Z(mm)')
        
    
#     def close(self):
#         return super().close()
    

# if __name__ == '__main__':
#     envs = DigitalEnv()
#     envs.reset()    
#     r_x_best = envs.needle.r_x      #最优弯曲半径
#     dis_max = 1     #距心脏和瓣膜的最大距离
#     catheter_points = np.array([[]])        #导管点云
#     while True:
#         envs.needle.r_x += 1
#         # envs.needle.r22_x = 68
#         envs.step()
        
#         # 合并了心脏和瓣膜的点云
#         obstacle = np.vstack((envs.sampled_vertices, envs.valve_unique_vertices))
#         # 弯曲部分与整个点云的最小距离及索引
#         dis1, obs_id1 = utility.calculate_min_dis(obstacle, envs.needle.catheter_bending_section[::100,:])
#         # 刚性部分与心脏点云的最小距离及索引
#         dis2, obs_id2 = utility.calculate_min_dis(envs.sampled_vertices, envs.needle.catheter_rigid_section[::50,:])
#         # 加权距离，有点好玩
#         dis = dis1 + 0.5*dis2
#         if dis > dis_max:
#             dis_max = dis
#             r_x_best = envs.needle.r_x
#             # 记录此时的导管位置
#             catheter_points = envs.needle.catheter_points
#         print('r_x: ', envs.needle.r_x, 'dis: ', dis, 'dis1: ', dis1, 'dis2: ', dis2)
#         # title = 'r_x: {}, dis: {}, dis1: {}, dis2: {}'.format(envs.needle.r_x, dis, dis1, dis2)
#         envs.render()

#         #plot min distance obs point
#         envs.ax.scatter(obstacle[obs_id1[0],0], 
#                         obstacle[obs_id1[0],1], 
#                         obstacle[obs_id1[0],2], s=50, c='black')
        
#         envs.ax.scatter(envs.sampled_vertices[obs_id2[0],0], 
#                 envs.sampled_vertices[obs_id2[0],1], 
#                 envs.sampled_vertices[obs_id2[0],2], s=50, c='black')

#         envs.ax.set_xlim([-150, 150])
#         envs.ax.set_ylim([-150, 150])
#         envs.ax.set_zlim([0, 300])
        
#         plt.pause(0.01)
#         if envs.needle.r_x == 100:
#             # envs.needle.r_x =10
#             break

#     # plot result
#     envs.needle.r_x = r_x_best
#     print('config: ', r_x_best)
#     np.savetxt("catheter_shape_plan_1.txt", catheter_points, fmt="%.6f", delimiter=" ")
#     envs.step()
#     envs.render()

#     envs.ax.set_xlim([-150,150])
#     envs.ax.set_ylim([-150,150])
#     envs.ax.set_zlim([0,300])
#     plt.pause(10000)


# """
# 其实导管模型考虑了两部分：
# 弯曲部分——用于导向瓣膜
# 刚性部分——用于进入血管        
# """


# def calculate_potential_field(envs, needle_points, centerline_points, k=10, epsilon=1e-6):
#         """
#         计算导管弯曲段的势场值
#         :param needle_points: 导管弯曲段的点云 (N, 3)
#         :param centerline_points: 中心线的点云 (M, 3)
#         :return: 势场值 (float)
#         """
#         # 计算导管点到中心线点的距离矩阵
#         distances = np.linalg.norm(needle_points[:, np.newaxis, :] - centerline_points[np.newaxis, :, :], axis=2)
        
#         # 对每个导管点，找到到中心线的最近距离
#         min_distances = np.min(distances, axis=1)
        
#         # 势场1值为最近距离的平方和
#         potential_1 = np.sum(min_distances**2)
        
#         # 合并了心脏和瓣膜的点云
#         obstacle = np.vstack((envs.sampled_vertices, envs.valve_unique_vertices))
        
#         # 弯曲部分与整个点云的最小距离及索引
#         min_dis1, obs_id1 = utility.calculate_min_dis(obstacle, envs.needle.catheter_bending_section[::100,:])
        
#         # 刚性部分与心脏点云的最小距离及索引
#         min_dis2, obs_id2 = utility.calculate_min_dis(envs.sampled_vertices, envs.needle.catheter_rigid_section[::50,:])
        
#         potential_2 = k / (min_dis1 + epsilon)
        
#         potential_3 = k / (min_dis2 + epsilon)
        
#         potential = potential_1 + potential_2 + potential_3
        
#         return potential



import gymnasium as gym
from gymnasium import spaces
import torch
import matplotlib.pyplot as plt
import numpy as np

from stl import mesh
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import utility
from Needle import Needle

class DigitalEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, 
                 filename_heart='./HeartModel/Heart_sim.STL',
                 filename_valve='./HeartModel/Valve_sim.STL',
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super(DigitalEnv, self).__init__()
        self.device = device
        # 心脏模型处理
        stl_mesh = mesh.Mesh.from_file(filename_heart)
        all_vertices = torch.from_numpy(np.vstack((stl_mesh.v0, stl_mesh.v1, stl_mesh.v2))).to(device)
        # self.sampled_vertices = torch.as_tensor(
        #     np.unique(all_vertices, axis=0),
        #     dtype=torch.float32,
        #     device=self.device
        # )
        self.sampled_vertices = torch.unique(
            all_vertices,
            dim=0,          # 按行去重（等价于np.unique的axis=0）
            sorted=True     # 保持与NumPy一致的默认排序行为
        ).to(dtype=torch.float32, device=self.device)
        
        # 面片采样
        self.sampled_faces = torch.as_tensor(
            np.ascontiguousarray(stl_mesh.vectors),
            dtype=torch.float32,
            device=self.device
        )

        # 瓣膜模型
        valve_mesh = mesh.Mesh.from_file(filename_valve)
        valve_vertices = torch.from_numpy(np.vstack((valve_mesh.v0, valve_mesh.v1, valve_mesh.v2))).to(device)
        self.valve_unique_vertices = torch.unique(valve_vertices, dim=0)
        self.valve_faces = torch.as_tensor(
            np.ascontiguousarray(valve_mesh.vectors),
            dtype=torch.float32,
            device=self.device
        )

        # 几何参数
        self.centerline = torch.tensor([[37.1, -0.46, 91.97],
                                      [-110.64, -62.12, 129.89]], 
                                       dtype=torch.float32,
                                       device=self.device)
        self.centroid = (self.centerline[0] + self.centerline[1]) / 2
        self.normal_vector = (self.centerline[1] - self.centroid)
        self.normal_vector /= torch.norm(self.normal_vector)

        # 沿法线生成点
        t = torch.linspace(-50, 50, 10, device=self.device)
        self.line_points = self.centroid + t[:, None] * self.normal_vector

        # 空间定义
        self.observation_space = spaces.Box(
            low=np.array([1, -150, 10, 10, 0, 1]),
            high=np.array([50, 150, 50, 100, 30, 50]))
        self.action_space = spaces.Box(
            low=np.array([-3]*6),
            high=np.array([3]*6))

        # 导管实例
        self.needle = Needle(device=device)
        self.fig = None
        self.destination = utility.get_destination(self.centerline[0], self.normal_vector, self.valve_faces)
        print("destination: ", self.destination)
        self.start = torch.tensor([0, 0, 0], dtype=torch.float32, device=self.device)

    def reset(self, dz1=0, theta_z=0, theta_x=0, theta_y=0, dz2=20, seed=None, options=None):
        self.needle.dz1 = torch.tensor(dz1, dtype=torch.float32, device=self.device, requires_grad=True) 
        self.needle.theta_z = torch.tensor(theta_z, dtype=torch.float32, device=self.device, requires_grad=True)
        self.needle.theta_x = torch.tensor(theta_x, dtype=torch.float32, device=self.device, requires_grad=True)
        self.needle.r_x = torch.tensor(0, dtype=torch.float32, device=self.device, requires_grad=True) if theta_x == 0 else (62.89620622886024 * 180) / (self.needle.theta_x * torch.pi)
        self.needle.theta_y = torch.tensor(theta_y, dtype=torch.float32, device=self.device, requires_grad=True)
        self.needle.dz2 = torch.tensor(dz2, dtype=torch.float32, device=self.device, requires_grad=True)
        self.needle.forward_kinematics()
        self.needle.calculate_shape()

    def step(self, action=0):
        # 保持原有计算逻辑
        z_axis = torch.tensor([0, 0, 1], dtype=torch.float32, device=self.device)
        angle_with_z = utility.angle_between_vectors(self.normal_vector, z_axis)
        self.needle.theta_x = angle_with_z

        theta_z_opt, d_z_opt, t_val = utility.solve_parameters(
            self.centroid, 
            -self.normal_vector,
            self.needle.theta_x,
            self.needle.r_x
        )
        self.needle.theta_z = theta_z_opt * 180 / torch.pi
        self.needle.dz1 = d_z_opt
        self.needle.theta_y = torch.tensor(0, dtype=torch.float32, device=self.device)

        self.needle.forward_kinematics()
        self.needle.calculate_shape()

        angle_with_y = utility.angle_between_vectors(
            self.normal_vector, 
            self.needle.T_12[0:3, 2]
        )
        self.needle.theta_y = -angle_with_y

        self.needle.forward_kinematics()
        self.needle.calculate_shape()



    def render(self):
        if self.fig is None:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(111, projection='3d')

        self.ax.cla()
        
        # 心脏模型可视化（需转NumPy）
        self.ax.add_collection3d(Poly3DCollection(
            self.sampled_faces.cpu().numpy(), 
            alpha=0.1, 
            edgecolor='gray'
        ))
        
        # 瓣膜模型可视化
        self.ax.add_collection3d(Poly3DCollection(
            self.valve_faces.cpu().numpy(), 
            alpha=0.1, 
            edgecolor='blue'
        ))
        
        # 中心线可视化
        self.ax.plot(
            self.centerline[:, 0].cpu().numpy(), 
            self.centerline[:, 1].cpu().numpy(), 
            self.centerline[:, 2].cpu().numpy(), 
            c='r', linewidth=2
        )
        
        # 导管可视化
        catheter_np = self.needle.catheter_points.detach().cpu().numpy()
        self.ax.scatter(
            catheter_np[:, 0], 
            catheter_np[:, 1], 
            catheter_np[:, 2]
        )
        
        # 坐标设置
        self.ax.set_xlim([-150, 150])
        self.ax.set_ylim([-150, 150])
        self.ax.set_zlim([0, 300])
        plt.pause(0.01)
        
    def close(self):
        return super().close()

def calculate_potential_field(envs, needle_points, centerline_points, k=10, epsilon=1e-6):
    """PyTorch版本的势场计算"""
    # 计算到中心线的距离
    distances = torch.cdist(needle_points, centerline_points)
    min_distances = torch.min(distances, dim=1).values
    potential_1 = torch.sum(min_distances**2)

    # 计算到障碍物的距离
    obstacle = torch.cat([envs.sampled_vertices, envs.valve_unique_vertices])
    obstacle_distances = torch.cdist(needle_points, obstacle)
    min_obstacle = torch.min(obstacle_distances, dim=1).values

    return potential_1 + k / (min_obstacle + epsilon)  


def artificial_potential_field_planning(env, max_steps=200, learning_rate_length=0.1, learning_rate_angle=0.1):
    """
    使用人工势场法对导管进行路径规划
    :param env: DigitalEnv 环境实例
    :param max_steps: 最大优化步数
    :param learning_rate: 学习率
    """
    # 势场权重
    k_attract = 0.1
    k_repel = 0.01
    k_direction = 0.1
    epsilon = 1e-6
    # last_attract_potential = 0.0
    # last_repel_potential = 0.0
    # last_direction_potential = 0.0

    # 获取目标点和方向
    destination = env.destination
    normal_vector = env.normal_vector

    for step in range(max_steps):
        # 获取导管末端位置和方向
        env.needle.forward_kinematics()
        env.needle.calculate_shape()
        end_point = env.needle.catheter_points[-1]
        end_direction = env.needle.T_12[0:3, 2]  # 导管末端方向向量
        # 排斥势场
        obstacle_points = torch.cat([env.sampled_vertices, env.valve_unique_vertices])
        distance1 = utility.calculate_loss(obstacle_points, env.needle.catheter_points)
        distance2,_ = utility.calculate_min_dis(obstacle_points, env.needle.catheter_points)
        distances = 1 * distance1 + 0.001 * distance2
        
        temp_attract_poteintial, temp_repel_poteintial, temp_direction_poteintial = calculate_poteintial(end_point, end_direction, distances, destination, normal_vector, k_attract= 0.1, k_repel= 0.01, k_direction=0.1, epsilon=1e-6)
        
        # 总势场
        total_potential = temp_attract_poteintial + temp_repel_poteintial + temp_direction_poteintial

        # 打印当前势场值
        print(f"Step {step}, Total Potential: {total_potential.item()}")

        # 计算梯度（数值梯度）
        gradients = calculate_gradient(env, total_potential, destination, normal_vector, k_attract, k_repel, k_direction, epsilon)
        
        
        env.needle.dz1 -= torch.clamp(learning_rate_length * gradients["dz1"],-10,10) 
        # env.needle.dz2 -= torch.clamp(learning_rate_length * gradients["dz2"],-10,10)
        env.needle.theta_x -= torch.clamp(learning_rate_angle * gradients["theta_x"],-2,2)
        env.needle.theta_y -= torch.clamp(learning_rate_angle * gradients["theta_y"],-2,2)
        env.needle.theta_z -= torch.clamp(learning_rate_angle * gradients["theta_z"],-2,2)

        # 更新 r_x
        env.needle.r_x = torch.tensor((62.89620622886024 * 180) / (env.needle.theta_x * torch.pi+epsilon),
                                      dtype=torch.float32, device=env.device)

        print("theta_x:", env.needle.theta_x.item(),"theta_y:", env.needle.theta_y.item(),"theta_z:", env.needle.theta_z.item(), "r_x:", env.needle.r_x.item(), "dz1:", env.needle.dz1.item(), "dz2:", env.needle.dz2.item())    
        # 可视化
        env.render()
        
        # if last_attract_potential == 0.0 and last_repel_potential == 0.0 and last_direction_potential == 0.0:
        #     last_attract_potential = temp_attract_poteintial
        #     last_repel_potential = temp_repel_poteintial
        #     last_direction_potential = temp_direction_poteintial
        #     continue
        
        # if temp_attract_poteintial < last_attract_potential:
        #     k_attract *= 1.0001
        # else:
        #     k_attract /= 1.0001
        
        # if temp_repel_poteintial < last_repel_potential:
        #     k_repel *= 1.0001
        # else:
        #     k_repel /= 1.0001
            
        # if temp_direction_poteintial < last_direction_potential:
        #     k_direction *= 1.0001
        # else:
        #     k_direction /= 1.0001
            

        # 判断是否到达目标点
        if torch.norm(end_point - destination) < 1.0 and torch.norm(end_direction - normal_vector) < 0.1:
            print("Reached destination!")
            break


# def artificial_potential_field_planning(env, max_steps=100, learning_rate=0.1):
#     """
#     使用人工势场法对导管进行路径规划（PyTorch 优化器版本）
#     :param env: DigitalEnv 环境实例
#     :param max_steps: 最大优化步数
#     :param learning_rate: 学习率
#     """
#     k_attract = 1.0
#     k_repel = 100.0
#     k_direction = 1.0
#     epsilon = 1e-6
#     destination = env.destination
#     normal_vector = env.normal_vector

#     # 确保参数启用梯度跟踪（初始化时已设置）
#     optimizable_params = [
#         env.needle.dz1,
#         env.needle.dz2,
#         env.needle.theta_x,
#         env.needle.theta_y,
#         env.needle.theta_z
#     ]
#     optimizer = torch.optim.Adam(optimizable_params, lr=learning_rate)

#     for step in range(max_steps):
#         optimizer.zero_grad()  # 梯度清零

#         # 前向传播与势场计算
#         env.needle.forward_kinematics()
#         env.needle.calculate_shape()
#         end_point = env.needle.catheter_points[-1]
#         end_direction = env.needle.T_12[0:3, 2]

#         # 势场计算（与原始代码一致）
#         attract_potential = 0.5 * k_attract * torch.norm(end_point - destination) ** 2
#         obstacle_points = torch.cat([env.sampled_vertices, env.valve_unique_vertices])
#         distances = utility.calculate_loss(obstacle_points, env.needle.catheter_points)
#         repel_potential = k_repel / (distances ** 2 + epsilon)
#         direction_potential = k_direction * torch.norm(end_direction - normal_vector) ** 2
#         total_potential = attract_potential + repel_potential + direction_potential

#         print(f"Step {step}, Total Potential: {total_potential.item()}")

#         # 反向传播与参数更新
#         total_potential.backward()
#         optimizer.step()

#         # 更新 r_x（保持计算图）
#         with torch.no_grad():
#             env.needle.r_x = (62.89620622886024 * 180) / (env.needle.theta_x * torch.pi+epsilon)

#         env.render()

#         # 终止条件
#         if torch.norm(end_point - destination) < 1.0 and torch.norm(end_direction - normal_vector) < 0.1:
#             print("Reached destination!")
#             break


def calculate_gradient(env, original_poteintial, destination, normal_vector, k_attract, k_repel, k_direction, epsilon):
    """
    计算势场函数相对于导管控制参数的梯度
    :param env: DigitalEnv 环境实例
    :param destination: 目标点
    :param normal_vector: 目标方向向量
    :param k_attract: 吸引势场权重
    :param k_repel: 排斥势场权重
    :param k_direction: 方向势场权重
    :param epsilon: 防止分母为零的小值
    :return: 梯度字典
    """
    gradients = {}
    delta = 1e-4  # 用于有限差分的微小增量

    # 保存当前参数
    original_params = {
        "dz1": env.needle.dz1.clone(),
        # "dz2": env.needle.dz2.clone(),
        "theta_x": env.needle.theta_x.clone(),
        "theta_y": env.needle.theta_y.clone(),
        "theta_z": env.needle.theta_z.clone(),
    }

    # 对每个参数计算梯度
    for param in original_params.keys():
        # 增加 delta
        setattr(env.needle, param, original_params[param] - delta)
        env.needle.forward_kinematics()
        env.needle.calculate_shape()

        # 计算新的势场值
        end_point = env.needle.catheter_points[-1]
        end_direction = env.needle.T_12[0:3, 2]
        obstacle_points = torch.cat([env.sampled_vertices, env.valve_unique_vertices])
        distance1 = utility.calculate_loss(obstacle_points, env.needle.catheter_points)
        distance2,_ = utility.calculate_min_dis(obstacle_points, env.needle.catheter_points)
        distances = 1 * distance1 + 0.001 * distance2
        
        new_attract_poteintial, new_repel_poteintial, new_direction_poteintial = calculate_poteintial(end_point, end_direction, distances, destination, normal_vector, k_attract, k_repel, k_direction, epsilon)
        new_potential = new_attract_poteintial + new_repel_poteintial + new_direction_poteintial

        # 恢复原始参数
        setattr(env.needle, param, original_params[param])

        # 计算梯度
        gradients[param] = (original_poteintial - new_potential) / delta

    return gradients


def calculate_poteintial(end_point, end_direction, distances, destination, normal_vector, k_attract, k_repel, k_direction, epsilon):
    """
    计算势场值
    :param end_point: 导管末端位置
    :param end_direction: 导管末端方向向量
    :param distances: 导管末端到障碍物的距离
    :param destination: 目标点
    :param normal_vector: 目标方向向量
    :param k_attract: 吸引势场权重
    :param k_repel: 排斥势场权重
    :param k_direction: 方向势场权重
    :param epsilon: 防止分母为零的小值
    :return: 势场值
    """
    attract_potential = 0.5 * k_attract * torch.norm(end_point - destination) ** 2

    repel_potential = k_repel / (distances ** 2 + epsilon)
    
    direction_potential = k_direction * torch.norm(end_direction - normal_vector) ** 2
    
    return attract_potential, repel_potential, direction_potential



if __name__ == '__main__':
    envs = DigitalEnv(device='cuda') if torch.cuda.is_available() else DigitalEnv()
    envs.reset()    
#     r_x_best = envs.needle.r_x      #最优弯曲半径
#     dis_max = 1     #距心脏和瓣膜的最大距离
#     catheter_points = torch.tensor([[]], device=envs.device)        #导管点云
#     while True:
#         envs.needle.r_x += 1
#         # envs.needle.r22_x = 68
#         envs.step()
        
#         # 合并了心脏和瓣膜的点云
#         obstacle = torch.cat([envs.sampled_vertices, envs.valve_unique_vertices])
#         # 弯曲部分与整个点云的最小距离及索引
#         dis1, obs_id1 = utility.calculate_min_dis(obstacle, envs.needle.catheter_bending_section[::100,:])
#         # 刚性部分与心脏点云的最小距离及索引
#         dis2, obs_id2 = utility.calculate_min_dis(envs.sampled_vertices, envs.needle.catheter_rigid_section[::50,:])
#         # 加权距离，有点好玩
#         dis = dis1 + 0.5*dis2
#         if dis > dis_max:
#             dis_max = dis
#             r_x_best = envs.needle.r_x
#             # 记录此时的导管位置
#             catheter_points = envs.needle.catheter_points
#         print('r_x: ', envs.needle.r_x, 'dis: ', dis, 'dis1: ', dis1, 'dis2: ', dis2)
#         # title = 'r_x: {}, dis: {}, dis1: {}, dis2: {}'.format(envs.needle.r_x, dis, dis1, dis2)
#         envs.render()

#         #plot min distance obs point
#         envs.ax.scatter(obstacle[obs_id1[0],0], 
#                         obstacle[obs_id1[0],1], 
#                         obstacle[obs_id1[0],2], s=50, c='black')
        
#         envs.ax.scatter(envs.sampled_vertices[obs_id2[0],0], 
#                 envs.sampled_vertices[obs_id2[0],1], 
#                 envs.sampled_vertices[obs_id2[0],2], s=50, c='black')

#         envs.ax.set_xlim([-150, 150])
#         envs.ax.set_ylim([-150, 150])
#         envs.ax.set_zlim([0, 300])
        
#         plt.pause(0.01)
#         if envs.needle.r_x == 100:
#             # envs.needle.r_x =10
#             break

#     # plot result
#     envs.needle.r_x = r_x_best
#     print('config: ', r_x_best)
#     np.savetxt("catheter_shape_plan_1.txt", catheter_points, fmt="%.6f", delimiter=" ")
#     envs.step()
#     envs.render()

#     envs.ax.set_xlim([-150,150])
#     envs.ax.set_ylim([-150,150])
#     envs.ax.set_zlim([0,300])
#     plt.pause(10000)


# """
# 其实导管模型考虑了两部分：
# 弯曲部分——用于导向瓣膜
# 刚性部分——用于进入血管        
# """  

    artificial_potential_field_planning(envs, max_steps=900, learning_rate_length=0.4, learning_rate_angle=0.2)
    # 保存最终导管形状
    np.savetxt("planned_catheter_shape.txt", envs.needle.catheter_points.cpu().numpy(), fmt="%.6f", delimiter=" ")
    plt.show()