import gymnasium as gym
from gymnasium import spaces    # 强化学习环境标准接口
import numpy as np
import matplotlib.pyplot as plt

from stl import mesh
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import utility
from Needle import Needle


# 强化学习环境类
class DigitalEnv(gym.Env):
    metadata = {"render_modes": ["human"]} # 可视化模型配置

    def __init__(self, 
                 filename_heart = './HeartModel/Heart_sim.STL',
                 filename_valve = './HeartModel/Valve_sim.STL'):
        super(DigitalEnv, self).__init__()
        
        # load heart, adapted to every STL file
        # 心脏模型加载与处理
        stl_mesh = mesh.Mesh.from_file(filename_heart)
        # tl_mesh = mesh.Mesh.from_file('./HeartModel/Heart1.STL')
        all_vertices = np.vstack((stl_mesh.v0, stl_mesh.v1, stl_mesh.v2))
        unique_vertices = np.unique(all_vertices, axis=0)

        # sample
        # 顶点采样配置，有问题，可能出现模型拓扑结构的破坏
        '''
        sample_ratio [0.0, 1.0]：采样比例，决定采样的顶点数量，比如当其为0.5时，随机保留50%的顶点
        目的：降低计算复杂度，防止算法偏差，优化可视化性能
        可能性：训练时不断增强采样率，开始时鼓励探索，后期确保精度
        '''
        sample_ratio = 1
        num_samples = int(sample_ratio * len(unique_vertices))
        self.sampled_vertices = unique_vertices[np.random.choice(len(unique_vertices), num_samples, replace=False)]
        
        # 面片采样配置
        face_ratio = 1
        num_faces = int(len(stl_mesh.vectors) * face_ratio)
        self.sampled_faces = stl_mesh.vectors[np.random.choice(len(stl_mesh.vectors), num_faces, replace=False)]

        # 瓣膜模型加载
        valve_mesh = mesh.Mesh.from_file(filename_valve)
        # valve_mesh = mesh.Mesh.from_file('./HeartModel/Valve0.STL')
        valve_vertices = np.vstack((valve_mesh.v0, valve_mesh.v1, valve_mesh.v2))
        self.valve_unique_vertices = np.unique(valve_vertices, axis=0)
        self.valve_faces = valve_mesh.vectors
        # 瓣膜就不采样了
        

        # find centroid of the valve
        # 目标平面参数计算,这里是计算了心脏中心线的方向向量，用参数方程表示了距质心长度-50 到 50 之间的十个点的坐标
        # 心脏中心线，希望导管沿着它走
        self.centerline = np.array([[37.1,-0.46,91.97],
                              [-110.64,-62.12,129.89]])
        # self.centroid, self.normal_vector = utility.fit_plane_and_find_normal_line(self.valve_unique_vertices)
        
        #心脏质心
        self.centroid = (self.centerline[0]+self.centerline[1])/2
        self.normal_vector = self.centerline[1]-self.centroid
        self.normal_vector = self.normal_vector/np.linalg.norm(self.normal_vector)

        # Compute points along the normal line using the parametric equation:：X = centroid + t * normal_vector
        t = np.linspace(-50, 50, 10)  
        self.line_x = self.centroid[0] + t * self.normal_vector[0]
        self.line_y = self.centroid[1] + t * self.normal_vector[1]
        self.line_z = self.centroid[2] + t * self.normal_vector[2]

        # plot
        # 可视化初始化
        fig = plt.figure()
        self.ax = fig.add_subplot(111, projection='3d') #构建3D坐标轴

        # needle
        # 造个实例
        self.needle = Needle()
        # 定义观测空间与动作空间，6个维度
        self.observation_space = spaces.Box(low=np.array([1,-150,10,10,0,1]), 
                                            high=np.array([50,150,50,100,30,50]))
        
        self.action_space = spaces.Box(low=np.array([-3]*6), 
                                       high=np.array([3]*6))


    def reset(self, dz1=30, theta_z=0, theta_x=80, theta_y=0, dz2=20, seed=None, options=None):
        self.needle.dz1 = dz1
        self.needle.theta_z = theta_z
        self.needle.theta_x = theta_x
        self.needle.r_x = (62.89620622886024 * 180) / (self.needle.theta_x * np.pi)
        self.needle.theta_y = theta_y
        self.needle.dz2 = dz2
        self.needle.forward_kinematics()
        self.needle.calculate_shape()
        # 完成这部分后调用needle.catheter_points即可得到导管的点云
        '''
        此处需要添加返回值，返回初始观测值与可选的字典信息
        '''
        

    def step(self, action=0):
        """处理动作输入，更新导管状态"""
        
        # -------此处待更改，这里theta_x，theta_z, dz1, theta_y都是根据经验求出的-------
        
        # 计算导管与法线的角度关系
        # find theta_x
        z_axis = np.array([0, 0, 1]) 
        angle_with_z = utility.angle_between_vectors(self.normal_vector, z_axis)
        self.needle.theta_x = angle_with_z
        # 这里意义不明，中心线与z轴夹角恒定不变，不用更新
        # 此处的目的是希望导管的x轴与心脏中心线的方向一致，theta_x的含义还是
        
        
        # find theta_z, dz
        theta_z_opt, d_z_opt, t_val = utility.solve_parameters(self.centroid, -self.normal_vector,
                                 self.needle.theta_x, self.needle.r_x)
        self.needle.theta_z = theta_z_opt*180/np.pi
        self.needle.dz1 = d_z_opt
        self.needle.theta_y = 0
        # update
        self.needle.forward_kinematics()
        self.needle.calculate_shape()
        
        # find theta_y
        angle_with_y = utility.angle_between_vectors(self.normal_vector, self.needle.T_12[0:3,2])
        self.needle.theta_y = -angle_with_y
        
        # -------改到这里-------
        '''
        这里我需要将观测输入Actor网络，得到theta_x, theta_y, dz1, theta_z，这里似乎和r_x没有关系
        '''
        
        # update
        self.needle.forward_kinematics()
        self.needle.calculate_shape()

        '''
        此处需要添加返回值：
        observation: 当前的观测值。
        reward: 当前的奖励值。
        done: 一个布尔值，表示当前回合是否结束。
        info: 一个字典，包含调试信息（可选）。
        '''


    '''
    需要支持mode="human"，并在需要时显示环境的当前状态。
    '''
    def render(self):
        # 3D渲染
        # plot heart
        self.ax.cla()   # 清空画布
        # self.ax.scatter(self.sampled_vertices[:, 0], 
        #                 self.sampled_vertices[:, 1], 
        #                 self.sampled_vertices[:, 2], c='#4D5793', s=1)
        
        # 绘制心脏
        self.ax.add_collection3d(Poly3DCollection(self.sampled_faces, alpha=0.1, edgecolor='gray'))
        
        # plot_valve
        # self.ax.scatter(self.valve_unique_vertices[:, 0], 
        #                 self.valve_unique_vertices[:, 1], 
        #                 self.valve_unique_vertices[:, 2], c='r', s=1)
        
        # 绘制瓣膜
        self.ax.add_collection3d(Poly3DCollection(self.valve_faces, 
                                                  alpha=0.1, edgecolor='r'))
      
        # self.ax.plot(self.line_x, self.line_y, self.line_z, color='r', linewidth=2, label="Fitted Normal Line")
        # centerline
        # 中心线
        self.ax.plot(self.centerline[:,0],self.centerline[:,1],self.centerline[:,2], c='r', linewidth=2)
        
        # plot needle
        # 导管
        self.ax.scatter(self.needle.catheter_points[:,0],
                        self.needle.catheter_points[:,1],
                        self.needle.catheter_points[:,2])

        
        # self.ax.set_title(title)

        self.ax.set_xlabel('X(mm)')
        self.ax.set_ylabel('Y(mm)')
        self.ax.set_zlabel('Z(mm)')
        
    
    def close(self):
        return super().close()
    

if __name__ == '__main__':
    envs = DigitalEnv()
    envs.reset()    
    r_x_best = envs.needle.r_x      #最优弯曲半径
    dis_max = 1     #距心脏和瓣膜的最大距离
    catheter_points = np.array([[]])        #导管点云
    while True:
        envs.needle.r_x += 1
        # envs.needle.r22_x = 68
        envs.step()
        
        # 合并了心脏和瓣膜的点云
        obstacle = np.vstack((envs.sampled_vertices, envs.valve_unique_vertices))
        # 弯曲部分与整个点云的最小距离及索引
        dis1, obs_id1 = utility.calculate_min_dis(obstacle, envs.needle.catheter_bending_section[::100,:])
        # 刚性部分与心脏点云的最小距离及索引
        dis2, obs_id2 = utility.calculate_min_dis(envs.sampled_vertices, envs.needle.catheter_rigid_section[::50,:])
        # 加权距离，有点好玩
        dis = dis1 + 0.5*dis2
        if dis > dis_max:
            dis_max = dis
            r_x_best = envs.needle.r_x
            # 记录此时的导管位置
            catheter_points = envs.needle.catheter_points
        print('r_x: ', envs.needle.r_x, 'dis: ', dis, 'dis1: ', dis1, 'dis2: ', dis2)
        # title = 'r_x: {}, dis: {}, dis1: {}, dis2: {}'.format(envs.needle.r_x, dis, dis1, dis2)
        envs.render()

        #plot min distance obs point
        envs.ax.scatter(obstacle[obs_id1[0],0], 
                        obstacle[obs_id1[0],1], 
                        obstacle[obs_id1[0],2], s=50, c='black')
        
        envs.ax.scatter(envs.sampled_vertices[obs_id2[0],0], 
                envs.sampled_vertices[obs_id2[0],1], 
                envs.sampled_vertices[obs_id2[0],2], s=50, c='black')

        envs.ax.set_xlim([-150, 150])
        envs.ax.set_ylim([-150, 150])
        envs.ax.set_zlim([0, 300])
        
        plt.pause(0.01)
        if envs.needle.r_x == 100:
            # envs.needle.r_x =10
            break

    # plot result
    envs.needle.r_x = r_x_best
    print('config: ', r_x_best)
    np.savetxt("catheter_shape_plan_1.txt", catheter_points, fmt="%.6f", delimiter=" ")
    envs.step()
    envs.render()

    envs.ax.set_xlim([-150,150])
    envs.ax.set_ylim([-150,150])
    envs.ax.set_zlim([0,300])
    plt.pause(10000)


"""
其实导管模型考虑了两部分：
弯曲部分——用于导向瓣膜
刚性部分——用于进入血管        
"""