# import torch
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# class Needle():
#     def __init__(self, x_base=0, y_base=0, dz1=0, theta_z=0,
#                  theta_x=80, theta_y=0, dz2=0, 
#                  device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):  # 新增device参数
#         """ 初始化导管参数（支持GPU设备） """
#         self.device = device  # 存储设备信息
        
#         # 基础位置（转换为张量）
#         self.x_base = torch.tensor(x_base, dtype=torch.float32, device=device)
#         self.y_base = torch.tensor(y_base, dtype=torch.float32, device=device)

#         # 运动学参数（保持为张量）
#         self.dz1 = torch.tensor(dz1, dtype=torch.float32, device=device, requires_grad=True)
#         self.theta_z = torch.tensor(theta_z, dtype=torch.float32, device=device, requires_grad=True)
#         self.theta_x = torch.tensor(theta_x, dtype=torch.float32, device=device, requires_grad=True)
#         self.r_x =  torch.tensor(0, dtype=torch.float32, device=device, requires_grad=True) if theta_x == 0 else (62.89620622886024 * 180) / (self.theta_x * torch.pi)
#         self.theta_y = torch.tensor(theta_y, dtype=torch.float32, device=device, requires_grad=True)
#         self.dz2 = torch.tensor(dz2, dtype=torch.float32, device=device, requires_grad=True)

#         # 初始化变换矩阵（直接在目标设备上创建）
#         self.T_0 = self.transl(self.x_base, self.y_base, 0)  # 基准坐标系
#         self.T_01 = torch.eye(4, dtype=torch.float32, device=device)  # 各段变换矩阵
#         self.T_12 = torch.eye(4, dtype=torch.float32, device=device)
#         self.T_23 = torch.eye(4, dtype=torch.float32, device=device)
        
#         # 点云容器初始化到设备
#         self.catheter_points = torch.tensor(
#             [[self.x_base, self.y_base, 0.]], 
#             dtype=torch.float32,
#             device=device
#         )
#         self.catheter_bending_section = torch.tensor(
#             [[self.x_base, self.y_base, 0.]], 
#             dtype=torch.float32,
#             device=device
#         )
#         self.catheter_rigid_section = torch.tensor([[]], dtype=torch.float32, device=device)

#     def forward_kinematics(self):
#         """ 正向运动学计算（支持GPU加速） """
#         # 使用类内存储的device信息
#         self.T_01 = self.T_0 @ self.transformation_matrix_z(
#             self.theta_z, self.dz1
#         )
#         self.T_12 = self.T_01 @ self.constance_curve_matrix_x(
#             self.theta_x, self.r_x
#         ) @ self.transformation_matrix_y(self.theta_y, 0)

#     def calculate_shape(self, strait_number_1=100, strait_number_2=100, bending_number=500):
#         """ 形状计算（全GPU张量操作） """
#         # 初始化点云容器（保持设备一致）
#         self.catheter_points = torch.tensor(
#             [[self.x_base, self.y_base, 0.]], 
#             device=self.device
#         )
        
#         # 第一部分：直线段（设备感知的linspace）
#         z_values = torch.linspace(
#             0, self.dz1, strait_number_1, 
#             device=self.device
#         )
#         section_one_points = torch.stack((
#             torch.full_like(z_values, self.x_base),
#             torch.full_like(z_values, self.y_base),
#             z_values
#         ), dim=1)
        
#         # 第二部分：弯曲段（设备感知的三角函数）
#         length = self.r_x * self.theta_x * torch.pi / 180
#         length_values = torch.linspace(
#             0, length, bending_number, 
#             device=self.device
#         )
#         x_values = torch.zeros(bending_number, device=self.device)
#         y_values = -self.r_x + self.r_x * torch.cos(length_values/self.r_x)
#         z_values = self.r_x * torch.sin(length_values/self.r_x)
#         one_values = torch.ones(bending_number, device=self.device)
        
#         section_two_points = torch.stack(
#             (x_values, y_values, z_values, one_values), 
#             dim=0
#         )
#         section_two_points = (self.T_01 @ section_two_points).T[:, :3]
        
#         # 第三部分：刚性段（设备感知的操作）
#         z_values = torch.linspace(
#             0, self.dz2, strait_number_2, 
#             device=self.device
#         )
#         x_values = torch.zeros(strait_number_2, device=self.device)
#         y_values = torch.zeros(strait_number_2, device=self.device)
#         one_values = torch.ones(strait_number_2, device=self.device)
#         section_three_points = torch.stack(
#             (x_values, y_values, z_values, one_values), 
#             dim=0
#         )
#         section_three_points = (self.T_12 @ section_three_points).T[:, :3]
        
#         # 合并点云（设备自动继承）
#         self.catheter_points = torch.cat((
#             self.catheter_points, 
#             section_one_points, 
#             section_two_points,
#             section_three_points
#         ), dim=0)
        
#         self.catheter_bending_section = torch.cat((self.catheter_points, 
#                                           section_one_points, 
#                                           section_two_points))
        
#         self.catheter_rigid_section = section_three_points

#     # ====================== 修改后的变换矩阵方法 ======================
#     def transformation_matrix_z(self, theta, d):
#         """ Z轴变换矩阵（支持设备） """
#         theta = theta * torch.pi/180.0
#         return torch.tensor([  # 直接返回设备感知的矩阵
#             [torch.cos(theta), -torch.sin(theta), 0., 0.],
#             [torch.sin(theta),  torch.cos(theta), 0., 0.],
#             [0.,               0.,               1., d],
#             [0.,               0.,               0., 1.]
#         ], device=self.device)

#     def constance_curve_matrix_x(self, theta, r):
#         """ X轴曲率矩阵（设备感知） """
#         theta = theta * torch.pi/180.0
#         return torch.tensor([
#             [1., 0.,               0.,              0.],
#             [0., torch.cos(theta), -torch.sin(theta), -r + r*torch.cos(theta)],
#             [0., torch.sin(theta), torch.cos(theta), r*torch.sin(theta)],
#             [0., 0.,               0.,              1.]
#         ], device=self.device)
    
#     def transl(self, x, y, z):
#         """ 平移矩阵（设备感知） """
#         return torch.tensor([
#             [1.,0.,0.,x],
#             [0.,1.,0.,y],
#             [0.,0.,1.,z],
#             [0.,0.,0.,1.]
#         ], device=self.device)
    
#     def transformation_matrix_y(self, theta, d):
#         """ Y轴变换矩阵（设备感知） """
#         theta = theta * torch.pi/180.0
#         return torch.tensor([
#             [torch.cos(theta),  0., torch.sin(theta), 0.],
#             [0.,               1., 0.,               d],
#             [-torch.sin(theta),0., torch.cos(theta), 0.],
#             [0.,               0., 0.,               1.]
#         ], device=self.device)




import torch

class NeedleSegmented:
    def __init__(self, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.reset()

    def reset(self):
        self.catheter_points = [torch.tensor([0.0, 0.0, 0.0], device=self.device)]
        self.directions = [torch.tensor([0.0, 0.0, 1.0], device=self.device)]
        self.current_position = torch.tensor([0.0, 0.0, 0.0], device=self.device)
        self.current_direction = torch.tensor([0.0, 0.0, 1.0], device=self.device)
        self.current_rotation = torch.eye(3, device=self.device)
        self.total_length = 0.0

    def step(self, delta_dz1, delta_theta_x, delta_theta_y, delta_theta_z):
        # 更新旋转矩阵
        R_x = torch.tensor([
            [1, 0, 0],
            [0, torch.cos(delta_theta_x), -torch.sin(delta_theta_x)],
            [0, torch.sin(delta_theta_x), torch.cos(delta_theta_x)]
        ], device=self.device)

        R_y = torch.tensor([
            [torch.cos(delta_theta_y), 0, torch.sin(delta_theta_y)],
            [0, 1, 0],
            [-torch.sin(delta_theta_y), 0, torch.cos(delta_theta_y)]
        ], device=self.device)

        R_z = torch.tensor([
            [torch.cos(delta_theta_z), -torch.sin(delta_theta_z), 0],
            [torch.sin(delta_theta_z), torch.cos(delta_theta_z), 0],
            [0, 0, 1]
        ], device=self.device)

        R = R_z @ R_y @ R_x
        self.current_rotation = R @ self.current_rotation
        self.current_direction = self.current_rotation @ torch.tensor([0.0, 0.0, 1.0], device=self.device)
        self.current_direction = self.current_direction / torch.norm(self.current_direction)

        # 推进位置
        next_position = self.current_position + delta_dz1 * self.current_direction

        self.catheter_points.append(next_position)
        self.directions.append(self.current_direction)
        self.current_position = next_position
        self.total_length += delta_dz1

    def get_points(self):
        return torch.stack(self.catheter_points)

    def get_tip(self):
        return self.catheter_points[-1], self.directions[-1]

    def num_steps(self):
        return len(self.catheter_points) - 1