# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# class Needle():
#     def __init__(self):
#         # base
#         self.x_base = 0
#         self.y_base = 0

#         # configration
#         self.dz1 = 30
#         self.theta_z = -20
#         self.r_x = 70
#         self.theta_x = 80
#         self.theta_y = 0
#         self.dz2 = 20

#         # transform
#         self.T_0 = self.transl(self.x_base, self.y_base, 0)
#         self.T_01, self.T_12, self.T_23 = np.eye(4), np.eye(4), np.eye(4)
#         self.catheter_points = np.array([[self.x_base, self.y_base, 0]])
#         self.catheter_bending_section = np.array([[self.x_base, self.y_base, 0]])
#         self.catheter_rigid_section = np.array([[]])

#     def forward_kinematics(self):
#         self.T_01 = self.T_0 @ self.transformation_matrix_z(self.theta_z, self.dz1)
#         self.T_12 = self.T_01 @ self.constance_curve_matrix_x(self.theta_x, self.r_x) \
#             @ self.transformation_matrix_y(self.theta_y, 0)

#     def calculate_shape(self):
#         self.catheter_points = np.array([[self.x_base, self.y_base, 0]])
#         # section one
#         z_values = np.linspace(1, self.dz1, int(self.dz1))
#         section_one_points = np.column_stack((np.full_like(z_values, self.x_base), 
#                           np.full_like(z_values, self.y_base), 
#                           z_values))
        
#         # section two
#         length = self.r_x*self.theta_x*np.pi/180
#         length_values = np.linspace(1,length, int(length))
#         x_values = np.array([0]*int(length))
#         y_values = -self.r_x+self.r_x*np.cos(length_values/self.r_x)
#         z_values = self.r_x*np.sin(length_values/self.r_x)
#         one_values = np.array([1]*int(length))
#         section_two_points = np.vstack((x_values, y_values, z_values, one_values))
#         section_two_points = ((self.T_01 @ section_two_points).T)[:,0:3]

#         # section three
#         z_values = np.linspace(1, self.dz2, int(self.dz2))
#         x_values = np.array([0]*int(self.dz2))
#         y_values = np.array([0]*int(self.dz2))
#         one_values = np.array([1]*int(self.dz2))
#         section_three_points = np.vstack((x_values, y_values, z_values, one_values))
#         section_three_points = ((self.T_12 @ section_three_points).T)[:,0:3]


#         self.catheter_points = np.vstack((self.catheter_points, 
#                                           section_one_points, 
#                                           section_two_points,
#                                           section_three_points))
#         self.catheter_bending_section = np.vstack((self.catheter_points, 
#                                           section_one_points, 
#                                           section_two_points))
#         self.catheter_rigid_section = section_three_points
#         # print(self.catheter_points)
    
#     @staticmethod
#     def transformation_matrix_z(theta, d):
#         theta = theta*np.pi/180.0
#         cos_theta = np.cos(theta)
#         sin_theta = np.sin(theta)
        
#         Tz = np.array([
#             [cos_theta, -sin_theta, 0, 0],
#             [sin_theta,  cos_theta, 0, 0],
#             [0,          0,         1, d],
#             [0,          0,         0, 1]
#         ])
        
#         return Tz

#     @staticmethod
#     def constance_curve_matrix_x(theta, r):
#         theta = theta*np.pi/180.0
#         cos_theta = np.cos(theta)
#         sin_theta = np.sin(theta)
        
#         Tx = np.array([
#             [1, 0,         0,          0],
#             [0, cos_theta, -sin_theta, -r+r*cos_theta],
#             [0, sin_theta, cos_theta,  r*sin_theta],
#             [0, 0,         0,          1]
#         ])
        
#         return Tx
    
#     @staticmethod
#     def transl(x,y,z):
#         T = np.array([
#             [1,          0,         0, x],
#             [0,          1,         0, y],
#             [0,          0,         1, z],
#             [0,          0,         0, 1],
#         ])

#         return T
    
#     @staticmethod
#     def transformation_matrix_y(theta, d):
#         theta = theta*np.pi/180.0
#         cos_theta = np.cos(theta)
#         sin_theta = np.sin(theta)
        
#         Ty = np.array([
#             [ cos_theta,  0, sin_theta, 0],
#             [ 0,          1, 0,         d],
#             [-sin_theta,  0, cos_theta, 0],
#             [ 0,          0, 0,         1]
#         ])
        
#         return Ty


# if __name__ == '__main__':
#     needle = Needle()
#     needle.forward_kinematics()
#     needle.calculate_shape()

#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     ax.scatter(needle.catheter_points[:,0],
#                needle.catheter_points[:,1],
#                needle.catheter_points[:,2])
#     ax.set_xlabel('X(mm)')
#     ax.set_ylabel('Y(mm)')
#     ax.set_zlabel('Z(mm)')

#     ax.set_xlim([-50,50])
#     ax.set_ylim([-50,50])
#     ax.set_zlim([0,100])

#     plt.show()

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ====================== 导管针状物类定义 ======================
class Needle():
    def __init__(self, x_base=0, y_base=0, dz1=30, theta_z=-20, r_x=70, theta_x=80, theta_y=0, dz2=20):
        """ 初始化导管基础参数和变换矩阵 """
        # 基础位置
        self.x_base = x_base  # X轴基准坐标
        self.y_base = y_base  # Y轴基准坐标

        # 运动学参数配置
        self.dz1 = dz1     # 第一段直线长度（沿Z轴）
        self.theta_z = theta_z  # Z轴旋转角度（度）
        self.r_x = r_x     # X方向弯曲半径
        self.theta_x = theta_x  # X方向弯曲角度（度）
        self.theta_y = theta_y   # Y轴旋转角度（预留）
        self.dz2 = dz2     # 第二段直线长度（弯曲后的延伸）

        # 变换矩阵初始化
        self.T_0 = self.transl(self.x_base, self.y_base, 0)  # 基准坐标系
        self.T_01, self.T_12, self.T_23 = np.eye(4), np.eye(4), np.eye(4)  # 各段变换矩阵
        self.catheter_points = np.array([[self.x_base, self.y_base, 0]])  # 导管点云容器
        self.catheter_bending_section = np.array([[self.x_base, self.y_base, 0]])  # 弯曲段
        self.catheter_rigid_section = np.array([[]])  # 刚性段

    def forward_kinematics(self):
        """ 正向运动学计算 """
        # 第一段变换：Z轴旋转+平移
        self.T_01 = self.T_0 @ self.transformation_matrix_z(self.theta_z, self.dz1)
        # 第二段变换：恒定曲率弯曲+Y轴旋转
        self.T_12 = self.T_01 @ self.constance_curve_matrix_x(self.theta_x, self.r_x) \
            @ self.transformation_matrix_y(self.theta_y, 0)

    def calculate_shape(self):
        """ 计算导管三维形状 """
        # 初始化点云容器
        self.catheter_points = np.array([[self.x_base, self.y_base, 0]])
        
        # 第一部分：直线段（沿Z轴）
        z_values = np.linspace(1, self.dz1, int(self.dz1))  # 生成等间距Z值
        section_one_points = np.column_stack((  # 构建(x,y,z)坐标
            np.full_like(z_values, self.x_base),  # X坐标保持基准
            np.full_like(z_values, self.y_base),  # Y坐标保持基准
            z_values))  # Z方向线性增长
        
        # 第二部分：弯曲段（恒定曲率圆弧）
        length = self.r_x * self.theta_x * np.pi / 180  # 计算弧长
        length_values = np.linspace(1, length, int(length))  # 沿弧长采样
        # 参数化圆弧（在局部坐标系中）
        x_values = np.zeros(int(length))  # X方向无变化
        y_values = -self.r_x + self.r_x * np.cos(length_values/self.r_x)  # Y方向余弦变化
        z_values = self.r_x * np.sin(length_values/self.r_x)  # Z方向正弦变化
        one_values = np.ones(int(length))  # 齐次坐标扩充
        # 组合齐次坐标并进行坐标变换
        section_two_points = np.vstack((x_values, y_values, z_values, one_values))
        section_two_points = ((self.T_01 @ section_two_points).T)[:,0:3]  # 应用变换矩阵
        
        # 第三部分：刚性延伸段
        z_values = np.linspace(1, self.dz2, int(self.dz2))  # 生成延伸段Z值
        x_values = np.zeros(int(self.dz2))  # X保持零
        y_values = np.zeros(int(self.dz2))  # Y保持零
        one_values = np.ones(int(self.dz2))  # 齐次坐标
        section_three_points = np.vstack((x_values, y_values, z_values, one_values))
        section_three_points = ((self.T_12 @ section_three_points).T)[:,0:3]  # 应用变换
        
        # 合并所有点云
        self.catheter_points = np.vstack((self.catheter_points, 
                                        section_one_points, 
                                        section_two_points,
                                        section_three_points))
        # 分离存储弯曲段和刚性段
        self.catheter_bending_section = np.vstack((self.catheter_points, 
                                        section_one_points, 
                                        section_two_points))
        self.catheter_rigid_section = section_three_points

    # ====================== 变换矩阵工具方法 ======================
    @staticmethod
    def transformation_matrix_z(theta, d):
        """ Z轴旋转平移矩阵 """
        theta = theta * np.pi/180.0  # 角度转弧度
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        return np.array([  # 标准Z轴旋转平移矩阵
            [cos_theta, -sin_theta, 0, 0],
            [sin_theta,  cos_theta, 0, 0],
            [0,          0,         1, d],
            [0,          0,         0, 1]
        ])

    @staticmethod
    def constance_curve_matrix_x(theta, r):
        """ 恒定曲率弯曲矩阵（绕X轴） """
        theta = theta * np.pi/180.0
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        return np.array([  # 包含曲率补偿的变换矩阵
            [1, 0,         0,          0],
            [0, cos_theta, -sin_theta, -r + r*cos_theta],  # Y方向补偿
            [0, sin_theta, cos_theta,  r*sin_theta],       # Z方向补偿
            [0, 0,         0,          1]
        ])
    
    @staticmethod
    def transl(x,y,z):
        """ 纯平移矩阵 """
        return np.array([  # 标准平移变换矩阵
            [1,0,0,x],
            [0,1,0,y],
            [0,0,1,z],
            [0,0,0,1]
        ])
    
    @staticmethod
    def transformation_matrix_y(theta, d):
        """ Y轴旋转平移矩阵 """
        theta = theta * np.pi/180.0
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        return np.array([  # 标准Y轴旋转矩阵
            [ cos_theta, 0, sin_theta, 0],
            [ 0,         1, 0,         d],
            [-sin_theta, 0, cos_theta, 0],
            [ 0,         0, 0,         1]
        ])

# ====================== 主程序 ======================
if __name__ == '__main__':
    needle = Needle()  # 创建导管对象
    needle.forward_kinematics()  # 计算运动学
    needle.calculate_shape()  # 生成形状点云

    # 创建3D可视化
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(needle.catheter_points[:,0],  # X坐标
               needle.catheter_points[:,1],  # Y坐标
               needle.catheter_points[:,2])  # Z坐标
    
    # 设置坐标轴标签
    ax.set_xlabel('X(mm)')
    ax.set_ylabel('Y(mm)')
    ax.set_zlabel('Z(mm)')
    
    # 设置显示范围
    ax.set_xlim([-50,50])
    ax.set_ylim([-50,50])
    ax.set_zlim([0,100])

    plt.show()