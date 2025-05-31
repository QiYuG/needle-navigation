import torch 
import numpy as np

# ====================== 损失计算函数 ======================
def calculate_loss(heart_vertices, catheter_line, threshold=25):
    """
    计算心脏表面顶点与导管线之间的惩罚性距离损失
    参数：
        heart_vertices: 心脏表面顶点坐标张量 (N,3)
        catheter_line: 导管线坐标张量 (M,3)
        threshold: 距离阈值，超过该值的点不计入计算
    返回：
        total_distance: 惩罚后的最大距离值
    """
    # 计算距离矩阵 (N,M) [6,9](@ref)
    distances = torch.cdist(heart_vertices.unsqueeze(0), catheter_line.unsqueeze(0)).squeeze(0)
    
    # 筛选并计算惩罚距离 [3](@ref)
    mask = distances <= threshold
    valid_distances = distances[mask] + 1e-8  # 防止除零
    total_distance = torch.max(1.0 / valid_distances) if valid_distances.numel() > 0 else torch.tensor(0.0)
    
    return total_distance

# ====================== 最小距离计算函数 ======================
def calculate_min_dis(heart_vertices, catheter_line):
    """
    计算心脏表面顶点与导管线之间的最小距离及其位置索引
    参数：
        heart_vertices: 心脏表面顶点坐标张量 (N,3)
        catheter_line: 导管线坐标张量 (M,3)
    返回：
        min_distance: 最小欧氏距离
        ids: 最小距离对应的顶点和导管点索引元组 (vertex_idx, catheter_point_idx)
    """
    heart_vertices = heart_vertices[torch.isfinite(heart_vertices).all(dim=1)]
    catheter_line = catheter_line[torch.isfinite(catheter_line).all(dim=1)]
    # 计算距离矩阵 [6,7](@ref)
    distances = torch.cdist(heart_vertices.unsqueeze(0), catheter_line.unsqueeze(0)).squeeze(0)
    if torch.isnan(distances).any() or torch.isinf(distances).any():
        print("distances contains NaN or Inf values.")
    # 获取最小距离和索引 [9](@ref)
    min_distance, flat_idx = torch.min(distances.view(-1), dim=0)
    idx = (flat_idx // distances.size(1), flat_idx % distances.size(1))
    
    return min_distance, idx

# ====================== 平面拟合与法线计算函数 ======================
def fit_plane_and_find_normal_line(points, threshold=150, modify=torch.tensor([0,10,-3])):
    """
    通过点云拟合平面并计算法线方向的直线参数
    参数：
        points: 输入点云坐标张量 (N,3)
        threshold: 距离阈值，用于筛选有效点
        modify: 质心位置调整偏移量
    返回：
        centroid: 调整后的质心坐标
        normal_vector: 平面法线向量
    """
    # 计算调整后的质心 [7](@ref)
    centroid = torch.mean(points, dim=0) + modify

    # 筛选有效点 [10](@ref)
    distances = torch.norm(points - centroid, dim=1)
    centered_points = points[distances < threshold] - centroid

    # SVD分解拟合平面 [7](@ref)
    _, _, Vh = torch.linalg.svd(centered_points)
    normal_vector = Vh[-1]  # 法线方向

    return centroid, normal_vector

# ====================== 向量夹角计算函数 ======================
def angle_between_vectors(vec1, vec2):
    """
    计算两个三维向量之间的夹角（单位：度）
    参数：
        vec1: 向量1
        vec2: 向量2
    返回：
        theta_degrees: 夹角角度（0-180度）
    """
    # 转换为numpy数组
    vec1 = torch.asarray(vec1)
    vec2 = torch.asarray(vec2)
    
    # 计算点积
    dot_product = torch.dot(vec1, vec2)
    
    # 计算向量模长
    norm_vec1 = torch.linalg.norm(vec1)
    norm_vec2 = torch.linalg.norm(vec2)
    
    # 计算余弦值并限制在[-1,1]范围内（防止数值误差）
    cos_theta = torch.clip(dot_product / (norm_vec1 * norm_vec2), -1.0, 1.0)
    
    # 转换为角度制
    theta = torch.arccos(cos_theta)
    theta_degrees = theta*180/torch.pi
    
    return theta_degrees

# ====================== 几何参数求解函数 ======================
def solve_parameters(centroid, normal_vector, theta_x, r_x):
    """
    根据平面参数求解几何配置参数
    参数：
        centroid: 心脏中心线质心坐标张量
        normal_vector: 心脏中心线方向张量
        theta_x: 导管x轴与世界系z轴的夹角（度）
        r_x: 输入半径参数
    返回：
        theta_z: 求解的z轴旋转角度（弧度）
        d_z: 心脏中心线相对世界坐标系z轴方向偏移量
        t: 直线参数方程中的比例系数
    """
    # 分解坐标分量 [13](@ref)
    Cx, Cy, Cz = centroid
    d_x, d_y, d_z_line = normal_vector
    
    theta_x_rad = torch.deg2rad(torch.tensor(theta_x))
    A = r_x * (1 - torch.cos(theta_x_rad))
    
    # 计算几何参数 [11](@ref)
    K = (Cx * d_y - Cy * d_x) / A
    R = torch.sqrt(d_x**2 + d_y**2)
    phi = torch.atan2(d_x, d_y)
    
    theta_z = torch.arcsin(torch.clamp(K/R, -1.0, 1.0)) - phi
    t = (A * torch.sin(theta_z) - Cx) / (d_x + 1e-8)
    d_z = Cz + t * d_z_line - r_x * torch.sin(theta_x_rad)
    
    return theta_z, d_z, t

def get_destination(start_point, normal_vector, valve_points, step=0.1, max_steps=10000):
    """
    沿着中心线方向找到一个点，使其尽可能接近valve点云的质心
    :param start_point: 起始点 (3,)
    :param normal_vector: 中心线的方向向量 (3,)
    :param valve_points: 瓣膜点云 (N, 3)
    :param step: 每次沿中心线移动的步长
    :param max_steps: 最大迭代次数
    :return: 最优点 (3,)
    """
    # 计算valve点云的质心
    valve_centroid = torch.mean(valve_points, dim=0)

    # 初始化搜索点
    current_point = start_point.clone()
    min_distance = float('inf')
    best_point = current_point.clone()

    for _ in range(max_steps):
        # 计算当前点到valve质心的距离
        distance = torch.norm(current_point - valve_centroid)

        # 如果找到更小的距离，则更新最优点
        if distance < min_distance:
            min_distance = distance
            best_point = current_point.clone()
        else:
            # 如果距离开始增大，停止搜索
            break

        # 沿着中心线方向移动
        current_point += step * normal_vector

    return best_point

def calculate_potential(end_point, end_direction, distances, destination, normal_vector, k_attract, k_repel, k_direction, epsilon):
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
   
    position_bias = torch.norm(end_point - destination)

    attract_potential = 0.5 * k_attract * position_bias ** 2

    repel_potential = k_repel / (distances ** 2 + epsilon)
    
    direction_bias = torch.norm(end_direction - normal_vector)
    
    direction_potential = k_direction * direction_bias ** 2
    
    return attract_potential, repel_potential, direction_potential