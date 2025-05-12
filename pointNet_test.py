import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PointNetSetAbstraction(nn.Module):
    """
    PointNet++ 的 Set Abstraction 模块，用于采样、分组和特征提取
    """
    def __init__(self, npoint, radius, nsample, in_channel, mlp):
        """
        初始化 Set Abstraction 模块
        :param npoint: 采样的点数
        :param radius: 分组的半径
        :param nsample: 每组的最大点数
        :param in_channel: 输入通道数
        :param mlp: 多层感知机的通道列表
        """
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample

        # 定义 MLP 层
        layers = []
        last_channel = in_channel
        for out_channel in mlp:
            layers.append(nn.Conv2d(last_channel, out_channel, 1))
            layers.append(nn.BatchNorm2d(out_channel))
            layers.append(nn.ReLU())
            last_channel = out_channel
        self.mlp = nn.Sequential(*layers)

    def forward(self, xyz, points):
        """
        前向传播
        :param xyz: 输入点云的坐标 (B, N, 3)
        :param points: 输入点云的特征 (B, N, C)
        :return: 采样后的点云坐标和特征
        """
        # 采样点
        sampled_xyz = self.farthest_point_sample(xyz, self.npoint)  # (B, npoint, 3)

        # 分组
        grouped_xyz, grouped_points = self.group_points(xyz, points, sampled_xyz)

        # 特征提取
        new_points = self.mlp(grouped_points)  # (B, C', npoint, nsample)
        new_points = torch.max(new_points, 3)[0]  # 最大池化 (B, C', npoint)

        return sampled_xyz, new_points

    def farthest_point_sample(self, xyz, npoint):
        """
        最远点采样
        :param xyz: 输入点云的坐标 (B, N, 3)
        :param npoint: 采样点数
        :return: 采样后的点云坐标 (B, npoint, 3)
        """
        B, N, _ = xyz.shape
        centroids = torch.zeros(B, npoint, dtype=torch.long).to(xyz.device)
        distance = torch.ones(B, N).to(xyz.device) * 1e10
        farthest = torch.randint(0, N, (B,), dtype=torch.long).to(xyz.device)
        batch_indices = torch.arange(B, dtype=torch.long).to(xyz.device)

        for i in range(npoint):
            centroids[:, i] = farthest
            centroid = xyz[batch_indices, farthest, :].unsqueeze(1)  # (B, 1, 3)
            dist = torch.sum((xyz - centroid) ** 2, -1)  # (B, N)
            mask = dist < distance
            distance[mask] = dist[mask]
            farthest = torch.max(distance, -1)[1]
        return xyz[batch_indices, centroids, :]

    def group_points(self, xyz, points, sampled_xyz):
        """
        分组点云
        :param xyz: 输入点云的坐标 (B, N, 3)
        :param points: 输入点云的特征 (B, N, C)
        :param sampled_xyz: 采样点的坐标 (B, npoint, 3)
        :return: 分组后的点云坐标和特征
        """
        B, N, _ = xyz.shape
        _, npoint, _ = sampled_xyz.shape

        # 计算每个点到采样点的距离
        dist = torch.cdist(sampled_xyz, xyz)  # (B, npoint, N)
        idx = dist.argsort()[:, :, :self.nsample]  # 取前 nsample 个点的索引

        grouped_xyz = xyz.gather(1, idx.unsqueeze(-1).expand(-1, -1, -1, 3))  # (B, npoint, nsample, 3)
        grouped_xyz -= sampled_xyz.unsqueeze(2)  # 相对坐标

        if points is not None:
            grouped_points = points.gather(1, idx.unsqueeze(-1).expand(-1, -1, -1, points.shape[-1]))
            grouped_points = torch.cat([grouped_xyz, grouped_points], dim=-1)  # 拼接坐标和特征
        else:
            grouped_points = grouped_xyz

        return grouped_xyz, grouped_points


class PointNetFeatureExtractor(nn.Module):
    """
    简化版 PointNet++ 特征提取器
    """
    def __init__(self, feature_dim=128):
        super(PointNetFeatureExtractor, self).__init__()
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=3, mlp=[64, 64, 128])
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128, mlp=[128, 128, feature_dim])

    def forward(self, xyz):
        """
        前向传播
        :param xyz: 输入点云的坐标 (B, N, 3)
        :return: 提取的特征 (B, feature_dim)
        """
        B, N, C = xyz.shape
        points = None

        # 第一层 Set Abstraction
        xyz, points = self.sa1(xyz, points)

        # 第二层 Set Abstraction
        xyz, points = self.sa2(xyz, points)

        return points.squeeze(1)  # 返回特征向量


# 示例：加载点云并提取特征
if __name__ == "__main__":
    # 初始化特征提取器
    extractor = PointNetFeatureExtractor(feature_dim=128)

    # 示例点云数据 (B, N, 3)
    point_cloud = torch.rand(1, 1024, 3)  # 模拟点云

    # 提取特征
    features = extractor(point_cloud)

    # 打印特征向量
    print("Extracted Features:", features.shape)



















































# from stl import mesh
# import numpy as np
# filename_heart = './HeartModel/Heart_sim.STL'
# filename_valve = './HeartModel/Valve_sim.STL'
# stl_mesh = mesh.Mesh.from_file(filename_heart)
# # tl_mesh = mesh.Mesh.from_file('./HeartModel/Heart1.STL')
# all_vertices = np.vstack((stl_mesh.v0, stl_mesh.v1, stl_mesh.v2))
# unique_vertices = np.unique(all_vertices, axis=0)

# # sample
# # 顶点采样配置，有问题，可能出现模型拓扑结构的破坏
# '''
# sample_ratio [0.0, 1.0]：采样比例，决定采样的顶点数量，比如当其为0.5时，随机保留50%的顶点
# 目的：降低计算复杂度，防止算法偏差，优化可视化性能
# 可能性：训练时不断增强采样率，开始时鼓励探索，后期确保精度
# '''
# sample_ratio = 1
# num_samples = int(sample_ratio * len(unique_vertices))
# sampled_vertices = unique_vertices[np.random.choice(len(unique_vertices), num_samples, replace=False)]

# # 面片采样配置
# face_ratio = 1
# num_faces = int(len(stl_mesh.vectors) * face_ratio)
# sampled_faces = stl_mesh.vectors[np.random.choice(len(stl_mesh.vectors), num_faces, replace=False)]

# # 瓣膜模型加载
# valve_mesh = mesh.Mesh.from_file(filename_valve)
# # valve_mesh = mesh.Mesh.from_file('./HeartModel/Valve0.STL')
# valve_vertices = np.vstack((valve_mesh.v0, valve_mesh.v1, valve_mesh.v2))
# valve_unique_vertices = np.unique(valve_vertices, axis=0)
# valve_faces = valve_mesh.vectors
# # 瓣膜就不采样了

