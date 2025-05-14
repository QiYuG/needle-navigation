import torch
import numpy as np
import matplotlib.pyplot as plt

class PotentialField(torch.nn.Module):
    def __init__(self, k_att=5.0, k_rep=100.0, infl_radius=3.0):
        super().__init__()
        # 可学习参数（通过requires_grad=True启用梯度计算）
        self.k_att = torch.nn.Parameter(torch.tensor(k_att), requires_grad=True)
        self.k_rep = torch.nn.Parameter(torch.tensor(k_rep), requires_grad=True)
        self.infl_radius = torch.tensor(infl_radius)

    def forward(self, pos, goal, obstacles):
        """计算给定位置的总势场力"""
        # 引力计算（使用L2范数）
        att_force = self.k_att * (goal - pos)
        
        # 斥力计算
        rep_force = torch.zeros_like(pos)
        for obs in obstacles:
            vec_to_obs = pos - obs
            distance = torch.norm(vec_to_obs)
            if distance < self.infl_radius and distance > 1e-5:
                rep_force += self.k_rep * (1/distance - 1/self.infl_radius) * \
                           (vec_to_obs / distance**3)
        
        return att_force + rep_force

def apf_path(start, goal, obstacles, max_iter=200, step_size=0.1):
    # 转换为PyTorch张量
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    current = torch.tensor(start, dtype=torch.float32, device=device, requires_grad=True)
    goal = torch.tensor(goal, device=device)
    obstacles = [torch.tensor(o, device=device) for o in obstacles]
    
    # 初始化势场模型
    model = PotentialField().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    path = [current.detach().cpu().numpy()]
    for _ in range(max_iter):
        optimizer.zero_grad()
        
        # 计算总势场力
        total_force = model(current, goal, obstacles)
        
        # 梯度下降更新位置
        current = current + step_size * total_force / torch.norm(total_force)
        
        # 记录路径
        path.append(current.detach().cpu().numpy())
        
        # 到达判断
        if torch.norm(current - goal) < 0.1:
            break
            
    return np.array(path)

# 参数设置
start = [0, 0]
goal = [8, 9]
obstacles = [[2, 3], [5, 5], [6, 7]]

# 路径规划
path = apf_path(start, goal, obstacles)

# 可视化
plt.figure(figsize=(8,6))
plt.scatter(*start, marker='o', s=100, label='Start')
plt.scatter(*goal, marker='*', s=200, label='Goal')
plt.scatter(np.array(obstacles)[:,0], np.array(obstacles)[:,1], 
           marker='x', s=100, c='red', label='Obstacles')
plt.plot(path[:,0], path[:,1], 'b-', lw=2, label='Path')
plt.legend()
plt.grid(True)
plt.show()