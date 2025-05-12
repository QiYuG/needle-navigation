# import open3d as o3d
# import numpy as np

# # 创建一个竖直放置的长方体
# box = o3d.geometry.TriangleMesh.create_box(width=1.0, height=3.0, depth=1.0)  # 高度 3.0，竖直方向
# box.paint_uniform_color([0, 0, 1])  # 颜色 (蓝色)
# box.translate((0, 1.5, 0))  # 初始位置 (底部对齐地面)

# # 记录长方体的当前位置信息
# current_position = np.array([0.0, 1.5, 0.0])  # (x, y, z)


# # 定义键盘回调函数
# def move_box(vis, action, mods):
#     global current_position
#     step = 0.2  # 移动步长

#     if action == 0:  # 按键按下时
#         key_to_move = {
#             ord("W"): np.array([0, step, 0]),   # W → 向上
#             ord("S"): np.array([0, -step, 0]),  # S → 向下
#             ord("A"): np.array([-step, 0, 0]),  # A → 左移
#             ord("D"): np.array([step, 0, 0]),   # D → 右移
#             ord("Q"): np.array([0, 0, step]),   # Q → 向前 (Z 方向)
#             ord("E"): np.array([0, 0, -step]),  # E → 向后 (Z 方向)
#         }

#         if action in key_to_move:
#             current_position += key_to_move[action]
#             box.translate(key_to_move[action], relative=True)
#             vis.update_geometry(box)
#             vis.poll_events()
#             vis.update_renderer()


# # 创建 Open3D 可视化窗口
# vis = o3d.visualization.VisualizerWithKeyCallback()
# vis.create_window(window_name="可移动长方体")


# # # 读取 STL 文件
# # stl_mesh = o3d.io.read_triangle_mesh("HeartModel/HeartModel.STL")

# # # 翻转模型（沿 Z 轴取反）
# # stl_mesh.vertices = o3d.utility.Vector3dVector(np.asarray(stl_mesh.vertices) * [1, 1, -1])

# # # 计算法线（可选）
# # stl_mesh.compute_vertex_normals()

# # # 可视化 STL
# # o3d.visualization.draw_geometries([stl_mesh], window_name="STL 可视化")

# # 绑定按键事件
# vis.register_key_callback(ord("W"), move_box)
# vis.register_key_callback(ord("S"), move_box)
# vis.register_key_callback(ord("A"), move_box)
# vis.register_key_callback(ord("D"), move_box)
# vis.register_key_callback(ord("Q"), move_box)
# vis.register_key_callback(ord("E"), move_box)

# # 添加几何体并启动交互
# vis.add_geometry(box)
# vis.run()
# vis.destroy_window()


# ====================== 导入库 ======================
import open3d as o3d  # 导入3D可视化库
import numpy as np    # 导入数值计算库

# ====================== 创建可移动长方体 ======================
# 创建竖直长方体（坐标系说明：Y轴垂直，X轴水平，Z轴深度）
box = o3d.geometry.TriangleMesh.create_box(
    width=1.0,   # X轴方向宽度
    height=3.0,   # Y轴方向高度（垂直方向）
    depth=1.0)    # Z轴方向深度
box.paint_uniform_color([0, 0, 1])  # 设置纯蓝色（RGB值范围0-1）
box.translate((0, 1.5, 0))  # 平移使底部对齐原点（Y方向移动高度的一半）

# 初始化位置记录（使用numpy数组便于计算）
current_position = np.array([0.0, 1.5, 0.0])  # [x, y, z]

# ====================== 键盘交互逻辑 ======================
def move_box(vis, action):
    """ 键盘回调函数 
    参数：
        vis: 可视化器对象
        action: 按键动作（0按下，1释放，2重复）
        mods: 修饰键状态
    """
    global current_position  # 声明使用全局位置变量
    step = 0.2  # 单次移动步长（单位：米）

    # 仅响应按键按下事件
    if action == 0:  
        # 定义按键映射字典（ASCII码->移动向量）
        key_to_move = {
            ord("W"): np.array([0, step, 0]),   # 沿+Y（上）
            ord("S"): np.array([0, -step, 0]),  # 沿-Y（下） 
            ord("A"): np.array([-step, 0, 0]),  # 沿-X（左）
            ord("D"): np.array([step, 0, 0]),   # 沿+X（右）
            ord("Q"): np.array([0, 0, step]),   # 沿+Z（前）
            ord("E"): np.array([0, 0, -step])   # 沿-Z（后）
        }

        # 执行移动操作
        if action in key_to_move:  # 实际上应判断按键值，此处有逻辑bug
            current_position += key_to_move[action]  # 更新坐标
            box.translate(key_to_move[action], relative=True)  # 相对平移
            vis.update_geometry(box)    # 更新几何体
            vis.poll_events()           # 处理事件队列
            vis.update_renderer()       # 刷新渲染

# ====================== 可视化系统初始化 ======================
# 创建带按键回调的可视化器
vis = o3d.visualization.VisualizerWithKeyCallback()
vis.create_window(window_name="可移动长方体")  # 创建显示窗口

# ====================== 事件绑定 ======================
# 注册6个按键的回调函数（实际应判断键值而非action）
keys = [ord("W"), ord("S"), ord("A"), ord("D"), ord("Q"), ord("E")]
for key in keys:
    vis.register_key_callback(key, move_box(box, 0))  # 绑定相同回调函数

# ====================== 场景配置与启动 ======================
vis.add_geometry(box)  # 将长方体添加到场景
vis.run()             # 启动主循环
vis.destroy_window()  # 关闭窗口后清理资源