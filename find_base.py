import numpy as np
from stl import mesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from utility import fit_plane_and_find_normal_line, calculate_loss





# read STL
stl_mesh = mesh.Mesh.from_file('./HeartModel/Heart.STL')
valve_mesh = mesh.Mesh.from_file('./HeartModel/Valve.STL')


# get all vertices
all_vertices = np.vstack((stl_mesh.v0, stl_mesh.v1, stl_mesh.v2))
unique_vertices = np.unique(all_vertices, axis=0)

valve_vertices = np.vstack((valve_mesh.v0, valve_mesh.v1, valve_mesh.v2))
valve_unique_vertices = np.unique(valve_vertices, axis=0)

# downsample vertices
sample_ratio = 1
num_samples = int(sample_ratio * len(unique_vertices))
sampled_vertices = unique_vertices[np.random.choice(len(unique_vertices), num_samples, replace=False)]

# downsample vectors
face_ratio = 0.5  
num_faces = int(len(stl_mesh.vectors) * face_ratio)
sampled_faces = stl_mesh.vectors[np.random.choice(len(stl_mesh.vectors), num_faces, replace=False)]

ideal_x, ideal_y = 0,0
# search for SVC center line, already done in SW
# min_loss = 50000
# for i in range(-20,20):
#     for j in range(-20,20):
#         catheter_line = np.array([[i,j,0],
#                                 [i,j,25]])
#         loss = calculate_loss(sampled_vertices[sampled_vertices[:, 2]<25], 
#                               catheter_line,
#                               threshold=50)
#         if loss < min_loss:
#             min_loss = loss
#             ideal_x, ideal_y = i,j
# print(min_loss)
# print(ideal_x, ideal_y)
catheter_line = np.array([[ideal_x, ideal_y,0],
                          [ideal_x, ideal_y,50]])


catheter_line_x = catheter_line[:, 0]  # 取第 0 列 (X 坐标)
catheter_line_y = catheter_line[:, 1]  # 取第 1 列 (Y 坐标)
catheter_line_z = catheter_line[:, 2]  # 取第 2 列 (Z 坐标)

calculate_loss(sampled_vertices, catheter_line)

# find centroid of the valve
centroid, normal_vector = fit_plane_and_find_normal_line(valve_unique_vertices)
centroid[0] -= 0
centroid[1] += 10
centroid[2] -= 3

# visualize
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(sampled_vertices[:, 0], sampled_vertices[:, 1], sampled_vertices[:, 2], c='#4D5793', s=1)
ax.scatter(valve_unique_vertices[:, 0], valve_unique_vertices[:, 1], valve_unique_vertices[:, 2], c='r', s=1)

print(centroid)
ax.scatter(centroid[0],centroid[1],centroid[2],c='b', s=5)


ax.plot(catheter_line_x, catheter_line_y, catheter_line_z)
ax.add_collection3d(Poly3DCollection(sampled_faces, alpha=0.1, edgecolor='gray'))
ax.add_collection3d(Poly3DCollection(valve_mesh.vectors, alpha=0.1, edgecolor='r'))

# Compute points along the normal line using the parametric equation:：X = centroid + t * normal_vector
t = np.linspace(0, 30, 10)  
line_x = centroid[0] + t * normal_vector[0]
line_y = centroid[1] + t * normal_vector[1]
line_z = centroid[2] + t * normal_vector[2]
ax.plot(line_x, line_y, line_z, color='green', linewidth=2, label="Fitted Normal Line")


ax.set_xlabel('X(mm)')
ax.set_ylabel('Y(mm)')
ax.set_zlabel('Z(mm)')

plt.show()





