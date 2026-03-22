# -----------
# > Imports <
# -----------
import open3d as o3d



def point_cloud_info(point_cloud):
    print(point_cloud)



def visualize_point_cloud(point_cloud):
    o3d.visualization.draw_geometries([point_cloud])







