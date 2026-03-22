# -----------
# > Imports <
# -----------
import os

import numpy as np
import laspy
import open3d as o3d
from torch.utils.data import Dataset, DataLoader



# ---------
# > Utils <
# ---------
def load_point_cloud(path, is_triangle_mesh=False):
    if path.endwith(".las") or path.endwith(".laz"):
        las = laspy.read(path)
        points = np.vstack((las.x, las.y, las.z)).transpose()

        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points)

        if hasattr(las, 'red'):
            colors = np.vstack((las.red, las.green, las.blue)).transpose()
            point_cloud.colors = o3d.utility.Vector3dVector(colors / 65535.0)

        # FIXME -> and other informations!?
    else:
        if is_triangle_mesh:
            point_cloud = o3d.io.read_triangle_mesh(path)
            point_cloud.compute_vertex_normals()
        else:
            point_cloud = o3d.io.read_point_cloud(path)
    
    return point_cloud



def save_point_cloud(path, point_cloud):
    if path.endwith(".las") or path.endwith(".laz"):
        header = laspy.LasHeader(point_format=3, version="1.2")
        export_las = laspy.LasData(header)

        export_las.x = np.asarray(point_cloud.points)[:, 0]
        export_las.y = np.asarray(point_cloud.points)[:, 1]
        export_las.z = np.asarray(point_cloud.points)[:, 2]

        if point_cloud.has_colors():
            colors = np.asarray(point_cloud.colors) * 65535
            export_las.red = colors[:, 0].astype(np.uint16)
            export_las.green = colors[:, 1].astype(np.uint16)
            export_las.blue = colors[:, 2].astype(np.uint16)

        # FIXME -> and other informations?!

        export_las.write(path)
    else:
        success = o3d.io.write_point_cloud(path, point_cloud)

        if success:
            print("Saving point cloud was successfull.")
        else:
            print("Saving point cloud was NOT successfull.")



class ParisLille3DDataset(Dataset):
    def __init__(self, path, testdata=False, transform=None):
        self.path = path
        self.testdata = testdata
        self.transform = transform

        if self.testdata:
            self.path = os.path.join(self.path, "test_10_classes")
        else:
            self.path = os.path.join(self.path, "training_10_classes")

        self.point_cloud_paths = []
        for cur_file in os.path.listdir(self.path):
            self.point_cloud_paths.append(os.path.join(self.path, cur_file))

        print(f"Uses {len(self.point_cloud_paths)} point clouds.")

    def __len__(self):
        return len(self.point_cloud_paths)

    def __getitem__(self, idx):
        point_cloud = load_point_cloud(self.point_cloud_paths[idx])

        if self.transform:
            point_cloud = self.transform(point_cloud)

        return point_cloud




def get_paris_data_loader(path, testdata=False, transform=None,
                          batch_size=32, shuffle=True, num_workers=4):
    dataset = ParisLille3DDataset(path=path, testdata=testdata, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)












