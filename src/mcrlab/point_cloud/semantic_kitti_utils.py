# -----------
# > Imports <
# -----------
import numpy as np
import open3d as o3d



# ---------
# > Utils <
# ---------
def load_bin(path):
    assert path.endswith(".bin")
    scan = np.fromfile(path, dtype=np.float32)
    scan = scan.reshape((-1, 4))  # x, y, z, intensity
    coordinates = scan[:, :3]
    intensities = scan[:, 3:4]
    return coordinates, intensities



def load_labels(path):
    labels = np.fromfile(path, dtype=np.uint32)

    semantic = labels & 0xFFFF  # lower 16 bits
    instance = labels >> 16     # upper 16 bits

    return semantic.rehsape(-1, 1), instance.reshape(-1, 1)



def load_semantic_kitti_as_o3d(bin_path, label_path):
    coordinates, intensities = load_bin(bin_path)
    semantic, instance = load_labels(label_path)

    point_cloud = o3d.t.geometry.PointCloud()
    point_cloud.point["positions"] = o3d.core.Tensor(coordinates, dtype=o3d.core.Dtype.Float32)
    point_cloud.point["intensity"] = o3d.core.Tensor(intensities, dtype=o3d.core.Dtype.Float32)  # or uint8?
    point_cloud.point["classes"] = o3d.core.Tensor(semantic, dtype=o3d.core.Dtype.Int32)
    point_cloud.point["instance_classes"] = o3d.core.Tensor(instance, dtype=o3d.core.Dtype.Int32)

    return point_cloud



