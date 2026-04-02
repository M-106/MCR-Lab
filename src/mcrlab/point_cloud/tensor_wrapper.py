# -----------
# > Imports <
# -----------
import numpy as np
import torch
import open3d as o3d
from torch_geometric.data import Data



# ----------
# > Helper <
# ----------
def o3d_tensor_type_to_numpy_type(o3d_tensor):
    if o3d_tensor is None:
        return None
    
    o3d_dtype = o3d_tensor.dtype
    mapping = {
        o3d.core.Dtype.Float32: np.float32,
        o3d.core.Dtype.Float64: np.float64,
        o3d.core.Dtype.Int32: np.int32,
        o3d.core.Dtype.Int64: np.int64,
        o3d.core.Dtype.UInt8: np.uint8,
        o3d.core.Dtype.Bool: np.bool_,
    }
    return mapping.get(o3d_dtype, None)



def torch_tensor_type_to_numpy_type(torch_tensor):
    if torch_tensor is None:
        return None

    torch_dtype = torch_tensor.dtype
    mapping = {
        torch.float32: np.float32,
        torch.float64: np.float64,
        torch.int32: np.int32,
        torch.int64: np.int64,
        torch.uint8: np.uint8,
        torch.bool: np.bool_,
    }
    return mapping.get(torch_dtype, None)



def numpy_tensor_type_to_torch_type(numpy_tensor):
    if numpy_tensor is None:
        return None
    
    np_dtype = numpy_tensor.dtype
    mapping = {
        np.dtype('float32'): torch.float32,
        np.dtype('float64'): torch.float64,
        np.dtype('int32'): torch.int32,
        np.dtype('int64'): torch.int64,
        np.dtype('uint8'): torch.uint8,
        np.dtype('bool'): torch.bool,
    }
    return mapping.get(np_dtype, None)



def numpy_to_torch_tensor(numpy_arr, dtype=None):
    if numpy_arr is None or isinstance(numpy_arr, torch.Tensor):
        return numpy_arr
    
    if dtype is not None:
        return torch.from_numpy(numpy_arr).to(dtype)
    else:
        return torch.from_numpy(numpy_arr)
    


def torch_tensor_to_numpy(tensor, dtype=np.float32):
    if tensor is None or isinstance(tensor, np.ndarray):
        return tensor
    
    tensor = tensor.detach().cpu().numpy()
    
    if dtype is not None:
        return tensor.astype(dtype, copy=False)
    else:
        return tensor



def ensure_2_dims(arr):
    # Ensure (N, C)
    if arr is None:
        return None
    
    if arr.ndim == 0:
        return arr.reshape(1, 1)
    
    if arr.ndim == 1:
        # arr = arr[:, None]
        arr = arr.reshape(-1, 1)
    return arr



def map_torch_device_to_o3d(device:str) -> str:
    device = device.lower()

    if device == "cpu":
        return "CPU:0"

    if device.startswith("cuda"):
        if ":" in device:
            index = device.split(":")[1]
        else:
            index = "0"
        return f"CUDA:{index}"

    raise ValueError(f"Open3D does not support '{device}' device.")



# ------------------------------
# > Own Point Cloud Data Class <
# ------------------------------
class PointCloudTensor(object):
    def __init__(self, coordinates, colors=None, 
                 intensities=None, normals=None, 
                 labels=None, is_torch_tensor=False,
                 bevs=None, meta=None):
        self.coordinates = coordinates
        self.colors = colors
        self.intensities = intensities
        self.normals = normals
        self.labels = labels

        self.bevs = bevs  # np.array
        self.meta = meta  # dict

        # can only be numpy or torch.Tensor
        self.is_torch_tensor = is_torch_tensor

    def to_torch(self, as_copy=False):
        coordinates_ = ensure_2_dims(numpy_to_torch_tensor(self.coordinates, dtype=numpy_tensor_type_to_torch_type(self.coordinates)))
        colors_ = ensure_2_dims(numpy_to_torch_tensor(self.colors, dtype=numpy_tensor_type_to_torch_type(self.colors)))
        intensities_ = ensure_2_dims(numpy_to_torch_tensor(self.intensities, dtype=numpy_tensor_type_to_torch_type(self.intensities)))
        normals_ = ensure_2_dims(numpy_to_torch_tensor(self.normals, dtype=numpy_tensor_type_to_torch_type(self.normals)))
        labels_ = ensure_2_dims(numpy_to_torch_tensor(self.labels, dtype=numpy_tensor_type_to_torch_type(self.labels)))

        if as_copy:
            return PointCloudTensor(
                coordinates=coordinates_,
                colors=colors_,
                intensities=intensities_,
                normals=normals_,
                labels=labels_,
                is_torch_tensor=True
            )
        else:
            self.is_torch_tensor = True
            self.coordinates = coordinates_
            self.colors = colors_
            self.intensities = intensities_
            self.normals = normals_
            self.labels = labels_

    def to_numpy(self, as_copy=False):
        coordinates_ = ensure_2_dims(torch_tensor_to_numpy(self.coordinates, dtype=torch_tensor_type_to_numpy_type(self.coordinates)))
        colors_ = ensure_2_dims(torch_tensor_to_numpy(self.colors, dtype=torch_tensor_type_to_numpy_type(self.colors)))
        intensities_ = ensure_2_dims(torch_tensor_to_numpy(self.intensities, dtype=torch_tensor_type_to_numpy_type(self.intensities)))
        normals_ = ensure_2_dims(torch_tensor_to_numpy(self.normals, dtype=torch_tensor_type_to_numpy_type(self.normals)))
        labels_ = ensure_2_dims(torch_tensor_to_numpy(self.labels, dtype=torch_tensor_type_to_numpy_type(self.labels)))

        if as_copy:
            return PointCloudTensor(
                coordinates=coordinates_,
                colors=colors_,
                intensities=intensities_,
                normals=normals_,
                labels=labels_,
                is_torch_tensor=False
            )
        else:
            self.is_torch_tensor = False
            self.coordinates = coordinates_
            self.colors = colors_
            self.intensities = intensities_
            self.normals = normals_
            self.labels = labels_

    def to_device(self, device):
        """
        The input device is expected as an string in PyTroch naming.

        In PyTorch, a device is represented by torch.device, and it can specify:
        - "cpu"
        - "cuda" (NVIDIA GPU)
        - "cuda:0", "cuda:1", … (specific GPU index)
        - "mps" (Apple Silicon GPU)
        - "xpu" (Intel)
        - "privateuseone" (TPU via XLA)

        These are therefore available strings.
        """
        # if device == torch.device("cuda:1"):
        # if device.type == "cuda":

        if not self.is_torch_tensor:
            device = map_torch_device_to_o3d(device)

        moved_coordinates = self.coordinates.to(device)

        if self.labels is not None:
            moved_labels = self.labels.to(device)
        else:
            moved_labels = None

        if self.intensities is not None:
            moved_intensities = self.intensities.to(device)
        else:
            moved_intensities = None

        if self.colors is not None:
            moved_colors = self.colors.to(device)
        else:
            moved_colors = None

        if self.normals is not None:
            moved_normals = self.normals.to(device)
        else:
            moved_normals = None

        return PointCloudTensor(
            coordinates=moved_coordinates,
            labels=moved_labels,
            intensities=moved_intensities,
            colors=moved_colors,
            normals=moved_normals,
            is_torch_tensor=self.is_torch_tensor
        )
            
    def get_as_one_tensor(self, include_color=False, 
                            include_intensity=False,
                            include_normals=False,
                            include_labels=False):
        if not self.is_torch_tensor:
            self.to_torch()

        features = [self.coordinates]
        if self.colors is not None and include_color:
            features.append(self.colors)
        if self.intensities is not None and include_intensity:
            features.append(self.intensities)
        if self.normals is not None and include_normals:
            features.append(self.normals)
        if self.labels is not None and include_labels:
            features.append(self.labels)

        return torch.cat(features, dim=1)
    
    def get_labels(self):
        return self.labels
    
    def get_as_o3d(self):
        o3d_pc = o3d.t.geometry.PointCloud()
    
        tensor_pc = self.to_numpy(as_copy=True)
        o3d_pc.point["positions"] = o3d.core.Tensor(tensor_pc.coordinates, dtype=o3d.core.float32)

        if tensor_pc.intensities is not None:
            o3d_pc.point["intensity"] = o3d.core.Tensor(tensor_pc.intensities, dtype=o3d.core.uint8)

        if tensor_pc.colors is not None:
            o3d_pc.point["colors"] = o3d.core.Tensor(tensor_pc.colors, dtype=o3d.core.float32)

        if tensor_pc.normals is not None:
            o3d_pc.point["normals"] = o3d.core.Tensor(tensor_pc.normals, dtype=o3d.core.float32)

        if tensor_pc.labels is not None:
            o3d_pc.point["labels"] = o3d.core.Tensor(tensor_pc.labels, dtype=o3d.core.int32)

        return o3d_pc

    def get_as_torch_geo_data(self, include_color=False, 
                        include_intensity=False,
                        include_normals=False,
                        include_labels=False):
        if not self.is_torch_tensor:
            self.to_torch()

        # Node positions
        pos = self.coordinates

        # Build feature matrix
        features = []
        if include_color and self.colors is not None:
            features.append(self.colors)
        if include_intensity and self.intensities is not None:
            features.append(self.intensities)
        if include_normals and self.normals is not None:
            features.append(self.normals)

        x = torch.cat(features, dim=1) if len(features) > 0 else None

        # Labels
        y = self.labels if (include_labels and self.labels is not None) else None
        # y = y.squeeze(-1)

        # Create Data object
        data = Data(pos=pos, x=x, y=y)

        return data

    # def save(self, path):
    #     o3d.t.io.write_point_cloud(path, self.get_as_o3d())
    






