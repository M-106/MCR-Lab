# -----------
# > Imports <
# -----------
import numpy as np
import torch
import open3d as o3d



# ----------
# > Helper <
# ----------
def numpy_to_torch_tensor(numpy_arr, as_float=True):
    if numpy_arr is None or isinstance(numpy_arr, torch.Tensor):
        return numpy_arr
    
    if as_float:
        return torch.from_numpy(numpy_arr).float()
    else:
        return torch.from_numpy(numpy_arr)
    


def torch_tensor_to_numpy(tensor, as_float=True):
    if tensor is None or isinstance(tensor, np.ndarray):
        return tensor
    
    tensor = tensor.detach().cpu().numpy()
    
    if as_float:
        return tensor.astype(np.float32, copy=False)
    else:
        return tensor.astype(np.int32, copy=False)



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



# ------------------------------
# > Own Point Cloud Data Class <
# ------------------------------
class PointCloudTensor(object):
    def __init__(self, coordinates, colors=None, 
                 intensities=None, normals=None, 
                 labels=None, is_torch_tensor=False):
        self.coordinates = coordinates
        self.colors = colors
        self.intensities = intensities
        self.normals = normals
        self.labels = labels

        # can only be numpy or torch.Tensor
        self.is_torch_tensor = is_torch_tensor

    def to_torch(self, as_copy=False):
        coordinates_ = ensure_2_dims(numpy_to_torch_tensor(self.coordinates, as_float=True))
        colors_ = ensure_2_dims(numpy_to_torch_tensor(self.colors, as_float=True))
        intensities_ = ensure_2_dims(numpy_to_torch_tensor(self.intensities, as_float=True))
        normals_ = ensure_2_dims(numpy_to_torch_tensor(self.normals, as_float=True))
        labels_ = ensure_2_dims(numpy_to_torch_tensor(self.labels, as_float=False))

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
        coordinates_ = ensure_2_dims(torch_tensor_to_numpy(self.coordinates, as_float=True))
        colors_ = ensure_2_dims(torch_tensor_to_numpy(self.colors, as_float=True))
        intensities_ = ensure_2_dims(torch_tensor_to_numpy(self.intensities, as_float=True))
        normals_ = ensure_2_dims(torch_tensor_to_numpy(self.normals, as_float=True))
        labels_ = ensure_2_dims(torch_tensor_to_numpy(self.labels, as_float=False))

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

    def get_as_vector(self, include_color=False, 
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
            # FIXME -> int ok?
            o3d_pc.point["intensity"] = o3d.core.Tensor(tensor_pc.intensities, dtype=o3d.core.int32)

        if tensor_pc.colors is not None:
            o3d_pc.point["colors"] = o3d.core.Tensor(tensor_pc.colors, dtype=o3d.core.float32)

        if tensor_pc.normals is not None:
            o3d_pc.point["normals"] = o3d.core.Tensor(tensor_pc.normals, dtype=o3d.core.float32)

        if tensor_pc.labels is not None:
            o3d_pc.point["labels"] = o3d.core.Tensor(tensor_pc.labels, dtype=o3d.core.int32)

        return o3d_pc
    



