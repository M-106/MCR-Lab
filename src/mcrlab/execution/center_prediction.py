import json
import torch
from torch.utils.data import Dataset


# ---------------------------------
# > Dataset for Center Prediction <
# ---------------------------------

# FIXME, does this work?

class ManholeCenterDataset(Dataset):
    """
    PyTorch Dataset mapping sliced/partial manhole point clouds to 3D GT center coordinates.
    """
    def __init__(self, data_loader, gt_json_path, max_points=512, transform=None):
        """
        Args:
            data_loader: Data loader supplying 3D point cloud patches.
            gt_json_path (str): Path to JSON file formatted with 'pointcloud-id'.
            max_points (int): Fixed number of points sampled per manhole batch.
            transform (callable, optional): Optional data augmentations.
        """
        self.samples = []
        self.max_points = max_points
        self.transform = transform

        with open(gt_json_path, "r", encoding="utf-8") as f:
            gt_data = json.load(f)

        # Lookup table using 'pointcloud-id'
        gt_lookup = {}
        for entry in gt_data:
            key = f"{entry.get('dataset', '')}_{entry.get('pointcloud-id', '')}"
            gt_lookup[key] = entry.get("centers", [])

        point_cloud_paths = data_loader.dataset.point_cloud_paths

        for idx, batch in enumerate(data_loader):
            pc_tensor = batch[0]
            pc_np = pc_tensor.squeeze(0).cpu().numpy() if hasattr(pc_tensor, "numpy") else np.array(pc_tensor)

            _, cur_pc_name = os.path.split(point_cloud_paths[idx])
            cur_pc_name = ".".join(cur_pc_name.replace("preprocessed_patch_", "").split(".")[:-1])
            cur_pc_id = cur_pc_name.split("_")[0]
            cur_dataset = getattr(data_loader.dataset, "dataset_name", "whu")

            manhole_mask = (pc_np[:, 3] == 1)
            manhole_xyz = pc_np[manhole_mask, :3]

            if len(manhole_xyz) < 5:
                continue

            lookup_key = f"{cur_dataset}_{cur_pc_id}"
            gt_centers = gt_lookup.get(lookup_key, [])

            if not gt_centers:
                continue

            # Convert JSON centers to numpy array [K, 3]
            centers_arr = np.array([[c["x"], c["y"], c["z"]] for c in gt_centers if c is not None], dtype=np.float32)

            if len(centers_arr) == 0:
                continue

            # If multiple centers exist in the global point cloud, pair with the center closest to the current patch points
            patch_mean = np.mean(manhole_xyz, axis=0)
            distances = np.linalg.norm(centers_arr - patch_mean, axis=1)
            closest_idx = np.argmin(distances)
            
            # Distance threshold (15m): filter out centers that belong to other patches far away
            if distances[closest_idx] < 15.0:
                target_center = centers_arr[closest_idx]
                
                self.samples.append({
                    "points": manhole_xyz,
                    "target_center": target_center,
                    "pc_id": cur_pc_id
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        pts = sample["points"]
        target = sample["target_center"]

        # Resample or pad to fixed size [max_points, 3]
        if np.random.random() > 0.6:
            num_pts = len(pts)
            if num_pts >= self.max_points:
                choice = np.random.choice(num_pts, self.max_points, replace=False)
                pts_sampled = pts[choice]
            else:
                choice = np.random.choice(num_pts, self.max_points - num_pts, replace=True)
                pts_sampled = np.concatenate([pts, pts[choice]], axis=0)
        else:
            pts_sampled = pts

        # FIXME
        # Remove random parts of the manhole (from one or two sides)
        # ...

        pts_tensor = torch.from_numpy(pts_sampled).float()
        target_tensor = torch.from_numpy(target).float()

        if self.transform:
            pts_tensor = self.transform(pts_tensor)

        return pts_tensor, target_tensor


# ---------
# > Model <
# ---------





# ------------
# > Pipeline <
# ------------





























