
import sys
sys.path.append(["."])


from point_cloud_io import get_paris_data_loader
from point_cloud_inspect import point_cloud_info, visualize_point_cloud, point_cloud_metrics


data_loader = get_paris_data_loader("../data/paris_lille_3d", testdata=False, transform=None,
                                    batch_size=1, shuffle=False, num_workers=1)


for batch in data_loader:
    point_cloud = batch[0]
    visualize_point_cloud(point_cloud)

