# -----------
# > Imports <
# -----------
import matplotlib.pyplot as plt

from ultralytics import YOLO
from huggingface_hub import hf_hub_download

from mcrlab.image.utils import one_channel_img_to_pil_rgb_img



# ------------------------------
# > 2D Object Detection Models <
# ------------------------------
hf_hub_download
class YOLOv12:
    def __init__(self, hf_token=None, device="cpu"): 
        model_path = hf_hub_download(repo_id="zakskyfighter/RGB_BEV_Kitti_Custom_Yolov12n", filename="best.pt")
        self.model = YOLO(model_path)
        self.model.to(device)

    def __call__(self, x):
        image = one_channel_img_to_pil_rgb_img(x)
        results = self.model(image, conf=0.25, verbose=False)
        return results[0]

    def predict(self, x):
        return self(x)
    
    def visualize(self, image_input, results):
        # get RGB numpy array
        img_array = one_channel_img_to_pil_rgb_img(image_input, return_numpy=True)

        plt.figure(figsize=(10, 10))
        plt.imshow(img_array)
        ax = plt.gca()

        # YOLO Daten extrahieren
        boxes = results.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
        scores = results.boxes.conf.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy()
        names = results.names # Klassen-Namen (z.B. Car, Pedestrian)

        for box, score, cls in zip(boxes, scores, classes):
            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1
            
            # color based on class
            color = plt.cm.get_cmap('tab10')(int(cls) % 10)
            
            # draw box
            rect = plt.Rectangle((x1, y1), w, h, fill=False, edgecolor=color, linewidth=2)
            ax.add_patch(rect)
            
            # write label to the box
            label = f"{names[int(cls)]} {score:.2f}"
            ax.text(x1, y1 - 5, label, color='white', fontsize=10, fontweight='bold',
                    bbox=dict(facecolor=color, alpha=0.6, pad=1))

        plt.axis('off')
        plt.title(f"Objects found: {len(boxes)}")
        plt.tight_layout()
        plt.show()







