# MCR-Lab

MCR-Lab is a pipeline for Manhole Center Regression evaluation of Deep Learning (via PyTorch) methods and classical approaches towards high-precision predictions in 3D LiDAR data (point clouds). 

This should be only a light-weight pipeline using most likely HuggingFace and 3D utils for road extraction and circle fitting.



<br><br>

---
### Setup

On Windows:
1. Start Docker Desktop and click on the console in the bottom-right
2. Navigate to the repo
    ```bash
    cd "D:\Studium\Master\Repos\MCR-Lab"
    ```
3. Build Docker (only once):
    ```bash
    docker build -f win.dev.Dockerfile -t mcr-lab .
    ```
4. Run container:
    ```bash
    docker run --gpus all -it --rm -v .:/app mcr-lab bash
    ```
5. Then connect with VSCode and the Docker Dev extension.

> For training directly sart train.py?



<br><br>

---
### Plan

Preparation-Phase:
- ✅ Make setup (with Docker)
- 📌 Add point-cloud loading utils
- Add point-cloud visualizing utils
- Inspect Paris-Lille-3D data (download it)
- Add Road Removing (which method is stable and easy?)
- Add BEV util + back-projection?
- Add Clustering util
Cold-Practise-Phase:
- Add unsupervised geometry method (see master repo method idea) -> using implemented utils
- Try loading Mask2Former instance sementation and SAM model via HuggingFace
- Inspect results from methods (manually -> labels not available)
- Check and label real data



<br><br>

---
### Methods

- Unsupervised Geometry Center Prediction
- [Mask2Former-Instance-Segmentation](https://github.com/M-106/AI/blob/main/docs/model_usage.md#segmentation-example) (+ RANSAC shape-fitting to get center?)
- [SAM-Segmentation + Classification-Head](https://github.com/M-106/AI/blob/main/docs/model_usage.md#segmentation-example) (manhole or not) (+ RANSAC shape-fitting to get center?)
- Point Transformer v3 (+ RANSAC shape-fitting to get center?) -> or direct regression?
    - maybe try https://huggingface.co/bryanchang/PTv3_laneline_segemenation_signal before you try use the official repo/pointcept


