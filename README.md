# MCR-Lab

MCR-Lab is a pipeline for Manhole Center Regression evaluation of Deep Learning (via PyTorch) methods and classical approaches towards high-precision predictions in 3D LiDAR data (point clouds). 

This should be only a light-weight pipeline using most likely HuggingFace and 3D utils for road extraction and circle fitting.



<br><br>

---
### Setup

On Windows via Docker:
1. Start Docker Desktop and click on the console in the bottom-right
2. Navigate to the repo
    ```bash
    cd "D:\Studium\Master\Repos\MCR-Lab"
    ```
3. Build Docker (only once):
    ```bash
    docker build -f win.dev.Dockerfile -t mcrlab-img .
    ```
4. Run container:
    ```bash
    docker run --gpus all --rm -v .:/app mcrlab-img --config config/test.yaml
    # or
    docker run --gpus all -it --rm -v .:/app mcrlab-img --config config/test.yaml bash
    ```
5. Then connect with VSCode and the Docker Dev extension.
6. Run code via jupyter or with the docker console `python ./src/main.py`.

> For training directly sart train.py?

On Windows via Anaconda:
1. Download[Anaconda](https://www.anaconda.com/download/success) -> `Anaconda Distribution`, recommendation: one person installation (not all)
2. Install Anaconda via donwloaded installer -> default settings are most likely fine
3. Run Anaconda Prompt
    ```bash
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/msys2
    ```
4. Install env
    ```bash
    conda create -n mcrlab python=3.12 pip -y
    conda activate mcrlab
    cd "C:\Users\tippolito\workspace\MCR-Lab"
    # you might want to install a specific PyTroch version before
    # https://pytorch.org/get-started/locally/
    pip install -e .
    ```
5. Start env (`conda env list`) + run python -> in anconda prompt
    ```bash
    conda activate mcrlab
    cd "C:\Users\tippolito\workspace\MCR-Lab"
    # cd "D:\Studium\Master\Repos\MCR-Lab" && D:
    # pip install -e .
    mcrlab --config "./configs/config.yaml"
    ```


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


