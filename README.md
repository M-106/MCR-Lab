# MCR-Lab

MCR-Lab is a pipeline for Manhole Center Regression evaluation of Deep Learning (via PyTorch) methods and classical approaches towards high-precision predictions in 3D LiDAR data (point clouds). 

This pipeline using HuggingFace and 3D utils for road extraction and circle fitting.



<br><br>

---
### Setup


**Python Env Setup:**
1. Download [Anaconda](https://www.anaconda.com/download/success) -> `Anaconda Distribution`, recommendation: one person installation (not all)
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


<br><br>

---
### Data Preprocessing

1. Download Datasets
    1. Visit [WHU Urban 3D Dataset](https://whu3d.com/download.html) and download the MLS dataset
    2. [Download the SUD-ROAD dataset](https://drive.google.com/file/d/17rQKcSWQhFsOPyQK3QrYZ7cSf0pjuTQr/view?usp=sharing)
2. Preprocess Datasets 
    1. Preprocess data and create patches
        - Change these seetings in the **MCR-Lab/configs/config.yaml**
            ```yaml
            mode: preprocessing
            # ...
            data: 
                name: "whu"
                path: "Patch/to/whu3d-dataset-wo-test-label"
                type: "all"  
            ```
        - Start env (`conda env list`) + run python -> in anconda prompt
            ```bash
            conda activate mcrlab
            cd "C:\Users\tippolito\workspace\MCR-Lab"  # with your path to the downloaded repo
            mcrlab --config "./configs/config.yaml"
            ```
        - Change the settings for SUD and run again if you want to also preprocess SUD dataset
        - In some cases you must change the rights of the new created files/folder which is inside the old dataset path a new folder called `preprocessed` 
    2. Create Center Ground Truths (if wanted) 
        - Change these seetings in the **MCR-Lab/configs/config.yaml**
            ```yaml
            mode: gt_extraction
            
            gt_extraction:
                generate_2d_gt_maps: False  # must be False
                names: ["whu", "sud"]
                data_paths: ["F:/Fraunhofer_Backup/data/whu3d-dataset-wo-test-label", "F:/Fraunhofer_Backup/data/SUD-ROAD"]
                save_path: "./output"  # where to save the gt, will crate a new folder inside of here
                center_algorithm: "squares"    # choose algorithm to extract the centers: "squares", "mean", "ransac", "mesqra"
            
            data: 
                type: "test" 
            ```
        - Start env (`conda env list`) + run python -> in anconda prompt
            ```bash
            conda activate mcrlab
            cd "C:\Users\tippolito\workspace\MCR-Lab"  # with your path to the downloaded repo
            mcrlab --config "./configs/config.yaml"
            ```
        - Centers will be at `your/save/path/center_gt` and some debugging plots at: `your/save/path/center_gt/plots`
    3. Create Heatmap Ground Truths (if wanted) 
        - Change these seetings in the **MCR-Lab/configs/config.yaml**
        ```yaml
        mode: gt_extraction
        
        gt_extraction:
            generate_2d_gt_maps: True  # must be True
            names: ["whu", "sud"]
            data_paths: ["F:/Fraunhofer_Backup/data/whu3d-dataset-wo-test-label", "F:/Fraunhofer_Backup/data/SUD-ROAD"]
            save_path: "./data"  # where to save the gt, will crate a new folder inside of here
            center_algorithm: "squares"    # choose algorithm to extract the centers: "squares", "mean", "ransac", "mesqra"
            debugging_path: "./output"  # if None, will not create debugging plots (only for heatmaps)

        data: 
            type: "all"
        ```
        - Start env (`conda env list`) + run python -> in anconda prompt
            ```bash
            conda activate mcrlab
            cd "C:\Users\tippolito\workspace\MCR-Lab"  # with your path to the downloaded repo
            mcrlab --config "./configs/config.yaml"
            ```
        - It is sometimes needed to change the rights of the new created files, depending on your specific system: your heatmaps are created under: 'your/save/path/heatmaps' and your debuggings under 'your/debugging/path/heatmaps_debugging'


<br><br>


---
### Running Lib

Adjust [config.yaml](./configs/config.yaml) and start env (`conda env list`) + run python -> in anconda prompt
```bash
conda activate mcrlab
cd "C:\Users\tippolito\workspace\MCR-Lab"
# cd "D:\Studium\Master\Repos\MCR-Lab" && D:
# pip install -e .
mcrlab --config "./configs/config.yaml"
```


<br><br>

Alternative you can also adjust the path in the "command" in [tasks.json](./.vscode/tasks.json), then you can open any python file, choosing the 'mcrlab' as env in the right bottom and click top right next to the start button and choose "Python Debugger: Debug using launch.json". This will install the local lib automatically.




