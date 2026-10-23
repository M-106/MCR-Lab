# ----------
# > Import <
# ----------
from typing import Union
from pydantic import BaseModel
import yaml



# -----------------------------
# > Config Classes and Helper <
# -----------------------------
def load_config(path):
        # auto relative path conversion -> relative to mcr-lab top folder
        # if path.startswith("./"):
        #     path = path.replace("./", "../../../")
        # elif path.startswith("/"):
        #     path = "../../.." + path

        with open(path) as file_:
            data = yaml.safe_load(file_)
        return Config(**data)



class TrainConfig(BaseModel):
    batch_size: int
    epochs: int
    learning_rate: float
    exp_name: str



class TestConfig(BaseModel):
    batch_size: int



class ModelConfig(BaseModel):
    name: str
    encoder: str
    check_point_path: Union[str, None]



class DataConfig(BaseModel):
    name: str
    path: str
    name_2: str
    path_2: str
    preprocessed: bool
    type: str
    heatmap_path: Union[str, None]
    used_heatmap_channel: int
    normalization: bool
    normalization_mode: str



class PreprocessingConfig(BaseModel):
    file_ending: str


class GTExtractionConfig(BaseModel):
    generate_2d_gt_maps: bool
    names: list
    data_paths: list
    save_path: str
    center_algorithm: str
    debugging_path: str


class CenterEvalConfig(BaseModel):
    center_extraction_method: str
    center_pred_path: str
    save_debug_plots: bool
    gt_json_file_paths: list
    min_input_points: int
    min_confidences: list
    min_confidence_peak: float
    candidate_min_points: int
    run_mode: str
    pred_save_load_mode_saving_name: str
    pred_save_load_mode_saving_path: str
    clear_pred_save_load_only_model_folder: bool
    clear_pred_save_load_folder: bool
    process_3d_mode: str


class InferenceConfig(BaseModel):
    checkpoint_path: str
    save_path: str


class Config(BaseModel):
    mode: str
    train: TrainConfig
    test: TestConfig
    inference: InferenceConfig
    preprocessing: PreprocessingConfig
    gt_extraction: GTExtractionConfig
    center_eval: CenterEvalConfig
    model: ModelConfig
    data: DataConfig








