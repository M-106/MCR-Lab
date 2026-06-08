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
    use_amp: bool
    scaler: Union[str, None]
    checkpoint_dir: str
    criterion: str
    optimizer: str
    checkpoint_best_model: bool
    val_steps: int
    lr_scheduler: Union[str, None]



class TestConfig(BaseModel):
    metrices: list
    batch_size: int



# class InferenceConfig(BaseModel):
#     pass



class ModelConfig(BaseModel):
    name: str
    check_point_path: Union[str, None]



class DataConfig(BaseModel):
    name: str
    path: str
    preprocessed: bool
    type: str



class PreprocessingConfig(BaseModel):
    file_ending: str


class EvalExtractionConfig(BaseModel):
    names: list
    data_paths: list
    preprocessed: bool
    type: str
    save_path: str

class InferenceConfig(BaseModel):
    checkpoint_path: str
    save_path: str

class Config(BaseModel):
    mode: str
    train: TrainConfig
    test: TestConfig
    inference: InferenceConfig
    # inference: InferenceConfig
    preprocessing: PreprocessingConfig
    eval_extraction: EvalExtractionConfig
    model: ModelConfig
    data: DataConfig
    device: Union[str, None]








