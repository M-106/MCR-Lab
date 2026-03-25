
from pydantic import BaseModel
import yaml



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



class TestConfig(BaseModel):
    metrices: list



class ModelConfig(BaseModel):
    name: str



class DataConfig(BaseModel):
    name: str
    path: str



class Config(BaseModel):
    mode: str
    train: TrainConfig
    test: TestConfig
    model: ModelConfig
    data: DataConfig








