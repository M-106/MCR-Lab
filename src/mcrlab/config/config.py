
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
    datapath: str



class TestConfig(BaseModel):
    datapath: str



class ModelConfig(BaseModel):
    name: str



class Config(BaseModel):
    mode: str
    train: TrainConfig
    test: TestConfig
    model: ModelConfig








