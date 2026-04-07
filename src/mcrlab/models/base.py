# -----------
# > Imports <
# -----------
from abc import ABC, abstractmethod



# --------------
# > Base Model <
# --------------
class BaseModel(ABC):
    def __init__(self, mode):
        self.mode = mode

    # def mode_to_train(self):
    #     self.mode = "train"

    # def mode_to_inference(self):
    #     self.mode = "inference"

    @abstractmethod
    def __call__(self, x):  # *args, **kwds
        pass

    @abstractmethod
    def predict(self, x):
        pass

    @abstractmethod
    def get_model(self):
        pass

    @abstractmethod
    def visualize(self, image, results):
        pass







