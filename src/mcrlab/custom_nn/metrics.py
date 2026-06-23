# -----------
# > Imports <
# -----------
from abc import ABC, abstractmethod
import copy

import numpy as np
import torch.nn as nn


# ----------
# > Getter <
# ----------
def get_metrics_aggregator(name:str, prename="", check_valid=True, **kwargs):
    name = name.lower()

    metrics_aggregator = None

    if name == "simple_loss":
        metrics_aggregator = SimpleAccLossAggregator(prename=prename, **kwargs)
    # ...

    if check_valid:
        if metrics_aggregator is None:
            raise ValueError(f"Could not create Metrics-Aggregator with name '{name}'.")

    return metrics_aggregator


# ---------------
# > Metrics Obj <
# ---------------
class MetricsAggregator(ABC):
    def __init__(self, prename):
        self.prename = prename

        if self.prename is None:
            self.prename = ""

    @abstractmethod
    def epoch_start(self, new_epoch):
        pass

    @abstractmethod
    def append(self, loss, inputs, labels, preds):
        pass

    @abstractmethod
    def epoch_end(self):
        pass

    @abstractmethod
    def get_metrics(self, with_prename=False):
        pass

    @abstractmethod
    def get_cur_metrics(self, with_prename=False):
        pass

    @abstractmethod
    def have_new_best_metric(self):
        pass

# Make per epoch, init before main loop
class SimpleAccLossAggregator(MetricsAggregator):
    def __init__(self, prename=""):
        super().__init__(prename=prename)
        self.cur_epoch = 1
        self.losses = []
        # self.all_losses = []
        self.last_loss = 999
        self.best_metric = float(inf)
        self.new_best_metric = False

    def epoch_start(self, new_epoch):
        self.cur_epoch = new_epoch
        self.epoch_losses = []

    def append(self, loss, inputs, labels, preds):
        self.epoch_losses.append(loss)
        self.last_loss = loss

    def epoch_end(self):
        avg_loss = np.array(self.epoch_losses).mean()
        self.losses.append(avg_loss)
        # self.all_losses += copy.deepcopy(self.epoch_losses)

        # self.cur_epoch += 1
        if avg_loss < self.best_metric:
            self.best_metric = avg_loss
            self.new_best_metric = True
        else:
            self.new_best_metric = False

    def get_metrics(self, with_prename=False):
        # avg_loss = np.mean(np.array(self.losses))
        if with_prename:
            prename = self.prename
            if len(self.prename) > 0:
                prename += "_"
        else:
            prename = ""

        return {
            # "all_losses": self.all_losses,
            f"{prename}_loss": self.losses,
            # "avg_loss": avg_loss
        }

    def get_cur_metrics(self, with_prename=False):
        if with_prename:
            prename = self.prename
            if len(self.prename) > 0:
                prename += "_"
        else:
            prename = ""

        return {
            # "all_losses": self.all_losses,
            f"{prename}_loss": self.last_loss,
            # "avg_loss": avg_loss
        }

    def have_new_best_metric(self):
        return self.new_best_metric





