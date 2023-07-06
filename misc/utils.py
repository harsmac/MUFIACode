import numpy as np
import torch
import torch.nn as nn


class NormalizeLayer(nn.Module):
    def __init__(self, mean, std):
        super(NormalizeLayer, self).__init__()
        self.register_buffer("mean", torch.Tensor(mean))
        self.register_buffer("std", torch.Tensor(std))

    def forward(self, input):
        # Broadcasting
        mean = self.mean.reshape(1, 3, 1, 1)
        std = self.std.reshape(1, 3, 1, 1)
        return (input - mean) / std


# For multiple output problem
class SelectOutput(nn.Module):
    def __init__(self):
        super(SelectOutput, self).__init__()

    def forward(self, x):
        out = x[0]
        return out


def run_name_generator(param):
    last_word = "test"
    run_name = param["atk_type"] + "_" + param["threat_model"] + "_" + last_word
    return run_name


def save_name_generator(param):
    save_name = (
        ("test")
        + "_"
        + param["atk_type"]
        + "_"
        + param["threat_model"]
        + "_"
        + param["model_name"]
        + "_iter_"
        + str(param["n_epochs"])
        + "_reg_"
        + str(param["lambda_mse"])
        + "_kappa_"
        + str(param["kappa"])
        + "_retain_dc_"
        + str(param["retain_dc"])
        + "_dc_"
        + str(param["dc"])
        + ".pt"
    )

    return save_name
