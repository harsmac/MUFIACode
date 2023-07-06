from typing import Any
from torchvision.models import *
from torchvision.utils import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchjpeg.dct as dct_lib
import math
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


class CosMisclassifyLoss:
    def __init__(self, kappa=0.8):
        super(CosMisclassifyLoss, self).__init__()
        self.kappa = kappa

    def __call__(self, outputs, targets):
        """
        Args:
            outputs: (N, C) where C = number of classes, after softmax of network
            target: (N) where each value is 0 <= targets[i] <= C-1
        """
        cosine_similarity = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        ce_loss = torch.mean(
            cosine_similarity(outputs, F.one_hot(targets, outputs.shape[1]).float())
        )
        net_loss = torch.max(self.kappa + ce_loss, torch.tensor(0.0).to(ce_loss.device))
        return net_loss


class DCTSimilarityLoss:
    def __init__(self, loss_type="cosine"):
        super(DCTSimilarityLoss, self).__init__()
        self.loss_type = loss_type
        if self.loss_type == "cosine":
            self.loss_fn = dct_cosine_loss
        else:
            raise NotImplementedError("Loss type not implemented")

    def __call__(self, y_orig_dct, y_adv_dct):
        return self.loss_fn(y_orig_dct, y_adv_dct)


def dct_cosine_loss(y_orig_dct, y_adv_dct):
    # y_orig_dct, y_adv_dct are the dct of the y channel of the original and adversarial images
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    # flatten the dct
    y_orig_dct = y_orig_dct.view(y_orig_dct.shape[0], -1)
    y_adv_dct = y_adv_dct.view(y_adv_dct.shape[0], -1)
    # get cosine similarity between y_orig_dct and y_adv_dct
    return torch.mean(1 - cos(y_orig_dct, y_adv_dct))


# DCT FUNCTIONS
def custom_dequantize(dct, mat):
    block_size = mat.shape[-1]
    dct_blocks = dct_lib.blockify(dct, block_size)
    # num_blocks = dct_blocks.shape[2]
    # dequantized_blocks = dct_blocks * mat.repeat(1, 1, num_blocks, 1, 1)
    dequantized_blocks = (
        dct_blocks * mat
    )  # .repeat(1, 1, num_blocks, 1, 1) torch does it automatically
    # print(dequantized_blocks.shape)
    dequantized = dct_lib.deblockify(dequantized_blocks, (dct.shape[2], dct.shape[3]))
    return dequantized

# Below functions inspired from TorchJPEG library 
def custom_batch_dct(x, block_size=8):
    size = (x.shape[2], x.shape[3])
    im_blocks = dct_lib.blockify(x, block_size)
    dct_blocks = dct_lib.block_dct(im_blocks)
    dct = dct_lib.deblockify(dct_blocks, size)
    return dct


def custom_batch_idct(coeff, block_size=8):
    size = (coeff.shape[2], coeff.shape[3])
    dct_blocks = dct_lib.blockify(coeff, block_size)
    im_blocks = dct_lib.block_idct(dct_blocks)
    im = dct_lib.deblockify(im_blocks, size)
    return im


def get_dct_func(
    ycbcr,
    y_quant=None,
    cb_quant=None,
    cr_quant=None,
    batch_size=1,
    device="cuda",
    block_size=8,
):
    # For quantization matrices to be valid if its not all ones ...
    if y_quant is None:
        y_quant = torch.ones((batch_size, 1, 1, block_size, block_size)).to(device)
    if cb_quant is None:
        # print("cb_quant is None")
        cb_quant = torch.ones((batch_size, 1, 1, block_size, block_size)).to(device)
    if cr_quant is None:
        # print("cr_quant is None")
        cr_quant = torch.ones((batch_size, 1, 1, block_size, block_size)).to(device)

    # original image is in range 0,1
    ycbcr = ycbcr * 255
    ycbcr = ycbcr - 128

    y_dct = custom_batch_dct(ycbcr[:, 0:1, :, :], block_size=block_size).to(device)
    cb_dct = custom_batch_dct(ycbcr[:, 1:2, :, :], block_size=block_size).to(device)
    cr_dct = custom_batch_dct(ycbcr[:, 2:3, :, :], block_size=block_size).to(device)

    y_dct = custom_dequantize(y_dct, y_quant)
    cb_dct = custom_dequantize(cb_dct, cb_quant)
    cr_dct = custom_dequantize(cr_dct, cr_quant)

    return y_dct, cb_dct, cr_dct


def get_rgb_func(y_dct, cb_dct, cr_dct, block_size=8):
    y_idct = custom_batch_idct(y_dct, block_size=block_size)
    cb_idct = custom_batch_idct(cb_dct, block_size=block_size)
    cr_idct = custom_batch_idct(cr_dct, block_size=block_size)

    ycbcr = torch.cat([y_idct, cb_idct, cr_idct], dim=1)

    # For quantization matrices to be valid
    ycbcr = (ycbcr + 128) / 255

    rgb = dct_lib.to_rgb(ycbcr, data_range=1.0)
    rgb = torch.clamp(rgb, 0, 1)
    return rgb
