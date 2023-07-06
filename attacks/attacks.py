import torch.nn as nn
from torchvision.models import *
from torchvision.utils import *
import torch
import torch.nn.functional as F
import torchjpeg.dct as dct_lib
from tqdm import tqdm
import sys

sys.path.append("../")
from attacks.attack_utils import *


class FilterAttack:
    def __init__(self, net, params, device="cuda"):
        super(FilterAttack, self).__init__()
        self.print_every = params["print_every"]
        self.lambda_reg = params["lambda_reg"]
        self.kappa = params["kappa"]
        self.device = device

        self.net = net
        # self.net.eval()
        self.n_epochs = params["n_epochs"]
        self.lr = params["lr"]
        self.block_size = params["block_size"]
        self.verbose = params["verbose"]
        self.misclassify_loss = CosMisclassifyLoss(kappa=self.kappa)
        self.similarity_loss = DCTSimilarityLoss(loss_type=params["sim_loss"])

    def __call__(self, x, targets):
        batch_size = x.shape[0]
        x, targets = x.to(self.device), targets.to(self.device)
        x = dct_lib.to_ycbcr(x, data_range=1.0)

        # initialize quantization matrices
        y_quantize = (
            torch.ones((batch_size, 1, 1, self.block_size, self.block_size))
            .detach()
            .to(self.device)
        )

        y_quantize.requires_grad = True

        y_orig_dct, _, _ = get_dct_func(
            x,
            y_quantize,
            cb_quant=None,
            cr_quant=None,
            batch_size=x.shape[0],
            device=self.device,
            block_size=self.block_size,
        )
        y_orig_dct = y_orig_dct.detach()

        optimizer = torch.optim.Adam([y_quantize], lr=self.lr, weight_decay=1e-5)

        if self.verbose:
            for epoch in tqdm(range(self.n_epochs)):
                y_quantize = self.step_solver(
                    x, targets, y_quantize, optimizer, epoch, y_orig_dct
                )
        else:
            for epoch in range(self.n_epochs):
                y_quantize = self.step_solver(
                    x, targets, y_quantize, optimizer, epoch, y_orig_dct
                )

        # Return final image
        new_y_dct, new_cb_dct, new_cr_dct = get_dct_func(
            x,
            y_quantize,
            cb_quant=None,
            cr_quant=None,
            batch_size=batch_size,
            device=self.device,
            block_size=self.block_size,
        )
        new_rgb = get_rgb_func(
            new_y_dct, new_cb_dct, new_cr_dct, block_size=self.block_size
        )
        return new_rgb, y_quantize

    def step_solver(self, x, targets, y_quantize, optimizer, epoch, y_orig_dct):
        # set gradients to zero
        y_quantize.requires_grad = True

        self.net.zero_grad()
        optimizer.zero_grad()

        new_y_dct, new_cb_dct, new_cr_dct = get_dct_func(
            x,
            y_quantize,
            cb_quant=None,
            cr_quant=None,
            batch_size=x.shape[0],
            device=self.device,
            block_size=self.block_size,
        )
        new_rgb = get_rgb_func(
            new_y_dct, new_cb_dct, new_cr_dct, block_size=self.block_size
        )
        outputs = self.net(new_rgb)

        # if tuple
        if isinstance(outputs, tuple):
            outputs = outputs[0]

        # get cosine similarity between outputs and targets
        ce_loss = self.misclassify_loss(outputs, targets)
        # increase the cosine similarity between dct
        energy_loss = self.similarity_loss(y_orig_dct, new_y_dct)

        loss = ce_loss + self.lambda_reg * energy_loss

        # loss = loss.mean()
        loss.backward()

        # update
        optimizer.step()

        if self.verbose and (
            epoch % self.print_every == 0 or epoch == self.n_epochs - 1
        ):
            # print(f"Attack Iteration {epoch}: Loss: {loss.item()}")
            print(
                {
                    "net_loss": loss.item(),
                    "epoch": epoch,
                    "misclassify loss": ce_loss.item(),
                    "energy loss": energy_loss.item(),
                }
            )

        # detach
        y_quantize.detach_()

        return y_quantize


class FilterAttackCE:
    def __init__(self, net, params, device="cuda"):
        super(FilterAttackCE, self).__init__()
        self.print_every = params["print_every"]
        self.lambda_reg = params["lambda_reg"]
        self.kappa = params["kappa"]
        self.device = device

        self.net = net
        # self.net.eval()
        self.n_epochs = params["n_epochs"]
        self.lr = params["lr"]
        self.block_size = params["block_size"]
        self.verbose = params["verbose"]
        self.misclassify_loss = torch.nn.CrossEntropyLoss()
        self.similarity_loss = DCTSimilarityLoss(loss_type=params["sim_loss"])

    def __call__(self, x, targets):
        batch_size = x.shape[0]
        x, targets = x.to(self.device), targets.to(self.device)
        x = dct_lib.to_ycbcr(x, data_range=1.0)

        # initialize quantization matrices
        y_quantize = (
            torch.ones((batch_size, 1, 1, self.block_size, self.block_size))
            .detach()
            .to(self.device)
        )

        y_quantize.requires_grad = True

        y_orig_dct, _, _ = get_dct_func(
            x,
            y_quantize,
            cb_quant=None,
            cr_quant=None,
            batch_size=x.shape[0],
            device=self.device,
            block_size=self.block_size,
        )
        y_orig_dct = y_orig_dct.detach()

        optimizer = torch.optim.Adam([y_quantize], lr=self.lr, weight_decay=1e-5)

        if self.verbose:
            for epoch in tqdm(range(self.n_epochs)):
                y_quantize = self.step_solver(
                    x, targets, y_quantize, optimizer, epoch, y_orig_dct
                )
        else:
            for epoch in range(self.n_epochs):
                y_quantize = self.step_solver(
                    x, targets, y_quantize, optimizer, epoch, y_orig_dct
                )

        # Return final image
        new_y_dct, new_cb_dct, new_cr_dct = get_dct_func(
            x,
            y_quantize,
            cb_quant=None,
            cr_quant=None,
            batch_size=batch_size,
            device=self.device,
            block_size=self.block_size,
        )
        new_rgb = get_rgb_func(
            new_y_dct, new_cb_dct, new_cr_dct, block_size=self.block_size
        )
        return new_rgb, y_quantize

    def step_solver(self, x, targets, y_quantize, optimizer, epoch, y_orig_dct):
        # set gradients to zero
        y_quantize.requires_grad = True

        self.net.zero_grad()
        optimizer.zero_grad()

        new_y_dct, new_cb_dct, new_cr_dct = get_dct_func(
            x,
            y_quantize,
            cb_quant=None,
            cr_quant=None,
            batch_size=x.shape[0],
            device=self.device,
            block_size=self.block_size,
        )
        new_rgb = get_rgb_func(
            new_y_dct, new_cb_dct, new_cr_dct, block_size=self.block_size
        )
        outputs = self.net(new_rgb)

        # if tuple
        if isinstance(outputs, tuple):
            outputs = outputs[0]

        # get cosine similarity between outputs and targets
        ce_loss = -1 * self.misclassify_loss(outputs, targets)
        # increase the cosine similarity between dct
        energy_loss = self.similarity_loss(y_orig_dct, new_y_dct)

        loss = ce_loss + self.lambda_reg * energy_loss

        # loss = loss.mean()
        loss.backward()

        # update
        optimizer.step()

        if self.verbose and (
            epoch % self.print_every == 0 or epoch == self.n_epochs - 1
        ):
            # print(f"Attack Iteration {epoch}: Loss: {loss.item()}")
            print(
                {
                    "net_loss": loss.item(),
                    "epoch": epoch,
                    "misclassify loss": ce_loss.item(),
                    "energy loss": energy_loss.item(),
                }
            )

        # detach
        y_quantize.detach_()

        return y_quantize
