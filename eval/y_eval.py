from torchattacks import *
from torchvision.models import *
from torchvision.utils import *
import torch
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
import torch
import matplotlib.pyplot as plt
import numpy as np
import sys
import clip
import wandb
import torchvision

sys.path.append("../")
from attacks.attacks import *
from attacks.attack_utils import *

# fix torch seed
torch.manual_seed(42)
# fix cuda seed
torch.cuda.manual_seed(42)
# fix cudnn seed for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

torch.cuda.empty_cache()


class Y_Evaluator:
    def __init__(self, device, model, params, logger=None):
        self.device = device
        self.model = model
        self.logger = logger

        # Done only once here
        self.model.eval()
        self.params = params

        self.lpips_metric = LearnedPerceptualImagePatchSimilarity(
            net_type="vgg", reduction="sum", normalize=True
        ).to(self.device)
        self.ssim_metric = StructuralSimilarityIndexMeasure(
            data_range=1.0, reduction="sum"
        ).to(self.device)
        self.psnr_metric = PeakSignalNoiseRatio(data_range=1.0, reduction="sum").to(
            self.device
        )

        self.clip_model, self.clip_preprocess = clip.load("RN50", device=self.device)

    def attack_model(self):

        if self.params["atk_type"] == "mufia":
            solver = FilterAttack(self.model, self.params, device=self.device)
        else:
            raise NotImplementedError

        return self.process_data(solver)

    def plot_y_avg(self, y_quantize_all):
        # get average of y_quantize_all
        y_quantize_all_avg = y_quantize_all.cpu().numpy()
        y_quantize_all_mean = np.mean(y_quantize_all_avg, axis=0)
        y_quantize_all_mean = y_quantize_all_mean[0, 0, :, :]

        fig = plt.figure()
        plt.imshow(y_quantize_all_mean, cmap="YlGnBu")
        plt.colorbar()
        self.logger.log({"y_quantize_all_avg": fig})

        y_quantize_all_median = np.median(y_quantize_all_avg, axis=0)
        y_quantize_all_median = y_quantize_all_median[0, 0, :, :]

        fig = plt.figure()
        plt.imshow(y_quantize_all_median, cmap="YlGnBu")
        plt.colorbar()
        self.logger.log({"y_quantize_all_median": fig})

    def save_y_quantize_all(
        self, y_quantize_all, dataset_name, save_dir, save_name, save=True
    ):
        if dataset_name == "cifar10":
            save_dir = save_dir + "CIFAR10/"
        elif dataset_name == "imagenet":
            save_dir = save_dir + "ImageNet/"
        elif dataset_name == "cifar100":
            save_dir = save_dir + "CIFAR100/"
        else:
            raise NotImplementedError

        save_name = save_name if save_name is not None else "new.pt"
        save_name = save_dir + save_name

        if save:
            torch.save(y_quantize_all, save_name)
            print("Saved at: ", save_name)
            # save locally also
            local_name = (
                self.params["threat_model"] + "_" + self.params["dataset"] + "_.pt"
            )
            torch.save(y_quantize_all, local_name)
            return
        else:
            return save_name

    def clip_score(self, data, data_processed):
        # clip needs 224x224 images
        resize_trans = torchvision.transforms.Resize((224, 224))
        clip_model, _ = clip.load("RN50", device=self.device)
        with torch.no_grad():
            # Resize image
            data = resize_trans(data)
            data_processed = resize_trans(data_processed)

            # Encode image
            data_encoding = clip_model.encode_image(data)
            data_processed_encoding = clip_model.encode_image(data_processed)

            # Normalize
            data_encoding /= data_encoding.norm(dim=-1, keepdim=True)
            data_processed_encoding /= data_processed_encoding.norm(
                dim=-1, keepdim=True
            )

            # similarity = data_processed_encoding.cpu().numpy() @ data_encoding.cpu().numpy().T
            similarity = data_processed_encoding @ data_encoding.T
            # get diagonal
            # clip_score = np.diag(similarity)
            clip_score = torch.diag(similarity)
            return clip_score

    def calc_metrics(self, data, data_processed, target):
        new_outputs = self.model(data_processed)
        new_pred = (
            new_outputs[0] if (type(new_outputs) is tuple) else new_outputs
        ).argmax(dim=1, keepdim=True)
        y_corrects_curr = new_pred.eq(target.view_as(new_pred)).sum().item()

        # calculate IQA metrics
        y_ssim_curr = self.ssim_metric(data_processed, data).item()
        y_psnr_curr = self.psnr_metric(data_processed, data).item()
        y_lpips_curr = self.lpips_metric(data_processed, data).item()

        # calculate CLIP metrics
        clip_score_curr = self.clip_score(data, data_processed)

        return y_corrects_curr, y_ssim_curr, y_psnr_curr, y_lpips_curr, clip_score_curr

    def apply_action(self, x, y_quant):
        # convert to ycbcr
        x = dct_lib.to_ycbcr(x, data_range=1.0)
        new_y_dct, new_cb_dct, new_cr_dct = get_dct_func(
            x,
            y_quant=y_quant,
            cb_quant=None,
            cr_quant=None,
            batch_size=x.shape[0],
            device=self.device,
            block_size=y_quant.shape[-1],
        )
        new_rgb = get_rgb_func(
            new_y_dct, new_cb_dct, new_cr_dct, block_size=y_quant.shape[-1],
        )
        return new_rgb

    def process_data(self, solver):
        y_adv_acc = 0
        y_ssim = 0
        y_psnr = 0
        y_lpips = 0
        y_clip = None

        y_quantize_all = None

        for batch_idx, (data, target) in enumerate(self.params["dataloader"]):
            print("Batch: ", batch_idx, " of ", len(self.params["dataloader"]))
            data, target = data.to(self.device), target.to(self.device)

            data_processed, y_quantize = solver(data, target)

            (
                y_correct,
                y_ssim_curr,
                y_psnr_curr,
                y_lpips_curr,
                y_clip_curr,
            ) = self.calc_metrics(data, data_processed, target)

            y_adv_acc += y_correct
            y_ssim += y_ssim_curr
            y_psnr += y_psnr_curr
            y_lpips += y_lpips_curr

            if y_clip is None:
                y_clip = y_clip_curr
            else:
                y_clip = torch.cat((y_clip, y_clip_curr), dim=0)

            del (
                data_processed,
                y_ssim_curr,
                y_psnr_curr,
                y_lpips_curr,
                y_clip_curr,
            )

            # if save_mat:
            if self.params["atk_type"] != "pgd":
                if y_quantize_all is None:
                    y_quantize_all = y_quantize
                else:
                    y_quantize_all = torch.cat((y_quantize_all, y_quantize), dim=0)

            # break

        # save the y_quantize_all
        if self.params["save_mat"] and self.params["atk_type"] != "pgd":
            self.save_y_quantize_all(
                y_quantize_all,
                self.params["dataset"],
                self.params["save_dir"],
                self.params["save_name"],
            )

        # y_clip = y_clip / (batch_idx + 1)
        y_ssim = y_ssim / len(self.params["dataloader"].dataset)
        y_psnr = y_psnr / len(self.params["dataloader"].dataset)
        y_lpips = y_lpips / len(self.params["dataloader"].dataset)
        y_adv_acc = y_adv_acc / len(self.params["dataloader"].dataset)

        if self.params["atk_type"] != "pgd":
            self.plot_y_avg(y_quantize_all)

        self.logger.log(
            {
                "y_adv_acc": 100.0 * y_adv_acc,
                "n_epochs": self.params["n_epochs"],
                "y_ssim": y_ssim,
                "y_psnr": y_psnr,
                "y_clip": y_clip,
                "y_lpips": y_lpips,
                "retain_dc": self.params["retain_dc"],
                "dc_comp": self.params["dc"],
                "block_size": self.params["block_size"],
            }
        )

        # log histogram of y_clip
        # convert to list
        y_clip = y_clip.cpu().numpy().tolist()
        y_clip = [[s] for s in y_clip]
        table = wandb.Table(data=y_clip, columns=["scores"])
        self.logger.log(
            {
                "CLIP Scores": wandb.plot.histogram(
                    table, "scores", title="CLIP Score Distribution"
                )
            }
        )

        return 100.0 * y_adv_acc, y_ssim, y_psnr, y_lpips, y_clip
