from torchvision.models import *
from torchvision.utils import *
import torch
import torch
from tqdm import tqdm
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms

# fix torch seed
torch.manual_seed(42)
# fix cuda seed
torch.cuda.manual_seed(42)
# fix cudnn seed for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.cuda.empty_cache()

from torchattacks import *
import sys

sys.path.append("../")


class Evaluator:
    def __init__(self, device, model, logger=None):
        self.device = device
        self.model = model
        self.logger = logger

    def clean_accuracy(self, clean_loader):
        """ Evaluate the model on clean dataset. """
        self.model.eval()

        correct = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(clean_loader):
                data, target = data.to(self.device), target.to(self.device)
                print(batch_idx)
                # print(target.shape)
                # exit()
                output = self.model(data)
                pred = (output[0] if (type(output) is tuple) else output).argmax(
                    dim=1, keepdim=True
                )
                correct += pred.eq(target.view_as(pred)).sum().item()
                # break

        acc = correct / len(clean_loader.dataset)
        print("Clean Test Acc {:.3f}".format(100.0 * acc))
        return acc
