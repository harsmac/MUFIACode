import sys
import os
import torch
import torchvision
import torch.nn as nn
from robustbench import load_model
from robustness import model_utils

# fix torch seed
torch.manual_seed(42)
# fix cuda seed
torch.cuda.manual_seed(42)

torch.cuda.empty_cache()
sys.path.append("../")
from misc import *  # needed for normalize_net
from models.cifar10_resnet import ResNet50
import configs


class MadryHelper:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std


class ModelLoader:
    def __init__(self, params, device="cuda"):
        self.device = device
        self.params = params
        self.parallel = params["parallel"]
        self.dataset = params["dataset"].lower()
        self.threat_model = params["threat_model"].lower()
        self.model_name = params["model_name"].lower()

        self.mean, self.std, self.path = self.get_dataset_stats()

    def get_model(self):
        if self.threat_model in ["untrained", "std"]:
            if self.dataset in ["cifar10", "cifar100", "cifar10c", "cifar100c"]:
                net = self.cifar_standard_models()
            elif self.dataset in ["imagenet", "imagenetc"]:
                net = self.imagenet_standard_models()
            net = self.normalize_net(
                net
            )  # robustbench adds a normalize layer so we dont need to do it here
        else:
            net = self.load_fancy_model()
        return net

    def get_custom_trained_model(self, ckpt_nm, ckpt_path):
        if self.dataset in ["cifar10", "cifar100", "cifar10c", "cifar100c"]:
            net = self.cifar_standard_models()
        elif self.dataset in ["imagenet", "imagenetc"]:
            net = self.imagenet_standard_models()
        net = self.normalize_net(net)

        ckpt_fnm = os.path.join(ckpt_path, ckpt_nm)
        sd = torch.load(ckpt_fnm)
        net.load_state_dict(sd["net"])
        return net

    def get_dataset_stats(self):
        if self.dataset == "cifar10" or self.dataset == "cifar10c":
            cifar10_mean = [0.4914, 0.4822, 0.4465]
            cifar10_std = [0.2023, 0.1994, 0.2010]
            path = configs.model_paths["cifar10"]
            return cifar10_mean, cifar10_std, path

        elif self.dataset == "cifar100" or self.dataset == "cifar100c":
            cifar100_mean = [0.5071, 0.4867, 0.4408]
            cifar100_std = [0.2675, 0.2565, 0.2761]
            path = configs.model_paths["cifar100"]
            return cifar100_mean, cifar100_std, path

        elif self.dataset == "imagenet" or self.dataset == "imagenetc":
            imagenet_mean = [0.485, 0.456, 0.406]
            imagenet_std = [0.229, 0.224, 0.225]
            path = configs.model_paths["imagenet"]
            return imagenet_mean, imagenet_std, path

    def normalize_net(self, net):
        norm_layer = NormalizeLayer(mean=self.mean, std=self.std)
        net = nn.Sequential(norm_layer, net).to(self.device)
        net.to(self.device)
        if self.parallel:
            net = torch.nn.DataParallel(net)
        return net

    def traditional_loading(self, ckpt_nm, net, key=None):
        model_path = os.path.join(self.path, self.threat_model)
        ckpt_fnm = os.path.join(model_path, ckpt_nm)
        if key is None:
            sd = torch.load(ckpt_fnm)
        else:
            sd = torch.load(ckpt_fnm)[key]
        net.load_state_dict(sd)
        return net

    def cifar_standard_models(self):
        # models with standard training but different architectures
        # https://github.com/chenyaofo/pytorch-cifar-models for cifar10 and cifar100
        pretrained_flag = False if self.threat_model in "untrained" else True
        torch.hub.set_dir(self.path)

        if self.model_name in ["resnet20", "resnet32", "resnet56"]:
            net = torch.hub.load(
                "chenyaofo/pytorch-cifar-models",
                self.dataset + "_" + self.model_name,
                pretrained=pretrained_flag,
            )

        elif self.model_name in "resnet50":
            net = ResNet50()
            if self.threat_model == "std":
                ds = MadryHelper(torch.tensor(self.mean), torch.tensor(self.std))
                self.threat_model = "std"
                net, _ = model_utils.make_and_restore_model(
                    arch=net,
                    dataset=ds,
                    resume_path=os.path.join(self.path, self.threat_model, "std.pt"),
                )
                net = net.model
        else:
            raise ValueError("Unknown model name")

        return net

    def imagenet_standard_models(self):
        # models with standard training but different architectures
        weights_flag = None if self.threat_model in "untrained" else "DEFAULT"
        torch.hub.set_dir(self.path)

        if self.model_name in ["alexnet", "resnet18", "resnet50"]:
            net = torchvision.models.get_model(self.model_name, weights=weights_flag)
        else:
            raise ValueError("Unknown ImageNet model name")

        return net

    def load_fancy_model(self):
        # models with fancy training and varying architectures
        model_path = self.path + self.threat_model + "/"

        print("NOTE: For Fancy models there is no model architecture choice !! ")

        if (
            self.dataset == "imagenetc"
            or self.dataset == "cifar10c"
            or self.dataset == "cifar100c"
        ):
            self.dataset = self.dataset[:-1]

        if self.threat_model in "prime":
            model_name = "Modas2021PRIMEResNet18"
            robustness_lib = False if self.dataset == "imagenet" else True
            ckpt_nm = "prime.pt" if self.dataset == "imagenet" else None
            threat_model = "prime" if self.dataset == "imagenet" else "corruptions"

        elif self.threat_model in "augmix":
            model_name = (
                "Hendrycks2020AugMix"
                if self.dataset == "imagenet"
                else "Hendrycks2020AugMix_ResNeXt"
            )
            robustness_lib = True
            threat_model = "corruptions"

        elif self.threat_model in "cc_sota":
            if self.dataset == "cifar100":
                model_name = "Diffenderfer2021Winning_LRR_CARD_Deck"
            elif self.dataset == "imagenet":
                model_name = "Tian2022Deeper_DeiT-B"
            elif self.dataset == "cifar10":
                model_name = "Diffenderfer2021Winning_LRR_CARD_Deck"
            else:
                raise ValueError("Unknown dataset for cc_sota")
            robustness_lib = True
            threat_model = "corruptions"

        elif self.threat_model in ["linf", "l2"]:
            # cifar10 and imagenet only available
            # For cifar10 and Imagenet, the architecture is Resnet50
            if self.dataset == "cifar100" or self.dataset == "cifar10":
                model_name = "Wang2023Better_WRN-70-16"
            model_name = "Engstrom2019Robustness"
            robustness_lib = True
            threat_model = self.threat_model.capitalize()

        elif self.threat_model in "xcit":
            model_name = "Debenedetti2022Light_XCiT-L12"
            robustness_lib = True
            threat_model = "Linf"

        else:
            raise ValueError("Unknown fancy threat model :(")

        if robustness_lib:
            net = load_model(
                model_name=model_name,
                threat_model=threat_model,
                dataset=self.dataset,
                model_dir=model_path,
            )
        else:

            if threat_model == "prime":
                # needed for prime imagenet model
                net = torchvision.models.get_model("resnet50", weights=None)
                net = self.traditional_loading(ckpt_nm, net)
                net = self.normalize_net(net)

        return net


# Write testing code here
if __name__ == "__main__":
    params = {
        "model_name": "resnet56",
        "threat_model": "std",
        "dataset": "cifar100",
        "parallel": False,
    }
    model = ModelLoader(params=params, device="cuda")
    print(model.get_model())
    from torchsummary import summary

    input_shape = (3, 32, 32)
    print(summary(model.get_model(), (input_shape)))
