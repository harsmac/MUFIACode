"""
Main file to evaluate the model on the test set and attack the model with MUFIA. Command line arguments are used to customize the attack as shown in run.sh file.
"""

import wandb
import sys
import argparse
import torch

# fix torch seed
torch.manual_seed(42)
# fix cuda seed
torch.cuda.manual_seed(42)
# fix cudnn seed for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

torch.cuda.empty_cache()
sys.path.pop()
sys.path.insert(0, "..")
sys.path.append("../")
from misc.utils import *
from data.data import *
from models.model_loader import *
from eval.classic_eval import Evaluator
from eval.y_eval import Y_Evaluator
import configs

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--threat_model",
        type=str,
        default="std",
        help="Norm for adversarially trained model",
        choices=["linf", "l2", "std", "untrained", "prime", "augmix", "cc_sota",],
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        help="Dataset",
        choices=["cifar10", "imagenet", "cifar100",],
    )
    parser.add_argument(
        "--batch_size", default=256, type=int, help="Batch size for model evaluation",
    )

    parser.add_argument(
        "--lr", default=0.1, type=float, help="Learning rate for MUFIA attack",
    )

    parser.add_argument(
        "--lambda_reg",
        default=20.0,
        type=float,
        help="Regularization for MUFIA attack",
    )

    parser.add_argument(
        "--save_mat",
        action="store_true",
        help="If true, save the adversarial filter bank tensor for each image",
    )

    parser.add_argument(
        "--n_epochs",
        default=100,
        type=int,
        help="Number of iterations for MUFIA attack",
    )

    parser.add_argument(
        "--print_every",
        default=10,
        type=int,
        help="Print losses for every n iterations of MUFIA attack",
    )

    parser.add_argument(
        "--atk_type",
        choices=["clean", "mufia"],
        default="mufia",
        help="Evaluation/Attack type",
    )

    parser.add_argument(
        "--block_size", default=32, type=int, help="number of blocks for DCT"
    )

    parser.add_argument("--model_name", default="resnet50", type=str, help="Model name")
    parser.add_argument(
        "--parallel", action="store_true", help="Use DataParallel for model"
    )
    parser.add_argument("--verbose", action="store_true", help="Print losses in attack")

    parser.add_argument(
        "--sim_loss",
        default="cosine",
        type=str,
        help="Type of loss for similarity of DCT",
    )
    parser.add_argument(
        "--kappa",
        default=0.9,
        type=float,
        help="Hinge Loss for Cosine Mis classification",
    )

    args = parser.parse_args()
    param = vars(args)

    config_wandb = dict(defense=param)
    run_name = run_name_generator(param)
    save_name = save_name_generator(param)
    param["save_name"] = save_name
    param["save_dir"] = configs.directory_names["save_dir"]

    logger = wandb.init(
        entity=configs.wandb_config["entity"],
        project=configs.wandb_config["project"],
        reinit=configs.wandb_config["reinit"],
        name=run_name,
        config=config_wandb,
    )

    """
    Set Dataloaders and Model
    """
    data_loading = DataLoading(params=param)
    _, _, testset, testloader = data_loading.get_data()

    param["dataloader"] = testloader
    param["logger"] = logger

    model_loading = ModelLoader(params=param, device=device)
    net = model_loading.get_model()
    net = net.to(device)
    net = net.eval()

    print("Model loaded")

    """
	Evaluations and Attacks
	"""
    eval = Y_Evaluator(device, net, param, logger)
    eval.attack_model()
    wandb.finish()
