directory_names = dict(save_dir="/SCRATCH2/machiraj/quantize_mats/",)

dataset_paths = dict(
    cifar10="/SCRATCH2/machiraj/datasets/cifar10/",
    cifar100="/SCRATCH2/machiraj/datasets/cifar100/",
    imagenet_val="/datasets2/ImageNet2012/val/",
    imagenet_train="/datasets2/ImageNet2012/train/",
)

model_paths = dict(
    cifar10="/SCRATCH2/machiraj/robust_models/CIFAR10/",
    cifar100="/SCRATCH2/machiraj/robust_models/CIFAR100/",
    imagenet="/SCRATCH2/machiraj/robust_models/IMAGENET/",
)

wandb_config = dict(entity="harshitha-machiraju", project="mufia", reinit=True,)
