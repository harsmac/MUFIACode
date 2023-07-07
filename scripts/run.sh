# Bash script to run the experiments
CUDA_VISIBLE_DEVICES=3 python test.py --atk_type mufia --threat_model std --batch_size 32 --dataset cifar10 --n_epochs 100 --lr 0.1 --model_name resnet56 --lambda_reg 20.0 --block_size 32 --verbose --kappa 0.99
