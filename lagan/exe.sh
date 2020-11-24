#!/bin/sh
# export CUDA_VISIBLE_DEVICES=1
# source activate <Env_name>

dataset=cifar10 # imagenet | cifar10 | svhn
arch=resnet     # ResNet

g_lr=2e-4       # type=float, default=0.0001
d_lr=2e-4       # type=float, default=0.0004
d_iters=5       # ratio of D updates for every update of G
gamma=-1        # def:-1 (float) param for a learning rate scheduler. Use negative to cancel. Typically: 0.99.

optim=adam      # Default: Adam. Type: str. Options: sgd | adam | radam.
beta=0
bsize=128       # 256 for imagenet, 128 for cifar
fid_freq=10000  # def: 10000; use < 0 to cancel computing it
model_save_step=${fid_freq}

# OPTIONS (run python main.py --help for full list):
# --img_size: Default img_size is 32
# --cont: continue training, assumes there is a backup sub-directory with timestamp.txt file in it
# --extra: activates extragradient, otherwise, vanilla GAN is used
# --momentum <int>:  Type: float; Default: 0; Used if SGD is selected
# Use `--version test` when testing. If None (default), the options yield the output directory.

# USE:
python main.py \
    --dataset ${dataset} --adv_loss hinge \
    --sample_step 5000 `# freq to store fake samples` \
    --backup_freq 1000 `# freq to backup the models` \
    --fid_freq ${fid_freq} \
    --model_save_step ${model_save_step} \
    --arch ${arch} \
    --num_workers 10 \
    --z_dim 128 \
    --g_lr ${g_lr} \
    --d_lr ${d_lr} \
    --d_iters ${d_iters} \
    --optim ${optim} `# optimizer`\
    --batch_size ${bsize} \
    --extra False \
    --lr_scheduler ${gamma} \
    --g_beta1 ${beta} --d_beta1 ${beta} \
    --lookahead_k 5 `# valid only if lookahead is activated` \
    --lookahead_alpha 0.5 `# valid only if lookahead is activated` \
    --lookahead True `# use True to activate it` \
    # --lookahead_super_slow_k 10000 \
    # --version test
    # --cont

#############################################################################################
echo "End at $(date +'%F %T')"
#############################################################################################

