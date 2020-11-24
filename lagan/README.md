# Experiments on CIFAR-10, SVHN \& ImageNet

## Dependencies
- Python 3.7 or later
- [PyTorch](https://pytorch.org/) 1.6
- \[ [Tensorflow](https://www.tensorflow.org/)   1.14.0 - for FID and IS metrics, see below \]

## Reproducing 

### CIFAR-10 \& SVHN

To reproduce our results, run `main.py` with corresponding options, 
see the `exe.sh` example script , or run:  `main.py --help` to list available options.


#### Dataset

(Recommended) The source code automatically downloads the dataset in `~/datasets/<dataset>/`, where `<dataset>` is either `cifar10` or `svhn`.

Alternatively, to specify the dataset path use `--image_path <path-to-dataset>`, assuming the dataset is in standard `torchvision.datasets` format.
If the dataset is not found, it will be downloaded.

#### Optimizer
The default optimizer--used in our experiments in the main paper is Adam. To modify it, use `--optim <SGD|Adam|Radam>`.

#### Example commands to run CIFAR-10 experiments
1. Alt-GAN:
```bash
g_lr=2e-4   	# type=float, default=0.0001
d_lr=2e-4       # type=float, default=0.0004
d_iters=5       # ratio of D updates for every update of G

optim=adam      # Default: Adam. Type: str. Options: sgd | adam | radam.
beta=0          # beta1 param for (R)Adam, default: 0
batch_size=128  # batchsize, 128 for cifar/svhn, default: 128
fid_freq=10000  # def: 10000; use < 0 to cancel computing it on the fly (models will be stored)
model_save_step=${fid_freq}

python main.py \
    --dataset cifar10 --adv_loss hinge \
    --sample_step 1000 `# freq to store fake samples` \
    --backup_freq 1000 `# freq to backup the models` \
    --fid_freq ${fid_freq} \
    --model_save_step ${model_save_step} \
    --num_workers 10 \
    --z_dim 128 \
    --g_lr ${g_lr} \
    --d_lr ${d_lr} \
    --d_iters ${d_iters} \
    --optim ${optim} `# optimizer`\
    --batch_size ${batch_size} \
    --extra False \
    --g_beta1 ${beta} --d_beta1 ${beta} \
    --lookahead False `# use True to activate it` \
    --lookahead_k 5 `# valid only if lookahead is activated` \
    --lookahead_alpha 0.5 `# valid only if lookahead is activated` \
```

2. LA-Alt-GAN:
```bash
python main.py \
    --dataset cifar10 --adv_loss hinge \
    --sample_step 1000 `# freq to store fake samples` \
    --backup_freq 1000 `# freq to backup the models` \
    --fid_freq 10000 \
    --model_save_step 10000 \
    --z_dim 128 \
    --g_lr 2e-4 --d_lr 2e-4 `# step size for G and D` \
    --d_iters ${d_iters} \
    --optim adam `# optimizer`\
    --extra False \
    --lookahead True `# use True to activate it` \
    --lookahead_k 5 `# valid only if la is activated` \
    --lookahead_alpha 0.5 `# valid only if la is activated` \
```

3. ExtraGradient:
```bash
python main.py \
    --dataset cifar10 --adv_loss hinge \
    --sample_step 1000 `# freq to store fake samples` \
    --backup_freq 1000 `# freq to backup the models` \
    --fid_freq 10000 \
    --model_save_step 10000 \
    --z_dim 128 \
    --g_lr 2e-4 --d_lr 2e-4 `# step size for G and D` \
    --d_iters ${d_iters} \
    --optim adam `# optimizer`\
    --extra True \
    --lookahead False `# use True to activate it`
```

4. LA-ExtraGradient
```bash
python main.py \
    --dataset cifar10 --adv_loss hinge \
    --sample_step 1000 `# freq to store fake samples` \
    --backup_freq 1000 `# freq to backup the models` \
    --fid_freq 10000 \
    --model_save_step 10000 \
    --z_dim 128 \
    --g_lr 2e-4 --d_lr 2e-4 `# step size for G and D` \
    --d_iters ${d_iters} \
    --optim adam `# optimizer`\
    --extra True \
    --lookahead True `# use True to activate it`
    --lookahead_k 5000 `# valid only if lookahead is activated` \
    --lookahead_alpha 0.5 `# valid only if lookahead is activated`
```


### ImageNet

Instructions to-be-updated.