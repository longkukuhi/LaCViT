# LaCViT: A Label-aware Contrastive Training Framework for Vision Transformers


--------

This repo holds code the paper: LaCViT: A Label-aware Contrastive Training Framework for Vision Transformers.


## Environment
The code is tested with python 3.9, torch 1.12.0 and timm 0.3.2. Please view `requirements.txt` for more details.
Attention: timm 0.3.2 needs a [fix](https://github.com/rwightman/pytorch-image-models/issues/420#issuecomment-776459842) is needed to work with PyTorch 1.8.1+. (See https://github.com/rwightman/pytorch-image-models/issues/420#issuecomment-776459842) 

## Dataset preparation
Pytorch will download all datasets this project uses except the CUB-200-2011. You can download it from https://www.vision.caltech.edu/datasets/cub_200_2011/ .
Then extracting downloaded files to  ''./datasets/cal_bird/' . The program reads images of the CUB-200-2011 dataset from './datasets/cal_bird/CUB_200_2011/images/' .

## Train LaCViT
The training of LaCViT consists two stages, (1) label-aware contrastive training stage, (2) task head training stage.
### Prepare pre-trained weight for vision transformers
In this paper, we experimented with ViT, MAE, SimMIM, Date2vec. You can download their pretrianed weight from their github pages.
We also provide thoes pretraiend weight we used in the experiments, [the link](https://drive.google.com/file/d/17Wgv3k8SADnkyNp5HZmy_VDgXQwiG7Sk/view?usp=sharing). Extracting the zip file to ''./checkpoints/' .

### LaCViT contrastive training

#### Run SimMIM with distributed training:
```bash
python  main_LaCViT.py --batch_size 128 --lr 0.001 --model 'simmim_base_vit' \
--log_dir './output_dir/simmim_LaCViT/cifar100/' --dataset 'cifar100' \
--data_folder './datasets/' --exchange_path "./LaCViT_simmim_exchange/" \
--simmim_chkpt "./checkpoints/simmim_finetune__vit_base.pth" \
--num_workers 4 --save_freq 20 --lr_decay_rate 1e-4 \
--method "LaCViT" --accum_iter 32 --cosine --epochs 500 --temp 0.1 \
--distributed True --dist_backend 'nccl' --dist_url 'tcp://127.0.0.1:12355'\
```
- The effective batch size = `batch_size`(batch size per gpu) *`accum_iter` * number of GPUs.
- `model` specifics the name of model we use in this experiments. Options includes 'simmim_base_vit','mae_cl_base', 'vit_base_patch16_224', 'data2vec_base', for SimMIM, MAE, ViT, data2vec respectively.
- `log_dir` is the folder dir that stores the ouput log.
- `dataset` specifics the dataset. Option includes 'cifar10', 'cifar100', 'flowers102', 'oxfordpet', 'bird', for datesets 'CIFAR-10', 'CIFAR-100', 'Oxford 102 Flower', 'Oxford-IIIT pet', 'CUB-200-2011' repectively.
- `data_folder` is the folder dir that stores the datasets.
- `simmim_chkpt` specifics the dir to pretrained weight of SimMIM model.
- `method` specifics the contrastive training framework. Option includes 'LaCViT', 'SimCLR', 'Npair', for method 'our proposed LaCViT', 'SimCLR', 'N-pari-loss' respectively.
- `cosine` decay learing rate in a cosine schema.
- `epoch` specifics the number of training epochs.
- `temp` temparature value for the contrastive loss.
- `distributed` enables the distributed training.


#### Train SimMIM without distributed training:
```bash
python  main_LaCViT.py --batch_size 128 --lr 0.001 --model 'simmim_base_vit' \
--log_dir './output_dir/simmim_LaCViT/cifar100/' --dataset 'cifar100' \
--data_folder '../../tmp/datasets/' --exchange_path "./LaCViT_simmim_exchange/" \
--simmim_chkpt "./checkpoints/simmim_finetune__vit_base.pth" \
--num_workers 4 --save_freq 20 --lr_decay_rate 1e-4 \
--method "LaCViT" --accum_iter 32 --cosine --epochs 500 --temp 0.1 \
```

#### Train MAE with distributed training:
```bash
python  main_LaCViT.py --batch_size 128 --lr 0.001 --model 'mae_cl_base' \
--log_dir './output_dir/mae_LaCViT/cifar100/' --dataset 'cifar100' \
--data_folder './datasets/' --exchange_path "./LaCViT_mae_exchange/" \
--mae_chkpt_dir './checkpoints/mae_finetuned_vit_base.pth' \
--num_workers 4 --save_freq 20 --lr_decay_rate 1e-4 \
--method "LaCViT" --accum_iter 32 --cosine --epochs 500 --temp 0.1 \
--distributed True --dist_backend 'nccl' --dist_url 'tcp://127.0.0.1:12355'\
```

#### Train MAE with distributed training:
```bash
python  main_LaCViT.py --batch_size 128 --lr 0.001 --model 'data2vec_base' \
--log_dir './output_dir/data2vec_LaCViT/cifar100/' --dataset 'cifar100' \
--data_folder './datasets/' --exchange_path "./LaCViT_data2vec_exchange/" \
--data2vec_chkpt "./checkpoints/data2vec_base.pth"  \
--num_workers 4 --save_freq 20 --lr_decay_rate 1e-4 \
--method "LaCViT" --accum_iter 32 --cosine --epochs 500 --temp 0.1 \
--distributed True --dist_backend 'nccl' --dist_url 'tcp://127.0.0.1:12355'\
```

### LaCViT task head training
#### Task head training with SimMIM
```bash
python  main_linear.py --lr 0.1 --batch_size 128 --epochs 100 --model 'simmim_base_vit' \
--ckpt "./LaCViT_simmim_exchange/cifar100last.pth" --dataset 'cifar100' \
--method "LaCViT" --cosine \
```

#### Task head training with MAE
```bash
python  main_linear.py --lr 0.1 --batch_size 128 --epochs 100 --model 'mae_cl_base' \
--ckpt "./LaCViT_mae_exchange/cifar100last.pth" --dataset 'cifar100' \
--method "LaCViT" --cosine \
```

## Cross-entropy fine-tuning
#### Fine-tuned with cross-entropy using SimMIM
```bash
python main_ce.py --lr 0.01 --model 'simmim_base_vit' \
--dataset "cifar100" --batch_size 128 --num_workers 1 \
--simmim_chkpt "./checkpoints/simmim_finetune__vit_base.pth" \
--epochs 100 --save_freq 5 --cosine \
```

## Acknowledgement
Parts of the code are modified from [SimCLR](https://github.com/google-research/simclr). We appreciate the authors for making it open-sourced.

## License
LaCViT is MIT licensed.