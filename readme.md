# LaCViT

LaCViT adapts a pretrained vision transformer to a labelled image dataset in two stages. It first tunes the representation with a label-aware contrastive objective, then trains a classification head using that representation. The repository also includes a conventional cross-entropy baseline. [Paper](https://arxiv.org/abs/2303.18013).

## Prepare the environment and data

The original environment used **Python 3.9, PyTorch 1.12.0 and timm 0.3.2**. Install the matching torchvision build for your CUDA environment, then install the remaining dependencies. The training script checks the timm version. Apply the [upstream timm compatibility fix](https://github.com/rwightman/pytorch-image-models/issues/420#issuecomment-776459842) when using this combination.

```bash
python -m pip install -r requirements.txt
```

- Extract the authors' [backbone checkpoints](https://drive.google.com/file/d/17Wgv3k8SADnkyNp5HZmy_VDgXQwiG7Sk/view?usp=sharing) under `checkpoints/`. The example below uses `simmim_finetune__vit_base.pth` (two underscores).
- The CIFAR example uses the torchvision dataset downloader and `./datasets/`. Other training options include `cifar10`, `flowers102`, `oxfordpet` and `bird`. For `bird`, obtain [CUB-200-2011](https://www.vision.caltech.edu/datasets/cub_200_2011/) and place the class folders under `datasets/cal_bird/CUB_200_2011/images/`.

## Train SimMIM on CIFAR-100

Run from the repository root. The first stage trains the encoder and writes `cifar100last.pth` to the specified exchange directory. The second stage opens that file to train and evaluate the task head.

```bash
python main_LaCViT.py --model simmim_base_vit --method LaCViT --dataset cifar100 \
  --data_folder ./datasets --simmim_chkpt ./checkpoints/simmim_finetune__vit_base.pth \
  --exchange_path ./LaCViT_simmim_exchange --log_dir ./output_dir/simmim_LaCViT/cifar100 \
  --batch_size 128 --lr 0.001 --num_workers 4 --save_freq 20 --lr_decay_rate 1e-4 \
  --accum_iter 32 --cosine --epochs 500 --temp 0.1

python main_linear.py --model simmim_base_vit --method LaCViT --dataset cifar100 \
  --ckpt ./LaCViT_simmim_exchange/cifar100last.pth \
  --lr 0.1 --batch_size 128 --epochs 100 --cosine
```

For the distributed contrastive recipe, append `--distributed True --dist_backend nccl --dist_url tcp://127.0.0.1:12355` to the first command. The script launches workers across the visible GPUs. Effective batch size is the per-GPU batch size × gradient accumulation steps × number of GPUs.

## Change the backbone or objective

| Backbone | Model name in both stages | Stage-one initialization |
| --- | --- | --- |
| SimMIM | `simmim_base_vit` | `--simmim_chkpt ./checkpoints/simmim_finetune__vit_base.pth` |
| MAE | `mae_cl_base` | `--mae_chkpt_dir ./checkpoints/mae_finetuned_vit_base.pth` |
| data2vec | `data2vec_base` | `--data2vec_chkpt ./checkpoints/data2vec_base.pth` |
| ViT | `vit_base_patch16_224` | See the [backbone constructors](models/umae_cl.py). |

Use a distinct exchange directory for each backbone and dataset, then pass its checkpoint to [main_linear.py](main_linear.py). [main_LaCViT.py](main_LaCViT.py) also exposes `SimCLR` and `Npair` as alternative objectives.

To train the SimMIM cross-entropy baseline directly:

```bash
python main_ce.py --model simmim_base_vit --dataset cifar100 \
  --simmim_chkpt ./checkpoints/simmim_finetune__vit_base.pth \
  --lr 0.01 --batch_size 128 --num_workers 1 --epochs 100 --save_freq 5 --cosine
```

## Reference and acknowledgments

Zijun Long, Zaiqiao Meng, Gerardo Aragon Camarasa and Richard McCreadie. *LaCViT: A Label-aware Contrastive Fine-tuning Framework for Vision Transformers*. [Paper](https://arxiv.org/abs/2303.18013).

Parts of this implementation adapt [SimCLR](https://github.com/google-research/simclr). The authors identify LaCViT as MIT-licensed in the original release.
