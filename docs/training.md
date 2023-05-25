# :runner: Training and Evaluation

## Available models

- set `${MODEL}` as one of [POEM, MVP, PEMeshTR, FTLMeshTR]
- set `${DATASET}` as one of [DexYCBMV, HO3Dv3MV, OakInkMV]

Download the pretrained checkpoints at [:link: ckpt_release](https://drive.google.com/drive/folders/1kWziuWaqevAd9F8Pa4ASmc1krIQe-LwN?usp=share_link) and move the contents to `./checkpoint`.

## Command line arguments

- `'-g', '--gpu_id'`, visible GPUs for training, e.g. `-g 0,1,2,3`. evaluation only supports single GPU.
- `'-w', '--workers'`, num_workers in reading data, e.g. `-w 4`, recommend set `-w` equals to `-g` on HO3Dv3MV.
- `'-p', '--dist_master_port'`, port for distributed training, e.g. `-p 60011`, set different `-p` for different training processes.
- `'-b', '--batch_size'`, e.g. `-b 32`, default is specified in config file, but will be overwritten if `-b` is provided.
- `--cfg`, config file for this experiment, e.g. `--cfg config/release/${MODEL}_${DATASET}.yaml`.
- `--exp_id` specify the name of experiment, e.g. `--exp_id ${EXP_ID}`. When `--exp_id` is provided, the code requires that no uncommitted changes is remained in the git repo. Otherwise, it defaults to 'default' for training and 'eval\_{cfg\*fname}' for evaluation. All results will be saved in `exp/${EXP_ID}_{timestamp}`.
- `--reload`, specify the path to the checkpoint (.pth.tar) to be loaded.

## Evaluation

Specify the `${PATH_TO_CKPT}` to `./checkpoint/${MODEL}_${DATASET}/checkpoint/{xxx}.pth.tar`. Then, run:

```shell
# use "--eval_extra" for extra evaluation.
#   "auc"        compute AUC in each batch.
#   "draw"       draw the predicted mesh in each batch.

$ python scripts/eval.py --cfg config/release/${MODEL}_${DATASET}.yaml -g 0 -b 8 --reload ${PATH_TO_CKPT}
```

The evaluation results will be saved at `exp/${EXP_ID}_{timestamp}/evaluations`.

## Training

```shell
$ python scripts/train_ddp.py --cfg config/release/${MODEL}_${DATASET}.yaml -g 0,1,2,3 -w 16
```

### Tensorboard

```shell
$ cd exp/${EXP_ID}_{timestamp}/runs/
$ tensorboard --logdir .
```

### Checkpoint

All the training checkpoints are saved at `exp/${EXP_ID}_{timestamp}/checkpoints/`
