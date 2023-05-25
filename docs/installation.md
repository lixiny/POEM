## :rocket: Installation

Follow the instructions below to setup the environment and dependencies.

&nbsp;

### Create a new conda env: POEM

```shell
$ conda env create -f environment.yml
$ conda activate POEM
```

### Install PyTorch 1.11.0 + cuda 11.3 (official pip)

```shell
$ pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
```

### Install dependencies

```shell
$ pip install -r requirement.txt
```

### Install thirdparty libraries

#### pytorch3d (only works for Python 3.8, Pytorch 1.11.0, Cuda 11.3)

```shell
$ pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu113_pyt1110/download.html
```

The above wheel only works for python 3.8, pytorch 1.11.0, and cuda 11.3.  
If your environment is different, you can build the pytorch3d from source, e.g.

```shell
# pip install git+https://github.com/facebookresearch/pytorch3d@v0.7.2
```

#### manotorch

```shell
$ pip install git+https://github.com/lixiny/manotorch.git@v0.0.2
```

#### neural_render

```shell
$ pip install git+https://github.com/KailinLi/neural_renderer.git
```

#### DexYCB: dex_ycb_toolkit

```shell
# clone the repo:
$ cd thirdparty
$ git clone -recursive https://github.com/NVlabs/dex-ycb-toolkit.git

# create a __init__.py in dex_ycb_toolkit
$ touch ./dex-ycb-toolkit/dex_ycb_toolkit/__init__.py

# install the repo (inside ./thirdparty):
$ pip install ./dex-ycb-toolkit

# check install success:
$ python -c "from dex_ycb_toolkit.dex_ycb import DexYCBDataset, _YCB_CLASSES"
```

#### OakInk: oikit (v1.1.0)

```shell
# clone the repo:
$ cd thirdparty
$ git clone https://github.com/oakink/OakInk.git

# install the repo
$ pip install ./OakInk

# check install success
$ python -c "from oikit.oi_image import OakInkImage"
```

#### Deformable (optional, only required for [MvP](https://arxiv.org/pdf/2111.04076.pdf) model)

Download MvP's **Deformable operation** code [:link: Deformable.zip](https://drive.google.com/file/d/1IEboRin84_HuDzP6Ucu3jqhCas1YHsiD/view?usp=share_link), unzip, and copy into `./thirdparty`.  
This Deformable archive is a backup of the `mvp/lib/models/ops` from MvP's [source page](https://github.com/sail-sg/mvp/tree/8b2ccc576a450841a5b344597cec26e5ac77eaf7/lib/models/ops).

```shell
$ cd thirdparty/Deformable
$ sh ./make.sh

# check install success
$ python -c "import torch; import Deformable as DF; print(DF.__file__)"
> ${CONDA_ENVS_DIRS}/POEM/lib/python3.8/site-packages/Deformable.cpython-38-x86_64-linux-gnu.so
```
