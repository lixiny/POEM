## Additions to run CMR

### Data

You need prepare extra data for CMR. Download from [official site](https://drive.google.com/drive/folders/1MIE0Jo01blG6RWo2trQbXlQ92tMOaLx*?usp=sharing).

```
template
├── MANO_RIGHT.pkl
├── template.ply
└── transform.pkl
```

Link the `template` to `lib/external/cmr/template`

### Environment

- openmesh==1.2.1

### 1. torch_scatter

download **torch_scatter** from https://pytorch-geometric.com/whl/.
in my case, choose [torch-1.11.0+cu113](), and download
[torch_scatter-2.0.9-cp38-cp38-linux_x86_64.whl](https://data.pyg.org/whl/torch-1.11.0%2Bcu113/torch_scatter-2.0.9-cp38-cp38-linux_x86_64.whl).

Then, install it in your current `neo` env;

```shell
$ pip install torch_scatter-2.0.9-cp38-cp38-linux_x86_64.whl
```

Next, use pip install:

```
torch_geometric==2.0.4
torch_sparse==0.6.13
```

the `torch_sparse` package will build for a long time.

### 2. MPI-IS's `mesh` libarary

into the thirdparty folder

```shell
$ cd thirdparty
$ git clone https://github.com/MPI-IS/mesh.git
$ cd mesh
```

change the `thirdparty/mesh/requirenment.txt` to match your current conda env:

You need to make sure all the below packages' version match your current packages' version (using `pip freeze`).
Otherwise, the `make all` process will uninstalled existing package and reinstall a default version from `pip`.

```
setuptools==??
numpy==??
matplotlib==??
scipy==??
pyopengl==??
pillow==??
pyzmq
pyyaml
opencv-python==??
```

Now, in your current conda env, execute:

```shell
$ BOOST_INCLUDE_DIRS=/usr/include/boost/ make all
```

Test:
`python -c "from psbody.mesh import Mesh"`

Success !
