## :floppy_disk: Datasets

Follow the instructions below to setup the datasets and assets.  
&nbsp;

### DexYCB

Download [DexYCB](https://arxiv.org/abs/2104.04631) dataset from the [official site](https://dex-ycb.github.io), unzip and link the dataset to `./data/DexYCB`.  
The `./data/DexYCB` directory should have the following structure:

```
├── DexYCB
│   ├── 20200709-subject-01
│   ├── ...
│   ├── 20201022-subject-10
│   ├── bop
│   ├── calibration
│   └── models
```

### HO3D v3

Download [**HO3D_v3**](https://arxiv.org/abs/2107.00887) from the [official site](https://www.tugraz.at/index.php?id=40231), unzip and link the dataset to `./data/HO3D_v3`.  
The `./data/HO3D_v3` should have the following structure:

```
├── HO3D_v3
│   ├── calibration
│   ├── evaluation
│   ├── evaluation.txt
│   ├── manual_annotations
│   ├── train
│   └── train.txt
```

#### :bell: Prepare HO3D-MV (HO3D_v3 in multi-view)

Sequence: 'GPMF1' and 'SB1' in HO3D v3's official testing set are included by HO3D-MV dataset.
We provide ground-truth MANO annotation for frames in 'GPMF1' and 'SB1' sequences.

Download the [:link: HO3D_v3_manual_test_gt.zip](https://drive.google.com/file/d/12T2Td0rTNy_l6fKi7KqTp0uttrRVS9kB/view?usp=share_link), unzip, and link the folder to `./data/HO3D_v3_manual_test_gt`.

```diff
├── HO3D_v3
+── HO3D_v3_manual_test_gt
```

### OakInk

Download [OakInk](https://arxiv.org/abs/2203.15709) dataset from the [official site](https://oakink.net), unzip and link the dataset in `./data/OakInk`.  
The `./data/OakInk` directory should have the following structure:

```
├── OakInk
│   ├── OakBase
│   ├── image
│   │   ├── anno
│   │   ├── obj
│   │   └── stream_release_v2
│   └── shape
│       ├── metaV2
│       ├── OakInkObjectsV2
│       ├── oakink_shape_v2
│       └── OakInkVirtualObjectsV2
```

#### :bell: Prepare OakInk-MV (OakInk in multi-view)

Step 1. extend the OakInk data splits for multi-view setting:

```shell
$ sh prepare/extend_oakink_mv_splits.sh
```

Step 2. pack annotations into a single archive for each sample (for quick loading):

```shell
$ sh prepare/pack_oakink_mv_anno.sh
```

After the above steps, the `./data/OakInk` directory should have the following structure:

```diff
├── OakInk
│   ├── OakBase
│   ├── image
│   │   ├── anno
+   │   ├── anno_mv
+   │   ├── anno_packed_mv
│   │   ├── obj
│   │   └── stream_release_v2
│   └── shape
```

##

:heavy_check_mark: After all three datasets have been successfully processed, you will have `./data` directory as:

```
├── data
│   ├── DexYCB
│   ├── HO3D_v3
│   ├── HO3D_v3_manual_test_gt
│   └── OakInk
```

&nbsp;

## :luggage: Assets

Download `mano_v1_2.zip` from the [MANO website](https://mano.is.tue.mpg.de) (Sign in -> Download -> Models & Code),  
unzip, and copy it to `assets/mano_v1_2`:

```Shell
$ mkdir assets
$ cp -r path/to/mano_v1_2 assets/
```
