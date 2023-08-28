# PointNeRF To Nerfstudio

Migration for **Point-NeRF: Point-based Neural Radiance Fields** to **Nerfstudio**

## Installation Requirements

### Environment:

- Linux (tested on Ubuntu 20.04, 22.04)
  -  According to the paper, it should be fine on Ubuntu 16.04 and 18.04.
- Python 3.8+
- Nerfstudio >= 0.3.0
- CUDA 11.7 or higher.

### Dependences

Install the dependent libraries as follows:

- Install the dependent python libraries:

```shell
pip install h5py imageio scikit-image plyfile
```

- Install Nerfstudio following: 

  https://docs.nerf.studio/en/latest/quickstart/installation.html

- Install torch_scatter following:
  
  https://github.com/rusty1s/pytorch_scatter

### Install
```sh
pip install git+https://github.com/SHUzhekiNg/pointnerf2studio
```

## Run

1. generate point cloud using MVSNet with:
   ```sh
   bash pointnerf/dev_scripts/w_n360/chair_points.sh
   ```

2. run nerfstudio with:
   ```sh
   ns-train pointnerf-original --pipeline.model.path-point-cloud ....../pointnerf2studio/checkpoints/nerfsynth/chair blender-data --data ....../pointnerf2studio/pointnerf/data_src/nerf/nerf_synthetic/chair
   ```
