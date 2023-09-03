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

- Install **Nerfstudio** following: 

  https://docs.nerf.studio/en/latest/quickstart/installation.html

- Install **torch_scatter** following:
  
  https://github.com/rusty1s/pytorch_scatter

### Install
```sh
pip install git+https://github.com/SHUzhekiNg/pointnerf2studio
```

## Run

1. Prepare data following the original **PointNeRF** repository's [README.md](https://github.com/Xharlie/pointnerf/tree/master#data-preparation)

   And the layout in **pointnerf2studio** should look like this.

   ```
   pointnerf2studio
   ├── pointnerf
       ├── data_src
       │   ├── dtu
           │   │   │──Cameras
           │   │   │──Depths
           │   │   │──Depths_raw
           │   │   │──Rectified
           ├── nerf
           │   │   │──nerf_synthetic
           │   │   │──nerf_synthetic_colmap
           ├── TanksAndTemple
           ├── scannet
           │   │   │──scans 
           |   │   │   │──scene0101_04
           |   │   │   │──scene0241_01
   ```

2. generate neural point cloud using MVSNet with:

   ```sh
   bash pointnerf/dev_scripts/w_<dataset_name>/<scene_name>_points.sh
   ```
   for example,
   ```sh
   bash pointnerf/dev_scripts/w_n360/chair_points.sh
   ```

3. run pointnerf2studio with blender datasets:
   ```sh
   ns-train pointnerf-original \
   --pipeline.model.path-point-cloud \
   PATH_TO_POINTNERF2STUDIO/checkpoints/nerfsynth/<scene_name>
   blender-data \
   --data \
   PATH_TO_POINTNERF2STUDIO/pointnerf/data_src/nerf/nerf_synthetic/<scene_name>
   ```



## Reference

This project is developed based on <strong>Point-NeRF: Point-based Neural Radiance Fields</strong>.

```
@inproceedings{xu2022point,
  title={Point-nerf: Point-based neural radiance fields},
  author={Xu, Qiangeng and Xu, Zexiang and Philip, Julien and Bi, Sai and Shu, Zhixin and Sunkavalli, Kalyan and Neumann, Ulrich},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={5438--5448},
  year={2022}
}
```
