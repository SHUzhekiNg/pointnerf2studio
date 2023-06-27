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

```
pip install h5py imageio scikit-image
```

- Install Nerfstudio following: 

  https://docs.nerf.studio/en/latest/index.html

- Install torch_scatter following:
  
  https://github.com/rusty1s/pytorch_scatter

### Install
```
pip install git+https://github.com/SHUzhekiNg/pointnerf2studio
```

## 
