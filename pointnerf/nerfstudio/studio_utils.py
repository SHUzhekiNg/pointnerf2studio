from nerfstudio.engine.schedulers import Scheduler, SchedulerConfig
from dataclasses import dataclass, field
from typing import Type
from torch.optim import Optimizer, lr_scheduler
try:
    from torch.optim.lr_scheduler import LRScheduler
except ImportError:
    # Backwards compatibility for PyTorch 1.x
    from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from nerfstudio.field_components.encodings import Encoding
from typing import Literal, Optional, Sequence
from jaxtyping import Float, Int, Shaped
from torch import Tensor
import torch
from ..models.rendering.diff_ray_marching import near_far_linear_ray_generation
from torch.utils.cpp_extension import load as load_cuda
import torch.nn as nn
import numpy as np
import os

@dataclass
class PointNerfSchedulerConfig(SchedulerConfig):
    """Config for multi step scheduler where lr decays by gamma every milestone"""

    _target: Type = field(default_factory=lambda: PointNerfScheduler)
    """target class to instantiate"""
    lr_decay_iters: int = 1000000
    """The maximum number of steps."""
    lr_decay_exp: float = 0.1
    """The learning rate decay factor."""


class PointNerfScheduler(Scheduler):
    """Multi step scheduler where lr decays by gamma every milestone"""

    config: PointNerfSchedulerConfig

    def get_scheduler(self, optimizer: Optimizer, lr_init: float) -> LRScheduler:
        def func(step):
            lr_l = pow(self.config.lr_decay_exp, step / self.config.lr_decay_iters)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=func)
        return scheduler
    
class PointNeRFEncoding(Encoding):
    def __init__(
        self,
        in_dim: int,
        num_frequencies: int,
        ori: bool = False,
    ) -> None:
        super().__init__(in_dim)
        self.num_frequencies = num_frequencies
        self.ori = ori        

    def forward(
        self, in_tensor: Float[Tensor, "*bs input_dim"], covs: Optional[Float[Tensor, "*bs input_dim input_dim"]] = None
    ) -> Float[Tensor, "*bs output_dim"]:
        freq_bands = (2**torch.arange(self.num_frequencies).float()).to(in_tensor.device)  # (F,)
        ori_c = in_tensor.shape[-1]
        pts = (in_tensor[..., None] * freq_bands).reshape(in_tensor.shape[:-1] + (self.num_frequencies * in_tensor.shape[-1], ))  # (..., DF)
        if self.ori:
            pts = torch.cat([in_tensor, torch.sin(pts), torch.cos(pts)], dim=-1).reshape(pts.shape[:-1]+(pts.shape[-1]*2+ori_c,))
        else:
            pts = torch.stack([torch.sin(pts), torch.cos(pts)], dim=-1).reshape(pts.shape[:-1]+(pts.shape[-1]*2,))
        return pts


class NeuralPoints(nn.Module):

    def __init__(self, state_dict, device, config):
        super().__init__()
        self.config = config
        self.device = device
        self.query_worldcoords_cuda = load_cuda(
            name='query_worldcoords_cuda',
            sources=[
                os.path.join(os.getcwd()+"/pointnerf/models/neural_points", path)
                for path in ['cuda/query_worldcoords.cpp', 'cuda/query_worldcoords.cu']],
            verbose=True)
        
        # TODO:
        self.points_xyz = nn.Parameter(state_dict["neural_points.xyz"].to(self.device)) # 
        self.points_embeding = nn.Parameter(state_dict["neural_points.points_embeding"].to(self.device))
        print("self.points_embeding", self.points_embeding.shape)
        self.points_conf = nn.Parameter(state_dict["neural_points.points_conf"].to(self.device))
        self.points_dir = nn.Parameter(state_dict["neural_points.points_dir"].to(self.device))
        self.points_color = nn.Parameter(state_dict["neural_points.points_color"].to(self.device))
        self.points_Rw2c = nn.Parameter(state_dict["neural_points.Rw2c"].to(self.device)) # 

        if self.points_xyz is not None:
            self.points_xyz.requires_grad = False
        if self.points_embeding is not None:
            self.points_embeding.requires_grad = config.feat_grad
        if self.points_conf is not None:
            self.points_conf.requires_grad = config.conf_grad
        if self.points_dir is not None:
            self.points_dir.requires_grad = config.dir_grad
        if self.points_color is not None:
            self.points_color.requires_grad = config.color_grad
        if self.points_Rw2c is not None:
            self.points_Rw2c.requires_grad = False
        
        # self.last_points_xyz = self.points_xyz.clone()
        # self.last_points_embeding = torch.zeros(self.points_embeding.shape, device=self.device, dtype=torch.float32)
        # self.last_points_color = torch.zeros(self.points_color.shape, device=self.device, dtype=torch.float32)
        # self.last_points_dir = torch.zeros(self.points_dir.shape, device=self.device, dtype=torch.float32)
        # self.last_points_conf = torch.zeros(self.points_conf.shape, device=self.device, dtype=torch.float32)
        
        self.reg_weight = 0.
        self.kernel_size = np.asarray(self.config.kernel_size, dtype=np.int32)
        self.kernel_size_tensor = torch.as_tensor(self.kernel_size, device=self.device, dtype=torch.int32)
        self.query_size = np.asarray(self.config.query_size, dtype=np.int32)
        self.query_size_tensor = torch.as_tensor(self.query_size, device=self.device, dtype=torch.int32)
        self.radius_limit_np = np.asarray(4 * max(self.config.vsize[0], self.config.vsize[1])).astype(np.float32)
        self.vscale_np = np.array(self.config.vscale, dtype=np.int32)
        self.scaled_vsize_np = (self.config.vsize * self.vscale_np).astype(np.float32)
        self.scaled_vsize_tensor = torch.as_tensor(self.scaled_vsize_np, device=self.device)

    def get_hyperparameters(self, vsize_np, point_xyz_w_tensor, ranges=None):
        min_xyz, max_xyz = torch.min(point_xyz_w_tensor, dim=-2)[0][0], torch.max(point_xyz_w_tensor, dim=-2)[0][0]
        ranges_min = torch.as_tensor(ranges[:3], dtype=torch.float32, device=min_xyz.device)
        ranges_max = torch.as_tensor(ranges[3:], dtype=torch.float32, device=min_xyz.device)
        if ranges is not None:
            # print("min_xyz", min_xyz.shape)
            # print("max_xyz", max_xyz.shape)
            # print("ranges", ranges)
            min_xyz, max_xyz = torch.max(torch.stack([min_xyz, ranges_min], dim=0), dim=0)[0], torch.min(torch.stack([max_xyz, ranges_max], dim=0), dim=0)[0]
        min_xyz = min_xyz - torch.as_tensor(self.scaled_vsize_np * self.config.kernel_size / 2, device=min_xyz.device, dtype=torch.float32)
        max_xyz = max_xyz + torch.as_tensor(self.scaled_vsize_np * self.config.kernel_size / 2, device=min_xyz.device, dtype=torch.float32)
        
        ranges_tensor = torch.cat([min_xyz, max_xyz], dim=-1)
        vdim_np = (max_xyz - min_xyz).cpu().numpy() / vsize_np
        scaled_vdim_np = np.ceil(vdim_np / self.vscale_np).astype(np.int32)
        return ranges_tensor, vsize_np, scaled_vdim_np

    def w2pers(self, point_xyz, camrotc2w, campos):
        point_xyz_shift = point_xyz[None, ...] - campos[:, None, :]
        xyz = torch.sum(camrotc2w[:, None, :, :] * point_xyz_shift[:, :, :, None], dim=-2)
        # print(xyz.shape, (point_xyz_shift[:, None, :] * camrot.T).shape)
        xper = xyz[:, :, 0] / xyz[:, :, 2]
        yper = xyz[:, :, 1] / xyz[:, :, 2]
        return torch.stack([xper, yper, xyz[:, :, 2]], dim=-1)
    
    def w2pers_loc(self, point_xyz_w, camrotc2w, campos):
        #     point_xyz_pers    B X M X 3
        xyz_w_shift = point_xyz_w - campos[:, None, :]
        xyz_c = torch.sum(xyz_w_shift[..., None,:] * torch.transpose(camrotc2w, 1, 2)[:, None, None,...], dim=-1)
        z_pers = xyz_c[..., 2]
        x_pers = xyz_c[..., 0] / xyz_c[..., 2]
        y_pers = xyz_c[..., 1] / xyz_c[..., 2]
        return torch.stack([x_pers, y_pers, z_pers], dim=-1)
    

    def null_grad(self):
        self.points_embeding.grad = None
        self.xyz.grad = None


    def reg_loss(self):
        return self.reg_weight * torch.mean(torch.pow(self.points_embeding, 2))

    def passfunc(self, input, vsize):
        return input


    def forward(self, ray_bundle):
        if ray_bundle.metadata["camrotc2w"].shape[0] != 3:
            cam_rot_tensor = ray_bundle.metadata["camrotc2w"][0].view(3, 3).unsqueeze(0).to(self.device)
        else:
            cam_rot_tensor = ray_bundle.metadata["camrotc2w"].unsqueeze(0).to(self.device)   # torch.Size([1, 3, 3])
        cam_pos_tensor = ray_bundle.origins[0].unsqueeze(0).to(self.device)              # torch.Size([1, 3])
        ray_dirs_tensor = ray_bundle.directions.unsqueeze(0).to(self.device)             # torch.Size([1, 4900, 3])
        near_depth = ray_bundle.nears[0].item()                          # float
        far_depth = ray_bundle.fars[0].item()                            # float
        # intrinsic = inputs["intrinsic"].cpu().numpy()

        
        point_xyz_w_tensor = self.points_xyz[None,...].to(self.device).detach()
        actual_numpoints_tensor = torch.ones([point_xyz_w_tensor.shape[0]], device=point_xyz_w_tensor.device, dtype=torch.int32) * point_xyz_w_tensor.shape[1]
        ranges_tensor, vsize_np, scaled_vdim_np = self.get_hyperparameters(self.config.vsize, point_xyz_w_tensor, ranges=self.config.ranges)


        # TODO: turn to nerfstudio.
        # raypos = campos[:, None, None, :] + raydir[:, :, None, :] * middle_point_ts[:, :, :, None]
        raypos_tensor, _, _, _ = near_far_linear_ray_generation(cam_pos_tensor, ray_dirs_tensor, self.config.z_depth_dim, near=near_depth, far=far_depth, jitter=0.3)

        D = raypos_tensor.shape[2]
        R = ray_dirs_tensor.shape[1]

        # sample_pidx_tensor: B, R, SR, K
        sample_pidx_tensor, sample_loc_w_tensor, ray_mask_tensor = \
            self.query_worldcoords_cuda.woord_query_grid_point_index(raypos_tensor,
                                                                        point_xyz_w_tensor,
                                                                        actual_numpoints_tensor,
                                                                        self.kernel_size_tensor,
                                                                        self.query_size_tensor,
                                                                        self.config.SR,
                                                                        self.config.K,
                                                                        R, D,
                                                                        torch.as_tensor(scaled_vdim_np,device=self.device).to(self.device),
                                                                        self.config.max_o,
                                                                        self.config.P,
                                                                        torch.as_tensor(self.radius_limit_np,device=self.device).to(self.device),
                                                                        ranges_tensor.to(self.device),
                                                                        self.scaled_vsize_tensor,
                                                                        self.config.gpu_maxthr,
                                                                        self.config.NN)

        sample_ray_dirs_tensor = torch.masked_select(ray_dirs_tensor, ray_mask_tensor[..., None]>0).reshape(ray_dirs_tensor.shape[0],-1,3)[...,None,:].expand(-1, -1, self.config.SR, -1).contiguous()

        sample_pnt_mask = sample_pidx_tensor >= 0
        B, R, SR, K = sample_pidx_tensor.shape
        sample_pidx_tensor = torch.clamp(sample_pidx_tensor, min=0).view(-1).long()

        # assert torch.equal(self.last_points_xyz, self.points_xyz)
        # self.last_points_xyz = self.points_xyz.clone()

        sample_loc_tensor = self.w2pers_loc(sample_loc_w_tensor, cam_rot_tensor, cam_pos_tensor)  # 
        point_xyz_pers_tensor = self.w2pers(self.points_xyz, cam_rot_tensor, cam_pos_tensor)  # 
        
        # assert not torch.equal(self.last_points_embeding, self.points_embeding)
        # self.last_points_embeding = self.points_embeding.clone()
        sampled_embedding = torch.index_select(torch.cat([self.points_xyz[None, ...], point_xyz_pers_tensor, self.points_embeding], dim=-1), 1, sample_pidx_tensor).view(B, R, SR, K, self.points_embeding.shape[2]+self.points_xyz.shape[1]*2)

        # assert not torch.equal(self.last_points_color,self.points_color)
        # self.last_points_color = self.points_color.clone()
        sampled_color = None if self.points_color is None else torch.index_select(self.points_color, 1, sample_pidx_tensor).view(B, R, SR, K, self.points_color.shape[2])

        # assert not torch.equal(self.last_points_dir, self.points_dir)
        # self.last_points_dir = self.points_dir.clone()
        sampled_dir = None if self.points_dir is None else torch.index_select(self.points_dir, 1, sample_pidx_tensor).view(B, R, SR, K, self.points_dir.shape[2])

        # assert not torch.equal(self.last_points_conf, self.points_conf)
        # self.last_points_conf = self.points_conf.clone()
        sampled_conf = None if self.points_conf is None else torch.index_select(self.points_conf, 1, sample_pidx_tensor).view(B, R, SR, K, self.points_conf.shape[2])

        sampled_Rw2c = self.points_Rw2c if self.points_Rw2c.dim() == 2 else torch.index_select(self.points_Rw2c, 0, sample_pidx_tensor).view(B, R, SR, K, self.points_Rw2c.shape[1], self.points_Rw2c.shape[2])

        return sampled_color, sampled_Rw2c, sampled_dir, sampled_embedding[..., 6:], sampled_embedding[..., 3:6], sampled_embedding[..., :3], sampled_conf, sample_loc_tensor, sample_loc_w_tensor, sample_pnt_mask, sample_ray_dirs_tensor, vsize_np, ray_mask_tensor