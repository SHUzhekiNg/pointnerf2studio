from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import os
from skimage.metrics import mean_squared_error
import numpy as np
import glob
import nerfstudio.utils
import torch
from nerfstudio.cameras.rays import RayBundle, RaySamples
from nerfstudio.field_components.encodings import NeRFEncoding
from nerfstudio.field_components.field_heads import (
    DensityFieldHead,
    FieldHeadNames,
    RGBFieldHead,
)
from nerfstudio.field_components.mlp import MLP
from nerfstudio.model_components import renderers
from nerfstudio.model_components.losses import MSELoss
from nerfstudio.model_components.ray_samplers import PDFSampler, Sampler, UniformSampler
from nerfstudio.model_components.renderers import (
    AccumulationRenderer,
    DepthRenderer,
    RGBRenderer,
)
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import colormaps, misc
from rich.console import Console
from skimage.metrics import structural_similarity
from torch import nn
from torch.nn import Parameter
from torch.utils.cpp_extension import load as load_cuda
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

# from ..utils.extension import TetrahedraTracer, interpolate_values
from ..models.helpers.networks import PointNeRFEncoding
from ..models.rendering.diff_ray_marching import near_far_linear_ray_generation_studio, near_far_linear_ray_generation
CONSOLE = Console(width=120)


def skimage_ssim(image, rgb):
    # Scikit implementation used in PointNeRF
    values = [
        structural_similarity(gt, img, win_size=11, multichannel=True, channel_axis=2, data_range=1.0)
        for gt, img in zip(image.cpu().permute(0, 2, 3, 1).numpy(), rgb.cpu().permute(0, 2, 3, 1).numpy())
    ]
    return sum(values) / len(values)

def skimage_rmse(image, rgb):
    # Scikit implementation used in PointNeRF
    values = [
        np.sqrt(mean_squared_error(gt, img)) for gt, img in zip(image.cpu().numpy(), rgb.cpu().numpy())
    ]
    return sum(values) / len(values)

def get_latest_epoch(resume_dir):
    os.makedirs(resume_dir, exist_ok=True)
    str_epoch = [file.split("_")[0] for file in os.listdir(resume_dir) if file.endswith("_states.pth")]
    int_epoch = [int(i) for i in str_epoch]
    return None if len(int_epoch) == 0 else str_epoch[int_epoch.index(max(int_epoch))]

@dataclass
class PointNerfConfig(ModelConfig):
    _target: Any = dataclasses.field(default_factory=lambda: PointNerf)
    path_point_cloud: Optional[Path] = None
    eval_num_rays_per_chunk: int = 4096

    num_pos_freqs: Optional[int] = 10
    num_viewdir_freqs: Optional[int] = 4
    num_feat_freqs: Optional[int] = 3
    num_dist_freqs: Optional[int] = 5

    agg_dist_pers: Optional[int] = 20
    point_features_dim: Optional[int] = 32

    point_color_mode: Optional[bool] = True  # False for only at features, True for color branch
    point_dir_mode: Optional[bool] = True

    num_samples: int = 80
    num_fine_samples: int = 256
    use_biased_sampler: bool = False
    field_dim: int = 64

    num_mlp_base_layers: Optional[int] = 2
    num_mlp_head_layers: Optional[int] = 2
    num_color_layers: Optional[int] = 3
    num_alpha_layers: Optional[int] = 1
    hidden_size: int = 256
    hidden_size_color: int = 128

    """USED PARAMS"""
    apply_pnt_mask: bool = True
    act_super: bool = True
    axis_weight: List[float] = dataclasses.field(default_factory=lambda: [1., 1., 1.])
    kernel_size: List[int] = dataclasses.field(default_factory=lambda: [3, 3, 3])
    vscale: List[float] = dataclasses.field(default_factory=lambda: [2, 2, 2])
    vsize: List[float] = dataclasses.field(default_factory=lambda: [0.004, 0.004, 0.004])
    query_size: List[float] = dataclasses.field(default_factory=lambda: [3, 3, 3])
    ranges: List[float] = dataclasses.field(default_factory=lambda: [-0.721, -0.695, -0.995, 0.658, 0.706, 1.050])
    z_depth_dim: int = 400  # num_coarse_sample

    
    # "query"
    SR: int = 80
    K: int = 8
    max_o: int = 410000
    P: int = 12
    NN: int = 2
    gpu_maxthr: int = 1024 # 'number of coarse samples'

    def __post_init__(self):
        if self.path_point_cloud is not None:
            if not self.path_point_cloud.exists():
                raise RuntimeError(f"PointCloud path {self.path_point_cloud} does not exist")


# Map from uniform space to transformed space
def map_from_real_distances_to_biased_with_bounds(num_bounds, bounds, samples):
    lengths = bounds[..., 1] - bounds[..., 0]
    sum_lengths = lengths.sum(-1)
    part_len = sum_lengths / num_bounds
    bounds_start = bounds[..., 0, 0]
    bounds_end = torch.gather(bounds[..., 1], 1, (num_bounds[:, None] - 1).clamp_min_(0)).squeeze(-1)
    rest = (samples - bounds_start[..., None]) / (bounds_end - bounds_start)[..., None]
    rest *= num_bounds[..., None]
    intervals = rest.floor().clamp_max_(num_bounds[..., None] - 1).clamp_min_(0)
    rest = rest - intervals
    intervals = intervals.long()
    cum_lengths = torch.cumsum(torch.cat((bounds_start[:, None], lengths), 1), 1)
    mapped_samples = torch.gather(cum_lengths, 1, intervals) + torch.gather(lengths, 1, intervals) * rest
    return mapped_samples

# TODO:
class PointNerfSampler(Sampler):
    """Sample points according to a function.

    Args:
        num_samples: Number of samples per ray
        train_stratified: Use stratified sampling during training. Defaults to True
    """

    def __init__(
        self,
        num_samples: Optional[int] = None,
        train_stratified=True,
    ) -> None:
        super().__init__(num_samples=num_samples)
        self.train_stratified = train_stratified

    def generate_ray_samples(
        self,
        ray_bundle: Optional[RayBundle] = None,
        num_samples: Optional[int] = None,
        *,
        num_visited_cells,
        hit_distances,
    ) -> RaySamples:
        """Generates position samples according to spacing function.

        Args:
            ray_bundle: Rays to generate samples for
            num_samples: Number of samples per ray

        Returns:
            Positions and deltas for samples along a ray
        """
        assert ray_bundle is not None
        assert ray_bundle.nears is not None
        assert ray_bundle.fars is not None

        num_samples = num_samples or self.num_samples
        assert num_samples is not None
        num_rays = ray_bundle.origins.shape[0]

        bins = torch.linspace(0.0, 1.0, num_samples + 1).to(ray_bundle.origins.device)[None, ...]  # [1, num_samples+1]

        # TODO More complicated than it needs to be.
        if self.train_stratified and self.training:
            t_rand = torch.rand((num_rays, num_samples + 1), dtype=bins.dtype, device=bins.device)
            bin_centers = (bins[..., 1:] + bins[..., :-1]) / 2.0
            bin_upper = torch.cat([bin_centers, bins[..., -1:]], -1)
            bin_lower = torch.cat([bins[..., :1], bin_centers], -1)
            bins = bin_lower + (bin_upper - bin_lower) * t_rand

        s_near, s_far = ray_bundle.nears, ray_bundle.fars
        spacing_to_euclidean_fn = lambda x: x * s_far + (1 - x) * s_near
        euclidean_bins = spacing_to_euclidean_fn(bins)
        euclidean_bins = map_from_real_distances_to_biased_with_bounds(
            num_visited_cells.long(), hit_distances, euclidean_bins
        )
        bins = (euclidean_bins - s_near) / (s_far - s_near)

        ray_samples = ray_bundle.get_ray_samples(
            bin_starts=euclidean_bins[..., :-1, None],
            bin_ends=euclidean_bins[..., 1:, None],
            spacing_starts=bins[..., :-1, None],
            spacing_ends=bins[..., 1:, None],
            spacing_to_euclidean_fn=spacing_to_euclidean_fn,
        )

        return ray_samples


class GradientScaler(torch.autograd.Function):
    @staticmethod
    def forward(ctx, colors, sigmas, ray_dist):
        ctx.save_for_backward(ray_dist)
        return colors, sigmas, ray_dist

    @staticmethod
    def backward(ctx, grad_output_colors, grad_output_sigmas, grad_output_ray_dist):
        (ray_dist,) = ctx.saved_tensors
        scaling = torch.square(ray_dist).clamp(0, 1)
        return grad_output_colors * scaling, grad_output_sigmas * scaling, grad_output_ray_dist


# pylint: disable=attribute-defined-outside-init
class PointNerf(Model):
    """PointNerf model

    Args:
        config: Basic NeRF configuration to instantiate model
    """

    config: PointNerfConfig

    def __init__(
        self,
        config: PointNerfConfig,
        cameras=None,
        **kwargs,
    ) -> None:
        super().__init__(
            config=config,
            **kwargs,
        )
        self._point_initialized = False
        self.cameras = cameras
        self._device = "cuda"
        self._init_pointnerf()
        # TODO:
        self.query_worldcoords_cuda = load_cuda(
            name='query_worldcoords_cuda',
            sources=[
                os.path.join("/home/zhenglicheng/Desktop/bootcamp/pointnerfstudio/pointnerf2studio/pointnerf/models/neural_points", path)
                for path in ['cuda/query_worldcoords.cpp', 'cuda/query_worldcoords.cu']],
            verbose=True)
        
    # load essential running things. (nerual point cloud)
    def _init_pointnerf(self):
        if self.config.path_point_cloud is not None:
            if not self.config.path_point_cloud.exists():
                raise RuntimeError(f"Specified point_cloud path {self.config.path_point_cloud} does not exist")
            # init
            if len([n for n in glob.glob(str(self.config.path_point_cloud) + "/*_net_ray_marching.pth") if os.path.isfile(n)]) == 0:
                raise RuntimeError(f"Cannot find any _net_ray_marching.pth in {self.config.path_point_cloud}")
            resume_iter = get_latest_epoch(self.config.path_point_cloud)
            load_filename = '{}_net_ray_marching.pth'.format(resume_iter)
            load_path = os.path.join(self.config.path_point_cloud, load_filename)
            CONSOLE.print('loading', load_filename, " from ", load_path)
            if not os.path.isfile(load_path):
                raise RuntimeError(f'cannot load {load_filename}')
            state_dict = torch.load(load_path, map_location=self.device)
            if isinstance(self, nn.DataParallel):
                self = self.module
            self.load_state_dict_from_init(state_dict)

            self.kernel_size = np.asarray(self.config.kernel_size, dtype=np.int32)
            self.kernel_size_tensor = torch.as_tensor(self.kernel_size, device=self._device, dtype=torch.int32)
            self.query_size = np.asarray(self.config.query_size, dtype=np.int32)
            self.query_size_tensor = torch.as_tensor(self.query_size, device=self._device, dtype=torch.int32)
            # radius_limit_scale == 4 
            self.radius_limit_np = np.asarray(4 * max(self.config.vsize[0], self.config.vsize[1])).astype(np.float32)
            self.vscale_np = np.array(self.config.vscale, dtype=np.int32)
            self.scaled_vsize_np = (self.config.vsize * self.vscale_np).astype(np.float32)
            self.scaled_vsize_tensor = torch.as_tensor(self.scaled_vsize_np, device=self._device)

            self._point_initialized = True
        else:
            raise RuntimeError("The point_cloud_path must be specified.")


    def populate_modules(self):
        """Set the fields and modules"""
        super().populate_modules()

        # fields
        # self.position_encoding = PointNeRFEncoding(  # NeRFEncoding
        #     in_dim=2,
        #     num_frequencies=self.config.num_pos_freqs, # 10
        #     ori=False
        # )
        self.direction_encoding = PointNeRFEncoding(  # NeRFEncoding
            in_dim=2,
            num_frequencies=self.config.num_viewdir_freqs,  # 4
            ori=True
        )
        self.feature_encoding = PointNeRFEncoding(  # NeRFEncoding
            in_dim=2,
            num_frequencies=self.config.num_feat_freqs,  # 3
            ori=False
        )
        self.dists_encoding = PointNeRFEncoding(  # NeRFEncoding
            in_dim=2,
            num_frequencies=self.config.num_dist_freqs,  # 5
            ori=False
        )

        # block1
        dist_dim = (4 if self.config.agg_dist_pers == 30 else 6) if self.config.agg_dist_pers > 9 else 3  # 6
        dist_xyz_dim = dist_dim if self.config.num_dist_freqs == 0 else 2 * abs(self.config.num_dist_freqs) * dist_dim
        mlp_in_dim = 2 * self.config.num_feat_freqs * self.config.point_features_dim + dist_xyz_dim + self.config.point_features_dim # simplified
        self.mlp_base = MLP(
            in_dim=mlp_in_dim,
            num_layers=self.config.num_mlp_base_layers,
            layer_width=self.config.hidden_size,
            activation=nn.LeakyReLU(),
            out_activation=nn.LeakyReLU(),
        )

        # block3
        mlp_in_dim = self.mlp_base.get_out_dim() + (3 if self.config.point_color_mode else 0) + (4 if self.config.point_dir_mode else 0)
        self.mlp_head = MLP(
            in_dim=mlp_in_dim,
            num_layers=self.config.num_mlp_head_layers,
            layer_width=self.config.hidden_size,
            activation=nn.LeakyReLU(),
            out_activation=nn.LeakyReLU(),
        )

        # color and density, w/ simplified.
        color_in_dim = self.mlp_head.get_out_dim() + 2 * self.config.num_viewdir_freqs * 3 
        self.mlp_color = MLP(
            in_dim=color_in_dim,
            num_layers=self.config.num_color_layers,
            layer_width=self.config.hidden_size_color,
            activation=nn.LeakyReLU(),
            out_activation=nn.LeakyReLU(),
        )
        self.field_output_color = RGBFieldHead(in_dim=self.mlp_color.get_out_dim())
        self.field_output_density = DensityFieldHead(in_dim=self.mlp_head.get_out_dim())

        self.density_super_act = torch.nn.Softplus()
        self.density_act = torch.nn.ReLU()
        self.color_act = torch.nn.Sigmoid()

        # TODO: samplers
        self.sampler_uniform = PointNerfSampler(num_samples=self.config.num_samples)
        # if self.config.num_fine_samples > 0:
        #     self.sampler_pdf = PDFSampler(num_samples=self.config.num_fine_samples)

        # TODO: is it right? just copy and paste from tetra's
        # renderers
        self._background_color = nerfstudio.utils.colors.WHITE
        self.renderer_rgb = RGBRenderer(background_color=self._background_color)
        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_depth = DepthRenderer()

        # TODO: losses, maybe 2.
        self.rgb_loss = MSELoss()

        # metrics, tracked.
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.skimage_ssim = skimage_ssim
        self.skimage_rmse = skimage_rmse
        self.lpips = LearnedPerceptualImagePatchSimilarity()
        self.lpips_vgg = LearnedPerceptualImagePatchSimilarity(net_type="vgg")


    def load_state_dict_from_init(self, state_dict):
        # # block1 -> mlp_base
        # self.mlp_base.layers[0].weight.data = state_dict["aggregator.block1.0.weight"]
        # self.mlp_base.layers[0].bias.data = state_dict["aggregator.block1.0.bias"]
        # self.mlp_base.layers[1].weight.data = state_dict["aggregator.block1.2.weight"]
        # self.mlp_base.layers[1].bias.data = state_dict["aggregator.block1.2.bias"]

        # # block3 -> mlp_head
        # self.mlp_head.layers[0].weight.data = state_dict["aggregator.block3.0.weight"]
        # self.mlp_head.layers[0].bias.data = state_dict["aggregator.block3.0.bias"]
        # self.mlp_head.layers[1].weight.data = state_dict["aggregator.block3.2.weight"]
        # self.mlp_head.layers[1].bias.data = state_dict["aggregator.block3.2.bias"]

        # # color_branch -> mlp_color
        # self.mlp_color.layers[0].weight.data = state_dict["aggregator.color_branch.0.weight"]
        # self.mlp_color.layers[0].bias.data = state_dict["aggregator.color_branch.0.bias"]
        # self.mlp_color.layers[1].weight.data = state_dict["aggregator.color_branch.2.weight"]
        # self.mlp_color.layers[1].bias.data = state_dict["aggregator.color_branch.2.bias"]
        # self.mlp_color.layers[2].weight.data = state_dict["aggregator.color_branch.4.weight"]
        # self.mlp_color.layers[2].bias.data = state_dict["aggregator.color_branch.4.bias"]

        # # color_branch.6 -> field_output_color
        # self.field_output_color.net.weight.data = state_dict["aggregator.color_branch.6.weight"]
        # self.field_output_color.net.bias.data = state_dict["aggregator.color_branch.6.bias"]

        # # alpha_branch.0 -> field_output_density
        # self.field_output_density.net.weight.data = state_dict["aggregator.alpha_branch.0.weight"]
        # self.field_output_density.net.bias.data = state_dict["aggregator.alpha_branch.0.bias"]

        # neural points
        self.points_xyz = state_dict["neural_points.xyz"].to(self._device)
        self.points_embeding = state_dict["neural_points.points_embeding"].to(self._device)
        self.points_conf = state_dict["neural_points.points_conf"].to(self._device)
        self.points_dir = state_dict["neural_points.points_dir"].to(self._device)
        self.points_color = state_dict["neural_points.points_color"].to(self._device)
        self.Rw2c = state_dict["neural_points.Rw2c"].to(self._device)

    # dont know
    # Just to allow for size reduction of the checkpoint
    def load_state_dict(self, state_dict, strict: bool = True):
        for k, v in self.lpips.state_dict().items():
            state_dict[f"lpips.{k}"] = v
        if hasattr(self, "lpips_vgg"):
            for k, v in self.lpips_vgg.state_dict().items():
                state_dict[f"lpips_vgg.{k}"] = v
        return super().load_state_dict(state_dict, strict)

    # dont know
    # Just to allow for size reduction of the checkpoint
    def state_dict(self, *args, prefix="", **kwargs):
        state_dict = super().state_dict(*args, prefix=prefix, **kwargs)
        for k in list(state_dict.keys()):
            if k.startswith(f"{prefix}lpips.") or k.startswith(f"{prefix}lpips_vgg."):
                state_dict.pop(k)
        return state_dict

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}
        if self.mlp_base is None:
            raise ValueError("populate_fields() must be called before get_param_groups")
        param_groups["fields"] = list(self.parameters())
        return param_groups

    def get_background_color(self):
        background_color = self._background_color
        if renderers.BACKGROUND_COLOR_OVERRIDE is not None:
            background_color = renderers.BACKGROUND_COLOR_OVERRIDE
        return background_color

    def get_outputs(self, ray_bundle: RayBundle):
        if self.mlp_base is None:
            raise ValueError("populate_fields() must be called before get_outputs")
        
        # pixel_idx_tensor = ray_bundle.metadata["pixel_idx"].to(torch.int32)  # sb xiede
        if ray_bundle.metadata["camrotc2w"].shape[0] != 3:
            cam_rot_tensor = ray_bundle.metadata["camrotc2w"][0].view(3, 3).unsqueeze(0).to(self._device)
        else:
            cam_rot_tensor = ray_bundle.metadata["camrotc2w"].unsqueeze(0).to(self._device)   # torch.Size([1, 3, 3])
        cam_pos_tensor = ray_bundle.origins[0].unsqueeze(0).to(self._device)              # torch.Size([1, 3])
        ray_dirs_tensor = ray_bundle.directions.unsqueeze(0).to(self._device)             # torch.Size([1, 4900, 3])
        near_depth = ray_bundle.nears[0].item()                          # float
        far_depth = ray_bundle.fars[0].item()                            # float
        # intrinsic = inputs["intrinsic"].cpu().numpy()

        # TODO: turn to nerfstudio.
        # raypos = campos[:, None, None, :] + raydir[:, :, None, :] * middle_point_ts[:, :, :, None]
        raypos_tensor, _, _, _ = near_far_linear_ray_generation(cam_pos_tensor, ray_dirs_tensor, self.config.z_depth_dim, near=near_depth, far=far_depth, jitter=0.3)

        
        point_xyz_w_tensor = self.points_xyz[None,...].to(self._device)
        actual_numpoints_tensor = torch.ones([point_xyz_w_tensor.shape[0]], device=point_xyz_w_tensor.device, dtype=torch.int32) * point_xyz_w_tensor.shape[1]
        ranges_tensor, vsize_np, scaled_vdim_np = self.get_hyperparameters(self.config.vsize, point_xyz_w_tensor, ranges=self.config.ranges)

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
                                                                        torch.as_tensor(scaled_vdim_np,device=self._device).to(self._device),
                                                                        self.config.max_o,
                                                                        self.config.P,
                                                                        torch.as_tensor(self.radius_limit_np,device=self._device).to(self._device),
                                                                        ranges_tensor.to(self._device),
                                                                        self.scaled_vsize_tensor,
                                                                        self.config.gpu_maxthr,
                                                                        self.config.NN)

        sample_ray_dirs_tensor = torch.masked_select(ray_dirs_tensor, ray_mask_tensor[..., None]>0).reshape(ray_dirs_tensor.shape[0],-1,3)[...,None,:].expand(-1, -1, self.config.SR, -1).contiguous()

        sample_pnt_mask = sample_pidx_tensor >= 0
        B, R, SR, K = sample_pidx_tensor.shape
        sample_pidx_tensor = torch.clamp(sample_pidx_tensor, min=0).view(-1).long()


        sample_loc_tensor = self.w2pers_loc(sample_loc_w_tensor, cam_rot_tensor, cam_pos_tensor)  # 
        point_xyz_pers_tensor = self.w2pers(self.points_xyz, cam_rot_tensor, cam_pos_tensor)  # 
        sampled_embedding = torch.index_select(torch.cat([self.points_xyz[None, ...], point_xyz_pers_tensor, self.points_embeding], dim=-1), 1, sample_pidx_tensor).view(B, R, SR, K, self.points_embeding.shape[2]+self.points_xyz.shape[1]*2)

        sampled_color = None if self.points_color is None else torch.index_select(self.points_color, 1, sample_pidx_tensor).view(B, R, SR, K, self.points_color.shape[2])

        sampled_dir = None if self.points_dir is None else torch.index_select(self.points_dir, 1, sample_pidx_tensor).view(B, R, SR, K, self.points_dir.shape[2])

        sampled_conf = None if self.points_conf is None else torch.index_select(self.points_conf, 1, sample_pidx_tensor).view(B, R, SR, K, self.points_conf.shape[2])

        sampled_Rw2c = self.Rw2c if self.Rw2c.dim() == 2 else torch.index_select(self.Rw2c, 0, sample_pidx_tensor).view(B, R, SR, K, self.Rw2c.shape[1], self.Rw2c.shape[2])

        sampled_xyz_pers = sampled_embedding[..., 3:6]
        sampled_xyz = sampled_embedding[..., :3]
        sampled_embedding = sampled_embedding[..., 6:]
        
        ray_valid = torch.any(sample_pnt_mask, dim=-1).view(-1)
        total_len = len(ray_valid)
        in_shape = sample_loc_w_tensor.shape
        # del sample_pidx_tensor, point_xyz_w_tensor, actual_numpoints_tensor, raypos_tensor

        if sampled_xyz_pers.shape[1] > 0:
            xdist = sampled_xyz_pers[..., 0] * sampled_xyz_pers[..., 2] - sample_loc_tensor[:, :, :, None, 0] * sample_loc_tensor[:, :, :, None, 2]
            ydist = sampled_xyz_pers[..., 1] * sampled_xyz_pers[..., 2] - sample_loc_tensor[:, :, :, None, 1] * sample_loc_tensor[:, :, :, None, 2]
            zdist = sampled_xyz_pers[..., 2] - sample_loc_tensor[:, :, :, None, 2]
            dists = torch.stack([xdist, ydist, zdist], dim=-1)
            dists = torch.cat([sampled_xyz - sample_loc_w_tensor[..., None, :], dists], dim=-1)
        else:
            B, R, SR, K, _ = sampled_xyz_pers.shape
            dists = torch.zeros([B, R, SR, K, 6], device=sampled_xyz_pers.device, dtype=sampled_xyz_pers.dtype)
        # del sample_loc_w_tensor, sampled_xyz_pers, sampled_xyz
        # weight:              B x valid R x SR         (1 * R * 80 * 8)
        # sampled_embedding:   B x valid R x SR x 32    (1 * R * 80 * 8 * 32)
        axis_weight = torch.as_tensor(self.config.axis_weight, dtype=torch.float32, device="cuda")[None, None, None, None, :]
        weight, sampled_embedding = self.linear(sampled_embedding, dists, sample_pnt_mask, axis_weight=axis_weight)
        weight = weight / torch.clamp(torch.sum(weight, dim=-1, keepdim=True), min=1e-8)
        def gradiant_clamp(sampled_conf, min=0.0001, max=1):
            diff = sampled_conf - torch.clamp(sampled_conf, min=min, max=max)
            return sampled_conf - diff.detach()
        conf_coefficient = gradiant_clamp(sampled_conf[..., 0], min=0.0001, max=1)
        # del sampled_conf

        # viewmlp
        pnt_mask_flat = sample_pnt_mask.view(-1)
        # pts = sample_loc_w_tensor.view(-1, sample_loc_w_tensor.shape[-1])
        viewdirs = sample_ray_dirs_tensor.view(-1, sample_ray_dirs_tensor.shape[-1])
        B, R, SR, K, _ = dists.shape
        sampled_Rw2c = sampled_Rw2c.transpose(-1, -2)
        pts_ray, pts_pnt = None, None
        viewdirs = viewdirs @ sampled_Rw2c
        viewdirs = self.direction_encoding.forward(viewdirs)
        ori_viewdirs, viewdirs = viewdirs[..., :3], viewdirs[..., 3:]
        viewdirs = viewdirs[ray_valid, :]

        dists_flat = dists.view(-1, dists.shape[-1])
        if self.config.apply_pnt_mask > 0:
            dists_flat = dists_flat[pnt_mask_flat, :]
        # dists_flat /= (
        #     1.0 if self.opt.dist_xyz_deno == 0. else float(self.opt.dist_xyz_deno * np.linalg.norm(vsize_np)))
        dists_flat[..., :3] = dists_flat[..., :3] @ sampled_Rw2c
        dists_flat = self.dists_encoding.forward(dists_flat)
        feat= sampled_embedding.view(-1, sampled_embedding.shape[-1])
        # del sampled_embedding, sample_ray_dirs_tensor
        # print("feat", feat.shape)
        feat = feat[pnt_mask_flat, :]
        feat = torch.cat([feat, self.feature_encoding.forward(feat)], dim=-1)
        feat = torch.cat([feat, dists_flat], dim=-1)
        weight = weight.view(B * R * SR, K, 1)
        pts = pts_pnt
        # print("feat",feat.shape) # 501
        feat = self.mlp_base(feat)


        sampled_color = sampled_color.view(-1, sampled_color.shape[-1])
        if self.config.apply_pnt_mask > 0:
            sampled_color = sampled_color[pnt_mask_flat, :]
        feat = torch.cat([feat, sampled_color], dim=-1)

        sampled_dir = sampled_dir.view(-1, sampled_dir.shape[-1])
        if self.config.apply_pnt_mask > 0:
            sampled_dir = sampled_dir[pnt_mask_flat, :]
            sampled_dir = sampled_dir @ sampled_Rw2c
        ori_viewdirs = ori_viewdirs[..., None, :].repeat(1, K, 1).view(-1, ori_viewdirs.shape[-1])
        if self.config.apply_pnt_mask > 0:
            ori_viewdirs = ori_viewdirs[pnt_mask_flat, :]
        feat = torch.cat([feat, sampled_dir - ori_viewdirs, torch.sum(sampled_dir*ori_viewdirs, dim=-1, keepdim=True)], dim=-1)
        feat = self.mlp_head(feat)
        # del sampled_dir, sampled_color

        alpha_in = feat
        alpha = self.raw2out_density(self.field_output_density(alpha_in))
        # print(alpha_in.shape, alpha_in)
        if self.config.apply_pnt_mask > 0:
            alpha_holder = torch.zeros([B * R * SR * K, alpha.shape[-1]], dtype=torch.float32, device=alpha.device)
            alpha_holder[pnt_mask_flat, :] = alpha
        else:
            alpha_holder = alpha
        alpha = alpha_holder.view(B * R * SR, K, alpha_holder.shape[-1])
        alpha = torch.sum(alpha * weight, dim=-2).view([-1, alpha.shape[-1]])[ray_valid, :] # alpha:


        if self.config.apply_pnt_mask > 0:
            feat_holder = torch.zeros([B * R * SR * K, feat.shape[-1]], dtype=torch.float32, device=feat.device)
            feat_holder[pnt_mask_flat, :] = feat
        else:
            feat_holder = feat
        feat = feat_holder.view(B * R * SR, K, feat_holder.shape[-1])
        feat = torch.sum(feat * weight, dim=-2).view([-1, feat.shape[-1]])[ray_valid, :]
        
        color_in = feat
        color_in = torch.cat([color_in, viewdirs], dim=-1)
        color_in = self.mlp_color(color_in)
        color_output = self.raw2out_color(self.field_output_color(color_in))
        output_mlp = torch.cat([alpha, color_output], dim=-1)


        # print("output_placeholder", output_placeholder.shape)
        # self.opt.shading_color_channel_num == 3
        output_placeholder = torch.zeros([total_len, 3 + 1], dtype=torch.float32, device=output_mlp.device)
        output_placeholder[ray_valid] = output_mlp

        decoded_features = output_placeholder.view(in_shape[:-1] + (3 + 1,))
        ray_valid = ray_valid.view(in_shape[:-1])

        ray_dist = torch.cummax(sample_loc_tensor[..., 2], dim=-1)[0]
        ray_dist = torch.cat([ray_dist[..., 1:] - ray_dist[..., :-1], torch.full((ray_dist.shape[0], ray_dist.shape[1], 1), vsize_np[2], device=ray_dist.device)], dim=-1)

        mask = ray_dist < 1e-8
        # if self.opt.raydist_mode_unit > 0:
        mask = torch.logical_or(mask, ray_dist > 2 * vsize_np[2])
        mask = mask.to(torch.float32)
        ray_dist = ray_dist * (1.0 - mask) + mask * vsize_np[2]
        ray_dist *= ray_valid.float()

        output = {}
        # output["queried_shading"] = torch.logical_not(torch.any(ray_valid, dim=-1, keepdims=True)).repeat(1, 1, 3).to(torch.float32)

        (
            ray_color,
            point_color,
            opacity,
            acc_transmission,
            blend_weight,
            background_transmission,
            _,
        ) = self.ray_march(ray_dist, ray_valid, decoded_features, self._background_color)
        # ray_color = self.tone_map(ray_color)
        output["coarse_raycolor"] = ray_color
        output["coarse_point_opacity"] = opacity
        output["coarse_is_background"] = background_transmission.squeeze(0)
        output["ray_mask"] = ray_mask_tensor
        # if weight is not None:
        #     output["weight"] = weight.detach()
        #     output["blend_weight"] = blend_weight.detach()
        #     output["conf_coefficient"] = conf_coefficient

        output = self.fill_invalid(output)
        output["ray_mask"] = output["ray_mask"].squeeze(0)
        return output

    # pylint: disable=unused-argument
    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, torch.Tensor]:
        # Scaling metrics by coefficients to create the losses.
        device = outputs["coarse_raycolor"].device
        image = batch["image"].to(device)

        masked_output = torch.masked_select(outputs["coarse_raycolor"], (outputs["ray_mask"] > 0)[..., None].expand(-1, 3)).reshape(-1, 3)
        masked_gt = torch.masked_select(image, (outputs["ray_mask"] > 0)[..., None].expand(-1, 3)).reshape(-1, 3)
        ray_masked_coarse_raycolor_loss = self.rgb_loss(masked_gt, masked_output)
        coarse_raycolor_loss = self.rgb_loss(image.unsqueeze(0), outputs["coarse_raycolor"])
        
        loss_dict = {
            "ray_masked_coarse_raycolor_loss": ray_masked_coarse_raycolor_loss,
            "coarse_raycolor_loss": coarse_raycolor_loss
        }
        loss_dict = misc.scale_dict(loss_dict, self.config.loss_coefficients)

        return loss_dict

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        image = batch["image"].to(outputs["coarse_raycolor"].device)
        rgb = outputs["coarse_raycolor"]
        # acc = colormaps.apply_colormap(outputs["accumulation"])
        # depth = colormaps.apply_depth_colormap(
        #     outputs["depth"],
        #     accumulation=outputs["accumulation"],
        # )

        combined_rgb = torch.cat([image, rgb], dim=1)
        # combined_acc = torch.cat([acc], dim=1)
        # combined_depth = torch.cat([depth], dim=1)

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        image = torch.moveaxis(image, -1, 0)[None, ...]
        rgb = torch.moveaxis(rgb, -1, 0)[None, ...]

        psnr = self.psnr(image, rgb).item()
        ssim = self.skimage_ssim(image, rgb)
        lpips = self.lpips(image, rgb)
        lpips_vgg = self.lpips_vgg(image, rgb)
        rmse = self.skimage_rmse(image, rgb)
        metrics_dict = {
            "psnr": float(psnr),
            "ssim": float(ssim),
            "lpips": float(lpips),
            "lpips_vgg": float(lpips_vgg),
            "rmse": float(rmse)
        }

        images_dict = {
            "img": combined_rgb,
            # "accumulation": combined_acc,
            # "depth": combined_depth,
        }
        return metrics_dict, images_dict
    
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
    
    def linear(self, embedding, dists, pnt_mask, axis_weight=None):
        # dists: B * R * SR * K * channel
        # return B * R * SR * K
        if axis_weight is None or (axis_weight[..., 0] == 1 and axis_weight[..., 2] ==1) :
            weights = 1. / torch.clamp(torch.norm(dists[..., :3], dim=-1), min= 1e-6)
        else:
            weights = 1. / torch.clamp(torch.sqrt(torch.sum(torch.square(dists[...,:2]), dim=-1)) * axis_weight[..., 0] + torch.abs(dists[...,2]) * axis_weight[..., 1], min= 1e-6)
        weights = pnt_mask * weights
        return weights, embedding
    
    def get_hyperparameters(self, vsize_np, point_xyz_w_tensor, ranges=None):
        '''
        :param l:
        :param h:
        :param w:
        :param zdim:
        :param ydim:
        :param xdim:
        :return:
        '''
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
    
    def raw2out_density(self, raw_density):
        if self.config.act_super > 0:
            return self.density_super_act(raw_density - 1)  # according to mip nerf, to stablelize the training
        else:
            return self.density_act(raw_density)
        
    def raw2out_color(self, raw_color):
        color = self.color_act(raw_color)
        if self.config.act_super > 0:
            color = color * (1 + 2 * 0.001) - 0.001 # according to mip nerf, to stablelize the training
        return color
            
    def ray_march(self, ray_dist, ray_valid, ray_features, bg_color=None):
        # ray_dist: N x Rays x Samples
        # ray_valid: N x Rays x Samples
        # ray_features: N x Rays x Samples x Features
        # Output
        # ray_color: N x Rays x 3
        # point_color: N x Rays x Samples x 3
        # opacity: N x Rays x Samples
        # acc_transmission: N x Rays x Samples
        # blend_weight: N x Rays x Samples x 1
        # background_transmission: N x Rays x 1

        point_color = ray_features[..., 1:4]

        # we are essentially predicting predict 1 - e^-sigma
        sigma = ray_features[..., 0] * ray_valid.float()
        opacity = 1 - torch.exp(-sigma * ray_dist)

        # cumprod exclusive
        acc_transmission = torch.cumprod(1. - opacity + 1e-10, dim=-1)
        temp = torch.ones(opacity.shape[0:2] + (1, )).to(
            opacity.device).float()  # N x R x 1

        background_transmission = acc_transmission[:, :, [-1]]
        acc_transmission = torch.cat([temp, acc_transmission[:, :, :-1]], dim=-1)
        
        def alpha_blend(opacity, acc_transmission):
            return opacity * acc_transmission
        blend_weight = alpha_blend(opacity, acc_transmission)[..., None]

        ray_color = torch.sum(point_color * blend_weight, dim=-2, keepdim=False)
        if bg_color is not None:
            ray_color += bg_color.to(opacity.device).float().view(
                background_transmission.shape[0], 1, 3) * background_transmission

        background_blend_weight = alpha_blend(1, background_transmission)

        return ray_color, point_color, opacity, acc_transmission, blend_weight, \
            background_transmission, background_blend_weight
    
    def fill_invalid(self, output):
        # ray_mask:             torch.Size([1, 1024])
        # coarse_is_background: torch.Size([1, 336, 1])  -> 1, 1024, 1
        # coarse_raycolor:      torch.Size([1, 336, 3])  -> 1, 1024, 3
        # coarse_point_opacity: torch.Size([1, 336, 24]) -> 1, 1024, 24
        ray_mask = output["ray_mask"]
        B, OR = ray_mask.shape
        ray_inds = torch.nonzero(ray_mask) # 336, 2
        coarse_is_background_tensor = torch.ones([B, OR, 1], dtype=output["coarse_is_background"].dtype, device=output["coarse_is_background"].device)
        # print("coarse_is_background", output["coarse_is_background"].shape)
        # print("coarse_is_background_tensor", coarse_is_background_tensor.shape)
        # print("ray_inds", ray_inds.shape, ray_mask.shape)
        coarse_is_background_tensor[ray_inds[..., 0], ray_inds[..., 1], :] = output["coarse_is_background"]
        output["coarse_is_background"] = coarse_is_background_tensor.squeeze(0)
        output['coarse_mask'] = (1 - coarse_is_background_tensor).squeeze(0)

        coarse_raycolor_tensor = torch.ones([B, OR, 3], dtype=output["coarse_raycolor"].dtype, device=output["coarse_raycolor"].device) * self._background_color[None, ...].to("cuda")
        coarse_raycolor_tensor[ray_inds[..., 0], ray_inds[..., 1], :] = output["coarse_raycolor"]
        output["coarse_raycolor"] = coarse_raycolor_tensor.squeeze(0)

        coarse_point_opacity_tensor = torch.zeros([B, OR, output["coarse_point_opacity"].shape[2]], dtype=output["coarse_point_opacity"].dtype, device=output["coarse_point_opacity"].device)
        coarse_point_opacity_tensor[ray_inds[..., 0], ray_inds[..., 1], :] = output["coarse_point_opacity"]
        output["coarse_point_opacity"] = coarse_point_opacity_tensor.squeeze(0)

        # queried_shading_tensor = torch.ones([B, OR, output["queried_shading"].shape[2]], dtype=output["queried_shading"].dtype, device=output["queried_shading"].device)
        # queried_shading_tensor[ray_inds[..., 0], ray_inds[..., 1], :] = output["queried_shading"]
        # output["queried_shading"] = queried_shading_tensor

        return output
    

