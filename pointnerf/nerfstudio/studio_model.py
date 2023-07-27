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
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

# from ..utils.extension import TetrahedraTracer, interpolate_values
from ..models.helpers.networks import PointNeRFEncoding

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
        np.sqrt(mean_squared_error(gt, img)) for gt, img in zip(image.cpu(), rgb.cpu())
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
    num_tetrahedra_vertices: Optional[int] = None
    num_tetrahedra_cells: Optional[int] = None

    num_pos_freqs: Optional[int] = 10
    num_viewdir_freqs: Optional[int] = 4
    num_feat_freqs: Optional[int] = 3
    num_dist_freqs: Optional[int] = 5

    agg_dist_pers: Optional[int] = 20
    point_features_dim: Optional[int] = 32

    point_color_mode: Optional[bool] = True  # False for only at features, True for color branch
    point_dir_mode: Optional[bool] = True

    max_intersected_triangles: int = 512  # TODO: try 1024
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

    input_fourier_frequencies: int = 0

    initialize_colors: bool = True
    use_gradient_scaling: bool = False
    """Use gradient scaler where the gradients are lower for points closer to the camera."""

    def __post_init__(self):
        if self.path_point_cloud is not None and self.num_tetrahedra_vertices is None:
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
        dataparser_transform=None,
        dataparser_scale=None,
        cameras=None,
        **kwargs,
    ) -> None:
        super().__init__(
            config=config,
            **kwargs,
        )
        self._point_initialized = False
        self.dataparser_transform = dataparser_transform
        self.dataparser_scale = dataparser_scale
        self.cameras = cameras
        self._init_pointnerf()
        


    # TODO:
    # load essential running things. (nerual point cloud)
    def _init_pointnerf(self):
        if self.config.path_point_cloud is not None:
            if not self.config.path_point_cloud.exists():
                raise RuntimeError(f"Specified point_cloud path {self.config.path_point_cloud} does not exist")
            # init
            if len([n for n in glob.glob(str(self.config.path_point_cloud) + "/*_net_ray_marching.pth") if os.path.isfile(n)]) == 0:
                raise RuntimeError(f"Cannot find any _net_ray_marching.pth in {self.config.path_point_cloud}")

            # best_PSNR=0.0
            # best_iter=0
            resume_iter = get_latest_epoch(self.config.path_point_cloud)
            # states = torch.load(
            #     os.path.join(self.config.path_point_cloud, '{}_states.pth'.format(resume_iter)), map_location=torch.device("cuda"))
            # epoch_count = states['epoch_count']
            # total_steps = states['total_steps']
            # best_PSNR = states['best_PSNR'] if 'best_PSNR' in states else best_PSNR
            # best_iter = states['best_iter'] if 'best_iter' in states else best_iter
            # best_PSNR = best_PSNR.item() if torch.is_tensor(best_PSNR) else best_PSNR
            # CONSOLE.print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
            # CONSOLE.print('Continue training from {} epoch'.format(resume_iter))
            # CONSOLE.print(f"Iter: {total_steps}")
            # CONSOLE.print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
            # del states

            # opt.mode = 2
            # opt.load_points=1  # ?????
            # opt.resume_iter = resume_iter
            # opt.is_train=True
            # model = create_model(opt)

            load_filename = '{}_net_ray_marching.pth'.format(resume_iter)
            load_path = os.path.join(self.config.path_point_cloud, load_filename)
            CONSOLE.print('loading', load_filename, " from ", load_path)
            if not os.path.isfile(load_path):
                raise RuntimeError(f'cannot load {load_filename}')
            state_dict = torch.load(load_path, map_location=self.device)
            # if resume_iter=="best" and name == "ray_marching" and self.opt.default_conf > 0.0 and self.opt.default_conf <= 1.0 and self.neural_points.points_conf is not None:
            #     assert "neural_points.points_conf" not in state_dict
            #     state_dict["neural_points.points_conf"] = torch.ones_like(self.net_ray_marching.module.neural_points.points_conf) * self.opt.default_conf
            if isinstance(self, nn.DataParallel):
                self = self.module
            self.load_state_dict_from_init(state_dict)
            # self.update_to_step(total_steps)
            # if self.config.maximum_step is not None and total_steps >= self.config.maximum_step:

            self._point_initialized = True
        else:
            raise RuntimeError("The point_cloud_path must be specified.")


    # optional.
    # def get_tetrahedra_tracer(self):
    #     device = self.tetrahedra_field.device
    #     if device.type != "cuda":
    #         raise RuntimeError("Tetrahedra tracer is only supported on a CUDA device")
    #     if self._tetrahedra_tracer is not None:
    #         if self._tetrahedra_tracer.device == device:
    #             return self._tetrahedra_tracer
    #         del self._tetrahedra_tracer
    #         self._tetrahedra_tracer = None
    #     if not self._point_initialized:
    #         self._init_pointnerf()
    #     self._tetrahedra_tracer = TetrahedraTracer(device)
    #     self._tetrahedra_tracer.load_tetrahedra(self.tetrahedra_vertices, self.tetrahedra_cells)
    #     return self._tetrahedra_tracer

    def populate_modules(self):
        """Set the fields and modules"""
        super().populate_modules()

        # fields
        self.position_encoding = PointNeRFEncoding(  # NeRFEncoding
            in_dim=2,
            num_frequencies=self.config.num_pos_freqs, # 10
            ori=False
        )
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


    def load_state_dict_from_init(self, state_dict, strict: bool = True):
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
        self.points_xyz = state_dict["neural_points.xyz"]
        self.points_embeding = state_dict["neural_points.points_embeding"]
        self.points_conf = state_dict["neural_points.points_conf"]
        self.points_dir = state_dict["neural_points.points_dir"]
        self.points_color = state_dict["neural_points.points_color"]
        self.Rw2c = state_dict["neural_points.Rw2c"]

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

        
        return outputs

    # pylint: disable=unused-argument
    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, torch.Tensor]:
        # Scaling metrics by coefficients to create the losses.
        device = outputs["rgb"].device
        image = batch["image"].to(device)

        rgb_loss = self.rgb_loss(image, outputs["rgb"])

        loss_dict = {"rgb_loss": rgb_loss}
        loss_dict = misc.scale_dict(loss_dict, self.config.loss_coefficients)
        return loss_dict

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        image = batch["image"].to(outputs["rgb"].device)
        rgb = outputs["rgb"]
        acc = colormaps.apply_colormap(outputs["accumulation"])
        depth = colormaps.apply_depth_colormap(
            outputs["depth"],
            accumulation=outputs["accumulation"],
        )

        combined_rgb = torch.cat([image, rgb], dim=1)
        combined_acc = torch.cat([acc], dim=1)
        combined_depth = torch.cat([depth], dim=1)

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
            "accumulation": combined_acc,
            "depth": combined_depth,
        }
        return metrics_dict, images_dict
