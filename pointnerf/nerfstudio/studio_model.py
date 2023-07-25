from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import os
from skimage.metrics import mean_squared_error
import numpy as np
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

from ..utils.extension import TetrahedraTracer, interpolate_values
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


@dataclass
class PointNerfConfig(ModelConfig):
    _target: Any = dataclasses.field(default_factory=lambda: PointNerf)
    point_cloud_path: Optional[Path] = None
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

    
    num_mlp_base_layers: int = 2
    num_mlp_head_layers: int = 2
    num_color_layers: int = 1
    num_alpha_layers: int = 1
    hidden_size: int = 256

    input_fourier_frequencies: int = 0

    initialize_colors: bool = True
    use_gradient_scaling: bool = False
    """Use gradient scaler where the gradients are lower for points closer to the camera."""

    def __post_init__(self):
        if self.tetrahedra_path is not None and self.num_tetrahedra_vertices is None:
            if not self.tetrahedra_path.exists():
                raise RuntimeError(f"Tetrahedra path {self.tetrahedra_path} does not exist")
            tetrahedra = torch.load(self.tetrahedra_path)
            self.num_tetrahedra_vertices = len(tetrahedra["vertices"])
            self.num_tetrahedra_cells = len(tetrahedra["cells"])


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
    """Tetrahedra NeRF model

    Args:
        config: Basic NeRF configuration to instantiate model
    """

    config: PointNerfConfig

    def __init__(
        self,
        config: PointNerfConfig,
        # dataparser_transform=None,
        # dataparser_scale=None,
        **kwargs,
    ) -> None:
        super().__init__(
            config=config,
            **kwargs,
        )
        self._point_initialized = False

    # idk whats this
    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        will_initialize = False
        if (
            f"{prefix}tetrahedra_vertices" in state_dict
            and f"{prefix}tetrahedra_cells" in state_dict
            and f"{prefix}tetrahedra_field" in state_dict
        ):
            will_initialize = True
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        if will_initialize:
            self._tetrahedra_initialized = True

    # TODO:
    # load essential running things. (nerual point cloud)
    def _init_pointnerf(self):
        if self.config.point_cloud_path is not None:
            if not self.config.point_cloud_path.exists():
                raise RuntimeError(f"Specified point_cloud path {self.config.point_cloud_path} does not exist")
            pointcloud = torch.load(str(self.config.point_cloud_path), map_location=torch.device("cpu"))
            
            # init


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
            out_activation=nn.LeakyReLU(),
        )

        # block3
        mlp_in_dim += (3 if self.config.point_color_mode else 0) + (4 if self.config.point_dir_mode else 0)
        self.mlp_head = MLP(
            in_dim=mlp_in_dim,
            num_layers=self.config.num_mlp_head_layers,
            layer_width=self.config.hidden_size,
            out_activation=nn.LeakyReLU(),
        )
        # color and density, w/ simplified.
        color_in_dim = self.mlp_head.get_out_dim() + 2 * self.config.num_viewdir_freqs * 3 
        density_in_dim = self.mlp_base.get_out_dim()
        self.field_output_color = RGBFieldHead(in_dim=color_in_dim)
        self.field_output_density = DensityFieldHead(in_dim=density_in_dim)

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

        tracer = self.get_tetrahedra_tracer()
        tracer_output = tracer.trace_rays(
            ray_bundle.origins.contiguous(),
            ray_bundle.directions.contiguous(),
            self.config.max_intersected_triangles,
        )
        num_visited_cells = tracer_output["num_visited_cells"]
        nears = tracer_output["hit_distances"][:, 0, 0][:, None]
        fars = torch.gather(
            tracer_output["hit_distances"][:, :, 1],
            1,
            (num_visited_cells[:, None].long() - 1).clamp_min_(0),
        )

        # Reduce everything to nonempty rays
        ray_mask = tracer_output["num_visited_cells"] > 0
        nears_r = nears[ray_mask]
        fars_r = fars[ray_mask]
        if nears_r.shape[0] > 0:
            ray_bundle_modified_r = dataclasses.replace(ray_bundle[ray_mask], nears=nears_r, fars=fars_r)

            # Apply biased sampling
            if isinstance(self.sampler_uniform, TetrahedraSampler):
                ray_samples_r: RaySamples = self.sampler_uniform(
                    ray_bundle_modified_r,
                    num_visited_cells=tracer_output["num_visited_cells"][ray_mask],
                    hit_distances=tracer_output["hit_distances"][ray_mask],
                )
            else:
                ray_samples_r: RaySamples = self.sampler_uniform(ray_bundle_modified_r)
            distances_r = (ray_samples_r.frustums.ends + ray_samples_r.frustums.starts) / 2

            # Trace matched cells and interpolate field
            traced_cells = tracer.find_visited_cells(
                tracer_output["num_visited_cells"][ray_mask],
                tracer_output["visited_cells"][ray_mask],
                tracer_output["barycentric_coordinates"][ray_mask],
                tracer_output["hit_distances"][ray_mask],
                distances_r.squeeze(-1),
            )
            barycentric_coords = traced_cells["barycentric_coordinates"]
            field_values = interpolate_values(
                traced_cells["vertex_indices"],
                barycentric_coords,
                self.tetrahedra_field,
            )

            if self.config.num_fine_samples > 0:
                # apply MLP on top
                encoded_abc = self.position_encoding(field_values)
                base_mlp_out = self.mlp_base(encoded_abc)

                # Apply dense, fine sampling
                density_coarse = self.field_output_density(base_mlp_out)
                weights = ray_samples_r.get_weights(density_coarse)
                # pdf sampling
                ray_samples_r = self.sampler_pdf(ray_bundle_modified_r, ray_samples_r, weights)
                distances_r = (ray_samples_r.frustums.ends + ray_samples_r.frustums.starts) / 2

                traced_cells = tracer.find_visited_cells(
                    tracer_output["num_visited_cells"][ray_mask],
                    tracer_output["visited_cells"][ray_mask],
                    tracer_output["barycentric_coordinates"][ray_mask],
                    tracer_output["hit_distances"][ray_mask],
                    distances_r.squeeze(-1),
                )
                barycentric_coords = traced_cells["barycentric_coordinates"]
                field_values = interpolate_values(
                    traced_cells["vertex_indices"],
                    barycentric_coords,
                    self.tetrahedra_field,
                )

            encoded_abc = self.position_encoding(field_values)
            base_mlp_out = self.mlp_base(encoded_abc)

            field_outputs = {}
            field_outputs[self.field_output_density.field_head_name] = self.field_output_density(base_mlp_out)
            encoded_dir = self.direction_encoding(ray_samples_r.frustums.directions)
            mlp_out = self.mlp_head(torch.cat([encoded_dir, base_mlp_out], dim=-1))  # type: ignore
            field_outputs[self.field_output_color.field_head_name] = self.field_output_color(mlp_out)

            colors = field_outputs[FieldHeadNames.RGB]
            sigmas = field_outputs[FieldHeadNames.DENSITY]
            if self.config.use_gradient_scaling:
                # NOTE: we multiply the ray distance by 2 because according to the
                # Radiance Field Gradient Scaling for Unbiased Near-Camera Training
                # paper, it is the distance to the object center
                ray_dist = ray_samples_r.spacing_ends + ray_samples_r.spacing_starts
                colors, sigmas, ray_dist = GradientScaler.apply(colors, sigmas, ray_dist)

            weights = ray_samples_r.get_weights(sigmas)
            rgb_r = self.renderer_rgb(
                rgb=colors,
                weights=weights,
            )
            accumulation_r = self.renderer_accumulation(weights)
            depth_r = self.renderer_depth(weights, ray_samples_r)

        # Expand rendered values back to the original shape
        device = ray_mask.device
        rgb = (
            self.get_background_color()
            .to(device=device, dtype=torch.float32)
            .view(1, 3)
            .repeat_interleave(ray_mask.shape[0], 0)
        )
        # rgb = torch.zeros((ray_mask.shape[0], 3), dtype=torch.float32, device=device)
        accumulation = torch.zeros((ray_mask.shape[0], 1), dtype=torch.float32, device=device)
        depth = torch.full(
            (ray_mask.shape[0], 1),
            self.collider.far_plane,
            dtype=torch.float32,
            device=device,
        )
        if nears_r.shape[0] > 0:
            rgb[ray_mask] = rgb_r
            accumulation[ray_mask] = accumulation_r
            depth[ray_mask] = depth_r

        outputs = {
            "rgb": rgb,
            "accumulation": accumulation,
            "depth": depth,
            "ray_mask": ray_mask,
        }
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
