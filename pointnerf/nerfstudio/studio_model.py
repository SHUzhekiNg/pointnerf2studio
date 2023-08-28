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
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from .studio_utils import PointNeRFEncoding, NeuralPoints
CONSOLE = Console(width=120)
torch.autograd.set_detect_anomaly(True)

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

    feat_grad: bool = True
    conf_grad: bool = True
    dir_grad: bool = True
    color_grad: bool = True

    num_pos_freqs: Optional[int] = 10
    num_viewdir_freqs: Optional[int] = 4
    num_feat_freqs: Optional[int] = 3
    num_dist_freqs: Optional[int] = 5

    agg_dist_pers: Optional[int] = 20
    point_features_dim: Optional[int] = 32

    point_color_mode: Optional[bool] = True  # False for only at features, True for color branch
    point_dir_mode: Optional[bool] = True

    num_samples: int = 80
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
    act_super: bool = False
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

    zero_epsilon: float = 1e-3
    zero_one_loss_weights: float = 0.0001
    def __post_init__(self):
        if self.path_point_cloud is not None:
            if not self.path_point_cloud.exists():
                raise RuntimeError(f"PointCloud path {self.path_point_cloud} does not exist")


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
            self.neural_points = NeuralPoints(state_dict, self._device, self.config)

            # block1 -> mlp_base
            # assert type(self.mlp_base)
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

            self._point_initialized = True
        else:
            raise RuntimeError("The point_cloud_path must be specified.")


    def populate_modules(self):
        """Set the fields and modules"""
        super().populate_modules()
        
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
            activation=nn.LeakyReLU(0.1, True),
            out_activation=nn.LeakyReLU(0.1, True),
        )

        # block3
        mlp_in_dim = self.mlp_base.get_out_dim() + (3 if self.config.point_color_mode else 0) + (4 if self.config.point_dir_mode else 0)
        self.mlp_head = MLP(
            in_dim=mlp_in_dim,
            num_layers=self.config.num_mlp_head_layers,
            layer_width=self.config.hidden_size,
            activation=nn.LeakyReLU(0.1, True),
            out_activation=nn.LeakyReLU(0.1, True),
        )

        # color and density, w/ simplified.
        color_in_dim = self.mlp_head.get_out_dim() + 2 * self.config.num_viewdir_freqs * 3 
        self.mlp_color = MLP(
            in_dim=color_in_dim,
            num_layers=self.config.num_color_layers,
            layer_width=self.config.hidden_size_color,
            activation=nn.LeakyReLU(0.1, True),
            out_activation=nn.LeakyReLU(0.1, True),
        )
        self.field_output_color = RGBFieldHead(in_dim=self.mlp_color.get_out_dim())
        self.field_output_density = DensityFieldHead(in_dim=self.mlp_head.get_out_dim())

        self.density_super_act = torch.nn.Softplus()
        self.density_act = torch.nn.ReLU()
        self.color_act = torch.nn.Sigmoid()

        self._background_color = nerfstudio.utils.colors.WHITE
        
        # losses
        self.miss_loss = MSELoss()
        self.mask_loss = MSELoss()
        self.rgb_loss = MSELoss()

        # metrics, tracked.
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.skimage_ssim = skimage_ssim
        self.skimage_rmse = skimage_rmse
        # self.lpips = LearnedPerceptualImagePatchSimilarity()
        # self.lpips_vgg = LearnedPerceptualImagePatchSimilarity(net_type="vgg")


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

    def get_background_color(self):
        background_color = self._background_color
        if renderers.BACKGROUND_COLOR_OVERRIDE is not None:
            background_color = renderers.BACKGROUND_COLOR_OVERRIDE
        return background_color

    def get_outputs(self, ray_bundle: RayBundle):
        if self.mlp_base is None:
            raise ValueError("populate_fields() must be called before get_outputs")
        
        # pixel_idx_tensor = ray_bundle.metadata["pixel_idx"].to(torch.int32)  # sb xiede
        sampled_color, sampled_Rw2c, sampled_dir, sampled_embedding, sampled_xyz_pers, sampled_xyz, sampled_conf, sample_loc_tensor, sample_loc_w_tensor,\
          sample_pnt_mask, sample_ray_dirs_tensor, vsize_np, ray_mask_tensor = self.neural_points(ray_bundle)

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
        weight = self.linear(dists, sample_pnt_mask, axis_weight=axis_weight)
        weight = weight / torch.clamp(torch.sum(weight, dim=-1, keepdim=True), min=1e-8)

        if self.training:
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
        # pts = pts_pnt
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

        alpha_in = self.field_output_density(feat)
        alpha = self.raw2out_density(alpha_in)
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
        color_in = self.field_output_color(color_in)
        color_output = self.raw2out_color(color_in)
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
        
        output = self.fill_invalid(output)
        output["ray_mask"] = output["ray_mask"].squeeze(0)
        if self.training:
            output["conf_coefficient"] = conf_coefficient
            

        return output
    
    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}
        net_params = []
        neural_params = []
        param_lst = list(self.named_parameters())
        if self.mlp_base is None:
            raise ValueError("populate_fields() must be called before get_param_groups")
        net_params = net_params + [par[1] for par in param_lst if not par[0].startswith("neural_points.points")]
        neural_params = neural_params + [par[1] for par in param_lst if par[0].startswith("neural_points.points")]

        param_groups["neural_points"] = neural_params
        param_groups["fields"] = net_params
        return param_groups
    
    # pylint: disable=unused-argument
    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, torch.Tensor]:
        # Scaling metrics by coefficients to create the losses.
        device = outputs["coarse_raycolor"].device
        image = batch["image"].to(device)

        masked_output = torch.masked_select(outputs["coarse_raycolor"], (outputs["ray_mask"] > 0)[..., None].expand(-1, 3)).reshape(-1, 3)
        masked_gt = torch.masked_select(image, (outputs["ray_mask"] > 0)[..., None].expand(-1, 3)).reshape(-1, 3)
        miss_output = torch.masked_select(outputs["coarse_raycolor"], (outputs["ray_mask"] == 0)[..., None].expand(-1, 3)).reshape(-1, 3)
        miss_gt = torch.masked_select(image, (outputs["ray_mask"] == 0)[..., None].expand(-1, 3)).reshape(-1, 3)
        ray_masked_coarse_raycolor_loss = self.mask_loss(masked_gt, masked_output) + 1e-6
        ray_miss_coarse_raycolor_loss = self.miss_loss(miss_gt, miss_output) * miss_gt.shape[0]
        # coarse_raycolor_loss = self.rgb_loss(image, outputs["coarse_raycolor"])
        
        #conf_coefficient_loss = conf_coefficient_tensor.new_full((outputs["ray_mask"].shape[0],1), conf_coefficient_tensor.item())
        # print("self.output[name]",torch.min(self.output[name]), torch.max(self.output[name]))
        # loss_total = conf_coefficient_loss + ray_masked_coarse_raycolor_loss
        
        loss_dict = {
            # "loss_total": loss_total,
            "ray_masked_coarse_raycolor_loss": ray_masked_coarse_raycolor_loss,
            # "ray_miss_coarse_raycolor_loss": ray_miss_coarse_raycolor_loss,
            # "coarse_raycolor_loss": coarse_raycolor_loss,
        }
        if self.training:
            val = torch.clamp(outputs["conf_coefficient"], self.config.zero_epsilon, 1 - self.config.zero_epsilon)
            conf_coefficient_loss = torch.mean(torch.log(val) + torch.log(1 - val)) * self.config.zero_one_loss_weights
            loss_dict["conf_coefficient_loss"] = conf_coefficient_loss
        loss_dict = misc.scale_dict(loss_dict, self.config.loss_coefficients)
        return loss_dict

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        image = batch["image"].to(outputs["coarse_raycolor"].device)
        outputs['ray_masked_coarse_raycolor'] = outputs["coarse_raycolor"].reshape(800, 800, 3)
        # outputs['ray_masked_coarse_raycolor'][outputs["ray_mask"].view(800, 800) <= 0,:] = 0.0
        rgb = outputs["ray_masked_coarse_raycolor"]
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
        # lpips = self.lpips(image, rgb)
        # lpips_vgg = self.lpips_vgg(image, rgb)
        rmse = self.skimage_rmse(image, rgb)
        metrics_dict = {
            "psnr": float(psnr),
            "ssim": float(ssim),
            # "lpips": float(lpips),
            # "lpips_vgg": float(lpips_vgg),
            "rmse": float(rmse)
        }

        images_dict = {
            "img": combined_rgb,
            # "accumulation": combined_acc,
            # "depth": combined_depth,
        }
        return metrics_dict, images_dict
    
    
    def linear(self, dists, pnt_mask, axis_weight=None):
        # dists: B * R * SR * K * channel
        # return B * R * SR * K
        if axis_weight is None or (axis_weight[..., 0] == 1 and axis_weight[..., 2] ==1) :
            weights = 1. / torch.clamp(torch.norm(dists[..., :3], dim=-1), min= 1e-6)
        else:
            weights = 1. / torch.clamp(torch.sqrt(torch.sum(torch.square(dists[...,:2]), dim=-1)) * axis_weight[..., 0] + torch.abs(dists[...,2]) * axis_weight[..., 1], min= 1e-6)
        weights = pnt_mask * weights
        return weights
    
    
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

        return output
    

