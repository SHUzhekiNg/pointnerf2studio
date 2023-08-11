# Copyright 2022 The Nerfstudio Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Datamanager.
"""

from __future__ import annotations

import os.path as osp
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Type, Union

import torch
import yaml
import numpy as np
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.utils.nerfstudio_collate import nerfstudio_collate
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes
from nerfstudio.model_components.ray_generators import RayGenerator
from nerfstudio.utils.misc import IterableWrapper
from rich.progress import Console

CONSOLE = Console(width=120)

from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManager, VanillaDataManagerConfig


@dataclass
class PointNerfDataManagerConfig(VanillaDataManagerConfig):
    _target: Type = field(default_factory=lambda: VanillaDataManager)
    # patch_tile_size_range: Tuple[int, int] = (0.05, 0.5)
    # patch_tile_size_res: int = 7
    # patch_stride_scaler: float = 0.5
    
    # from global.parser

    # from nerf_synth360_ft_dataset.parser
    random_sample: str = "none"     # random sample pixels
    random_sample_size: int = 70  # number of random samples
    init_view_num: int = 3          # number of random samples
    shape_id: int = 0               # shape id
    trgt_id: int = 0                # shape id
    num_nn: int = 1                 # number of nearest views in a batch
    near_plane: float = 2.0       # Near clipping plane, by default it is computed according to the distance of the camera 
    far_plane: float = 6.0        # Far clipping plane, by default it is computed according to the distance of the camera
    # bg_color: str = "white"         # background color, white|black(None)|random|rgb (float, float, float)
    # bg_filtering: int = 0           # 0 for alpha channel filtering, 1 for background color filtering
    # scan: str = "scan1"
    # full_comb: int = 0              
    # inverse_gamma_image: int = -1   # de-gamma correct the input image
    # pin_data_in_memory: int = -1    # load whole data in memory
    # normview: int = 0               # load whole data in memory
    # id_range: Tuple[int, int, int] = (0, 385, 1)    # the range of data ids selected in the original dataset. The default is range(0, 385). If the ids cannot be generated by range, use --id_list to specify any ids.
    # id_list: Optional[List[int]] = None     # the list of data ids selected in the original dataset. The default is range(0, 385).
    # split: str = "train"            # train, val, test
    # half_res: bool = False          # load blender synthetic data at 400x400 instead of 800x800
    # testskip: int = 8               # will load 1/N images from test/val sets, useful for large datasets like deepvoxels
    # dir_norm: int = 0               # normalize the ray_dir to unit length or not, default not
    # train_load_num: int = 0         # normalize the ray_dir to unit length or not, default not



class PointNerfDataManager(VanillaDataManager):  # pylint: disable=abstract-method
    """Basic stored data manager implementation.

    This is pretty much a port over from our old dataloading utilities, and is a little jank
    under the hood. We may clean this up a little bit under the hood with more standard dataloading
    components that can be strung together, but it can be just used as a black box for now since
    only the constructor is likely to change in the future, or maybe passing in step number to the
    next_train and next_eval functions.

    Args:
        config: the DataManagerConfig used to instantiate class
    """

    config: PointNerfDataManagerConfig

    def __init__(
        self,
        config: PointNerfDataManagerConfig,
        device: Union[torch.device, str] = "cpu",
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        **kwargs,  # pylint: disable=unused-argument
    ):
        super().__init__(
            config=config, device=device, test_mode=test_mode, world_size=world_size, local_rank=local_rank, **kwargs
        )


    def next_train(self, step: int) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data from the train dataloader."""
        self.train_count += 1
        image_batch = next(self.iter_train_image_dataloader)
        image_idx = (self.train_count-1) % image_batch["image_idx"].shape[0]
        image_batch = {
            "image_idx": torch.tensor(image_idx).unsqueeze(0),
            "image": image_batch["image"][torch.nonzero(image_batch["image_idx"] == image_idx).squeeze()].unsqueeze(0)
        }  # next(self.iter_train_image_dataloader)
        assert self.train_pixel_sampler is not None
        batch = self.train_pixel_sampler.sample(image_batch)
        ray_indices = batch["indices"]
        ray_bundle = self.train_ray_generator(ray_indices)

        # h = self.train_dataset.cameras[0].height.item()
        # w = self.train_dataset.cameras[0].width.item()
        # px = np.random.randint(0, w, size=(self.config.random_sample_size,
        #                                  self.config.random_sample_size)).astype(np.float32)
        # py = np.random.randint(0, h, size=(self.config.random_sample_size,
        #                                  self.config.random_sample_size)).astype(np.float32)
        # ray_bundle.metadata["pixel_idx"] = np.stack((px, py), axis=-1).astype(np.float32)
        ray_bundle.metadata["camrotc2w"] = self.train_dataset.cameras[ray_bundle.camera_indices.cpu()].camera_to_worlds[0][0][0:3, 0:3]
        # ray_bundle.metadata["h"] = h
        # ray_bundle.metadata["w"] = w

        return ray_bundle, batch
    
    def next_eval(self, step: int) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data from the train dataloader."""
        self.eval_count += 1
        image_batch = next(self.iter_eval_image_dataloader)
        image_idx = (self.eval_count-1) % image_batch["image_idx"].shape[0]
        image_batch = {
            "image_idx": torch.tensor(image_idx).unsqueeze(0),
            "image": image_batch["image"][torch.nonzero(image_batch["image_idx"] == image_idx).squeeze()].unsqueeze(0)
        }  # next(self.iter_train_image_dataloader)
        assert self.eval_pixel_sampler is not None
        batch = self.eval_pixel_sampler.sample(image_batch)
        ray_indices = batch["indices"]
        ray_bundle = self.eval_ray_generator(ray_indices)

        # h = self.train_dataset.cameras[0].height.item()
        # w = self.train_dataset.cameras[0].width.item()
        # px = np.random.randint(0, w, size=(self.config.random_sample_size,
        #                                  self.config.random_sample_size)).astype(np.float32)
        # py = np.random.randint(0, h, size=(self.config.random_sample_size,
        #                                  self.config.random_sample_size)).astype(np.float32)
        # ray_bundle.metadata["pixel_idx"] = np.stack((px, py), axis=-1).astype(np.float32)
        ray_bundle.metadata["camrotc2w"] = self.eval_dataset.cameras[ray_bundle.camera_indices.cpu()].camera_to_worlds[0][0][0:3, 0:3]
        # ray_bundle.metadata["h"] = h
        # ray_bundle.metadata["w"] = w

        return ray_bundle, batch
    
    def next_eval_image(self, step: int) -> Tuple[int, RayBundle, Dict]:
        for camera_ray_bundle, batch in self.eval_dataloader:
            assert camera_ray_bundle.camera_indices is not None
            image_idx = int(camera_ray_bundle.camera_indices[0, 0, 0])
            camera_ray_bundle.metadata["camrotc2w"] = self.eval_dataset.cameras[image_idx].camera_to_worlds[0:3, 0:3].unsqueeze(0).unsqueeze(0).expand(800, 800, -1, -1).reshape(800, 800, -1)
            return image_idx, camera_ray_bundle, batch
        raise ValueError("No more eval images")
        # self.eval_count += 1
        # image_batch = next(self.iter_eval_image_dataloader)
        # image_idx = (self.eval_count-1) % image_batch["image_idx"].shape[0]
        # image_batch = {
        #     "image_idx": torch.tensor(image_idx).unsqueeze(0),
        #     "image": image_batch["image"][torch.nonzero(image_batch["image_idx"] == image_idx).squeeze()].unsqueeze(0)
        # }  # next(self.iter_train_image_dataloader)
        # assert self.eval_pixel_sampler is not None
        # batch = self.eval_pixel_sampler.sample(image_batch)
        # ray_indices = batch["indices"]
        # ray_bundle = self.eval_ray_generator(ray_indices)

        # # h = self.train_dataset.cameras[0].height.item()
        # # w = self.train_dataset.cameras[0].width.item()
        # # px = np.random.randint(0, w, size=(self.config.random_sample_size,
        # #                                  self.config.random_sample_size)).astype(np.float32)
        # # py = np.random.randint(0, h, size=(self.config.random_sample_size,
        # #                                  self.config.random_sample_size)).astype(np.float32)
        # # ray_bundle.metadata["pixel_idx"] = np.stack((px, py), axis=-1).astype(np.float32)
        # ray_bundle.metadata["camrotc2w"] = self.eval_dataset.cameras[ray_bundle.camera_indices.cpu()].camera_to_worlds[0][0][0:3, 0:3]
        # return image_idx, ray_bundle, batch
        