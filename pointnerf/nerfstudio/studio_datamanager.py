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

from dataclasses import dataclass, field
from typing import Dict, Literal, Tuple, Type, Union

import torch
import random
from nerfstudio.cameras.rays import RayBundle
from rich.progress import Console

CONSOLE = Console(width=120)

from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManager, VanillaDataManagerConfig


@dataclass
class PointNerfDataManagerConfig(VanillaDataManagerConfig):
    _target: Type = field(default_factory=lambda: VanillaDataManager)
    
    # from nerf_synth360_ft_dataset.parser
    random_image_idx: bool = True     # random input image.
    near_plane: float = 2.0         # Near clipping plane, by default it is computed according to the distance of the camera 
    far_plane: float = 6.0          # Far clipping plane, by default it is computed according to the distance of the camera
    

class PointNerfDataManager(VanillaDataManager): 

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
        if self.config.random_image_idx:
            image_idx = random.randint(0, image_batch["image_idx"].shape[0] - 1)
        else:
            image_idx = (self.train_count-1) % image_batch["image_idx"].shape[0]
        image_batch = {
            "image_idx": torch.tensor(image_idx).unsqueeze(0),
            "image": image_batch["image"][torch.nonzero(image_batch["image_idx"] == image_idx).squeeze()].unsqueeze(0)
        }
        assert self.train_pixel_sampler is not None
        batch = self.train_pixel_sampler.sample(image_batch)
        ray_indices = batch["indices"]
        ray_bundle = self.train_ray_generator(ray_indices)

        ray_bundle.metadata["camrotc2w"] = self.train_dataset.cameras[ray_bundle.camera_indices.cpu()].camera_to_worlds[0][0][0:3, 0:3]

        return ray_bundle, batch
    
    def next_eval(self, step: int) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data from the train dataloader."""
        self.eval_count += 1
        image_batch = next(self.iter_eval_image_dataloader)
        if self.config.random_image_idx:
            image_idx = random.randint(0, image_batch["image_idx"].shape[0] - 1)
        else:
            image_idx = (self.train_count-1) % image_batch["image_idx"].shape[0]
        image_batch = {
            "image_idx": torch.tensor(image_idx).unsqueeze(0),
            "image": image_batch["image"][torch.nonzero(image_batch["image_idx"] == image_idx).squeeze()].unsqueeze(0)
        } 
        assert self.eval_pixel_sampler is not None
        batch = self.eval_pixel_sampler.sample(image_batch)
        ray_indices = batch["indices"]
        ray_bundle = self.eval_ray_generator(ray_indices)

        ray_bundle.metadata["camrotc2w"] = self.eval_dataset.cameras[ray_bundle.camera_indices.cpu()].camera_to_worlds[0][0][0:3, 0:3]

        return ray_bundle, batch
    
    def next_eval_image(self, step: int) -> Tuple[int, RayBundle, Dict]:
        for camera_ray_bundle, batch in self.eval_dataloader:
            assert camera_ray_bundle.camera_indices is not None
            image_idx = int(camera_ray_bundle.camera_indices[0, 0, 0])
            camera_ray_bundle.metadata["camrotc2w"] = self.eval_dataset.cameras[image_idx].camera_to_worlds[0:3, 0:3].unsqueeze(0).unsqueeze(0).expand(800, 800, -1, -1).reshape(800, 800, -1)
            return image_idx, camera_ray_bundle, batch
        raise ValueError("No more eval images")
        