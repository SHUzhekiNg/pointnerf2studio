import dataclasses

from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManagerConfig
from nerfstudio.data.dataparsers.minimal_dataparser import MinimalDataParserConfig
from nerfstudio.engine.optimizers import RAdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
from nerfstudio.plugins.types import MethodSpecification

from .studio_model import PointNerf, PointNerfConfig
from .studio_pipeline import PointNerfPipeline

pointnerf_original_config = TrainerConfig(
    method_name="pointnerf_original_config",
    pipeline=VanillaPipelineConfig(
        _target=PointNerfPipeline,
        datamanager=VanillaDataManagerConfig(
            # _target=RayPruningDataManager,
            dataparser=MinimalDataParserConfig(),
            eval_num_rays_per_batch=4096,
            train_num_rays_per_batch=4096,
        ),
        model=PointNerfConfig(_target=PointNerf),
    ),
    max_num_iterations=300000,
    steps_per_save=25000,
    steps_per_eval_batch=1000,
    steps_per_eval_image=2000,
    steps_per_eval_all_images=50000,
    optimizers={
        "fields": {
            "optimizer": RAdamOptimizerConfig(lr=0.001),
            "scheduler": ExponentialDecaySchedulerConfig(  # little differ from src.
                lr_final=0.0001,
                max_steps=30000,
            ),
        },
    },
)


tetranerf_original = MethodSpecification(
    config=pointnerf_original_config, description="Implementation of Point-NeRF to Nerfstudio."
)
