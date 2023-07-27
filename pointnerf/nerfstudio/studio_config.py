import dataclasses

from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManagerConfig
# from nerfstudio.data.dataparsers.minimal_dataparser import BlenderDataParserConfig
from nerfstudio.data.dataparsers.minimal_dataparser import MinimalDataParserConfig
from nerfstudio.engine.optimizers import RAdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
from nerfstudio.plugins.types import MethodSpecification

from .studio_model import PointNerf, PointNerfConfig
from .studio_pipeline import PointNerfPipeline
from ..data.studio_datamanager import PointNerfDataManagerConfig

pointnerf_config = TrainerConfig(
    method_name="pointnerf-original",
    pipeline=VanillaPipelineConfig(
        _target=PointNerfPipeline,
        datamanager=VanillaDataManagerConfig(
            # _target=RayPruningDataManager,
            dataparser=MinimalDataParserConfig(),
            eval_num_rays_per_batch=4096,
            train_num_rays_per_batch=4096,
        ),
        # datamanager=PointNerfDataManagerConfig(
        #     dataparser=BlenderDataParserConfig(),
        #     eval_num_rays_per_batch=400,
        #     train_num_rays_per_batch=400,
        # ),
        model=PointNerfConfig(
        	_target=PointNerf
    	),
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


pointnerf_original = MethodSpecification(
    config=pointnerf_config, description="Implementation of Point-NeRF to Nerfstudio."
)
