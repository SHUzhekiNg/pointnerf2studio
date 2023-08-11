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
from .studio_datamanager import PointNerfDataManagerConfig, PointNerfDataManager

pointnerf_config = TrainerConfig(
    method_name="pointnerf-original",
    pipeline=VanillaPipelineConfig(
        _target=PointNerfPipeline,
        # datamanager=VanillaDataManagerConfig(
        #     # _target=RayPruningDataManager,
        #     dataparser=MinimalDataParserConfig(),
        #     eval_num_rays_per_batch=4096,
        #     train_num_rays_per_batch=4096,
        # ),
        datamanager=PointNerfDataManagerConfig(
            _target=PointNerfDataManager,
            eval_num_rays_per_batch=8192,
            train_num_rays_per_batch=8192,
        ),
        model=PointNerfConfig(
        	_target=PointNerf,
            eval_num_rays_per_chunk=8192,
    	),
    ),
    max_num_iterations=200000,
    steps_per_save=25000,
    steps_per_eval_batch=1000,  # 1000
    steps_per_eval_image=2000,  
    steps_per_eval_all_images=1000000,  # set to a very large number so we don't eval with all images
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
