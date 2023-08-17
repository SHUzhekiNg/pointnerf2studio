import dataclasses

from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
from nerfstudio.plugins.types import MethodSpecification

from .studio_model import PointNerf, PointNerfConfig
from .studio_pipeline import PointNerfPipeline
from .studio_datamanager import PointNerfDataManagerConfig, PointNerfDataManager
from .studio_scheduler import PointNerfSchedulerConfig

pointnerf_config = TrainerConfig(
    method_name="pointnerf-original",
    pipeline=VanillaPipelineConfig(
        _target=PointNerfPipeline,
        datamanager=PointNerfDataManagerConfig(
            _target=PointNerfDataManager,
            eval_num_rays_per_batch=4096,
            train_num_rays_per_batch=4096,
        ),
        model=PointNerfConfig(
        	_target=PointNerf,
            eval_num_rays_per_chunk=8192,
    	),
    ),
    max_num_iterations=50000, # 200000
    steps_per_save=25000,
    steps_per_eval_batch=1000,  # 1000
    steps_per_eval_image=2000,  
    steps_per_eval_all_images=1000000,  # set to a very large number so we don't eval with all images
    optimizers={
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=0.001),
            "scheduler": PointNerfSchedulerConfig(
                lr_decay_exp=0.1,
                lr_decay_iters=1000000,
            ),
        },
    },
)


pointnerf_original = MethodSpecification(
    config=pointnerf_config, description="Implementation of Point-NeRF to Nerfstudio."
)
