import dataclasses

from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
from nerfstudio.plugins.types import MethodSpecification

from .studio_model import PointNerf, PointNerfConfig
from .studio_pipeline import PointNerfPipeline
from .studio_datamanager import PointNerfDataManagerConfig, PointNerfDataManager
from .studio_utils import PointNerfSchedulerConfig

pointnerf_config = TrainerConfig(
    method_name="pointnerf-original",
    experiment_name="pointnerf2studio",
    pipeline=VanillaPipelineConfig(
        _target=PointNerfPipeline,
        datamanager=PointNerfDataManagerConfig(
            _target=PointNerfDataManager,
            eval_num_rays_per_batch=3600,
            train_num_rays_per_batch=3600,
        ),
        model=PointNerfConfig(
        	_target=PointNerf,
            eval_num_rays_per_chunk=2306,
    	),
    ),
    max_num_iterations=200000, # 200000
    steps_per_save=25000,
    steps_per_eval_batch=1000,  # 1000
    steps_per_eval_image=1000,  
    steps_per_eval_all_images=100000,
    optimizers={
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=0.0005),
            "scheduler": PointNerfSchedulerConfig(
                lr_decay_exp=0.1,
                lr_decay_iters=1000000,
            ),
        },
        "neural_points": {
            "optimizer": AdamOptimizerConfig(lr=0.002),
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
