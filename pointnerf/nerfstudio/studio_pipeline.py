import typing
from .studio_datamanager import PointNerfDataManagerConfig, PointNerfDataManager
from torch.cuda.amp.grad_scaler import GradScaler
from nerfstudio.pipelines.base_pipeline import (
    DDP,
    Model,
    Pipeline,
    VanillaPipeline,
    VanillaPipelineConfig,
    dist,
)
from typing_extensions import Literal


class PointNerfPipeline(VanillaPipeline):
    def __init__(
        self,
        config: PointNerfDataManagerConfig,
        device: str,
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        grad_scaler: typing.Optional[GradScaler] = None,
    ):
        Pipeline.__init__(self)
        self.config = config
        self.test_mode = test_mode

        self.datamanager: PointNerfDataManager = config.datamanager.setup(
            device=device,
            test_mode=test_mode,
            world_size=world_size,
            local_rank=local_rank,
        )

        self.datamanager.to(device)
        assert self.datamanager.train_dataset is not None, "Missing input dataset"
        kwargs = {}
        self._model = config.model.setup(
            scene_box=self.datamanager.train_dataset.scene_box,
            num_train_data=len(self.datamanager.train_dataset),
            cameras=self.datamanager.train_dataset.cameras,
            **kwargs,
        )
        self.model.to(device)

        self.world_size = world_size
        if world_size > 1:
            self._model = typing.cast(
                Model,
                DDP(self._model, device_ids=[local_rank], find_unused_parameters=True),
            )
            dist.barrier(device_ids=[local_rank])
