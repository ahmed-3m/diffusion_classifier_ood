from dataclasses import dataclass, field
from typing import Tuple
from datetime import datetime


@dataclass
class ModelConfig:
    sample_size: int = 32
    in_channels: int = 3
    out_channels: int = 3
    layers_per_block: int = 2
    block_out_channels: Tuple[int, ...] = (128, 256, 256, 256)
    down_block_types: Tuple[str, ...] = (
        "DownBlock2D",
        "DownBlock2D",
        "AttnDownBlock2D",
        "DownBlock2D",
    )
    up_block_types: Tuple[str, ...] = (
        "UpBlock2D",
        "AttnUpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
    )
    num_class_embeds: int = 2
    num_train_timesteps: int = 1000
    beta_schedule: str = "squaredcos_cap_v2"
    prediction_type: str = "epsilon"


@dataclass
class TrainingConfig:
    batch_size: int = 64
    learning_rate: float = 1e-4
    max_epochs: int = 200
    precision: str = "16-mixed"
    accumulate_grad_batches: int = 2
    gradient_clip_val: float = 1.0
    warmup_epochs: int = 5
    weight_decay: float = 0.01


@dataclass
class DataConfig:
    data_dir: str = "./data"
    num_workers: int = 4
    id_class: int = 0
    pin_memory: bool = True


@dataclass
class EvalConfig:
    num_trials: int = 10
    eval_interval: int = 10
    num_inference_steps: int = 50


@dataclass
class LoggingConfig:
    project_name: str = "diffusion-classifier-ood"
    output_dir: str = "./outputs"
    save_top_k: int = 1
    log_every_n_steps: int = 50
    huggingface_repo: str = ""
    
    def generate_run_name(self, tag: str = "run") -> str:
        now = datetime.now()
        return f"{now:%Y-%m-%d}/{now:%H-%M-%S}_{tag}"


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
