import transformers
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="Qwen/Qwen2.5-VL-3B-Instruct")
    tune_mm_llm: bool = field(default=False)
    tune_mm_mlp: bool = field(default=False)
    tune_mm_vision: bool = field(default=False)
    use_local_vision_branch: bool = field(default=False)
    tune_mm_local_vision: Optional[bool] = field(default=None)
    tune_mm_local_mlp: Optional[bool] = field(default=None)
    local_branch_model_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Local full-checkpoint overlay containing model.local_visual.*; "
                "GRPO uses this to restore the SFT policy before training."
            )
        },
    )
    local_branch_strict_load: bool = field(
        default=False,
        metadata={
            "help": (
                "Require exact full-overlay compatibility and complete, shape-matched "
                "model.local_visual.* weights."
            )
        },
    )


@dataclass
class DataArguments:
    dataset_use: str = field(default="")
    data_flatten: bool = field(default=False)
    data_packing: bool = field(default=False)
    enable_local_image: bool = field(default=False)
    base_interval: int = field(default=2)
    max_pixels: int = field(default=28 * 28 * 576)
    min_pixels: int = field(default=28 * 28 * 16)
    video_max_frames: Optional[int] = field(default=8)
    video_min_frames: Optional[int] = field(default=4)
    video_max_pixels: int = field(default=1024 * 28 * 28)
    video_min_pixels: int = field(default=256 * 28 * 28)
    video_fps: float = 2


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    mm_projector_lr: Optional[float] = None
    vision_tower_lr: Optional[float] = None
    local_mm_projector_lr: Optional[float] = None
    local_vision_tower_lr: Optional[float] = None

    ## Lora config
    lora_enable: bool = field(default=False)
    lora_r: int = field(default=64)
    lora_alpha: int = field(default=128)
    lora_dropout: float = field(default=0.0)


@dataclass
class GRPOArguments:
    # GRPO core
    num_generations: int = field(
        default=4, metadata={"help": "G: number of completions sampled per prompt"}
    )
    max_completion_length: int = field(default=512)
    clip_eps: float = field(default=0.2)
    beta_kl: float = field(default=0.04)
    reward_eps: float = field(default=1e-6)
    inner_epochs: int = field(
        default=1, metadata={"help": "mu: PPO-style epochs per rollout"}
    )
    minibatch_size: int = field(default=8)
    ref_logprob_micro_batch_size: int = field(
        default=0,
        metadata={
            "help": (
                "If > 0, compute reference-model completion logprobs in micro-batches of this many "
                "completions and offload each chunk to CPU immediately."
            )
        },
    )
    # sampling
    temperature: float = field(default=1.0)
    top_p: float = field(default=0.95)
    vllm_max_model_len: Optional[int] = field(default=None)

    # rollout logging
    log_completions: bool = field(default=False)

    vllm_model_name_or_path: Optional[str] = field(
        default="Qwen/Qwen2.5-VL-3B-Instruct"
    )
    ref_model_name_or_path: Optional[str] = field(default=None)

    # ZeRO-3 generation gather
    ds3_gather_for_generation: bool = field(
        default=False, metadata={"help": "gather ZeRO-3 params for faster generate"}
    )


@dataclass
class VLLMArguments:
    # vLLM generation acceleration hooks
    use_vllm: bool = field(default=False)
    vllm_mode: str = field(default="server", metadata={"help": "server|colocate"})
    vllm_served_model_name: Optional[str] = field(default=None)
    vllm_server_base_url: str = field(default="http://127.0.0.1:8000/v1")

    # colocate config
    vllm_colocate_device: str = field(default="cuda:0")
    vllm_tensor_parallel_size: int = field(default=1)
    vllm_gpu_memory_utilization: float = field(default=0.3)
    vllm_max_model_len: Optional[int] = field(default=None)

    # weight sync policy (UNSPECIFIED)
    vllm_sync_strategy: str = field(default="none", metadata={"help": "none|reinit"})
    vllm_sync_every_steps: int = field(default=100)
    vllm_sync_model_path: Optional[str] = field(default=None)
