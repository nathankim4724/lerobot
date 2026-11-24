# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

from dataclasses import dataclass, field

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature, PreTrainedConfig
from lerobot.optim import AdamWConfig, CosineDecayWithWarmupSchedulerConfig
from lerobot.utils.constants import OBS_IMAGES

from ..rtc.configuration_rtc import RTCConfig

# Batch key of the whole-grid NCA sequence used when `stage1_input == "seq"` (NCA pre-pretraining only).
OBS_SEQ = "observation.seq"


@PreTrainedConfig.register_subclass("smolvla")
@dataclass
class SmolVLAConfig(PreTrainedConfig):
    # Input / output structure.
    n_obs_steps: int = 1
    chunk_size: int = 50
    n_action_steps: int = 50

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    # Shorter state and action vectors will be padded
    max_state_dim: int = 32
    max_action_dim: int = 32

    # Image preprocessing
    resize_imgs_with_padding: tuple[int, int] = (512, 512)

    # Add empty images. Used by smolvla_aloha_sim which adds the empty
    # left and right wrist cameras in addition to the top camera.
    empty_cameras: int = 0

    # Converts the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model.
    adapt_to_pi_aloha: bool = False

    # Converts joint dimensions to relative values with respect to the current state before passing to the model.
    # Gripper dimensions will remain in absolute values.
    use_delta_joint_actions_aloha: bool = False

    # Tokenizer
    tokenizer_max_length: int = 48

    # Decoding
    num_steps: int = 10

    # Attention utils
    use_cache: bool = True

    # Finetuning settings
    freeze_vision_encoder: bool = True
    train_expert_only: bool = True
    train_state_proj: bool = True

    # Training presets
    optimizer_lr: float = 1e-4
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-10
    optimizer_grad_clip_norm: float = 10

    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 2.5e-6

    # Frames at the end of each episode never used as a training start (read by lerobot_train.py's
    # EpisodeAwareSampler, and applied to the held-out eval set too). 0 = use every frame, as before.
    # Set it to chunk_size - 1 to train only on starts whose whole action chunk is inside the episode.
    drop_n_last_frames: int = 0

    # --- NCA pre-pretraining on whole-grid sequences (Stage 1 only). Defaults = normal SmolVLA training. ---
    # "chunk": the usual action chunk. "seq": the expert's suffix is the sequence in batch["observation.seq"]
    # (seq_frames frames of tokens, in time order; tokens of a frame row by row) and actions are ignored.
    stage1_input: str = "chunk"
    # Loss on the sequence: "flow" (flow matching, continuous tokens), "mae" (masked tokens, cross-entropy,
    # bidirectional attention inside the sequence) or "ntp" (next-token prediction, cross-entropy, causal).
    stage1_objective: str = "flow"
    # "bin16": integer IDs of 4x4 binary patches, 64 per frame (mae / ntp). "rgb3": 3 numbers per patch, 64 per
    # frame. "rgb12": 2x2 neighbouring patches as one 12-number token, 16 per frame (both flow).
    seq_token: str = "bin16"
    seq_frames: int = 16
    seq_vocab: int = 65536
    # MAE: what is hidden. "token": random tokens; "frame": whole frames; "tube": the same patch positions in
    # every frame. mae_mask_ratio of the tokens / frames / positions are hidden in each sample.
    mae_mask: str = "token"
    mae_mask_ratio: float = 0.75
    # rgb3 / rgb12: per-channel (R, G, B) mean and std of the raw values, from the dataset's build record.
    seq_norm_mean: tuple[float, float, float] = (0.0, 0.0, 0.0)
    seq_norm_std: tuple[float, float, float] = (1.0, 1.0, 1.0)

    vlm_model_name: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"  # Select the VLM backbone.
    load_vlm_weights: bool = False  # Set to False in case of training the expert from scratch. True when init from pretrained SmolVLA weights

    add_image_special_tokens: bool = False  # Whether to use special image tokens around image features.

    attention_mode: str = "cross_attn"

    prefix_length: int = -1

    pad_language_to: str = "longest"  # "max_length"

    num_expert_layers: int = -1  # Less or equal to 0 is the default where the action expert has the same number of layers of VLM. Otherwise the expert have less layers.
    num_vlm_layers: int = 16  # Number of layers used in the VLM (first num_vlm_layers layers)
    self_attn_every_n_layers: int = 2  # Interleave SA layers each self_attn_every_n_layers
    expert_width_multiplier: float = 0.75  # The action expert hidden size (wrt to the VLM)

    min_period: float = 4e-3  # sensitivity range for the timestep used in sine-cosine positional encoding
    max_period: float = 4.0

    # Real-Time Chunking (RTC) configuration
    rtc_config: RTCConfig | None = None

    compile_model: bool = False  # Whether to use torch.compile for model optimization
    compile_mode: str = "max-autotune"  # Torch compile mode

    def __post_init__(self):
        super().__post_init__()

        """Input validation (not exhaustive)."""
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"The chunk size is the upper bound for the number of action steps per model invocation. Got "
                f"{self.n_action_steps} for `n_action_steps` and {self.chunk_size} for `chunk_size`."
            )
        if self.drop_n_last_frames < 0:
            raise ValueError(f"drop_n_last_frames must be >= 0, got {self.drop_n_last_frames}")
        if self.stage1_input not in ("chunk", "seq"):
            raise ValueError(f"stage1_input must be 'chunk' or 'seq', got {self.stage1_input!r}")
        if self.stage1_input == "seq":
            if self.stage1_objective not in ("flow", "mae", "ntp"):
                raise ValueError(
                    f"stage1_objective must be 'flow', 'mae' or 'ntp', got {self.stage1_objective!r}"
                )
            if self.seq_token not in ("bin16", "rgb3", "rgb12"):
                raise ValueError(f"seq_token must be 'bin16', 'rgb3' or 'rgb12', got {self.seq_token!r}")
            if (self.seq_token == "bin16") != (self.stage1_objective in ("mae", "ntp")):
                raise ValueError(
                    "seq_token 'bin16' goes with stage1_objective 'mae' / 'ntp'; 'rgb3' / 'rgb12' with 'flow'"
                )
            if self.mae_mask not in ("token", "frame", "tube"):
                raise ValueError(f"mae_mask must be 'token', 'frame' or 'tube', got {self.mae_mask!r}")
            if not 0.0 < self.mae_mask_ratio < 1.0:
                raise ValueError(f"mae_mask_ratio must be in (0, 1), got {self.mae_mask_ratio}")
            if self.seq_frames < 1 or self.seq_vocab < 2:
                raise ValueError("seq_frames must be >= 1 and seq_vocab >= 2")
            if len(self.seq_norm_mean) != 3 or len(self.seq_norm_std) != 3 or min(self.seq_norm_std) <= 0:
                raise ValueError("seq_norm_mean / seq_norm_std need 3 values each (R, G, B), std > 0")
            if self.seq_token_dim > self.max_action_dim:
                raise ValueError("seq tokens are wider than max_action_dim")
        if self.use_delta_joint_actions_aloha:
            raise NotImplementedError(
                "`use_delta_joint_actions_aloha` is used by smolvla for aloha real models. It is not ported yet in LeRobot."
            )

    def validate_features(self) -> None:
        for i in range(self.empty_cameras):
            key = f"{OBS_IMAGES}.empty_camera_{i}"
            empty_camera = PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, 480, 640),
            )
            self.input_features[key] = empty_camera

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self):
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    @property
    def seq_tokens_per_frame(self) -> int:
        """Tokens per frame of the 8x8-patch grid: 64 patches, or 16 groups of 2x2 patches."""
        return 16 if self.seq_token == "rgb12" else 64

    @property
    def seq_len(self) -> int:
        return self.seq_frames * self.seq_tokens_per_frame

    @property
    def seq_token_dim(self) -> int:
        """Numbers per continuous token (rgb3 / rgb12); 1 for integer IDs."""
        return {"bin16": 1, "rgb3": 3, "rgb12": 12}[self.seq_token]

    @property
    def observation_delta_indices(self) -> list:
        return [0]

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
