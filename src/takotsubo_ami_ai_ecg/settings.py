"""Explicit defaults for the author-confirmed final evaluation settings."""

from dataclasses import asdict, dataclass, field
import math

LABELS = ("tako", "AMI")
MODEL_NAMES = ("vit", "convnextv2", "mlpmixer")
IMAGE_SIZE = 480
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class ModelSettings:
    tta: int
    magnitude: int

    def validate(self):
        if type(self.tta) is not int or self.tta < 1:
            raise ValueError("TTA must be an integer >= 1; 1 means identity only.")
        # The original subclass used ten bins, not torchvision's default 31.
        if type(self.magnitude) is not int or not 0 <= self.magnitude <= 9:
            raise ValueError("Magnitude must be an integer from 0 to 9 (ten bins).")


@dataclass(frozen=True)
class InferenceSettings:
    vit: ModelSettings = field(default_factory=lambda: ModelSettings(24, 1))
    convnextv2: ModelSettings = field(default_factory=lambda: ModelSettings(1, 5))
    mlpmixer: ModelSettings = field(default_factory=lambda: ModelSettings(32, 3))
    ensemble_weights: tuple = (1.0, 1.0, 1.0)
    threshold: float | None = None
    seed: int = 1234
    input_policy: str = "strict480"
    precision: str = "float32"
    device: str = "auto"
    batch_size: int = 1
    num_workers: int = 4

    def validate(self):
        for name in MODEL_NAMES:
            getattr(self, name).validate()
        if len(self.ensemble_weights) != 3 or any(
            not math.isfinite(w) or w < 0 for w in self.ensemble_weights
        ) or sum(self.ensemble_weights) <= 0:
            raise ValueError("Use three finite, nonnegative ensemble weights with positive sum.")
        if self.threshold is not None and (not math.isfinite(self.threshold) or not 0 <= self.threshold <= 1):
            raise ValueError("Threshold must be between 0 and 1.")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32 - 100000:
            raise ValueError("Seed must be a nonnegative 32-bit integer below 4294867296.")
        if self.input_policy not in ("strict480", "resize"):
            raise ValueError("Input policy must be strict480 or resize.")
        if self.precision not in ("float32", "bfloat16"):
            raise ValueError("Precision must be float32 or bfloat16.")
        if self.device not in ("auto", "cpu", "cuda"):
            raise ValueError("Device must be auto, cpu or cuda.")
        if type(self.batch_size) is not int or self.batch_size < 1:
            raise ValueError("Batch size must be >= 1.")
        if type(self.num_workers) is not int or self.num_workers < 0:
            raise ValueError("Number of workers must be >= 0.")

    def to_dict(self):
        return asdict(self)

