"""The network: one ResNet-50 trunk, three heads, and a geometry side-channel.

::

    image ──> ResNet-50 (ImageNet init) ──> avgpool ──> 2048
                                                         │
    19 hand-crafted shadow features ─────────────────────>├──> concat (2067)
                                                         │
                                             Linear 512 + BN + ReLU + Dropout
                                                         │
                        ┌────────────────────────────────┼────────────────────┐
                        │                                │                    │
                  side (2 logits)          regression (4 values)      direction (2 logits)
                  left / right         distance, width, height, y     into / out of frame

Why one trunk and three heads rather than three models: the three predictions
share almost all of their evidence. Where the shadow points determines the side;
how long and how faint it is determines the distance; the shape of the near end
carries whatever direction signal exists. Splitting them would triple the
compute and lose the shared representation, and the whole thing trains in ten
minutes on one GPU as it is.

The shared projection down to 512 before the heads is what lets the 19-dimension
feature vector matter at all: concatenated straight onto a 2048-vector and fed
to a linear head, its contribution would be swamped. Forcing everything through
a narrow bottleneck makes the network spend capacity on it.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torchvision import models

from shadow_detection.features import NUM_FEATURES


class ShadowNet(nn.Module):
    """Predicts ``(side, [distance, width, height, y_center], direction)``.

    Args:
        num_features: width of the hand-crafted feature vector. Pass ``0`` to
            train an image-only ablation.
        dropout: applied in the shared projection and in each head.
        pretrained: load ImageNet weights for the trunk. Always ``True`` for
            real runs -- with 1692 training images, training a ResNet-50 from
            scratch is hopeless -- and ``False`` in tests so they need no
            network access.
    """

    def __init__(
        self,
        num_features: int = NUM_FEATURES,
        dropout: float = 0.3,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.num_features = num_features

        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet50(weights=weights)
        # Drop the classifier; keep the global average pool so any input
        # resolution collapses to 2048 features. That is what allowed the same
        # architecture to run at both 384x384 and native 720x480.
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        trunk_dim = 2048

        self.shared_proj = nn.Sequential(
            nn.Linear(trunk_dim + num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.side_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2),
        )
        self.regression_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 4),
        )
        self.direction_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2),
        )

    def forward(
        self,
        image: torch.Tensor,
        features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns ``(side_logits, regression, direction_logits)``."""
        pooled = self.backbone(image).flatten(1)
        if self.num_features:
            if features is None:
                raise ValueError(
                    f"model expects {self.num_features} hand-crafted features but got None"
                )
            pooled = torch.cat([pooled, features], dim=1)
        shared = self.shared_proj(pooled)
        return self.side_head(shared), self.regression_head(shared), self.direction_head(shared)

    def parameter_groups(self, lr: float, backbone_lr: float) -> list[dict]:
        """Split parameters so the pretrained trunk gets a gentler learning rate.

        The heads start from noise and need a large step; the trunk starts from
        ImageNet and mostly needs nudging. A single learning rate either crawls
        or destroys the pretrained features in the first few batches.
        """
        backbone_params, head_params = [], []
        for name, param in self.named_parameters():
            (backbone_params if name.startswith("backbone") else head_params).append(param)
        return [
            {"params": backbone_params, "lr": backbone_lr},
            {"params": head_params, "lr": lr},
        ]

    @classmethod
    def from_checkpoint(
        cls,
        path: Path,
        device: torch.device | str = "cpu",
        num_features: int = NUM_FEATURES,
    ) -> ShadowNet:
        """Load weights saved by :mod:`shadow_detection.train`.

        The trunk is built without downloading ImageNet weights, since the
        checkpoint is about to overwrite them.
        """
        model = cls(num_features=num_features, pretrained=False)
        state = torch.load(path, map_location=device, weights_only=True)
        model.load_state_dict(state)
        return model.to(device).eval()

    def export_torchscript(self, path: Path, input_size: tuple[int, int] = (384, 384)) -> Path:
        """Trace the model to a self-contained TorchScript archive.

        A trace carries the graph as well as the weights, so it loads without
        this package installed -- which is what makes a released artifact
        usable by a web backend that should not depend on the training code.
        """
        self.eval()
        example = (
            torch.randn(1, 3, *input_size),
            torch.randn(1, max(self.num_features, 1)),
        )
        traced = torch.jit.trace(self, example if self.num_features else example[:1])
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.jit.save(traced, str(path))
        return path


def load_for_inference(
    path: Path,
    device: torch.device | str = "cpu",
    num_features: int = NUM_FEATURES,
) -> torch.nn.Module:
    """Load a checkpoint for prediction, whichever form it is saved in.

    Accepts both of the shapes this project produces:

    * a ``state_dict`` written by :mod:`shadow_detection.train`, and
    * a **TorchScript** archive, which is how the team's released weights ship
      so that a deployment does not have to install the training package.

    Both come back as something callable as ``model(image, features)``
    returning ``(side_logits, regression, direction_logits)``, so
    :func:`~shadow_detection.predict.predict_with_model` neither knows nor
    cares which it was handed.
    """
    path = Path(path)
    try:
        # A TorchScript archive is a zip with a code/ directory; torch.load
        # rejects it, and torch.jit.load rejects a plain state_dict, so trying
        # one and falling back is a reliable discriminator.
        model = torch.jit.load(str(path), map_location=device)
    except RuntimeError:
        return ShadowNet.from_checkpoint(path, device=device, num_features=num_features)
    return model.to(device).eval()
