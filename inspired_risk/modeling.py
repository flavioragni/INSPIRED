import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


# =========================
# Model / transform classes
# =========================

class CustomTransforms:
    class CLAHETransform:
        def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
            self.clahe = cv2.createCLAHE(
                clipLimit=clip_limit,
                tileGridSize=tile_grid_size
            )

        def __call__(self, img):
            img = np.array(img)
            lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            l = self.clahe.apply(l)
            lab = cv2.merge((l, a, b))
            img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            return img

    class RandomScaleCenterCrop:
        def __init__(self, scale_range=(0.9, 1.0)):
            self.scale_range = scale_range

        def __call__(self, img):
            input_w, input_h = img.size
            scale = np.random.uniform(*self.scale_range)
            output_w, output_h = int(input_w * scale), int(input_h * scale)
            transform = transforms.CenterCrop((output_h, output_w))
            img = transform(img)
            return img

    def __init__(self, train=True, mean=None, std=None):
        self.train = train
        self.mean = mean or [0.485, 0.456, 0.406]
        self.std = std or [0.229, 0.224, 0.225]

    def __call__(self, img):
        transforms_list = [
            self.CLAHETransform(clip_limit=2.0, tile_grid_size=(8, 8)),
            transforms.ToPILImage(),
        ]

        if self.train:
            transforms_list.append(
                transforms.RandomApply(
                    [self.RandomScaleCenterCrop(scale_range=(0.9, 1.0))],
                    p=0.1,
                )
            )

        transforms_list.extend([
            transforms.Resize(256),
            transforms.CenterCrop(224),
        ])

        if self.train:
            transforms_list.extend([
                transforms.RandomHorizontalFlip(p=0.1),
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.02)],
                    p=0.1,
                ),
                transforms.RandomApply(
                    [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))],
                    p=0.1,
                ),
            ])

        transforms_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std),
        ])

        transform = transforms.Compose(transforms_list)
        return transform(img)


class OldInspiredModel(nn.Module):
    def __init__(self, n_unfreeze=-1, n_fc_layers=1, dropouts=None, device=torch.device("cpu")):
        super().__init__()

        self.pre_trained_weights = models.ResNet50_Weights.DEFAULT
        self.model = models.resnet50(weights=self.pre_trained_weights)

        if n_fc_layers < 1:
            raise ValueError("n_fc_layers must be a positive integer greater than 0.")

        self.original_fc = self.model.fc
        self.model.fc = nn.Identity()

        self.fc_layers = nn.ModuleList()

        dropouts = dropouts or [0.0]
        next_input_dim = self.original_fc.out_features

        self.fc_layers.append(
            nn.Sequential(
                nn.ReLU(),
                nn.Dropout(dropouts[0] if len(dropouts) > 0 else 0.0),
            )
        )

        for i in range(n_fc_layers - 1):
            self.fc_layers.append(
                nn.Sequential(
                    nn.Linear(next_input_dim, next_input_dim // 2),
                    nn.ReLU(),
                    nn.Dropout(dropouts[i + 1] if i + 1 < len(dropouts) else 0.0),
                )
            )
            next_input_dim = next_input_dim // 2

        self.fc_layers.append(nn.Linear(next_input_dim, 1))

        self.unfreeze_blocks(n_unfreeze)
        self.device = device
        self.to(self.device)

    def unfreeze_blocks(self, n_unfreeze):
        for fc_layer in self.fc_layers:
            for param in fc_layer.parameters():
                param.requires_grad = True

        if n_unfreeze == -1:
            for param in self.model.parameters():
                param.requires_grad = True
            for param in self.original_fc.parameters():
                param.requires_grad = True
            return

        if n_unfreeze < -1:
            raise ValueError("n_unfreeze must be an integer greater than or equal to -1.")

        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.original_fc.parameters():
            param.requires_grad = False

        if n_unfreeze == 0:
            return

        learnable_blocks = [
            block for block in self.model.children()
            if sum(p.numel() for p in block.parameters()) > 0
        ] + [self.original_fc]

        for block in learnable_blocks[-n_unfreeze:]:
            for param in block.parameters():
                param.requires_grad = True

    def forward(self, x):
        x = self.model(x)
        x = self.original_fc(x)
        for fc_layer in self.fc_layers:
            x = fc_layer(x)
        return x


class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, x):
        return x / self.temperature.clamp(min=1e-6)


# =========================
# Helper structs / parsing
# =========================

@dataclass
class ModelSpec:
    n_unfreeze: int
    n_fc_layers: int
    dropouts: list[float]


def _load_json(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_checkpoint_payload(path: str | Path, device: torch.device) -> Any:
    return torch.load(path, map_location=device)


def _extract_state_dict(payload: Any) -> dict[str, torch.Tensor]:
    if isinstance(payload, dict):
        for key in ("state_dict", "model_state_dict"):
            if key in payload and isinstance(payload[key], dict):
                return payload[key]
        if all(isinstance(v, torch.Tensor) for v in payload.values()):
            return payload
    raise RuntimeError("Could not extract a valid state_dict from checkpoint.")


def _strip_known_wrapper_prefixes(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    prefixes = ("module.", "state_dict.",)
    out = dict(state_dict)
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if out and all(k.startswith(prefix) for k in out.keys()):
                out = {k[len(prefix):]: v for k, v in out.items()}
                changed = True
    return out


def _looks_like_old_inspired_state_dict(state_dict: dict[str, torch.Tensor]) -> bool:
    return any(
        k.startswith(("model.", "original_fc.", "fc_layers."))
        for k in state_dict.keys()
    )


def _looks_like_plain_resnet_state_dict(state_dict: dict[str, torch.Tensor]) -> bool:
    return any(
        k.startswith(("conv1.", "bn1.", "layer1.", "layer2.", "layer3.", "layer4.", "fc."))
        for k in state_dict.keys()
    )


def _remap_plain_resnet_to_old_inspired(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    remapped: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if key.startswith("fc."):
            new_key = "original_fc." + key[len("fc."):]
        else:
            new_key = "model." + key
        remapped[new_key] = value
    return remapped


def _parse_dropout_config(raw_dropout: Any, n_fc_layers: int) -> list[float]:
    if isinstance(raw_dropout, list):
        return [float(x) for x in raw_dropout]
    if raw_dropout is None:
        return [0.0 for _ in range(max(n_fc_layers, 1))]
    return [float(raw_dropout) for _ in range(max(n_fc_layers, 1))]


def _load_model_spec_from_json(model_config_path: str | Path) -> ModelSpec:
    cfg = _load_json(model_config_path)
    n_unfreeze = int(cfg.get("n_unfreeze", -1))
    n_fc_layers = int(cfg.get("n_fc_layers", 1))
    dropouts = _parse_dropout_config(cfg.get("dropout", 0.0), n_fc_layers)
    return ModelSpec(
        n_unfreeze=n_unfreeze,
        n_fc_layers=n_fc_layers,
        dropouts=dropouts,
    )


def _infer_model_spec(state_dict: dict[str, torch.Tensor]) -> ModelSpec:
    fc_layer_ids = set()
    for key in state_dict.keys():
        if key.startswith("fc_layers."):
            parts = key.split(".")
            if len(parts) > 1 and parts[1].isdigit():
                fc_layer_ids.add(int(parts[1]))

    if not fc_layer_ids:
        # Fallback. Works if checkpoint is plain ResNet only.
        return ModelSpec(n_unfreeze=-1, n_fc_layers=1, dropouts=[0.0])

    n_fc_layers = max(fc_layer_ids)
    # In OldInspiredModel, with n_fc_layers=1 we still have fc_layers.0 and fc_layers.1
    # so max index already equals logical n_fc_layers when final layer exists at last index.
    return ModelSpec(
        n_unfreeze=-1,
        n_fc_layers=n_fc_layers,
        dropouts=[0.0 for _ in range(max(n_fc_layers, 1))],
    )


def _is_effectively_empty_fc_layer_tensor(t: torch.Tensor) -> bool:
    return t.numel() == 0


# =========================
# Main inference wrapper
# =========================

class TorchDRClassifier:
    def __init__(
        self,
        checkpoint_path: str | Path,
        model_config_path: Optional[str | Path] = None,
        temperature_checkpoint_path: Optional[str | Path] = None,
        device: str = "cpu",
    ):
        self.checkpoint_path = str(checkpoint_path)
        self.model_config_path = str(model_config_path) if model_config_path else None
        self.temperature_checkpoint_path = (
            str(temperature_checkpoint_path) if temperature_checkpoint_path else None
        )
        self.device = torch.device(device)
        self.transform = CustomTransforms(train=False)
        self.model = self._load_model()
        self.temperature_scaler = self._load_temperature_scaler()

    def _load_model(self) -> nn.Module:
        payload = _load_checkpoint_payload(self.checkpoint_path, self.device)
        state_dict = _extract_state_dict(payload)
        state_dict = _strip_known_wrapper_prefixes(state_dict)

        if _looks_like_plain_resnet_state_dict(state_dict) and not _looks_like_old_inspired_state_dict(state_dict):
            state_dict = _remap_plain_resnet_to_old_inspired(state_dict)

        if self.model_config_path:
            spec = _load_model_spec_from_json(self.model_config_path)
        else:
            spec = _infer_model_spec(state_dict)

        model = OldInspiredModel(
            n_unfreeze=spec.n_unfreeze,
            n_fc_layers=spec.n_fc_layers,
            dropouts=spec.dropouts,
            device=self.device,
        )

        missing, unexpected = model.load_state_dict(state_dict, strict=False)

        # Ignore batchnorm tracking buffers mismatch if any
        missing = [k for k in missing if not k.endswith("num_batches_tracked")]
        unexpected = [k for k in unexpected if not k.endswith("num_batches_tracked")]

        if unexpected:
            raise RuntimeError(
                "Unexpected keys while loading model checkpoint: "
                + ", ".join(unexpected[:50])
                + (" ..." if len(unexpected) > 50 else "")
            )

        # If checkpoint was plain ResNet, fc_layers are expected to be missing because they do not exist there.
        required_missing = [
            k for k in missing
            if not (
                k.startswith("fc_layers.")
                or k.startswith("original_fc.")
            )
        ]

        if required_missing:
            raise RuntimeError(
                "Model checkpoint could not be loaded cleanly into OldInspiredModel. "
                "Missing required keys: "
                + ", ".join(required_missing[:50])
                + (" ..." if len(required_missing) > 50 else "")
            )

        model.eval()
        return model

    def _load_temperature_scaler(self) -> Optional[nn.Module]:
        if not self.temperature_checkpoint_path:
            return None

        scaler = TemperatureScaler().to(self.device)
        payload = _load_checkpoint_payload(self.temperature_checkpoint_path, self.device)
        state_dict = _extract_state_dict(payload)
        state_dict = _strip_known_wrapper_prefixes(state_dict)

        missing, unexpected = scaler.load_state_dict(state_dict, strict=False)
        missing = [k for k in missing if not k.endswith("num_batches_tracked")]
        unexpected = [k for k in unexpected if not k.endswith("num_batches_tracked")]

        if missing or unexpected:
            raise RuntimeError(
                f"Temperature scaler checkpoint mismatch. Missing: {missing}; Unexpected: {unexpected}"
            )

        scaler.eval()
        return scaler

    def preprocess_image(self, image_path: str | Path) -> torch.Tensor:
        with Image.open(image_path) as img:
            img = img.convert("RGB")
        x = self.transform(img)
        return x

    @torch.no_grad()
    def predict_logits_from_tensor(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(0).to(self.device)
        logits = self.model(x)
        logits = logits.view(-1, 1)
        if self.temperature_scaler is not None:
            logits = self.temperature_scaler(logits)
        return logits.squeeze(0)

    @torch.no_grad()
    def predict_proba_from_path(self, image_path: str | Path) -> tuple[float, float]:
        x = self.preprocess_image(image_path)
        logits = self.predict_logits_from_tensor(x)
        p_any_dr = torch.sigmoid(logits.squeeze()).item()
        p_no_dr = 1.0 - p_any_dr
        return float(p_no_dr), float(p_any_dr)