"""Native YOLO26n detector with controlled head-sharing interventions."""
from __future__ import annotations

import argparse
import copy
import math
from typing import Literal

import torch
from torch import nn
from ultralytics.nn.modules.head import Detect
from ultralytics.nn.tasks import DetectionModel

from . import ALL_ARMS, LEGACY_ARMS, LOOPED_ARMS
from .head_spec import canonicalize_head_spec

Arm = Literal["native", "independent_k2", "shared_k2", "loop_ab2", "loop_a3"]


class PointTower(nn.Module):
    def __init__(self, widths: tuple[int, int, int], output: int, arm: str):
        super().__init__()
        width = widths[0]
        self.adapters = nn.ModuleList(nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.Conv2d(channels, width, 1),
        ) for channels in (64, 128, 256))
        self.scale_id = nn.ParameterList([nn.Parameter(torch.zeros(width)) for _ in range(3)])
        self.norm_a = nn.LayerNorm(width, eps=1e-5)
        if arm == "loop_ab2":
            self.norm_b = nn.LayerNorm(width, eps=1e-5)
            self.a = nn.Conv2d(width, width, 1)
            self.b = nn.Conv2d(width, width, 1)
            self.passes = 2
        else:
            self.a = nn.Conv2d(width, width, 1)
            self.passes = 3
        self.variant = "baseline"
        self.readout = nn.Conv2d(width, output, 1)
        nn.init.zeros_(self.readout.bias)
        self.scale_bias = nn.ParameterList([nn.Parameter(torch.zeros(output)) for _ in range(3)])

    @staticmethod
    def _preact(x: torch.Tensor, norm: nn.LayerNorm, gains=None, level_indices=None, counts=None) -> torch.Tensor:
        # The channel dimension is normalized independently at every point.
        normalized = norm(x.permute(0, 3, 2, 1))
        if gains is not None:
            calibrated = []
            offset = 0
            for scale, count in zip(level_indices, counts, strict=True):
                calibrated.append(normalized[:, offset:offset + count] * gains[scale])
                offset += count
            normalized = torch.cat(calibrated, dim=1)
        return torch.nn.functional.silu(normalized).permute(0, 3, 2, 1)

    def _updates(self, h: torch.Tensor, arm: str, level_indices=None, counts=None, observe=None) -> torch.Tensor:
        for index in range(self.passes):
            if arm == "loop_ab2":
                norm_a = self.norm_a_steps[index] if self.variant == "step_norm" else self.norm_a
                norm_b = self.norm_b_steps[index] if self.variant == "step_norm" else self.norm_b
                gains_a = self.gain_a if self.variant == "scale_gain" else None
                gains_b = self.gain_b if self.variant == "scale_gain" else None
                hidden = self._preact(h, norm_a, gains_a, level_indices, counts)
                update = self.a(hidden)
                update = self._preact(update, norm_b, gains_b, level_indices, counts)
                update = self.b(update)
            else:
                hidden = self._preact(h, self.norm_a)
                update = self.a(hidden)
            coefficient = self.alpha[index] if self.variant == "learned_alpha" else 0.1
            if observe is not None:
                observe(index, h, update, coefficient)
            h = h + coefficient * update
        return h

    @torch.no_grad()
    def update_diagnostics(self, features: list[torch.Tensor], arm: str) -> list[dict]:
        points = [(self.adapters[index](feature) + self.scale_id[index][None, :, None, None])
                  .flatten(2).unsqueeze(2) for index, feature in enumerate(features)]
        h = torch.cat(points, dim=-1)
        records = []
        epsilon = torch.finfo(h.dtype).eps
        def observe(index, hidden, update, coefficient):
            hidden_norm = torch.linalg.vector_norm(hidden)
            scaled_norm = torch.linalg.vector_norm(coefficient * update)
            alpha = float(coefficient.detach().cpu()) if isinstance(coefficient, torch.Tensor) else float(coefficient)
            records.append({"update": index + 1, "h_l2": float(hidden_norm.cpu()),
                            "alphaF_l2": float(scaled_norm.cpu()),
                            "ratio": float((scaled_norm / hidden_norm.clamp_min(epsilon)).cpu()),
                            "alpha": alpha})
        self._updates(h, arm, list(range(len(features))), [point.shape[-1] for point in points], observe)
        return records

    def forward(self, features: list[torch.Tensor], arm: str, *, level_indices=None) -> torch.Tensor:
        points = []
        indices = list(range(len(features)) if level_indices is None else level_indices)
        for index, feature in zip(indices, features, strict=True):
            adapted = self.adapters[index](feature) + self.scale_id[index][None, :, None, None]
            points.append(adapted.flatten(2).unsqueeze(2))
        h = torch.cat(points, dim=-1)
        h = self._updates(h, arm, indices, [point.shape[-1] for point in points])
        output = self.readout(h)
        counts = [item.shape[-2] * item.shape[-1] for item in features]
        outputs = []
        offset = 0
        for index, count in zip(indices, counts, strict=True):
            outputs.append(output[..., offset:offset + count] + self.scale_bias[index][None, :, None, None])
            offset += count
        return torch.cat(outputs, dim=-1).squeeze(2)


@torch.no_grad()
def diagnose_ab2_updates(model: "HeadStudyModel", features: list[torch.Tensor] | tuple[torch.Tensor, ...]) -> dict[str, list[dict]]:
    """Explicitly inspect the actual two residual updates without changing model state."""
    if model.arm != "loop_ab2":
        raise ValueError("update diagnostics require loop_ab2")
    quantized = model.quantize_features(features)
    towers = (("cv2", model.detect.cv2[0]), ("cv3", model.detect.cv3[0]),
              ("one2one_cv2", model.detect.one2one_cv2[0]), ("one2one_cv3", model.detect.one2one_cv3[0]))
    return {name: tower.update_diagnostics(quantized, model.arm) for name, tower in towers}


class LoopedDetect(Detect):
    def bias_init(self):
        """Initialize each shared readout with Detect's native per-stride priors."""
        with torch.no_grad():
            for box_head, cls_head in ((self.cv2, self.cv3), (self.one2one_cv2, self.one2one_cv3)):
                if box_head is None or cls_head is None:
                    continue
                for tower in box_head:
                    tower.readout.bias.zero_()
                    for bias in tower.scale_bias:
                        bias.fill_(2.0)
                for tower in cls_head:
                    tower.readout.bias.zero_()
                    for index, stride in enumerate(self.stride.tolist()):
                        tower.scale_bias[index].fill_(math.log(5 / self.nc / (640 / stride) ** 2))

    def forward_head(self, x, box_head=None, cls_head=None):
        if box_head is None or cls_head is None:
            return {}
        width_arm = self.loop_arm
        boxes = box_head[0](x, width_arm)
        scores = cls_head[0](x, width_arm)
        return {"boxes": boxes, "scores": scores, "feats": x}


def validate_imgsz(imgsz: int) -> int:
    if isinstance(imgsz, bool) or not isinstance(imgsz, int) or imgsz <= 0 or imgsz % 32:
        raise ValueError("imgsz must be a positive multiple of 32 (for example 512 or 640)")
    return imgsz


def _routes(module: nn.Module, x: torch.Tensor, saved: list[torch.Tensor | None]):
    source = module.f
    if source != -1:
        x = saved[source] if isinstance(source, int) else [x if i == -1 else saved[i] for i in source]
    return x


class HeadStudyModel(nn.Module):
    """Fresh native YOLO26n heads on a frozen trunk with a common Q boundary."""

    architecture = "yolo26n"
    quantization = "fp32_fp16_fp32_v1"

    def __init__(self, arm: Arm = "native", seed: int = 11, imgsz: int = 640, *, model: DetectionModel | None = None, head_spec=None):
        super().__init__()
        validate_imgsz(imgsz)
        if arm not in ALL_ARMS:
            raise ValueError(f"unsupported arm: {arm}")
        canonical_spec = canonicalize_head_spec(head_spec)
        if canonical_spec is not None and ((canonical_spec["version"] == 1 and arm != "loop_ab2") or
                                           (canonical_spec["version"] == 2 and arm != "native")):
            raise ValueError("head_spec version is not supported for this arm")
        self.head_spec = canonical_spec
        self.arm, self.seed, self.imgsz = arm, int(seed), imgsz
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed)
            self.detector = model or DetectionModel("yolo26n.yaml", ch=3, nc=80, verbose=False)
        self.detector.end2end = True
        self.detect = self.detector.model[-1]
        self._assert_native_structure()
        if canonical_spec is not None and canonical_spec["version"] == 2:
            self._install_spatial_variant(canonical_spec["variant"])
        if arm in LEGACY_ARMS and arm != "native":
            self._replace_second_units(shared=arm == "shared_k2")
        elif arm in LOOPED_ARMS:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(self.seed)
                self._install_looped_heads(arm)
        for parameter in self.detector.model[:-1].parameters():
            parameter.requires_grad_(False)
        self.detector.model[:-1].eval()
        self.train(True)

    def _assert_native_structure(self) -> None:
        d = self.detect
        assert d.nc == 80 and d.reg_max == 1 and d.nl == 3
        assert tuple(self.detector.model[-1].stride.tolist()) == (8.0, 16.0, 32.0)
        assert len(d.cv2) == len(d.cv3) == len(d.one2one_cv2) == len(d.one2one_cv3) == 3
        assert tuple(layer[0].conv.in_channels for layer in d.cv2) == (64, 128, 256)
        assert tuple(layer[0].conv.out_channels for layer in d.cv2) == (16, 16, 16)
        assert tuple(layer[0][0].conv.in_channels for layer in d.cv3) == (64, 128, 256)
        assert tuple(layer[0][1].conv.out_channels for layer in d.cv3) == (80, 80, 80)

    def _install_spatial_variant(self, variant: str) -> None:
        d = self.detect
        with torch.random.fork_rng(devices=[]):
            for box_name, cls_name in (("cv2", "cv3"), ("one2one_cv2", "one2one_cv3")):
                for modules, sites, groups in ((getattr(d, box_name), ((1,),), 8),
                                                (getattr(d, cls_name), ((1, 0), (1, 1)), 16)):
                    sources = [modules[0][site[0]] if len(site) == 1 else modules[0][site[0]][site[1]] for site in sites]
                    source_modules = [modules[level][site[0]] if len(site) == 1 else modules[level][site[0]][site[1]]
                                     for site in sites for level in range(3)]
                    for index, source in enumerate(sources):
                        family = source_modules[index * 3:index * 3 + 3]
                        canonical = family[0]
                        for target in family:
                            conv = target.conv
                            if variant == "spatial_sepbn_k5" and conv.kernel_size == (3, 3):
                                replacement = nn.Conv2d(conv.in_channels, conv.out_channels, 5, stride=conv.stride,
                                                        padding=2, dilation=conv.dilation, groups=conv.groups,
                                                        bias=conv.bias is not None, padding_mode=conv.padding_mode,
                                                        device=conv.weight.device, dtype=conv.weight.dtype)
                                with torch.no_grad():
                                    replacement.weight.zero_()
                                    replacement.weight[:, :, 1:4, 1:4].copy_(conv.weight)
                                    if conv.bias is not None: replacement.bias.copy_(conv.bias)
                                target.conv = replacement
                        for level in range(1, 3):
                            target = family[level]
                            if variant == "spatial_untied":
                                with torch.no_grad():
                                    target.conv.weight.copy_(canonical.conv.weight)
                            else:
                                target.conv.weight = canonical.conv.weight
                        if variant == "spatial_gn":
                            channels = canonical.bn.num_features
                            norm = nn.GroupNorm(groups, channels, eps=canonical.bn.eps, affine=True,
                                                device=canonical.bn.weight.device, dtype=canonical.bn.weight.dtype)
                            for target in family:
                                target.bn = norm
                if variant == "spatial_sepbn_sharedpred":
                    for modules in (getattr(d, box_name), getattr(d, cls_name)):
                        for level in (1, 2):
                            modules[level][-1].weight = modules[0][-1].weight

    @staticmethod
    def _configure_looped_tower(tower: PointTower, variant: str, inner_width: int) -> None:
        width = tower.a.in_channels
        if inner_width != width:
            tower.a = nn.Conv2d(width, inner_width, 1)
            tower.b = nn.Conv2d(inner_width, width, 1)
            tower.norm_b = nn.LayerNorm(inner_width, eps=1e-5)
        if variant == "step_norm":
            norm_a = tower.norm_a
            norm_b = tower.norm_b
            tower.norm_a_steps = nn.ModuleList([norm_a, copy.deepcopy(norm_a)])
            tower.norm_b_steps = nn.ModuleList([norm_b, copy.deepcopy(norm_b)])
            del tower.norm_a
            del tower.norm_b
        elif variant == "scale_gain":
            tower.gain_a = nn.ParameterList([nn.Parameter(torch.ones(width)) for _ in range(3)])
            tower.gain_b = nn.ParameterList([nn.Parameter(torch.ones(inner_width)) for _ in range(3)])
        elif variant == "learned_alpha":
            tower.alpha = nn.Parameter(torch.full((2,), 0.1))
        tower.variant = variant
        if variant == "dw5":
            for adapter in tower.adapters:
                dw3, pointwise = adapter
                channels = dw3.in_channels
                dw5 = nn.Conv2d(channels, channels, 5, padding=2, groups=channels, bias=False)
                with torch.no_grad():
                    dw5.weight.zero_()
                    dw5.weight[:, :, 1:4, 1:4].copy_(dw3.weight)
                adapter[0] = dw5

    def _install_looped_heads(self, arm: str) -> None:
        d = self.detect
        # Preserve Detect's registered state while installing its native-compatible forward override.
        d.__class__ = LoopedDetect
        d.loop_arm = arm
        widths = (24, 24, 24) if arm == "loop_ab2" else (28, 28, 28)
        class_widths = (96, 96, 96) if arm == "loop_ab2" else (108, 108, 108)
        towers = []
        for box_name, cls_name in (("cv2", "cv3"), ("one2one_cv2", "one2one_cv3")):
            box_tower = PointTower(widths, 4, arm)
            cls_tower = PointTower(class_widths, d.nc, arm)
            setattr(d, box_name, nn.ModuleList([box_tower]))
            setattr(d, cls_name, nn.ModuleList([cls_tower]))
            towers.extend((box_tower, cls_tower))
        if self.head_spec is not None:
            variant = self.head_spec["variant"]
            inner_width = self.head_spec["class_inner"]
            # Construct every baseline tower first; variant-specific layers must not shift later towers' draws.
            with torch.random.fork_rng(devices=[]):
                for index, tower in enumerate(towers):
                    is_class_tower = index % 2 == 1
                    self._configure_looped_tower(tower, variant, inner_width if is_class_tower else tower.a.in_channels)
        d.bias_init()

    def _replace_second_units(self, *, shared: bool) -> None:
        d = self.detect
        for branch_box, branch_cls in ((d.cv2, d.cv3), (d.one2one_cv2, d.one2one_cv3)):
            for box_seq, cls_seq in zip(branch_box, branch_cls, strict=True):
                box_native = box_seq[1]
                cls_native = cls_seq[1]
                box_f, cls_f = copy.deepcopy(box_native), copy.deepcopy(cls_native)
                if shared:
                    box_seq[1] = nn.Sequential(box_f, box_f)
                    cls_seq[1] = nn.Sequential(cls_f, cls_f)
                else:
                    box_seq[1] = nn.Sequential(box_f, copy.deepcopy(box_f))
                    cls_seq[1] = nn.Sequential(cls_f, copy.deepcopy(cls_f))

    def train(self, mode: bool = True):
        if mode and getattr(self, "arm", None) in LOOPED_ARMS and self.detect.cv2 is None:
            raise RuntimeError("fused looped heads are inference-only")
        super().train(mode)
        # Trunk BN statistics are invariant; train mode only affects head modules.
        for module in self.detector.model[:-1]:
            module.eval()
        return self

    @staticmethod
    def quantize_features(features: list[torch.Tensor] | tuple[torch.Tensor, ...]) -> list[torch.Tensor]:
        return [(feature.to(torch.float16).to(torch.float32)) for feature in features]

    @torch.no_grad()
    def extract_features(self, images: torch.Tensor) -> list[torch.Tensor]:
        x = images.to(dtype=torch.float32)
        saved: list[torch.Tensor | None] = []
        modules = self.detector.model
        with torch.autocast(device_type=x.device.type, enabled=False):
            for module in modules[:-1]:
                x = _routes(module, x, saved)
                x = module(x)
                saved.append(x if module.i in self.detector.save else None)
            detect = modules[-1]
            routed = _routes(detect, x, saved)
        return self.quantize_features(routed)

    def forward_features(self, features: list[torch.Tensor] | tuple[torch.Tensor, ...]):
        if len(features) != 3:
            raise ValueError("YOLO26n expects three feature maps")
        quantized = self.quantize_features(features)
        return self.detect(quantized)

    def forward_images(self, images: torch.Tensor):
        return self.forward_features(self.extract_features(images))

    def forward(self, images: torch.Tensor):
        return self.forward_images(images)

    def deployment(self) -> nn.Module:
        """Return an inference-only image predictor that cannot bypass common Q."""
        owner = self

        class QuantizedDeployment(nn.Module):
            def __init__(self):
                super().__init__()
                self.owner = owner

            def forward(self, images: torch.Tensor):
                was_training = owner.training
                owner.eval()
                try:
                    return owner.forward_images(images)
                finally:
                    owner.train(was_training)

        return QuantizedDeployment().eval()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inspect", action="store_true", help="print sanitized architecture facts")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--arm", choices=ALL_ARMS, default="native")
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args(argv)
    validate_imgsz(args.imgsz)
    if args.inspect:
        model = HeadStudyModel(args.arm, args.seed, args.imgsz)
        print({"architecture": model.architecture, "arm": model.arm, "imgsz": model.imgsz,
               "channels": [64, 128, 256], "strides": [8, 16, 32], "quantization": model.quantization})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
