"""Native YOLO26n detector with controlled head-sharing interventions."""
from __future__ import annotations

import argparse
import copy
from typing import Literal

import torch
from torch import nn
from ultralytics.nn.tasks import DetectionModel

Arm = Literal["native", "independent_k2", "shared_k2"]


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

    def __init__(self, arm: Arm = "native", seed: int = 11, imgsz: int = 640, *, model: DetectionModel | None = None):
        super().__init__()
        validate_imgsz(imgsz)
        if arm not in ("native", "independent_k2", "shared_k2"):
            raise ValueError(f"unsupported arm: {arm}")
        self.arm, self.seed, self.imgsz = arm, int(seed), imgsz
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed)
            self.detector = model or DetectionModel("yolo26n.yaml", ch=3, nc=80, verbose=False)
        self.detector.end2end = True
        self.detect = self.detector.model[-1]
        self._assert_native_structure()
        if arm != "native":
            self._replace_second_units(shared=arm == "shared_k2")
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
    parser.add_argument("--arm", choices=("native", "independent_k2", "shared_k2"), default="native")
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
