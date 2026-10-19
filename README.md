# Same compute, 20.5% fewer head parameters, +2.10% development mAP

**Result.** On the frozen YOLO26n trunk, one detection head shares its later convolution weights and its predictor kernels across the P, M and G scales, while each scale keeps its own BatchNorm. Against the native three-head YOLO26n, that head raised development AP50:95 by **2.10%** (**+0.319 points**) and removed **20.5%** of the unique head parameters. The ConvMAC count did not change. The gain was positive in **5 of 5** paired seeds.

| Development AP50:95 ×100, mean ± sample SD, 5 paired seeds | `native` | `spatial_sepbn_sharedpred` |
|---|---:|---:|
| **AP50:95** | 15.173 ± 0.151 | **15.492 ± 0.165** |
| Paired difference vs `native` | n/a | **+0.319 pp** |
| Sample SD of that difference | n/a | 0.206 |
| 95% t interval, df = 4 | n/a | [+0.062, +0.575] |
| Seeds above native | n/a | **5 of 5** |
| Unique head parameters, both internal branches | 309,656 | **246,104 (−20.5%)** |
| Head ConvMACs, 512 px, batch 1 | 384.606 M | 384.606 M (**0%**) |
| Full-detector ConvMACs, 512 px, batch 1 | 1,906.680 M | 1,906.680 M (**0%**) |

AP values are ×100, same convention as the rest of this repository. The interval describes these five paired seeds after the head was chosen on the first two of them. It is not a COCO val2017 result and not a claim about unseen images. Details and the exact values are below.

## What changed

YOLO26n detects from three scales, P, M and G. The native head gives each scale its own weights. The new head keeps the first block of every scale native, then shares the later weights:

```text
Native, one scale family:
  P → first block → its own later kernel / its own BN → its own predictor + bias
  M → first block → its own later kernel / its own BN → its own predictor + bias
  G → first block → its own later kernel / its own BN → its own predictor + bias

spatial_sepbn_sharedpred:
  P → first block → shared later kernel / its own BN → shared predictor + its own bias
  M → first block → shared later kernel / its own BN → shared predictor + its own bias
  G → first block → shared later kernel / its own BN → shared predictor + its own bias
```

Box and class stay separate. The one-to-many and one-to-one branches stay separate. Nothing is shared across those four families. Inside one family, the shared kernel is still executed once per scale, so the operation count does not fall. What falls is the number of unique weights: **309,656 → 246,104**.

BatchNorm is the part that stays per scale: affine weights, bias, and the running statistics. The predictor kernel is shared, and the predictor bias is not. That is the whole difference from native. No width change, no extra loop, no 5×5 kernel.

## Five-seed development accuracy

Five head seeds, 11, 22, 33, 44 and 55. Every fit ran 40 epochs. The checkpoint is the earliest best development AP50:95, measured every 5 epochs. A selected epoch of 35 means that checkpoint scored best, not that training stopped early.

| Seed | `native` | `spatial_sepbn_sharedpred` | Difference | Selected epochs |
|---:|---:|---:|---:|---|
| 11 | 15.351 | 15.684 | **+0.333** | 40 / 40 |
| 22 | 15.181 | 15.368 | **+0.187** | 35 / 35 |
| 33 | 15.179 | 15.278 | **+0.099** | 40 / 40 |
| 44 | 15.220 | 15.552 | **+0.332** | 35 / 40 |
| 55 | 14.934 | 15.576 | **+0.642** | 35 / 40 |
| **Mean** | **15.173** | **15.492** | **+0.319** | |

Seeds 11 and 33 were the screen that chose this head. Seeds 22, 44 and 55 were new fits of the already chosen head and of fresh native controls. Those three confirmation seeds average **+0.387 points**, so the gain is not only the two seeds that picked the winner.

Mean paired difference **+0.319 points**, sample SD **0.206**, 95% t interval with df = 4 **[+0.062, +0.575]**. All five differences are positive. Exact rows: [`results/shared-spatial/five_seed.csv`](results/shared-spatial/five_seed.csv).

## How this head was chosen

Six heads were screened on seeds 11 and 33 only. The untied row copies the shared head's initialization but does not share weights. It is a control and could not win. The winner is the registered candidate with a strictly positive paired development mean. No other candidate was retrained.

| Head | Unique parameters | Mean AP, seeds 11 and 33 | vs `native` |
|---|---:|---:|---:|
| `native` | 309,656 | 15.265 | |
| Untied control | 309,656 | 15.347 | +0.082 |
| Shared later conv, separate BN | 271,960 | 15.408 | +0.143 |
| Shared later conv and GroupNorm | 270,552 | 14.935 | −0.331 |
| Shared later conv, separate BN, 5×5 | 282,712 | 15.191 | −0.075 |
| **Shared later conv, separate BN, shared predictor** | **246,104** | **15.481** | **+0.216** |

GroupNorm and the 5×5 kernel were worse than native on both screen seeds. Sharing the later convolutions alone was already a little higher than native. Sharing the predictor kernel as well was higher still, and it is the head in the five-seed table. Screen rows: [`results/shared-spatial/screen.csv`](results/shared-spatial/screen.csv).

## Compute

Counted on the selected seed-11 checkpoints at 512 px, with this repository's convolution counter. One MAC is one multiply-accumulate. Normalization, activation, pooling and decoding are not counted.

| | `native` | `spatial_sepbn_sharedpred` |
|---|---:|---:|
| Unique head parameters | 309,656 | 246,104 |
| Head ConvMACs, batch 1 | 384,606,208 | 384,606,208 |
| Full detector ConvMACs, batch 1 | 1,906,679,808 | 1,906,679,808 |
| Convolutions executed, head / full detector | 48 / 126 | 48 / 126 |

The head is 20.2% of the detector's ConvMACs. Those MACs stay, because each scale still runs the shared kernel. The parameter cut is storage, not fewer operations.

Latency and peak memory on the training GPU were **not measured** for this head. There is no speed claim.

## Method

- **Same trunk as the published looped-head study.** One frozen YOLO26n trunk, artifact SHA-256 `f01ec41445fac9ef7078b48ef1b73473e6314bcb1a77e77c20666d6ecfcd9016`, shared with [v0.2.0](https://github.com/Hmenez3s/detection-head-weight-sharing/releases/tag/v0.2.0). It was trained from scratch on the reduced COCO train split. These seeds are five heads on that trunk, not five backbones.
- **Frozen features.** Only the head is trained, on cached 512 px features. This is not end-to-end training.
- **Fixed dose.** 40 epochs, batch 32, development AP every 5 epochs, 2,000-image development split for selection, 1,000-image training probe. The probe is not held-out accuracy.
- **Eighteen closed fits, no failures.** Twelve screen fits (six heads × seeds 11 and 33) and six confirmation fits (`native` and the winner × seeds 22, 44 and 55). Every fit ran on one RTX 5070 Ti from commit `4272ae2`.
- **No COCO val2017 score.** This study did not run the holdout. Development AP selected the checkpoint and the head, so it is the measurement used to make those choices.

## What this number is not

- **Not the published val2017 number.** On COCO val2017 the native head of the earlier study scored 14.333. The 15.173 here is development AP on a different split. Do not subtract them.
- **Not proof on unseen images.** The development split chose both the checkpoint and, for two of the five seeds, the head. The three confirmation seeds were not used to choose the head, and they were positive too. That still is not val2017.
- **Not a speed result.** ConvMACs are unchanged. GPU latency was not measured.
- **One trunk, five seeds.** The interval is wide: the lower end is +0.062 points.
- **The screen spent its comparisons.** GroupNorm and 5×5 lost on two seeds and were not retrained. The untied control matches initialization of the shared convolutions only. It does not isolate every difference in the predictor.
- **Scope.** YOLO26n, one COCO subset, 512 px, frozen features, this recipe. No retune and no longer training.

## Previous result

The previous experiment in this repository asked a different question and got a negative answer. A looped head shared by P, M and G lost to native on COCO val2017: **14.333 vs 13.744 (−0.589 points)** for `loop_ab2`, and **13.600 (−0.733)** for `loop_a3`, each worse in 5 of 5 seeds. Those loops also used about 64% more head ConvMACs. That result is unchanged and is written up in [docs/looped-result.md](docs/looped-result.md). Release [v0.2.0](https://github.com/Hmenez3s/detection-head-weight-sharing/releases/tag/v0.2.0) has its checkpoints and evidence.

The head above does not repeat that design. It keeps the native blocks, shares compatible weights, and leaves BatchNorm per scale.

## Build this head

```python
from head_sharing.model import HeadStudyModel

model = HeadStudyModel(
    "native",
    11,
    512,
    head_spec={"version": 2, "variant": "spatial_sepbn_sharedpred"},
)
```

`head_spec=None` is the native head. The other screened variants are `spatial_sepbn`, `spatial_gn`, `spatial_sepbn_k5`, and the non-shared control `spatial_untied`. The measured scores in this README are the committed tables, not a promise that constructing the module reproduces the fit. COCO images, feature caches and the selected checkpoints are not in this repository.

Code is AGPL-3.0. See [LICENSE](LICENSE) and [NOTICE.md](NOTICE.md) for Ultralytics YOLO26 and for COCO terms.
