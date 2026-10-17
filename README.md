# One looped head for every scale

<p align="center">
  <img src="assets/looped-head.png" width="100%" alt="Three stacked diagrams of the YOLO26n detection head, one internal copy each. Top, native: separate P, M and G heads, each with two blocks per box and class branch. Middle, loop_ab2: a separable 3×3 adapter and learned scale ID per scale and tower, then all 5,376 points in one list through a shared two-layer A→B loop applied twice. Bottom, loop_a3: the same adapters, then a shared single-layer A loop applied three times. Box and class towers stay separate and a shared 1×1 readout splits the points back to P, M and G. Below, the path of one location through the class tower and a table of widths and head MACs.">
</p>

**Question.** Can one looped detection head, shared by the P, M and G scales, replace YOLO26n's three separate heads? By reusing what it learns about objects at one size for the others, could it detect more accurately for a moderate increase in compute?

> **Status: implementation in progress.** The two new looped arms have not been trained yet. Native counts come from the existing control; looped-head counts are estimates, not new accuracy or speed results.

## The idea

- **Today (`native`)**, the fine, intermediate and coarse feature maps P, M and G have separate head weights. They support detection at different object sizes, but are not rigid small/medium/large bins.
- **Proposed**, box and class keep separate towers, as in the native head, but **each tower is shared by all three scales and looped**. Box and class do not share weights with each other.
- Per scale and per tower, an **adapter** (a depthwise 3×3 on the input channels plus a 1×1 projection to the tower width) brings the map to one width, and a **learned scale ID** is added. The three adapters of a tower are independent, specific to each scale, and still run three times, once per scale.
- The points of the three maps are flattened and concatenated into **one list of 5,376 points** (4,096 P · 1,024 M · 256 G). The loop uses only 1×1 convs with per-point channel norm and SiLU, so P, M and G points go through it together. Each pass applies the shared layers once to the whole list with a residual update `h ← h + 0.1·F(h)`. A shared 1×1 readout produces the box and class outputs, which are split back to the native P, M and G geometry.
- **The goal is accuracy and cross-scale learning at moderate extra compute, not fewer weights.** The scale ID lets the shared loop behave differently when scale matters, but it does not guarantee that conflicts between scales disappear.

## The three arms

| | `native` | `loop_ab2` | `loop_a3` |
|---|---|---|---|
| Heads | 3, one per scale | 1, shared by P, M and G | 1, shared by P, M and G |
| Box / class | separate | separate | separate |
| Width | box 16 · class 80 | box 24 · class 96 | box 28 · class 108 |
| Reused layers | none across scales | two layers A → B per tower, shared by P, M and G | one layer A per tower, shared by P, M and G |
| Passes | 2 blocks, once | 2 passes of `F = B(A(h))` | 3 passes of `F = A(h)` |
| Applications per point | 2 blocks | 4 | 3 |
| Conv applications per internal copy | 24 | 22 | 20 |
| Spatial view of the input feature map | 5×5 | 3×3 (adapter only) | 3×3 (adapter only) |
| Head MACs, both internal branches, 512 px | 384.6 M (counted) | ≈ 631.4 M (estimate) | ≈ 636.9 M (estimate) |

The native head takes P, M and G with 64, 128 and 256 channels at 64×64, 32×32 and 16×16 for a 512 px image. Its box branch is two conventional 3×3 convs at 16 channels, then a 1×1 to 4 outputs. Its class branch is two separable 3×3 blocks at 80 channels (each a depthwise 3×3 plus a pointwise 1×1, so two primitive convs), then a 1×1 to 80 outputs, with separate weights per scale.

**How to read the numbers.**
- YOLO26n is end-to-end and has two internal head branches (one2many and one2one). The table counts both; the figure draws one. A fused deployed head keeps one branch: about 192 M MACs native, 316 M `loop_ab2` and 318 M `loop_a3`.
- "Conv applications" counts layer applications per internal copy, not GPU kernels or latency. No speed claim is made.
- "Spatial view" is the 3×3 or 5×5 neighborhood on the head's own input feature map, not on the whole image. The only 3×3 in the looped arms is the adapter.
- Parameter counts are not predicted here; they will be reported from the actual implementation.

**Why box and class stay separate.** Locating an object and naming it can pull a shared layer in different directions. [YOLOX](https://arxiv.org/abs/2107.08430) found that separating them speeds up training and improves AP; in an end-to-end YOLO, like YOLO26n, a shared box-and-class head lost 4.2 AP versus 0.8 for separate ones. Merging them here would also blur what the experiment measures.

**Why two looped arms.** `loop_ab2` and `loop_a3` differ in how the same compute is spent: two distinct layers applied twice, or one layer applied three times. Their head MACs are within about 0.9% of each other. A separate CIFAR benchmark with spatial 3×3 layers and GroupNorm found AB2 ahead of A3 by 2.256 pp over five seeds at matched MACs ([benchmark](https://github.com/Hmenez3s/cnn-weight-sharing-generalization/tree/53d8caf/benchmarks/conv-granularity-20261017)). That is motivation only: it uses a different task, spatial layers and normalization, and says nothing about the YOLO outcome.

## Planned comparison

- **`loop_ab2` vs `loop_a3`:** does depth through distinct layers beat repeating one layer at about the same cost?
- **Each loop arm vs `native`:** does a shared looped head beat the original three heads, and is the extra compute worth it?

All arms use the same frozen YOLO26n trunk with features cached at 512 px, 40 epochs, batch size 32, development-split evaluation every 5 epochs for checkpoint selection, and five paired seeds. The `native` control is reused when its recipe is unchanged. AP, latency and memory are reported for every arm.

**Evaluation caveat.** COCO val2017 was already used in the earlier study in this repository. It is a reused evaluation benchmark here, not an untouched held-out confirmation.

## Repository

The code in `src/` and release [v0.1.0](https://github.com/Hmenez3s/detection-head-weight-sharing/releases/tag/v0.1.0) come from an earlier experiment. It tested a different, narrower change — repeating one layer inside each scale's head — and does not address this question.

The figure is generated by `scripts/draw_looped_head.py`. Code is licensed under AGPL-3.0; see [LICENSE](LICENSE) and [NOTICE](NOTICE.md).
