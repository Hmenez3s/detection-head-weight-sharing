# Attribution and artifact licenses

## Code and detector implementation

This repository is licensed under the GNU Affero General Public License, version 3 ([LICENSE](LICENSE)). It uses **Ultralytics YOLO26**, copyright Ultralytics and its contributors, distributed under AGPL-3.0. The exact upstream version, source revision, wheel digest, and inspected file digests are recorded in [upstream.json](upstream.json). Upstream source and license: <https://github.com/ultralytics/ultralytics>.

The new study's detector checkpoints are derived from that implementation and are distributed with the same AGPL-3.0 license and their own provenance manifests. Dataset access is not a grant of rights to dataset photographs.

## COCO annotations and photographs

COCO annotations are copyright the COCO Consortium and licensed under [Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/). The official terms are <https://cocodataset.org/#termsofuse>. Dataset citation: Tsung-Yi Lin et al., *Microsoft COCO: Common Objects in Context*, ECCV 2014, <https://arxiv.org/abs/1405.0312>.

COCO does **not** own the copyright of its photographs. Individual image licenses apply. The dataset, image archives, and full annotation files are not included in git. Published example images must have individual source, creator attribution, license, and modification notices; the repository's code license does not replace those image licenses.

The COCO API software is copyright the COCO Consortium and uses its BSD-style software terms; see <https://cocodataset.org/#termsofuse> and <https://github.com/cocodataset/cocoapi>. Other installed dependencies retain their respective licenses and are not relicensed by this repository.
