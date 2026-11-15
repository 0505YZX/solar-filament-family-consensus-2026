# Family-balanced consensus for solar filament segmentation

2026 competition solution by Zhixiang Yang. Final displayed public score: **0.41**. Last verified public rank: **147/1548** (2026-11-15); private rank unknown.

This is an attributed deterministic fusion and geometry postprocessor of public model predictions. It is **not** a claim to train or reconstruct all underlying models. Read [REPRODUCIBILITY.md](REPRODUCIBILITY.md) before using this package for final judging.

## Run

Tested with Python 3.13.5 on Windows, numpy 2.1.3, pandas 2.2.3, scipy 1.15.3 and pycocotools 2.0.11. A fresh Python environment with these packages and the Jupyter frontend of your choice is sufficient for the fusion notebook. No GPU, weights, credentials, network calls, retraining or raw images are used by this replay.

```
python -m pip install -r requirements.txt
python run.py --output replay_output
```

The output directory must not already exist. Alternatively run `pipeline.ipynb` from the repository root. Expected final output: 1212 rows across all 180 image IDs; SHA256 `68b26327bd9770d621753144ec29220af0e2b40094ee5756953af41409c94a09`. The final byte hash is required, in addition to ID, canonical RLE, overlap, empty-image and CSV readback checks. Five intermediate CSV hashes are also asserted. Submission content has already been scored by Kaggle; this replay does not submit again.

## Method

Reproduce prvsiyan V18 existence voting, compute equal-family pixel consensus with mutual-best source matches and conservative geometry/topology/occupancy guards, update the direct-YOLO family, add 13 independently corroborated detections, then average SOLFIRE inside the existing UNet family for a final bounded contour update. Correlated UNet models share one family; dual direct YOLOs share another. There are five family votes, not one vote per model. Every threshold is recorded in `pipeline/*protocol.json`.

## Provenance and licensing

See [ATTRIBUTION.md](ATTRIBUTION.md) and `input_manifest.json`. Original model authors retain credit. Apache-2.0 was observed for the prvsiyan and SOLFIRE notebooks. No blanket relicensing claim is made for third-party weights, images, predictions or code. Source use remains subject to its original license and competition rules. No full third-party weights are bundled. AI assistance was used for implementation, debugging, documentation and experiment analysis; the methods and limitations are explicitly described.

## Report

`technical_report.pdf` contains four body pages plus the mandated acknowledgment and references. Editable official-template sources are in `report/`. `final_submission.csv` is the already-scored final file and is provided for inspection; `run.py` does not use it as an input.
