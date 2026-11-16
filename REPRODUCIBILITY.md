# Reproducibility ledger

## Verified artifacts and computations

- Actual selected CSV from Kaggle V4 / 357558453: 180 images, 1281 rows, two legal empty images; canonical RLE, disjointness, exact IDs and readback pass. SHA256 is `85db516f7c7490e7d48a6dd953f408df7b28d4b6d2a47afb9207ed1528c9668d`.
- Twelve actual model files stored as 36 binary parts, with exact eight checkpoint identities and four scorer hashes. All parts reconstruct the original 553,083,408 bytes. The manifest is authoritative.
- Original dependency freeze, upstream checkpoint hashes and full 707-image fold metadata retrieved from the selected saved run. Counts are 177/176/177/177.
- Five inference functions extracted verbatim without training initialization; their hashes are in `inference_function_manifest.json`.
- Unchanged fusion rerun on all four saved candidate-record outputs reproduces the selected CSV byte for byte. All four learned scorers reproduce the saved candidate scores without fitting. See `evidence/saved_fusion_regression.json` and `evidence/saved_scorer_regression.json`.
- All eight neural checkpoints load with strict state matching and produce finite CPU diagnostic forwards. This verifies architecture/state compatibility, not full-image accuracy or CUDA numerical parity. See `evidence/model_loading_diagnostic.json`.
- Kaggle submission 57105017 (0.39) was confirmed as the sole final selection, 1/1.

## Scope and remaining execution limitation

The original successful Kaggle run completed training and full 180-image inference. The newly separated public image-to-CSV entry point has not been rerun end to end against the original images. Saved-record fusion regression does not replace that test. CUDA fp16/D4 are preserved; CPU or other-GPU bit-exact parity is not established. `--require-exact` makes a full reproduction fail unless its CSV SHA matches the scored reference. No new accuracy claim follows from packaging checks. Organizer form revision status is recorded separately from this computation ledger.

## Original outputs and attribution

- Own cascade V4: https://www.kaggle.com/code/zhixiangyang123123/solar-filament-pq-cascade-unet/output?scriptVersionId=357558453
- Public dense source, Apache-2.0 observed: https://www.kaggle.com/code/tushhaaarrrr/solar-filament-segmentation-2026-0-37?scriptVersionId=351734356

The former 0.41 method is historical and deselected. Prediction masks cannot reconstruct missing upstream code or parameters. Final eligibility and private ranking are decided by the organizers.
