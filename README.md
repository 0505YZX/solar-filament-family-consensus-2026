# Solar filament segmentation: selected four-fold U-Net++ cascade

The sole selected final-scoring entry is **0.39**, Kaggle submission **57105017**, from our saved **V4 / 357558453** run. It replaces the previously selected 0.41 saved-public-prediction ensemble. The repository name is retained to preserve the link already submitted to the organizers.

The actual selected CSV is `final_submission.csv` (180 images, 1281 rows, two legal empty images), SHA256:
`85db516f7c7490e7d48a6dd953f408df7b28d4b6d2a47afb9207ed1528c9668d`.

## Frozen model package

The actual twelve learned model files are included as 36 binary parts under `weights/` (553,083,408 original bytes). `python verify_models.py` checks each part, reconstructs the exact four dense backbones, four epoch-24 refiners and four fold-specific scorers, then checks each original SHA256. The files came from the pinned successful Kaggle outputs, not from prediction masks. Original dependency freeze and 707-image fold metadata are in `environment/`.

The original training notebook, all source modules, extracted saved-weight inference functions, validation, report and actual selected submission are included. `technical_report.pdf` describes this 0.39 method. No third-party prediction CSV is an inference input for this method.

## Saved-weight workflow

Use Python 3.13 with a CUDA 12.8 compatible GPU. Install the recorded inference dependencies, then run:

```sh
python -m pip install -r requirements.txt
python verify_models.py
python infer.py --images /path/to/test/test_images --output result --require-exact
python validate.py result/submission.csv
```

`inference.ipynb` provides the same workflow. CUDA fp16/D4 behavior is retained; CPU numerical parity has not been established. The separated image-to-CSV entry point has not been rerun end to end. The original complete Kaggle run is evidence for the scored image inference. Independently rerunning the unchanged final fusion on its saved four-fold candidate records reproduces the scored CSV byte for byte; the four saved scorers also reproduce their recorded scores without fitting. These checks are not a new full image-inference benchmark.

## Method and evidence

Three physical image channels feed four residual U-Net foreground/spine backbones. Fold-matched five-channel native-pixel U-Net++ refiners and learned candidate scorers produce instances; four-fold consensus resolves overlap. The four refiners each completed 24 epochs (26952 total updates). Development OOF PQ was 0.446304 versus 0.433080 for the dense control. Thresholds were selected on that same development set; it is not a hidden-test estimate. Public score is displayed as 0.39; private score is unknown.

The former 0.41 fusion and its documentation remain available in repository history at commit `fbb427a5f33e534e02cf9b603039c0bd5e9ba6d3`. That result depended on missing private upstream assets and is not the selected cascade's performance. Final masks cannot recover those original trained parameters.

See `REPRODUCIBILITY.md`, `ATTRIBUTION.md`, `model_manifest.json` and `evidence/`. Final eligibility is determined by the organizers.
