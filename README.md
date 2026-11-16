# Solar filament segmentation: selected four-fold U-Net++ cascade

The sole selected final-scoring entry is **0.39**, Kaggle submission **57105017**, from our saved **V4 / 357558453** run. It replaces the previously selected 0.41 saved-public-prediction ensemble. The repository name is retained to preserve the link already submitted to the organizers.

The actual selected CSV is `final_submission.csv` (180 images, 1281 rows, two legal empty images), SHA256:
`85db516f7c7490e7d48a6dd953f408df7b28d4b6d2a47afb9207ed1528c9668d`.

## Packaging status: weights not yet included

The original complete cascade ran successfully on Kaggle. Its four trained refiners and four scorers remain in our saved private output, and its four dense backbones remain in the attributed public source output. **This public checkout does not yet include those twelve model files.** Browser binary downloads failed; official CLI download is being prepared. The frozen dependency snapshot is also pending retrieval. This is a concrete packaging gap, not a claim of complete organizer-compliant reproduction. `model_manifest.json` deliberately refuses inference until the exact model bytes are included and verified.

The original training notebook, all source modules, extracted saved-weight inference functions, validation, report and actual selected submission are included. `technical_report.pdf` describes this 0.39 method. No third-party prediction CSV is an inference input for this method.

## Intended saved-weight workflow

After the model ledger is complete and the recorded environment has been installed:

```sh
python verify_models.py
python infer.py --images /path/to/test/test_images --output result --require-exact
python validate.py result/submission.csv
```

`inference.ipynb` provides the same workflow. CUDA fp16/D4 behavior is retained; CPU numerical parity has not been established. The newly separated entry point has not yet been rerun end to end; the original complete Kaggle run is the evidence for the scored result.

## Method and evidence

Three physical image channels feed four residual U-Net foreground/spine backbones. Fold-matched five-channel native-pixel U-Net++ refiners and learned candidate scorers produce instances; four-fold consensus resolves overlap. The four refiners each completed 24 epochs (26952 total updates). Development OOF PQ was 0.446304 versus 0.433080 for the dense control. Thresholds were selected on that same development set; it is not a hidden-test estimate. Public score is displayed as 0.39; private score is unknown.

The former 0.41 fusion and its documentation remain available in repository history at commit `fbb427a5f33e534e02cf9b603039c0bd5e9ba6d3`. That result depended on missing private upstream assets and is not the selected cascade's performance. Final masks cannot recover those original trained parameters.

See `REPRODUCIBILITY.md`, `ATTRIBUTION.md`, `model_manifest.json` and `evidence/`. Final eligibility is determined by the organizers.
