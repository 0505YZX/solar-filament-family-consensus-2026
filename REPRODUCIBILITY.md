# Scope of reproducibility

## Included and executable

All nine pinned public prediction input CSVs; 180 image IDs; original voting-function excerpt; attributed deterministic geometry and recall code; frozen protocol JSON files; exact-version requirements; a runnable Jupyter Notebook and CLI; per-input SHA256 manifest; expected hashes for all five generated stages; decoded-mask and serialization validation. The seed CSV is regenerated from input voters, not shipped as a hidden answer. The final scored CSV is available as the replay output.

Portability changes relative to the experiment code: remove a workstation-specific dependency path and take image IDs from `image_ids.json`. Historical duplicate-file checks were replaced by pinned historical checksums or comparisons to the included generated stages, so unrelated old experiments are not required as inputs. No matching, fusion, clipping or geometric threshold is changed. A clean replay on 2026-11-15 reproduced all five expected CSV byte hashes and passed final decoded-mask validation; see verification.txt. `build_boundary_consensus_v1_frozen.py` is retained unchanged as a historical code reference and is not the driver.

## Not verified or not supplied

The complete original-image-to-prediction inference pipelines, all exact upstream trained model weights, upstream training data splits, and full dependency environments for every family are NOT reconstructed here. The saved public predictions are necessary inputs. The original networks include UNet-based systems, direct and crop-based YOLO systems and Mask R-CNN. Their saved outputs are used with attribution; they are not independently retrained models by this entrant.

Consequently this package demonstrates exact prediction-fusion replay but does **not establish compliance** with the competition final-judging requirement that the complete model pipeline and weights be publicly available and reproducible from images. Organizers must assess eligibility on this disclosed basis. A public leaderboard score does not waive those requirements.

There is no matched labelled out-of-fold set for this exact public ensemble. Unlabelled family IoU agreement is not accuracy, PQ, a confidence interval or an independent validation. Repeated public leaderboard experiments introduce selection bias. No private leaderboard score or final award eligibility is asserted.
