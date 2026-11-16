# Reproducibility ledger

## Verified
- Actual selected CSV retrieved from saved Kaggle V4 output and matched to the historical scored SHA256.
- 180 image IDs, 1281 rows, canonical compressed RLE, disjoint instances, two legal empty images and CSV readback validated.
- Original complete training/inference source retained, with original public source attribution and hashes.
- Five inference functions copied verbatim from selected-run source, without executing its training initialization. The new entry point explicitly loads frozen weights.
- Kaggle final selection switched to submission 57105017 (0.39), sole 1/1.

## Pending; not silently waived
- Four dense `.pt` weights, four epoch-24 refiner `.pt` weights, four scorer `.joblib` files physically included in this public checkout and verified.
- Saved `requirements_frozen.txt` and fold metadata retrieved from the exact original run.
- New standalone image-to-CSV entry point executed against those artifacts and compared with the scored CSV.
- Updated organizer attachments confirmed after replacement.

The original successful Kaggle run is not evidence that this public checkout already contains every required weight. An external link does not meet a requirement to store weight bytes in the repository. Do not claim full material compliance until this ledger and the model manifest record actual completion.

## Original outputs
- Own cascade V4: https://www.kaggle.com/code/zhixiangyang123123/solar-filament-pq-cascade-unet/output?scriptVersionId=357558453
- Public dense source: https://www.kaggle.com/code/tushhaaarrrr/solar-filament-segmentation-2026-0-37?scriptVersionId=351734356

The former 0.41 method is historical and deselected. Its final prediction masks cannot reconstruct the author's missing private code, trained parameters, or dependencies.
