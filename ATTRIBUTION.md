# Attribution and source ledger

The original model architectures and prediction files remain the work of their authors. Our contribution is the bounded family-balanced contour consensus, conservative recall supplement, packaging and audit. The original existence vote is adapted from prvsiyan, with original functions retained in `pipeline/upstream-cell-08.txt`.

- prvsiyan: Kaggle notebook `notebook24a4dce69e`, V18 / 368163500. https://www.kaggle.com/code/prvsiyan/notebook24a4dce69e?scriptVersionId=368163500 . Apache-2.0 observed on notebook page.
- _.tushhaaarrrr: UNet-family predictions, script versions 354107337 and 351734356. Files `tush040-v354107337.csv`, `tush037-v351734356.csv`.
- Dariush Afshar: historical UNet-family predictions, V2 / 359486068. File `dariush-v2-submission.csv`; pinning prevents the newer V3 output silently replacing the historical voter.
- HDJoJo: direct-YOLO V1 / 367892109 and V2 / 369397665. V2 combines YOLO26x and YOLOv8x, counted as one family. Files `hdjojo-v1-367892109.csv`, `hdjojo-latest-369397665.csv`.
- Muhammad Shoaib Altaf: MAGFiLO crop YOLO26 predictions, version 362037449. File `magfilo-v362037449.csv`.
- Tim Gluhkikh: crop YOLO11 predictions, version 342770268. File `tim-v342770268.csv`.
- Fususu / phuongncn and Nomannic19: public Mask R-CNN contribution and upstream credits. File `maskrcnn-thr070.csv` is the original threshold-0.70 intermediate. Exact network weights are not included here.
- Vedran Hrabar (vhrabar): SOLFIRE Final Pipeline V1 / 369652646, file `solfire-v1-369652646.csv`. Source https://github.com/vhrabar/solfire . Apache-2.0 observed on Kaggle notebook. This source remains within the UNet family, not a sixth independent family.

All prediction files were obtained through public Kaggle page/output interfaces; checksums are provided in `input_manifest.json`. Model-training details not independently verified are not claimed. If a source has additional license conditions, those conditions continue to apply. This ledger is attribution, not a blanket grant of rights to all third-party artifacts.
