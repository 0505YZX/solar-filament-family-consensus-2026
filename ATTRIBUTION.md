# Attribution and scope

`cascade/solarpq/` is the physical-channel residual U-Net source by **tushhaaarrrr**, extracted from visible notebook cells at fixed script version **351734356**, with observed **Apache-2.0** notebook license. Original source fingerprints and URL are in `cascade/public_source_manifest.json`. The original backbone and its pretrained fold weights are not claimed as our original architecture.

Our cascade adds fold-specific native-pixel PCA crops, five-channel SMP U-Net++ refinement, expected-match scoring, and four-fold instance consensus. The original trained source is retained, including source notices. `cascade/inference_core.py` extracts five original functions unchanged; `infer.py` separates inference from training side effects.

U-Net++: Zhou et al., 2018, DOI 10.1007/978-3-030-00889-5_1. ResNet: He et al., 2016, DOI 10.1109/CVPR.2016.90. The SMP implementation is MIT licensed. External dependencies retain their own licenses. The organizer-supplied ACM report template retains its disclaimer and acknowledgment; its inclusion is not ACM endorsement.

AI assistance supported implementation, analysis, debugging, packaging and report drafting. No private test labels were available. No ownership claim is made over the competition data or third-party architectures.
