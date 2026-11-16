"""solarpq -- PQ-aware, physics-informed solar filament segmentation.

Modules (each importable and testable on its own):
    config       hyper-parameters (single dataclass, serialisable)
    geometry     solar-disk fitting and radial (limb-darkening) flat-fielding
    features     physics-inspired input channels (flat field, local contrast, dark-ridge scale space)
    data         MAGFiLO/COCO parsing, multi-annotator soft targets, fold assignment, caching
    metrics      official-scorer-faithful PQ, fast RLE PQ, one-to-many / many-to-one statistics
    postprocess  probability map -> disjoint instances (hysteresis, closing, components)
    calibrate    decision-theoretic component scorer + PQ-optimal operating point
    model        residual U-Net
    train        fold training loop
    infer        D4 test-time-augmented, fold-ensembled inference
    export       RLE encoding, submission writer and validator
    classical    sparse-library logistic model (no deep learning) used as baseline / hedge
"""
__version__ = "1.1.0"   # + FN root-cause breakdown, dynamic session-clock training budget, finer OOM ladder
