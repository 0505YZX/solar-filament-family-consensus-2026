# Recorded environment

Observed original CUDA preflight: Python 3.13.15, PyTorch 2.11.0+cu128, segmentation-models-pytorch 0.5.0, Tesla T4 GPUs. Source explicitly pins pycocotools 2.0.11. The complete original `requirements_frozen.txt` is pending download. Other package versions are not invented or presented as fully pinned.

Required imports include numpy, pandas, scipy, opencv-python, scikit-learn, joblib, torch, segmentation-models-pytorch and pycocotools. Scikit-learn serialization versions must match the saved scorer environment. Do not treat a fresh unpinned installation as verified numerical reproduction.
