# Recorded environment

The exact selected Kaggle V4 output contains `environment/requirements_frozen.txt`. Python 3.13.15, CUDA PyTorch 2.11.0+cu128 and two Tesla T4 GPUs were recorded. `requirements.txt` pins the direct inference imports and model-library dependencies to this saved environment; it is not a fresh unpinned approximation. The full platform freeze also contains many unrelated preinstalled packages and multiple OpenCV variants. The focused installation uses opencv-python at the recorded version.

Install with Python 3.13 and a compatible NVIDIA GPU/driver. The selected recipe explicitly uses CUDA fp16 and D4 transforms. CPU numerical parity and cross-device bit-exactness are not established. Scikit-learn 1.6.1 is required for the own saved scorer state. No network download or training is needed for model initialization once this repository and dependencies are available. `build_refiner(False)` disables pretrained encoder downloads.
