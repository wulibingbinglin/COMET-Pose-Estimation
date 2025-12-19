# COMET: Continuous-time Trajectory-guided Temporal Modeling for Spacecraft Pose Estimation

## 1. Introduction

**COMET** is a framework designed for accurate 6-DoF pose estimation of non-cooperative spacecraft under challenging conditions (unknown geometry, extreme illumination, and rapid motion).

Existing approaches often struggle with generalization and scale ambiguity. We propose:

* **Trajectory-Guided Temporal Modeling (\mathfrak{T}_P & \mathfrak{T}_F):** Fuses local motion trajectories with continuous-time encodings to capture long-range dependencies.
* **Geometry-Aware Pose Regression (GAPR):** Jointly predicts rotation quaternions, image-plane translations, and relative depth to explicitly mitigate monocular scale ambiguity.

By unifying trajectory dynamics with temporal feature reasoning, COMET achieves state-of-the-art results on standard benchmarks.

## 2. Prerequisites

The code is tested on **Linux** (AutoDL) with **Python 3.10**.

### Core Dependencies

Please install the dependencies via `pip`. Key packages include:

* PyTorch (>= 1.12.0)
* OmegaConf (for configuration management)
* OpenCV-Python
* NumPy
* SciPy
* Matplotlib

### Installation

```bash
# Clone the repository
git clone [Your-Repository-URL]
cd COMET

# Install requirements
bash install.sh

```

## 3. Data & Checkpoints Preparation

Due to the large size of datasets and pre-trained weights, they are hosted on Baidu Netdisk. Please download them and organize the files as described below.

### 3.1 Download Links

* **Model Weights (Pre-trained Checkpoints)**
* **Download Link**: [点击下载权重 (Baidu Netdisk)](https://www.google.com/search?q=https://pan.baidu.com/s/1KdX2rlDw5IS_cOmikar7qQ%3F)
* **Password**: `2ftj`


* **Datasets (Evaluation & Demo Data)**
* **Download Link**: [点击下载数据集 (Baidu Netdisk)](https://www.google.com/search?q=https://pan.baidu.com/s/1KdX2rlDw5IS_cOmikar7qQ%3F)
* **Password**: `2ftj`



### 3.2 Directory Structure

After downloading, please organize your project root directory as follows:

```text
Project_Root/
├── models/
│   └── ckpt/                  <-- Place all downloaded .bin weights here
│       ├── best.bin           (Ours: Full COMET Model)
│       ├── abl_track.bin      (Ablation: w/o Trajectory Module)
│       ├── abl_time.bin       (Ablation: w/o Temporal Features)
│       ├── abl_uvz.bin        (Ablation: w/o GAPR Head)
│       └── abl_all.bin        (Baseline: w/o All Proposed Modules)
│
├── datasets/
│   ├── AMD/
│   │   └── AMD_eval/          <-- Evaluation Data for Ablation Studies
│   │
│   └── DCA_SpaceNet/
│       └── model1/            <-- Demo Data & Custom Testing Sequences

```

## 4. Usage

### 4.1 Quick Demo (Testing)

To evaluate the model on the sample sequence (DCA_SpaceNet), run the end-to-end inference script:

```bash
python test_e2epose2.py

```

**Configuration Options (in `test_e2epose2.py`):**
Modify these parameters in the script to customize the demo:

* `seqlen`: Input sequence length (Default: `16`).
* `visual_pose`: `True` to visualize predicted pose vs. Ground Truth.
* `demo_json`: `True` to save results in JSON format.

### 4.2 Training

If you want to train the COMET model from scratch or fine-tune it on your own dataset, please run the training script:

```bash
python train_e2epose2.py

```

**Training Tips:**

* Ensure your dataset is properly formatted in the `datasets/` directory.
* You can adjust training hyperparameters (learning rate, batch size, epochs) in the corresponding `.yaml` configuration file.
* Training logs and intermediate checkpoints will be saved in the `exp/` (or specified output) directory.

### 4.3 Reproduction of Results (Evaluation)

To evaluate the **Full COMET Model** and reproduce the main results reported in the paper:

```bash
python abl_ours.py

```

## 5. Ablation Studies

| Experiment | Module Removed | Description | Config File | Weight File |
| --- | --- | --- | --- | --- |
| **Ours (Full)** | None | Full COMET Framework | `abl_ours.py` | `best.bin` |
| **w/o \mathfrak{T}_P** | Trajectory Prior | Removes trajectory-guided modeling | `abl_track.yaml` | `abl_track.bin` |
| **w/o \mathfrak{T}_F** | Temporal Feat. | Removes long-range temporal dependencies | `abl_time.yaml` | `abl_time.bin` |
| **w/o GAPR** | Geometry Head | Replaces GAPR with direct regression | `abl_uvz.yaml` | `abl_uvz.bin` |
| **Baseline** | All | Removes all proposed modules | `abl_all.yaml` | `abl_all.bin` |

**How to run ablations:**
Modify the configuration loader in `abl_test.py`. For example, to test **w/o \mathfrak{T}_P**:

1. Open `abl_test.py`.
2. Load `abl_track.yaml`:
```python
if __name__ == '__main__':
    cfg = OmegaConf.load('abl_track.yaml')
    test_fn(cfg)

```


3. Run: `python abl_test.py`


## 6. Results

The following table presents the quantitative results of our ablation studies on the AMD evaluation dataset. These metrics validate the effectiveness of each proposed module in **COMET**.

| Experiment | Acc@5° (X) | Acc@5° (Y) | Acc@5° (Z) | R-Acc@5° | T-Acc@15° | **AUC @ 30** |
| --- | --- | --- | --- | --- | --- | --- |
| **Baseline (abl_all)** | 87.40% | 91.99% | 94.24% | 72.84% | 54.51% | 47.32% |
| **w/o \mathfrak{T}_P (abl_track)** | 87.13% | 92.57% | 93.72% | 75.08% | 57.05% | 49.77% |
| **w/o \mathfrak{T}_F (abl_time)** | 95.03% | 97.43% | 97.50% | 85.35% | 68.48% | 58.42% |
| **w/o GAPR (abl_xyz)** | 91.28% | 95.28% | 95.41% | 80.69% | 70.82% | 59.18% |
| **COMET (Ours)** | **95.82%** | **97.68%** | **97.41%** | **85.70%** | **77.48%** | **64.76%** |

### Key Observations:

* **GAPR Head Impact**: Comparing `ours` with `abl_xyz` (w/o GAPR), the AUC increases from **59.18% to 64.76%**, proving that explicitly modeling image-plane translations and relative depth effectively mitigates monocular scale ambiguity.
* **Temporal Reasoning**: The removal of long-range temporal features (`abl_time`) results in a significant drop in T-Acc@15° (**-9.0%**), highlighting the importance of temporal reasoning for stable translation estimation.
* **Overall Improvement**: Compared to the strong baseline (`abl_all`), **COMET** improves the overall AUC by **17.44%**, achieving state-of-the-art spacecraft pose estimation.

## Acknowledgments & Citations

If you use the datasets provided in this repository, please cite the original data source:

@article{liu2025tap,
  title={TAP-Track: Generalizable Spacecraft Pose Tracking by Tracking Any Points},
  author={Liu, Yating and Qi, Zhaoshuai and Chen, Pulin and others},
  journal={IEEE Transactions on Geoscience and Remote Sensing (TGRS)},
  year={2025},
  publisher={IEEE}
}

