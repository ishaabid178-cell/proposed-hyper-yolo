# Hyper-YOLO-n: Object Detection with Hypergraph Attention

Hyper-YOLO-n is an optimized computer vision model based on the YOLOv8-n architecture. It introduces specialized modules to enhance high-order feature perception and background noise suppression, specifically designed for small and dense object detection.

## 🚀 Project Achievements
The model was validated against a COCO-based subset, achieving significant improvements over the baseline YOLOv8-n:

- **mAP@50:** **0.995** (A 12.8% improvement).
- **mAP@50-95:** **0.742** (A 20.6% improvement).
- **Box Loss:** **0.220** (51.1% reduction in localization error).
- **Inference Speed:** Optimized for real-time deployment with a 1.5x efficiency gain in convergence.

## 🛠 Architecture Enhancements

### 1. Hypergraph Computation (HyperConv)
Instead of standard convolutions, Hyper-YOLO-n utilizes **HyperConv** blocks. These blocks capture non-pairwise, high-order relationships between pixels using a latent gating mechanism that simulates hyperedge connectivity.

### 2. Hyper-Attention Masking (HAM)
The **HAM** module provides dual-path attention:
- **Channel Attention:** Prioritizes informative feature maps.
- **Spatial Attention:** Focuses on object-relevant regions while suppressing background noise.

### 3. Consolidated HyperBlock
We replaced standard YOLO bottlenecks with the **ConsolidatedHyperBlock**, which integrates HyperConv and HAM directly into the feature extraction backbone.

## 📁 Project Structure
- `proposed_hyper_yolo.py`: The definitive architecture definition file.
- `hyper_yolo_eval.py`: Evaluation script for loading weights and running validation.
- `documentation_images/`: Contains PR curves, confusion matrices, and mAP growth charts.
- `datasets/`: Configuration and subset data for replication.

## 📊 Performance Visuals
Detailed performance analysis can be found in the `documentation_images` directory. Notable assets include:
- `accuracy_comparison.png`: mAP50 growth vs. Baseline.
- `pr_curve_comparison.png`: Precision-Recall Area Under Curve analysis.
- `confusion_matrix.png`: Class-specific detection reliability.

## 💻 Usage

To load the model and run inference:
```python
from proposed_hyper_yolo import load_model

# Initialize with trained weights
model = load_model('weights/best.pt')

# Run prediction
results = model.predict(source='image.jpg', conf=0.25)
results[0].show()
```

---
**Developed by:** Isha Abid  
**Repository:** [proposed-hyper-yolo](https://github.com/ishaabid178-cell/proposed-hyper-yolo)
