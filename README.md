# Hyper-YOLO-n: Optimized Object Detection

Hyper-YOLO-n is an enhanced object detection model based on YOLOv8-n, featuring **Hyper-Attention Masking (HAM)** and **High-Order Perception** modules.

## Key Achievements
- **mAP50:** 0.995 (12.8% improvement over baseline).
- **mAP50-95:** 0.742 (20.6% improvement).
- **Localization:** 51.1% reduction in Box Loss.

## Architecture Enhancements
1. **HyperC2Net:** Multi-scale feature fusion bridge.
2. **High-Order Information Perception:** Captures latent structural correlations.
3. **HAM (Hyper-Attention Masking):** Dual-path background suppression.

## Usage
To run evaluation:
```python
from hyper_yolo_eval import load_hyper_yolo
model = load_hyper_yolo('weights/best.pt')
results = model.val(data='coco_subset.yaml')
```

## Performance Visuals
Performance charts and confusion matrices can be found in the `documentation_images/` folder.
