# Dataset Documentation

This directory contains the datasets used to train and evaluate the object detection models for the self-checkout application.

## 1. COCO Dataset (Class 47 - Apple)

- **Source**: https://cocodataset.org/#home
- **Description**: A large-scale dataset containing over 200,000 labeled images across 80 categories. For this project, only the apple class (class ID 47) was used.
- **Preprocessing Steps**:
  - Images resized to 1280×1280
  - Normalization applied
  - Filtered to retain only apple class annotations
- **Augmentations**:
  - Mosaic augmentation
  - Random flipping
  - HSV color space adjustments

## 2. MinneApple Dataset

- **Source**: https://rsn.umn.edu/projects/orchard-monitoring/minneapple
- **Description**: A dataset focused on apples captured in orchard environments, particularly useful for detecting small and distant apples.
- **Usage in Project**:
  - Evaluated detection performance at various resolutions (320×320 to 1280×1280)
  - Conducted distance-based analysis using bounding box size
  - Helped derive recommendations for resolution selection based on object distance

These datasets were essential for both model training and evaluation in real-world use cases.
