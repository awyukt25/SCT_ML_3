# Cat vs Dog Image Classification using SVM

A Support Vector Machine (SVM) classifier that distinguishes between images of cats and dogs, built on a ~4,000-image subset of the Kaggle "Dogs vs. Cats" dataset.

## Overview

| | |
|---|---|
| **Task** | Binary image classification (Cat vs Dog) |
| **Approach** | Classical ML: SVM with hand-crafted features (not deep learning) |
| **Model** | `sklearn.svm.SVC` (RBF kernel) inside a scikit-learn `Pipeline`, tuned via `GridSearchCV` |
| **Dataset** | Kaggle Dogs vs. Cats (~4,000 images, ~2,000 per class) |

## Why HOG + Color Histograms Instead of Raw Pixels

A naive approach, resizing images and flattening raw pixel values directly into the SVM, caps out around ~70–72% accuracy. Raw pixels mostly encode noise (lighting, background, pose) rather than the actual shape information that separates a cat from a dog.

This project instead extracts:

- **HOG (Histogram of Oriented Gradients):** captures edge orientation and local shape structure (ear shape, face contours), which is what visually distinguishes the two classes.
- **Color histograms:** a compact 96-value summary of color distribution per image, as complementary signal HOG doesn't capture.

This combination gives the SVM meaningfully more informative features to learn from, without switching to a deep learning approach.


## Workflow

1. **Load dataset:** mount Google Drive, verify image counts per class and plot the class balance.
2. **Preview samples:** visually inspect a few cat/dog images before processing.
3. **Feature extraction:** resize to 128×128, extract HOG features (grayscale) + normalized color histograms (per channel), and concatenate them into one feature vector per image. Extraction runs in parallel and the features are cached to Drive, so re-runs are fast.
4. **Train/test split:** 80/20, stratified to preserve the 50/50 class balance. The test set is held out and never used for fitting, scaling, or tuning.
5. **Pipeline:** `StandardScaler` (SVM is distance-based) → `PCA` (dimensionality reduction) → `SVC`. Because all steps live in one pipeline, they are fit on training data only, which prevents data leakage from the test set.
6. **Hyperparameter tuning:** `GridSearchCV` over `C` and `gamma` (RBF kernel) with stratified 5-fold cross-validation, instead of using default values.
7. **Evaluation:** accuracy, classification report, confusion matrix, ROC curve with AUC, and the best cross-validation score from the tuning step.
8. **Prediction visualization:** display random test images with actual vs predicted labels, plus a gallery of misclassified images to see where the model struggles.
9. **Reuse:** the trained model is saved with `joblib`, and a `predict_image(path)` helper classifies any new photo.

## Results

Fill in with your actual run output from Colab:

| Metric | Score |
|---|---|
| Test Accuracy | ~ 0.73 |
| Average CV Accuracy | ~ 0.72 |

## Limitations

This is a classical ML approach. HOG and color histograms are hand-crafted features, not learned representations. It will not match the accuracy of a Convolutional Neural Network (CNN), which learns its own features directly from raw pixels. This project's goal is to demonstrate SVM-based classification and feature engineering, not to reach state-of-the-art image classification accuracy.

## How to Run

1. Upload the cat/dog image folder to Google Drive (filenames like `cat.123.jpg`, `dog.456.jpg`).
2. Open `cat_dog_svm_improved.ipynb` in Google Colab.
3. Update `DATASET_PATH` (and optionally `CACHE_PATH` / `MODEL_PATH`) in the configuration cell to point to your Drive folder.
4. Run all cells top to bottom (note: HOG extraction over ~4,000 images and `GridSearchCV` both take a few minutes).

## Requirements

```
opencv-python
numpy
matplotlib
seaborn
scikit-image
scikit-learn
joblib
```

Install with:

```bash
pip install opencv-python numpy matplotlib seaborn scikit-image scikit-learn joblib
```

(Google Colab already includes most of these.)
