The datasets used in the U-Net paper **"U-Net: Convolutional Networks for Biomedical Image Segmentation"** are:

1. **EM Segmentation Challenge Dataset (ISBI 2012)**
   - 30 images (512x512) from serial section transmission electron microscopy of *Drosophila* first instar larva ventral nerve cord (VNC).
   - Challenge page: [http://brainiac2.mit.edu/isbi_challenge/](http://brainiac2.mit.edu/isbi_challenge/)

2. **ISBI Cell Tracking Challenge 2014/2015**
   - **PhC-U373 Dataset** – Glioblastoma-astrocytoma U373 cells (phase contrast microscopy).
   - **DIC-HeLa Dataset** – HeLa cells (differential interference contrast microscopy).
   - Challenge page: [http://www.codesolorzano.com/celltrackingchallenge/Cell_Tracking_Challenge/Welcome.html](http://www.codesolorzano.com/celltrackingchallenge/Cell_Tracking_Challenge/Welcome.html)

---

### **Download Links:**
- EM Segmentation Challenge: [http://brainiac2.mit.edu/isbi_challenge/](http://brainiac2.mit.edu/isbi_challenge/)
  (You may need to create an account to access the dataset.)

- ISBI Cell Tracking Challenge: [http://www.celltrackingchallenge.net/2d-datasets/](http://www.celltrackingchallenge.net/2d-datasets/)
  (Download the **PhC-U373** and **DIC-HeLa** datasets.)

---

# Kaggle Competition: Carvana Image Masking Challenge

## Goal: To build an algorithm that can automatically remove the studio background from car images.

## Evaluation Metric: Mean Dice Coefficient

Formula:
\[ \text{Dice} = \frac{2 \cdot |X \cap Y|}{|X| + |Y|} \]

Where:
- \(X\) = predicted Mask
- \(Y\) = Ground Truth Mask

## Dataset:
- /train/ - contain training set images : 5088
- /test/ - contain test set images
- /train_masks/ - contain training set masks: 5088
- /sample_submission.csv - sample submission file
- /metadata.csv - basic information about all the cars

