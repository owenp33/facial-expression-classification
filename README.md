# Facial Expression Classification

How much does it help a model to understand the spatial layout of a face when recognizing expressions? This project compares four models on the FER2013 dataset of 48×48 grayscale face images. Each model captures more of the image's spatial structure than the one before:

1. An RBF-kernel **SVM** on PCA-reduced pixels
2. A fully connected **MLP**
3. A **CNN** built from scratch
4. A **ResNet-18** fine-tuned from ImageNet weights

Everything is in one notebook. The full write-up is in [Final-Report-4521.pdf](Final-Report-4521.pdf).

**Team:** Alistair White and Owen Peterson (CSCI 4521 final project, December 2025)

## Repository layout

```
.
├── FacialExpressionNotebook.ipynb   All data loading, models and evaluation, with saved outputs
├── Final-Report-4521.pdf            Final report: methods, results, discussion
└── README.md
```

The images aren't stored in this repository. See [Dataset](#dataset) for where to get them.

## Dataset

The project uses the [FER2013 dataset from Kaggle](https://www.kaggle.com/datasets/msambare/fer2013). It contains **35,887 grayscale face images, each 48×48 pixels**. The download is a zip file with two folders, `train/` and `test/`. Each folder has one subfolder per emotion, and **the subfolder's name is the image's label**:

```
train/                 test/
├── angry/             ├── angry/
├── disgust/           ├── disgust/
├── fear/              ├── fear/
├── happy/             ├── happy/
├── neutral/           ├── neutral/
├── sad/               ├── sad/
└── surprise/          └── surprise/
```

The notebook doesn't use Kaggle's train/test split. It pools both folders and makes its own random, stratified splits (see [Splits and class balancing](#splits-and-class-balancing)).

### Labels

| Label | Images | Share | Encoded as |
|---|---|---|---|
| `angry` | 4,953 | 13.8% | 0 |
| `disgust` | 547 | 1.5% | dropped |
| `fear` | 5,121 | 14.3% | 1 |
| `happy` | 8,989 | 25.0% | 2 |
| `neutral` | 6,198 | 17.3% | 3 |
| `sad` | 6,077 | 16.9% | 4 |
| `surprise` | 4,002 | 11.2% | 5 |

`disgust` has far fewer images than the other classes, so the final models leave it out and classify 6 emotions (35,340 images). `LabelEncoder` assigns the codes in alphabetical order.

The models see each image in one of two shapes:

- the SVM and the MLP get a flat vector of 2,304 pixel values
- the CNN and the ResNet get the 48×48 grid

## Notebook walkthrough

The notebook has four top-level sections:

| Section | What it does |
|---|---|
| **Data preparation** | Unzips the dataset, loads every image and its label, and plots the class counts and a t-SNE projection of the raw pixels |
| **Modeling (with disgust images)** | A first exploratory pass with all 7 classes, before `disgust` was dropped |
| **Dropping disgust images** | The final 6-class split and undersampling, then SVM + PCA, the MLP, and the CNN |
| **ResNet model [Extra Credit]** | ResNet-18 transfer learning on the 6 classes |

Each model is evaluated with:

- a classification report (per-class precision, recall and F1)
- a confusion matrix
- one-vs-rest ROC curves and AUC
- example images the model got wrong

The MLP, CNN and ResNet sections also show the images each model was most and least confident about.

### Splits and class balancing

Every split is stratified by label and uses `random_state=42`. The SVM and CNN split the data 60/20/20; the MLP and ResNet split it 80/20. Both splits give the **same 7,068 test images**, because the seed and the labels are the same.

| Model | Training data |
|---|---|
| SVM + PCA | 60% split, undersampled to 2,402 images per class (14,412 total). Tuned on the 20% validation split, then refit on the undersampled training set plus the validation split |
| MLP | 80% split (28,272). Each image is stored three times: original, flipped and rotated (84,816 total). Class-weighted loss |
| CNN | 60% + 20% splits combined (28,272), not undersampled. Random horizontal flips and ±15° rotations each epoch |
| ResNet-18 | The MLP's augmented training set (84,816) |

### Models

| Model | Architecture | Training |
|---|---|---|
| **SVM** | scikit-learn `SVC(kernel='rbf')` on PCA features that keep 95% of the variance (254 components) | Grid search over C ∈ {1, 5, 10, 20} with γ = scale, scored by macro F1 on the validation split |
| **MLP** | PyTorch: 2,304 → 128 → 64 → 6, with ReLU and dropout 0.3 | Adam (learning rate 1e-3, weight decay 1e-4), batch size 512, 30 epochs |
| **CNN** (`CNN_Emotion`) | Three blocks of [3×3 conv, BatchNorm, ReLU] × 2, followed by 2×2 max-pooling and dropout 0.25. Channels grow 32 → 64 → 128. Then a 4,608 → 256 dense layer (BatchNorm, dropout 0.5) and the output layer | Adam (learning rate 3e-4, weight decay 1e-4), batch size 64, up to 20 epochs. Stops when test loss goes up |
| **ResNet-18** (`ResNetClassifier`) | torchvision ResNet-18 with ImageNet weights. All layers are frozen except `layer4`. The final layer is replaced with a 512 → 256 → 6 head (BatchNorm, ReLU, dropout 0.5 and 0.3). Grayscale images are copied into 3 channels and normalized with ImageNet statistics | AdamW (head learning rate 1e-4, `layer4` learning rate 1e-5, weight decay 0.01) with a cosine schedule, batch size 64, up to 10 epochs. Stops after test loss fails to improve 3 epochs in a row |

No trained models are saved in the repository. The CNN training function writes `best_model.pt` to the working directory, and each CNN run overwrites it.

## Results

All scores are on the held-out test set of 7,068 images. The notebook's saved outputs and the final report come from different training runs, so both are shown.

| Model | Accuracy (notebook) | Macro F1 (notebook) | Accuracy (report) | Macro F1 (report) |
|---|---|---|---|---|
| SVM + PCA | 46.3% | 0.45 | 45.8% | 0.45 |
| MLP | 44.0% | 0.43 | 44.8% | 0.44 |
| CNN | **60.0%** | **0.58** | **64.7%** | **0.62** |
| ResNet-18 | 47.5% | missing | 48.7% | 0.47 |

In both runs the CNN did best, and `happy` and `surprise` were the easiest classes to recognize. `fear` was the hardest class for the three neural networks; the SVM did slightly worse on `angry`. When a model got a `fear` image wrong, its most common guess was `sad`, for all four models.

## Running the notebook

The notebook was written for Google Colab. A GPU makes the CNN and ResNet much faster.

1. Download the FER2013 zip from Kaggle. The zip must have `train/` and `test/` at its top level.
2. In the **Find files and unzip** cell, uncomment `file_path = "yourpath"` and set it to the zip's location. Mount Google Drive first if the zip is stored there; the mount lines are commented out in the first code cell.
3. Run all cells **from top to bottom** (see [Notes and known issues](#notes-and-known-issues)).

The next cell unzips to `/content/unzipped_data` using the shell command `unzip`. To run outside Colab, change `output_dir` and the two `base_dir` paths.

**Dependencies:** numpy, pandas, matplotlib, seaborn, Pillow, opencv-python, scikit-learn, imbalanced-learn, torch, torchvision

**Slow steps:**

- The t-SNE plot runs on all 35,887 images.
- An SVM on raw pixels was stopped after 1.5 hours without finishing, which is why the SVM uses PCA.
- The report says the CNN took about 40 minutes to train.

## Notes and known issues

- **The CNN and ResNet decide when to stop training based on test loss.** No separate validation set is used for this, so the test set influenced their training and their test scores are slightly optimistic. Also, the CNN's training function saves its lowest-test-loss weights to `best_model.pt`, but it returns and evaluates the weights from the final epoch.
- **The report describes a few settings differently from the code.**
  - It says the CNN was trained on undersampled data. The code trains it on the full training and validation splits.
  - It says the ResNet used label smoothing with learning rates of 1e-3 (head) and 1e-4 (`layer4`). The code uses no label smoothing, with learning rates of 1e-4 and 1e-5.
- **Run the notebook in order.** The MLP section overwrites `X_2d_train`, `y_train`, `X_2d_test`, `y_test` and `label_encoder`. The ResNet section depends on those new values.
- **Some images aren't real faces.** A few are pixel noise, cartoons, or stock photos with watermarks. They were found late in the project and left in the dataset.

## Citation

FER2013 was introduced in:

> Goodfellow, I. J., et al. (2013). "Challenges in Representation Learning: A report on three machine learning contests." *ICONIP 2013*. [arXiv:1307.0414](https://arxiv.org/abs/1307.0414)
