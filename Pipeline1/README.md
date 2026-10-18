# Human-Made Stone Structure Classification

**CS156 — Machine Learning Pipeline Assignment 1**

Over the past few years, my friends and I have collected nearly 10,000 photographs from Tibet. Most are organized by camera and date, so finding particular subjects can take a long time. For this project, I explored whether machine learning could help identify photographs containing **recognizable human-made stone structures**, including structures that blend into the surrounding rocks.

This is a **binary image-classification task**, not a method for verifying whether a structure is archaeologically authentic.

## Research questions

- **RQ1:** How does a Decision Tree using handcrafted image features compare with a classifier using pretrained DINOv2 visual features?
- **RQ2:** Does 10× training-data augmentation improve classification on unseen photographs?

## Data and preprocessing

I manually labeled **107 photographs**: 62 positive (a recognizable human-made stone structure is present) and 45 negative (no recognizable human-made stone structure).

| Dataset | Positive | Negative | Total |
| --- | ---: | ---: | ---: |
| Original training set | 49 | 36 | 85 |
| Held-out test set | 13 | 9 | 22 |
| 10× augmented training set | 490 | 360 | 850 |

The preprocessing workflow included converting Sony RAW/PNG images to JPEG, checking for near-duplicates with perceptual hashes, standardizing filenames, and creating a stratified 80/20 train/test split (seed `156`). No potential duplicate pairs were flagged in the 107-image screening. Each training image was kept along with nine variants using small changes in brightness, contrast, horizontal flipping, rotation, and Gaussian noise. **The 22 test images were never augmented.**

The original preprocessing code and its saved outputs are documented separately in [`notebooks/preprocessing_history.ipynb`](notebooks/preprocessing_history.ipynb). This is a record of the initial data preparation, not a step that needs to be rerun every time the classification notebook is opened.

## Machine learning pipeline

I compared four combinations of representation and training data:

| Representation | Classifier | Training condition |
| --- | --- | --- |
| 10 handcrafted grayscale, edge, and orientation features | Decision Tree | Original / 10× augmented |
| 384-dimensional embeddings from frozen DINOv2 Small (ViT-S/14) | L2-regularized Logistic Regression | Original / 10× augmented |

The Decision Tree uses Gini impurity and searches `max_depth = [2, 3, 4, 5, 6]`. For DINOv2, the pretrained visual backbone is **frozen**; a `StandardScaler` and Logistic Regression are fitted on the embeddings, searching `C = [0.01, 0.1, 1, 10]`.

Both approaches use the same **5-fold stratified cross-validation** on original training images. In each fold, 68 original images are used for training and 17 originals for validation. For the augmented condition, only variants of the 68 training-source images are included (680 training images per fold), preventing augmentation leakage. Hyperparameters are selected by **mean positive-class validation F1**, and the held-out test set is used only for final evaluation.

The full workflow and mathematical explanations are in [`notebooks/CS156_Pipeline_Assignment_1.ipynb`](notebooks/CS156_Pipeline_Assignment_1.ipynb).

## Results

All conditions were evaluated on the same **22 test images**.

| Model | Training data | Test accuracy | Positive-class F1 |
| --- | --- | ---: | ---: |
| Decision Tree | Original | 0.4545 | 0.6000 |
| Decision Tree | 10× augmented | 0.5455 | 0.6429 |
| DINOv2 + Logistic Regression | Original | 0.9545 | 0.9630 |
| DINOv2 + Logistic Regression | 10× augmented | 0.9545 | 0.9630 |
| Majority-class baseline (always positive) | — | 0.5909 | 0.7429 |

Cross-validation selected tree depths **5** (original) and **4** (augmented), and Logistic Regression `C` values **0.01** (original) and **0.1** (augmented).

On this small test set, the DINOv2 pipelines performed substantially better than the handcrafted-feature pipelines. Both DINOv2 classifiers correctly identified all 13 positives and 8 of 9 negatives; their only false positive was `negative_037.jpg`. Augmentation modestly improved the Decision Tree's test F1 but did not change DINOv2's test F1.

**Limitation:** The dataset includes only 85 independent training images and 22 test images. Augmented copies are not new independent scenes, and the two pipelines use different classifiers as well as different representations. These results do not establish statistical significance or broader superiority.

## Repository layout

```text
Pipeline1/
├── README.md
├── requirements.txt
├── notebooks/
│   ├── CS156_Pipeline_Assignment_1.ipynb   # Main experiment and results
│   └── preprocessing_history.ipynb         # Original data-preparation record
├── src/
│   ├── image_features.py                   # Handcrafted image features
│   └── dinov2_features.py                  # Frozen DINOv2 and embedding cache
├── results/
│   └── figures/
│       └── pipeline.png                    # Pipeline diagram
└── dataset/                                # Prepared photos (if provided)
    ├── train/{positive,negative}/
    ├── test/{positive,negative}/
    └── train_augmented/{positive,negative}/
```

The main notebook includes saved execution outputs, evaluation tables, and plots. Private photographs, including the false-positive example, are intentionally omitted. The source modules under `src/` are required for the main notebook to run.

## Running the notebooks

1. From the repository root, run `cd Pipeline1`. Use a Python 3.12 environment and install the dependencies with `python -m pip install -r requirements.txt`. The preprocessing notebook also uses `rawpy` and `ImageHash`.
2. Place the prepared images in the `dataset/` structure above. Open the project folder in VS Code or Jupyter and run `notebooks/CS156_Pipeline_Assignment_1.ipynb` from the top. The notebook expects `src/image_features.py` and `src/dinov2_features.py` to be present.
3. The DINOv2 extractor uses a pretrained model and caches embeddings locally under `.cache/`. Initial setup may require downloading model files; later runs can reuse cached features. On an Apple Silicon Mac, PyTorch can use the MPS backend when available.

**Data availability:** These are private personal photographs. Neither `data/` nor `dataset/` is published. Photo previews, original-to-renamed filename mappings, and model weights are also excluded. Derived feature caches are included as described below. The code and saved notebook outputs can be reviewed on GitHub, but rerunning the full experiment requires authorized access to the prepared images. Set `CS156_DATASET_DIR` to an alternative local dataset directory if needed. The historical preprocessing notebook writes output directories, so its paths and destinations must be reviewed before rerunning it.

### Included feature caches

The repository includes the existing numerical feature caches under [`.cache/`](.cache/):

- `decision_tree/augmented_features.csv`: handcrafted features for the 850 augmented training rows.
- `decision_tree/augmented_features_manifest.json`: the original cache provenance record, including relative image names, file sizes, modification times, and feature-source hashes.
- `dinov2/embeddings/<signature>/`: 384-dimensional embeddings for 85 original training images, 850 augmented training images, and 22 test images; a two-image smoke-check cache; and the shared content-addressed cache of 872 unique image hashes. Split caches include standardized filenames, class labels, source identifiers, and preprocessing configuration.
- `dinov2/model_download.json`: the pretrained model source, pinned revision, download URL, and checksum.

These files contain derived numerical features and metadata, not photograph pixels. Load an embedding archive with `numpy.load(path, allow_pickle=False)`; the `features` and `labels` arrays in a split archive can be used for classifier experiments without the photos. The main notebook still requires local images for dataset discovery, hashing, and handcrafted feature extraction. Automatic DINOv2 cache reuse also depends on matching the recorded model, preprocessing, source code, library versions, and device signature.

Downloaded third-party code and model weights (`.cache/dinov2/torch_hub/`), Matplotlib font caches, and temporary preservation records are not tracked. The extractor downloads the pinned DINOv2 model when needed.

## References

Key references include [DINOv2 (Oquab et al., 2024)](https://arxiv.org/abs/2304.07193), the [official DINOv2 code](https://github.com/facebookresearch/dinov2), [Classification and Regression Trees (Breiman et al., 1984)](https://books.google.com/books?id=JwQx-WOmSyQC), and [scikit-learn (Pedregosa et al., 2011)](https://jmlr.org/papers/v12/pedregosa11a.html). The full bibliography is included at the end of the main notebook.
