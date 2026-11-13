# Real-Time Facial Expression Recognition

A Streamlit app that detects faces with MediaPipe and classifies each one into one of seven emotions — **Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral** — using Keras models trained on the FER-2013 and CK+ datasets.

It works on three kinds of input:

- **Webcam** — live detection with a per-frame emotion chart
- **Image upload** — JPG / PNG
- **Video upload** — MP4 / MOV / AVI, with adjustable frame skipping

Multiple faces per frame are supported, and an optional MediaPipe face-mesh overlay can be drawn.

## Project structure

```
├── facial-expression-app.py        # Streamlit application
├── deep_v5_final.ipynb             # Training & evaluation notebook (Google Colab)
├── Facial_Expression_Recognition.pdf  # Project report
├── model_deep/                     # Put the trained .keras models here (not in git)
└── requirements.txt
```

## Models

All models take a 48×48×3 input (grayscale faces repeated over 3 channels, scaled to `[0, 1]`) and output 7 softmax classes.

| Model | Architecture |
|---|---|
| VGG16 / VGG19 | ImageNet-pretrained base + GlobalAveragePooling + Dense(7) |
| EfficientNetB7 | ImageNet-pretrained base + GAP + Dense(512) + Dropout + Dense(7) |
| Custom CNN | 3 conv blocks (32/64/128, BatchNorm, Dropout) + Dense(1024) + Dense(7) |

Training uses Adam (lr 1e-4), data augmentation (rotation, shift, shear, zoom, horizontal flip), early stopping and learning-rate reduction on plateau, with a 90 / 10 train/validation split.

### Results (FER-2013, validation split)

Numbers taken from the saved run in `deep_v5_final.ipynb`:

| Model | Accuracy | Weighted F1 |
|---|---|---|
| **VGG16** | **68.0 %** | 67.7 % |
| VGG19 | 66.3 % | 66.3 % |
| EfficientNetB7 | 60.4 % | 59.3 % |
| Custom CNN | 56.8 % | 53.9 % |

The validation split is also used for early stopping, so these figures are slightly optimistic; a held-out test set (e.g. FER-2013's `PrivateTest` rows) would give a fairer estimate.

### Getting the model files

The trained models (~290–400 MB each) are **not stored in this repository**. Download them from the
[Google Drive folder](https://drive.google.com/drive/folders/1vowDVZAALaRUlM_0Alf22OtY400ZHbwV?usp=sharing)
and place them in `model_deep/` with these names:

```
model_deep/vgg16_fer13_model.keras
model_deep/vgg16_ck+_model.keras
model_deep/vgg19_fer13_model.keras
```

The app also looks for `vgg19_ck+`, `efficientnet_*` and `custom_cnn_*` models and lists them as unavailable when they are missing.

## Installation

Requires Python 3.8 – 3.11.

```bash
git clone https://github.com/<your-account>/Real-Time-Face-Expression-Recognition.git
cd Real-Time-Face-Expression-Recognition
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux / macOS
pip install -r requirements.txt
```

## Usage

```bash
streamlit run facial-expression-app.py
```

Then open the URL Streamlit prints (usually http://localhost:8501), pick a tab and a model.

## Known issues

These are open problems in the current code:

- **Preprocessing shape** — `preprocess_face` adds an extra channel axis, producing `(1, 48, 48, 3, 1)` instead of `(1, 48, 48, 3)`, and feeds colour crops while the models were trained on grayscale.
- **Version mismatch** — the app calls `st.image(..., use_container_width=True)`, which Streamlit 1.33 (pinned in `requirements.txt`) does not support. The models were also saved with Keras 3 / mixed precision, which may not load under the pinned TensorFlow 2.13.
- **Performance** — the model is reloaded on every Streamlit rerun (no `st.cache_resource`), and a new matplotlib figure is created per frame without being closed.
- **Webcam release** — pressing *Stop* reruns the script without stopping the capture thread, so the camera can stay locked.
- **Training notebook** — with `ImageDataGenerator` + `steps_per_epoch`, every second epoch trains on a single batch; the XGBoost / LightGBM / SVM entries are defined but not wired into the training loop.

## Datasets

- **FER-2013** — 35,887 grayscale 48×48 face images, 7 classes (in-the-wild, imbalanced: only 547 *disgust* samples)
- **CK+** — ~900 face-cropped 48×48 images, 7 classes (posed, controlled conditions; the *contempt* class is removed)

