
````md
# CIFAR10 CNN Image Classifier 🧠📸

An end-to-end **PyTorch** project that trains a **Convolutional Neural Network (CNN)** on the CIFAR-10 dataset and serves it through a **FastAPI** web app where users can upload an image and get a prediction.

> ✅ From raw CIFAR-10 data → training → evaluation → CLI inference → web API with UI.

---

## ✨ Features

- **Dataset & Data Pipeline**
  - CIFAR-10 loading with `torchvision`
  - Train / validation / test splits
  - Data augmentation for training (random crop, flip, normalization)

- **Model**
  - Custom `SimpleCNN` built in PyTorch for 32×32 RGB images
  - Flexible `build_model()` factory (ready to plug in `resnet18` later)

- **Training**
  - Clean PyTorch training loop (`src/train.py`)
  - Tracks train/val loss & accuracy per epoch
  - Saves best model weights to `models/best_model_simple_cnn.pth`
  - Training & validation curves saved under `reports/`

- **Evaluation**
  - Test accuracy on holdout set (`src/eval.py`)
  - Confusion matrix (PNG)
  - Per-class accuracy bar plot
  - Metrics JSON for reproducibility

- **Inference**
  - CLI script (`src/inference.py`) for single-image prediction
  - FastAPI app (`app/main.py`) with:
    - `/` → web UI: upload image, preview, see top prediction & class probabilities
    - `/predict-image` → JSON API for programmatic access

---

## 🧱 Tech Stack

- **Language:** Python 3.x
- **DL Framework:** PyTorch, torchvision
- **API:** FastAPI, Uvicorn
- **Data:** CIFAR-10 (via `torchvision.datasets.CIFAR10`)
- **Visuals:** Matplotlib, Seaborn
- **Frontend:** Vanilla HTML/CSS/JS served from FastAPI (`app/static`)

---

## 📁 Project Structure

```bash
.
├── app/
│   ├── main.py                # FastAPI app (serves UI + prediction endpoint)
│   └── static/
│       ├── index.html         # Frontend page (upload + preview + results)
│       └── style.css          # Styling for the UI
│
├── src/
│   ├── dataset.py             # CIFAR-10 loaders + transforms
│   ├── model.py               # SimpleCNN + build_model factory
│   ├── train.py               # Training loop (saves best model)
│   ├── eval.py                # Evaluation on test set, confusion matrix, per-class acc
│   └── inference.py           # CLI inference on a single image
│
├── models/                    # Saved model weights (created after training)
│   └── best_model_simple_cnn.pth
├── reports/                   # Training & evaluation artifacts
│   ├── loss_curve.png
│   ├── accuracy_curve.png
│   ├── confusion_matrix_simple_cnn.png
│   └── per_class_accuracy_simple_cnn.png
│
├── data/                      # CIFAR-10 is downloaded here automatically
├── requirements.txt
└── README.md
````

> Note: some folders (`models/`, `reports/`, `data/`) are created at runtime.

---

## ⚙️ Setup & Installation

### 1. Clone the repo

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

### 2. Create and activate a virtual environment (recommended)

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🏋️‍♂️ Training the Model

Train the CNN on CIFAR-10 and save the best model:

```bash
python -m src.train
```

What this does:

* Downloads CIFAR-10 (if not already present) into `./data`
* Trains `SimpleCNN` for the configured number of epochs
* Saves the best weights (based on validation accuracy) to:

```bash
models/best_model_simple_cnn.pth
```

* Saves training curves to:

```bash
reports/loss_curve.png
reports/accuracy_curve.png
```

---

## 📊 Evaluation on Test Set

After training, run:

```bash
python -m src.eval
```

This will:

* Load `models/best_model_simple_cnn.pth`
* Evaluate on the test split
* Print:

  * Overall test accuracy
  * A detailed classification report (precision/recall/F1 per class)
* Save:

  * Confusion matrix → `reports/figures/confusion_matrix_simple_cnn.png`
  * Per-class accuracy → `reports/figures/per_class_accuracy_simple_cnn.png`
  * JSON metrics → `reports/test_metrics_simple_cnn.json`

---

## 🧪 CLI Inference (Single Image)

You can run inference on any image from the command line.

Example:

```bash
python -m src.inference --image sample_images/truck.jpg
```

Output (example):

```text
Using device: cpu

Prediction result:
Image: sample_images/truck.jpg
Predicted class: truck (index 9)
Confidence: 0.94
```

---

## 🌐 FastAPI Web App (Upload & Predict)

The project includes a small web UI + JSON API.

### 1. Start the server

From the project root:

```bash
uvicorn app.main:app --reload
```

You should see:

```text
Starting API. Using device: cpu
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

### 2. Web UI (for humans 🧑‍💻)

Open:

> [http://127.0.0.1:8000/](http://127.0.0.1:8000/)

You’ll see:

* File picker to upload an image
* Image preview
* Predicted CIFAR-10 class (`truck`, `cat`, `airplane`, etc.)
* Confidence percentage
* Table of probabilities for all 10 classes with small bars

### 3. JSON API (for programs 🤖)

Endpoint:

* `POST /predict-image`
* Form-data key: `file` (image file)

Example with `curl`:

```bash
curl -X POST "http://127.0.0.1:8000/predict-image" \
  -H "accept: application/json" \
  -F "file=@sample_images/truck.jpg"
```

Example JSON response:

```json
{
  "predicted_class": "truck",
  "predicted_index": 9,
  "confidence": 0.94,
  "all_probabilities": {
    "airplane": 0.04,
    "automobile": 0.001,
    "bird": 0.0007,
    "cat": 0.0006,
    "deer": 0.0006,
    "dog": 0.0001,
    "frog": 0.0000,
    "horse": 0.006,
    "ship": 0.009,
    "truck": 0.94
  },
  "filename": "truck.jpg"
}
```

You can also explore and test the API via the automatically generated docs:

> [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 🧩 Notes & Tips

* Training on CPU will be slow; if you have a GPU and proper CUDA setup, PyTorch will use it automatically.
* You can tweak:

  * Batch size, learning rate, epochs in `src/train.py`
  * Model architecture in `src/model.py`
  * Augmentations / transforms in `src/dataset.py`
* The code is written to be beginner-friendly if you're coming from **Keras/TensorFlow** and learning **PyTorch**.

---

## 🚀 Roadmap / Next Steps

* [ ] Add `resnet18` transfer learning option and compare results
* [ ] Improve training with better hyperparameters
* [ ] Dockerize the API for easier deployment
* [ ] Deploy to a cloud platform (Render / Railway / Fly.io / Hugging Face Spaces)

---

