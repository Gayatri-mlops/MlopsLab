#  Breast Cancer Diagnosis Prediction App

A **Dockerized Flask Web Application** that predicts whether a breast tumor is **benign or malignant** using the **Breast Cancer Wisconsin dataset** from scikit-learn.
The model is trained using **TensorFlow/Keras** and served as an interactive web form for real-time predictions.

##  Project Overview

This project demonstrates a complete **end-to-end MLOps workflow** — from model training to deployment — within a single Docker container.
It’s divided into two main stages:

1. **Model Training (Stage 1)**

   * Loads the Breast Cancer dataset
   * Scales features using `StandardScaler`
   * Trains a neural network with TensorFlow/Keras
   * Saves the trained model (`cancer_model.keras`), scaler (`scaler.pkl`), and metadata (`metadata.json`)

2. **Model Serving (Stage 2)**

   * Uses Flask to create a RESTful web app
   * Hosts an HTML form for user input (30 numeric features)
   * Returns model predictions (Benign or Malignant) with confidence scores

##  Tech Stack

* **Programming Language:** Python 3.9
* **Libraries:**

  * TensorFlow / Keras
  * scikit-learn
  * Flask
  * NumPy / Pandas
* **Containerization:** Docker
* **Deployment:** Exposes Flask app on port `4000`

##  Project Structure

```
MLOpsLab/
│
├── src/
│   ├── model_training.py     # Model training & preprocessing
│   ├── main.py               # Flask app for inference
│   ├── templates/
│   │   └── predict.html      # Frontend UI for prediction
│   └── statics/              # Static CSS or JS (optional)
│
├── requirements.txt          # Dependencies
├── Dockerfile                # Multi-stage Docker build
└── README.md                 # Project documentation
```

##  Model Details

* **Dataset:** Breast Cancer Wisconsin (Diagnostic) Dataset (`sklearn.datasets.load_breast_cancer`)
* **Input Features:** 30 numerical features extracted from digitized breast tumor images
* **Output Classes:**

  * `Malignant`
  * `Benign`
* **Architecture:**

  * Input: 30 neurons
  * Hidden layers: 64 → 32 (ReLU activation, dropout for regularization)
  * Output: 1 neuron (sigmoid activation for binary classification)

##  Installation & Setup

### 1️ Clone the Repository

```bash
git clone https://github.com/<your-username>/MLOpsLab.git
cd MLOpsLab
```

### 2️ Build the Docker Image

```bash
docker build -t breast-cancer-app .
```

### 3️ Run the Container

```bash
docker run -p 4000:4000 breast-cancer-app
```

### 4️ Access the Web App

Open your browser and go to:

```
http://localhost:4000/predict
```
##  API Endpoint (Optional Use)

You can also send POST requests directly to the `/predict` endpoint with JSON data.

**Example Request:**

```bash
curl -X POST http://localhost:4000/predict \
  -F mean_radius=14.1 \
  -F mean_texture=19.3 \
  -F mean_perimeter=92.2 \
  ... (remaining 27 features)
```

**Example Response:**

```json
{
  "diagnosis": "benign",
  "confidence": 0.93,
  "probabilities": {
    "malignant": 0.07,
    "benign": 0.93
  }
}
```
##  Example Prediction UI

The web form automatically loads all 30 feature fields.
You can also click **“Load Example”** to autofill demo values for quick testing.

##  Requirements

If you’re running locally (without Docker), install dependencies manually:

```bash
pip install -r requirements.txt
python src/model_training.py
python src/main.py
```

##  Screenshots

**Training Stage (Docker build logs):**

* Shows dataset loading, model training progress, and evaluation metrics.

**Serving Stage:**

* User-friendly prediction page built with HTML, CSS, and JavaScript.
* Displays predicted class with confidence bar visualization.

