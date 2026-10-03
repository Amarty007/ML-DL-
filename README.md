# 🤖 ML + DL Playground

Ye project ek interactive Streamlit app hai jisme aap CSV dataset upload karke
multiple machine learning aur deep learning models ko train aur compare kar sakte ho.

## ✨ Features

- CSV upload support for tabular classification datasets
- Target column selection
- Train/test split controls
- Automatic dataset preview and column information
- Model comparison for multiple classifiers
- Deep learning ANN model with live epoch-wise training curve
- Evaluation metrics such as accuracy, precision, recall, F1 score
- Confusion matrix and classification report
- Comparison table for all trained models
- Local SQLite dataset library: every uploaded CSV is saved and can be reused

## 🧠 Included Models

### Machine Learning
- Logistic Regression
- Ridge Classifier
- SGD Classifier
- Passive Aggressive Classifier
- Decision Tree
- Extra Tree
- Random Forest
- Extra Trees
- AdaBoost
- Gradient Boosting
- HistGradientBoosting
- K-Nearest Neighbors (KNN)
- SVM / Linear SVM
- Gaussian Naive Bayes
- Multinomial Naive Bayes
- Bernoulli Naive Bayes
- MLP Classifier

### Deep Learning
- Artificial Neural Network (ANN) built with scikit-learn MLP
- Live training progression tracking with accuracy/loss visualization

## 🚀 Setup

### 1. Python install hona chahiye
Recommended version: Python 3.9 to 3.11

### 2. Virtual environment create karo

```bash
python -m venv .venv
source .venv/bin/activate
```

Windows ke liye:

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Dependencies install karo

```bash
pip install -r requirements.txt
```

### 4. App start karo

```bash
streamlit run app.py
```

Browser me app normally `http://localhost:8501` par khulega.

## 📝 Usage

1. CSV file upload karo; dataset automatically local SQLite database me save hota hai
2. Purana dataset library se select karke reuse kar sakte ho
3. Target column select karo
4. Test size configure karo
5. ML model choose karke train karo
6. ANN ke settings adjust karke deep learning model train karo
7. Results aur model comparison table dekho

Database file `data/ml_dl_playground.db` me create hoti hai. Isme uploaded CSV ka
content aur basic metadata store hota hai, isliye Streamlit restart ke baad bhi
saved datasets available rehte hain.

## 📊 Sample Datasets

Agar demo ke liye dataset chahiye to inme se ek try kar sakte ho:
- Iris dataset
- Titanic dataset
- Any custom classification CSV

## ⚠️ Notes

- App mainly classification tasks ke liye designed hai
- Large files are limited to keep the app responsive
- For a clean setup, use the included virtual environment

## 🔧 Troubleshooting

- If dependency install fails, try reinstalling with:
  ```bash
  pip install --upgrade pip
  pip install -r requirements.txt
  ```
- Agar port busy ho to:
  ```bash
  streamlit run app.py --server.port 8502
  ```
- Agar TensorFlow installation issue aaye to:
  ```bash
  pip install tensorflow-cpu
  ```
