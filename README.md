# 🤖 ML + DL Playground

Ek interactive web app jisme CSV data upload karke Machine Learning aur Deep
Learning dono models ko train/test kar sakte ho, aur training process **live**
dekh sakte ho.

## ✨ Features

- **CSV Upload**: Koi bhi tabular dataset upload karo (Iris, Titanic, ya
  apna khud ka data)
- **Auto Preprocessing**: Missing values, categorical encoding, feature
  scaling — sab automatic
- **Machine Learning Models**:
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - K-Nearest Neighbors (KNN)
  - Support Vector Machine (SVM)
  - Naive Bayes
- **Deep Learning Model**: Configurable Artificial Neural Network (ANN)
  - Hidden layers, neurons, epochs, batch size — sab adjust kar sakte ho
  - **Live epoch-by-epoch training graph** (accuracy + loss)
- **Evaluation**: Accuracy, Precision, Recall, F1 Score, Confusion Matrix,
  Classification Report
- **Model Comparison**: Sab trained models ki accuracy ek table/chart mein
  compare karo

## 🚀 Kaise Run Karein

### 1. Python installed hona chahiye (3.9 - 3.11 recommended)

### 2. Virtual environment banao (optional but recommended)

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

### 3. Required libraries install karo

```bash
pip install -r requirements.txt
```

### 4. App run karo

```bash
streamlit run app.py
```

Ye command chalate hi browser mein automatically app khul jayega
(usually `http://localhost:8501`).

## 📝 Kaise Use Karein

1. Apni CSV file upload karo
2. Target column select karo (jo predict karna hai)
3. Test data ka percentage set karo
4. ML models select karke "Train karo" button dabao
5. DL model ke settings adjust karke train karo — live graph dekhoge
6. Neeche comparison table mein sab models ka result dekho

## 💡 Testing ke liye Sample Datasets

Agar apna data nahi hai, to ye try kar sakte ho:
- Iris dataset: https://archive.ics.uci.edu/dataset/53/iris
- Titanic dataset: Kaggle pe search karo "Titanic - Machine Learning from Disaster"

## 🔧 Troubleshooting

- **TensorFlow install issue**: Agar tensorflow install nahi ho raha, to
  `pip install tensorflow-cpu` try karo
- **Port already in use**: `streamlit run app.py --server.port 8502`
