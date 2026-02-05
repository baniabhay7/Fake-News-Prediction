# Fake News Detection System 📰🤖

A robust, full-stack web application designed to detect and analyze news articles for misinformation using a 4-model machine learning ensemble.

## 🚀 Key Features

- **Professional UI Dashboard**: High-fidelity analysis dashboard with real-time metrics.
- **Ensemble Learning**: Uses a majority-vote consensus from:
  - Naive Bayes
  - Logistic Regression (90.0% Test Accuracy)
  - Random Forest (88.0% Test Accuracy)
  - SVM (90.4% Test Accuracy)
- **Reliability Metrics**: Distinguishes between High, Medium, and Low reliability based on model agreement.
- **Detailed Analytics**: Cross-model probability comparison and individual model stats.
- **Style Analysis**: Capable of spotting "Style Mimics" where tone is professional but facts are incorrect.

## 🛠️ Technology Stack

- **Backend**: Python, Flask
- **Machine Learning**: Scikit-learn, Joblib, NumPy, Pandas
- **Natural Language Processing**: NLTK (Tokenization, Lemmatization, Stopword removal)
- **Frontend**: Vanilla CSS, JavaScript, Chart.js
- **Environment**: Virtual environment support (.venv)

## 📦 Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/baniabhay7/Fake-News-Prediction.git
   cd Fake-News-Prediction
   ```

2. **Set up the Virtual Environment**:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Application**:
   ```bash
   python app.py
   ```
   Open `http://127.0.0.1:5000` in your browser.

## 📊 Training Results
The models were trained on a balanced dataset of 11,632 articles (2016-2023).

- **Logistic Regression**: 90.0% Accuracy
- **SVM**: 90.4% Accuracy
- **Random Forest**: 88.0% Accuracy
- **Naive Bayes**: 82.9% Accuracy

## 📄 License
This project is for educational purposes as part of a Major Project submission.
