# 🔍 Fake News Detection System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-2.0%2B-green)](https://flask.palletsprojects.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An AI-powered web application that uses Natural Language Processing (NLP) and Machine Learning to detect fake news articles with high accuracy. This project demonstrates end-to-end ML model development—from sophisticated data preprocessing to a modern web deployment—featuring a robust ensemble classification system.

## 🌟 Features

- **🤖 Ensemble ML Models**: 4 models (Naive Bayes, Random Forest, Logistic Regression, SVM) with a majority-voting consensus for maximum reliability.
- **🔍 AI Fact-Checker**: Entity verification via knowledge bases, numerical claim validation, and scam pattern detection.
- **🧠 Advanced NLP Pipeline**: Detailed text preprocessing including tokenization, lemmatization, stopword removal, and TF-IDF vectorization.
- **🎨 Modern Professional UI**: A beautiful, glassmorphism-inspired Flask-based dashboard with real-time analysis metrics.
- **⚡ Real-time Analytics**: Instant classification with dynamic confidence scores, model agreement stats, and reliability indicators.
- **🛡️ Smart Validation**: Edge case handling for varying text lengths, non-English characters, and formatting anomalies.
- **🧪 Production Ready**: Clean modular architecture with dedicated folders for data, models, and source code.

## 📁 Project Structure

```bash
fake-news-prediction/
├── app.py                 # Flask web application & Prediction API
├── config.py              # Configuration settings & Model paths
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
├── .gitignore             # Git exclusion rules
├── templates/             # UI Components
│   └── index_professional.html # Main dashboard
├── src/                   # Backend Intelligence
│   ├── data_processing.py # NLP preprocessing & Vectorization
│   ├── fact_checker.py    # Multi-layer claim verification
│   └── utils.py           # Helper & utility functions
├── data/                  # Storage Layer
│   ├── training_data.csv  # Combined training dataset
│   └── sample_data.csv    # Demo/test articles
├── models/                # Trained Intelligence
│   ├── *.joblib           # Pre-trained model & vectorizer files
│   └── training_results.txt # Performance benchmark logs
└── tests/                 # Quality Assurance
    └── test_model.py      # Unit tests for prediction logic
```

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Git

### Step 1: Clone the Repository
```bash
git clone https://github.com/baniabhay7/Fake-News-Prediction.git
cd Fake-News-Prediction
```

### Step 2: Create Virtual Environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS/Linux
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Setup NLP Resources
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4'); nltk.download('punkt_tab')"
```

## 💻 Usage

### Running the Web Application
```bash
python app.py
```
Open your browser and navigate to: `http://localhost:5000`

### Training Models
To retrain models with a custom dataset:
```bash
# Ensure your data is in data/training_data.csv
python retrain_models.py
```

## 📊 Dataset

### Training Data Coverage (2016-2023)
The models are trained on **11,632 professionally labeled articles** spanning multiple years to ensure temporal robustness:

- **2016-2017**: Political contexts and ISOT dataset articles.
- **2020-2021**: Health misinformation and COVID-19 related news.
- **2022-2023**: Modern fake news patterns from diverse digital sources.

**Dataset Composition:**
- Real News: 5,816 articles (50%)
- Fake News: 5,816 articles (50%)
- **Total**: 11,632 perfectly balanced articles.

## 🎯 Model Performance

Benchmarked results on **11,632 articles**:

| Model | Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **SVM** | **90.4%** | 90% | 90% | 91% |
| **Logistic Regression** | **90.0%** | 90% | 90% | 90% |
| **Random Forest** | **88.0%** | 88% | 88% | 88% |
| **Naive Bayes** | **83.1%** | 84% | 83% | 83% |

**Ensemble Strategy**: A majority-vote system that triggers more detailed analysis when models disagree, ensuring a "Safety-First" prediction.

## 🛠️ Technologies Used

- **Back-End**: Python, Flask
- **Machine Learning**: Scikit-Learn (NB, LR, RF, SVM)
- **Data Science**: Pandas, NumPy, Joblib
- **NLP**: NLTK (Natural Language Toolkit)
- **Front-End**: Vanilla JavaScript, CSS3 (Custom Glassmorphism), Chart.js

## 🧠 How It Works

1.  **Preprocessing**: Text is cleaned (removing URLs, special chars), tokenized, and lemmatized.
2.  **Vectorization**: The `TfidfVectorizer` converts text into numerical features using an N-gram (1,2) range.
3.  **Ensemble Prediction**: All 4 models run parallel predictions.
4.  **Result Aggregation**: The system calculates the consensus, average probability, and reliability score.

## 🎓 Key Skills Demonstrated

- ✅ **End-to-End ML Pipeline**: From raw text to a live dashboard.
- ✅ **Ensemble Methods**: Implementing majority-voting for improved generalization.
- ✅ **NLP Sophistication**: Advanced text cleaning and feature engineering.
- ✅ **Full-Stack Development**: Integrating Python logic with modern UI designs.

## ⚠️ Disclaimer & Limitations

### Use Responsibly
This system is an **educational tool**. It identifies patterns in writing style and linguistic features; it does not "know" the absolute truth.

### Two-Layer Detection
- **Layer 1 (ML)**: Spots emotional manipulation, sensationalism, and conspiracy patterns.
- **Layer 2 (Fact-Checker)**: Flags numerical impossibilities and cross-verifies entities.

## 👨‍💻 Author

**Abhay Bani**
- **GitHub**: [@baniabhay7](https://github.com/baniabhay7)
- **Project**: [Fake News Prediction](https://github.com/baniabhay7/Fake-News-Prediction)

## 📄 License
This project is licensed under the MIT License.

## 🙏 Acknowledgments
- **Scikit-Learn** & **NLTK** communities for the tools.
- **Kaggle** for providing professional misinformation datasets.

---
**Made with ❤️ and Python**
