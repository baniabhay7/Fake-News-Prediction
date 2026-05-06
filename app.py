from flask import Flask, render_template, request, jsonify
import os
import re
import joblib
import numpy as np
from src.data_processing import TextPreprocessor
import config

try:
    from src.fact_checker import FactChecker
    FACT_CHECKER_AVAILABLE = True
except (ImportError, OSError) as e:
    print(f"⚠️ Fact Checker not available: {e}")
    print("   Install spacy with: pip install spacy && python -m spacy download en_core_web_sm")
    FactChecker = None
    FACT_CHECKER_AVAILABLE = False

app = Flask(__name__)

# Global variables for models and vectorizer
models = {}
vectorizer = None
fact_checker = None

SAMPLE_NEWS = {
    "real_1": {
        "title": "India Launches Gaganyaan Mission Successfully",
        "text": "The Indian Space Research Organisation successfully launched the Gaganyaan-1 unmanned test flight from Sriharikota on Monday morning. The mission marks a crucial milestone in India's human spaceflight program, testing critical systems including the crew module and escape mechanisms."
    },
    "real_2": {
        "title": "Mumbai Metro Line 3 Opens After Nine Years of Construction",
        "text": "The Mumbai Metro Rail Corporation inaugurated the 33.5-kilometer underground Metro Line 3 connecting Colaba to SEEPZ on Sunday. Chief Minister Eknath Shinde and Union Railway Minister flagged off the first train from the Bandra-Kurla Complex station."
    },
    "real_3": {
        "title": "Women's Cricket World Cup Victory",
        "text": "The Indian women's cricket team secured a historic victory in the World Cup final against Australia. The match went down to the last over, with clinical performances in both departments leading to a joyous celebration across the nation."
    },
    "fake_1": {
        "title": "SHOCKING: Government Announces 200% Tax on All Bank Deposits",
        "text": "BREAKING NEWS! Finance Ministry has announced a shocking 200 percent tax on all bank savings and fixed deposits starting next week! This means if you have 1 lakh rupees in your account, you will have to pay 2 lakh rupees as tax!"
    },
    "fake_2": {
        "title": "Hot Water Cures Diseases (Fake)",
        "text": "Incredible discovery! Doctors have found that drinking hot water at exactly 100 degrees Celsius can cure all known diseases including cancer and the common cold instantly! Share this with everyone to save lives!"
    }
}
AVAILABLE_MODELS = {
    'Naive Bayes': 'naive_bayes_model.joblib',
    'Logistic Regression': 'logistic_regression_model.joblib',
    'Random Forest': 'random_forest_model.joblib',
    'SVM': 'svm_model.joblib'
}

def load_models():
    """Load all available models and vectorizer"""
    global models, vectorizer
    
    try:
        # Load vectorizer
        vectorizer_path = os.path.join(config.MODELS_DIR, 'vectorizer.joblib')
        if os.path.exists(vectorizer_path):
            vectorizer = joblib.load(vectorizer_path)
            print("Vectorizer loaded successfully!")
        else:
            print("No vectorizer found.")
            return False
        
        # Load each model
        loaded_count = 0
        for name, filename in AVAILABLE_MODELS.items():
            model_path = os.path.join(config.MODELS_DIR, filename)
            if os.path.exists(model_path):
                models[name] = joblib.load(model_path)
                print(f"Model {name} loaded successfully!")
                loaded_count += 1
            else:
                print(f"Model {name} not found at {model_path}")
        
        if loaded_count == 0:
            print("No models were loaded. Please train models first.")
            return False
            
        print(f"Successfully loaded {loaded_count} models!")
        return True
            
    except Exception as e:
        print(f"Error loading models: {e}")
        return False

# Load models at startup
models_loaded = load_models()

if FACT_CHECKER_AVAILABLE:
    try:
        fact_checker = FactChecker()
        print("Fact checker initialized.")
    except Exception as e:
        print(f"Fact checker init failed: {e}")
        fact_checker = None
        FACT_CHECKER_AVAILABLE = False

@app.route('/')
def home():
    """Render the professional UI as default home page"""
    return render_template('index_professional.html',
                         model_loaded=models_loaded,
                         available_models=list(models.keys()),
                         sample_news=SAMPLE_NEWS)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests using ensemble of models"""
    
    try:
        if not models_loaded:
            return jsonify({
                'error': 'Models not loaded. Please train models first.',
                'success': False
            })
        
        # Get text from request
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided', 'success': False})
            
        text = data.get('text', '').strip()
        if not text:
            return jsonify({'error': 'Empty text provided', 'success': False})
        
        # Initialize preprocessor
        preprocessor = TextPreprocessor()
        processed_text = preprocessor.preprocess(text)
        print(f"\n--- Prediction Request ---")
        print(f"Original Text Length: {len(text)}")
        print(f"Processed Text: {processed_text[:100]}...")
        
        # Vectorize text
        text_vectorized = vectorizer.transform([processed_text])
        
        # Collect predictions from all models
        individual_results = {}
        fake_votes = 0
        real_votes = 0
        total_fake_prob = 0
        total_real_prob = 0
        
        print("Individual Model Scores:")
        for name, model in models.items():
            prediction = model.predict(text_vectorized)[0]
            prediction_proba = model.predict_proba(text_vectorized)[0]
            
            real_p = float(prediction_proba[0] * 100)
            fake_p = float(prediction_proba[1] * 100)
            
            total_real_prob += real_p
            total_fake_prob += fake_p
            
            pred_text = 'FAKE NEWS' if prediction == 1 else 'REAL NEWS'
            print(f"  - {name:20}: {pred_text} (R={real_p:.1f}%, F={fake_p:.1f}%)")
            
            if prediction == 1:
                fake_votes += 1
            else:
                real_votes += 1
                
            individual_results[name] = {
                'prediction': pred_text,
                'confidence': fake_p if prediction == 1 else real_p,
                'fake_probability': round(fake_p, 2),
                'real_probability': round(real_p, 2)
            }
        
        num_models = len(models)
        avg_fake_prob = total_fake_prob / num_models
        avg_real_prob = total_real_prob / num_models
        
        # Ensemble verdict
        final_verdict = 'FAKE NEWS' if fake_votes > real_votes else 'REAL NEWS'
        if fake_votes == real_votes:
            # Tie-break with average probabilities
            final_verdict = 'FAKE NEWS' if avg_fake_prob > avg_real_prob else 'REAL NEWS'
            
        final_confidence = avg_fake_prob if final_verdict == 'FAKE NEWS' else avg_real_prob
        
        print(f"Ensemble Result: {final_verdict}")
        print(f"Votes: Fake={fake_votes}, Real={real_votes}")
        print(f"Avg Probabilities: Fake={avg_fake_prob:.1f}%, Real={avg_real_prob:.1f}%\n")
        
        # Optional advisory fact check on the RAW text (digits still present)
        fact_check_result = None
        if FACT_CHECKER_AVAILABLE and fact_checker is not None:
            try:
                fact_check_result = fact_checker.analyze(text)
                if fact_check_result.get('warnings'):
                    print(f"Fact-check warnings: {fact_check_result['warnings']}")
            except Exception as fc_err:
                print(f"Fact checker skipped: {fc_err}")
                fact_check_result = None

        # Prepare response
        result = {
            'success': True,
            'prediction': final_verdict,
            'confidence': round(final_confidence, 2),
            'fake_probability': round(avg_fake_prob, 2),
            'real_probability': round(avg_real_prob, 2),
            'fake_votes': fake_votes,
            'real_votes': real_votes,
            'individual_results': individual_results,
            'model_used': 'Ensemble (Majority Vote)',
            'decision_type': 'Multi-Model Consensus',
            'fact_check': fact_check_result,
        }

        return jsonify(result)
        
    except Exception as e:
        import traceback
        print(f"PREDICT ERROR: {str(e)}")
        print(traceback.format_exc())
        return jsonify({
            'error': f'An error occurred: {str(e)}',
            'success': False
        })

@app.route('/about')
def about():
    """Render the about page"""
    return render_template('about.html')

@app.route('/api/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': models_loaded,
        'available_models': list(models.keys())
    })

if __name__ == '__main__':
    # Run the app
    port = int(os.environ.get('PORT', 4321))
    app.run(host='0.0.0.0', port=port, debug=True)
