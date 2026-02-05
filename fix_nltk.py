import nltk
import os

nltk_data_dir = os.path.abspath(".venv/nltk_data")
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir)

nltk.data.path.append(nltk_data_dir)
print(f"Downloading NLTK data to {nltk_data_dir}...")

try:
    for item in ['punkt', 'stopwords', 'wordnet', 'omw-1.4', 'punkt_tab']:
        print(f"Downloading {item}...")
        nltk.download(item, download_dir=nltk_data_dir, quiet=False)
    
    from nltk.tokenize import word_tokenize
    print("Tokenizing test:", word_tokenize("This is a test."))
    print("SUCCESS: NLTK is ready.")
except Exception as e:
    print(f"ERROR: {str(e)}")
