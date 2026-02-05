import nltk
try:
    print("Checking NLTK data...")
    nltk.download('punkt', quiet=False)
    nltk.download('stopwords', quiet=False)
    nltk.download('wordnet', quiet=False)
    nltk.download('omw-1.4', quiet=False)
    
    from nltk.tokenize import word_tokenize
    print("Tokenizing test:")
    print(word_tokenize("This is a test."))
    
    from nltk.corpus import stopwords
    print("Stopwords test:")
    print(stopwords.words('english')[:5])
    
    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    print("Lemmatizer test:")
    print(lemmatizer.lemmatize("running"))
    
    print("✓ NLTK is working correctly!")
except Exception as e:
    print(f"❌ NLTK error: {e}")
    import traceback
    traceback.print_exc()
