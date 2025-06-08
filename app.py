from flask import Flask, render_template, request, jsonify
import pickle
import re
import nltk
from nltk.corpus import stopwords
    

app = Flask(__name__, template_folder='template')

# Download NLTK resources (run this only once)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Load the trained model and vectorizer
with open('model/spam_classifier.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('model/vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Define the cleaning function (same as in training)
def clean_text(text):
    text = text.lower()                              # 1. Lowercase
    text = re.sub(r'[^\w\s]', '', text)              # 2. Remove punctuation
    text = re.sub(r'\d+', '', text)                  # 3. Remove digits
    words = text.split()                             # 4. Split into words
    words = [word for word in words if word not in stopwords.words('english')]  # 5. Remove stopwords
    return ' '.join(words)                           # 6. Re-join to sentence

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == 'POST':
        # For AJAX/API requests
        if request.is_json:
            data = request.get_json()
            message = data.get("message", "")
        else:
            # For form submissions
            message = request.form.get("message", "")
        
        if not message:
            return jsonify({"error": "No message provided"}), 400
        
        # Clean the text
        cleaned_message = clean_text(message)
        
        # Vectorize
        message_vec = vectorizer.transform([cleaned_message])
        
        # Predict
        prediction = model.predict(message_vec)[0]
        prediction_proba = model.predict_proba(message_vec)[0][1]  # Probability of being spam
        
        # Format result
        result = {
            "message": message, 
            "is_spam": bool(prediction),
            "confidence": round(float(prediction_proba) * 100, 2) if prediction else round(float(1 - prediction_proba) * 100, 2)
        }
        
        # Return result based on request type
        if request.is_json:
            return jsonify(result)
        else:
            # For form submissions, render the template with results
            return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)