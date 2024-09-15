from flask import Flask, request, jsonify
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from flask_cors import CORS

# Load the model and tokenizer
model_name = "Salesforce/codet5-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

def summarize_code(code):
    inputs = tokenizer(code, return_tensors="pt")
    outputs = model.generate(**inputs)
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summary

@app.route('/summarize', methods=['POST'])
def summarize():
    data = request.json
    code = data.get('code')
    
    if not code:
        return jsonify({"error": "No code provided"}), 400
    
    # Validate the input code
    if not isinstance(code, str) or len(code.strip()) == 0:
        return jsonify({"error": "Invalid code provided"}), 400

    try:
        summary = summarize_code(code)
        return jsonify({"summary": summary})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
