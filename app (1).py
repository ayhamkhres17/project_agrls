from flask import Flask, request, jsonify
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import ast
from flask_cors import CORS

# Load the model and tokenizer
model_name = "Salesforce/codet5-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

def correct_syntax(code):
    inputs = tokenizer(code, return_tensors="pt")
    outputs = model.generate(**inputs)
    corrected_code = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return corrected_code

def analyze_syntax(code):
    try:
        # Parse the Python code to its AST (Abstract Syntax Tree)
        tree = ast.parse(code)
        analysis_result = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                analysis_result.append(f"Assignment found at line {node.lineno}.")
            if isinstance(node, ast.FunctionDef):
                analysis_result.append(f"Function '{node.name}' defined at line {node.lineno}.")
        return "\n".join(analysis_result) if analysis_result else "No syntax issues found."
    except SyntaxError as e:
        return f"Syntax Error: {str(e)}"

def correct_and_analyze_syntax(code):
    # Correct the syntax using CodeT5
    corrected_code = correct_syntax(code)
    
    # Analyze the corrected code using AST parsing
    analysis_result = analyze_syntax(corrected_code)
    
    return {
        "corrected_code": corrected_code,
        "analysis": analysis_result
    }

@app.route('/correct', methods=['POST'])
def correct_code():
    data = request.json
    code = data.get('code')
    
    if not code:
        return jsonify({"error": "No code provided"}), 400

    try:
        result = correct_and_analyze_syntax(code)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
