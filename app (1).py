from flask import Flask, request, jsonify
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import ast
from flask_cors import CORS

# Load the model and tokenizer for summarization
model_name = "Salesforce/codet5-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

def summarize_code(code):
    # Tokenize and encode the input code
    inputs = tokenizer(code, return_tensors="pt", truncation=True, max_length=512)
    outputs = model.generate(
        **inputs,
        max_length=150,  # Adjust length according to your needs
        num_beams=5,     # Beam search for better quality
        length_penalty=1.0,
        early_stopping=True
    )
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return clean_summary(summary)

def clean_summary(summary):
    # Clean up summary text
    summary = summary.strip()
    # Optionally, replace unwanted characters or patterns here
    return summary

def analyze_code(code):
    try:
        # Parse the code into an Abstract Syntax Tree (AST)
        tree = ast.parse(code)
        analysis = []

        # Walk through the AST and collect analysis data
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                analysis.append(f"A block of code named '{node.name}' starts here.")
            elif isinstance(node, ast.Assign):
                targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
                analysis.append(f"Data is stored in {', '.join(targets)} here.")
            elif isinstance(node, ast.For):
                analysis.append(f"Repetitive tasks are performed here.")
            elif isinstance(node, ast.If):
                analysis.append(f"Checks are performed here to decide what to do.")
        
        if not analysis:
            analysis.append("This code doesn't contain recognizable structural elements.")

        return "\n".join(analysis)

    except Exception as e:
        return f"Error during AST analysis: {str(e)}"

@app.route('/summarize', methods=['POST'])
def summarize():
    data = request.json
    code = data.get('code')

    if not code:
        return jsonify({"error": "No code provided"}), 400

    try:
        # Perform summarization and analysis
        summary = summarize_code(code)
        analysis = analyze_code(code)
        return jsonify({"summary": summary, "analysis": analysis})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
