from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from mangum import Mangum
import torch

app = Flask(__name__)
model_dir = "LLM_model"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

def predict(text):
    prompt = f"Complaint: {text}\nPredict:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=256)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

@app.route("/predict", methods=["POST"])
def predict_route():
    data = request.get_json()
    if "text" not in data:
        return jsonify({"error": "Please provide 'text' in request body."}), 400
    result = predict(data["text"])
    return jsonify({"prediction": result})

handler = Mangum(app)
