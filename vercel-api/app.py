import os, io, hmac
import numpy as np
from flask import Flask, request, jsonify
from PIL import Image
from ai_edge_litert.interpreter import Interpreter

API_KEY = os.environ["API_KEY"]
THRESHOLD = float(os.environ.get("THRESHOLD", "0.5"))
MODEL_NAME = os.environ.get("MODEL_NAME", "visionguard")
MODEL_VERSION = os.environ.get("MODEL_VERSION", "1")
IMG = 224

interp = Interpreter(model_path=os.path.join(os.path.dirname(__file__), "champion.tflite"))
interp.allocate_tensors()
inp, out = interp.get_input_details()[0], interp.get_output_details()[0]
app = Flask(__name__)

@app.get("/health")
def health():
    return jsonify(status="ok", model=MODEL_NAME, version=MODEL_VERSION)

@app.get("/")
def index():
    return jsonify(service="VisionGuard model API", model=MODEL_NAME, version=MODEL_VERSION,
                   endpoints={"GET /health": "status", "POST /predict": "send an image in form field 'file' with header X-API-Key"})

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return jsonify(message="Send a POST request with header X-API-Key and an image in form field 'file'."), 200
    if not hmac.compare_digest(request.headers.get("X-API-Key", ""), API_KEY):
        return jsonify(error="unauthorized"), 401
    f = request.files.get("file")
    if f is None:
        return jsonify(error="send an image in the 'file' field"), 400
    try:
        img = Image.open(io.BytesIO(f.read())).convert("RGB").resize((IMG, IMG), Image.BILINEAR)
    except Exception:
        return jsonify(error="could not read that image"), 400
    interp.set_tensor(inp["index"], np.asarray(img, dtype=np.float32)[None])
    interp.invoke()
    p = float(interp.get_tensor(out["index"])[0][0])
    label = "AI-generated" if p >= THRESHOLD else "Real"
    return jsonify(label=label, confidence=p if label == "AI-generated" else 1 - p, score=p,
                   model=MODEL_NAME, version=MODEL_VERSION)