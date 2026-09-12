"""
Application Flask — Classification de lésions cutanées (benign / malignant)
============================================================================
Sert le modèle ResNet50 fine-tuné entraîné dans le notebook, avec une
interface web permettant d'uploader une image et d'obtenir :
  - la classe prédite (benign / malignant)
  - le score de confiance
  - une visualisation Grad-CAM (zone de l'image ayant influencé la décision)

Lancement :
    pip install -r requirements.txt
    python app.py
Puis ouvrir http://127.0.0.1:5000
"""

import io
import os
import base64
import logging

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms, models
import torchvision.transforms.functional as TF
from torchcam.methods import SmoothGradCAMpp
import matplotlib
matplotlib.use("Agg")  # pas d'affichage interactif côté serveur
import matplotlib.pyplot as plt

from flask import Flask, request, jsonify, render_template

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
CLASS_NAMES = ["benign", "malignant"]     # ordre alphabétique, identique à ImageFolder
CHECKPOINT_PATH = os.environ.get("CHECKPOINT_PATH", "best_model.pth")
IMG_SIZE = 224
DECISION_THRESHOLD = float(os.environ.get("DECISION_THRESHOLD", 0.5))
MAX_UPLOAD_SIZE_MB = 8

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_SIZE_MB * 1024 * 1024


# --------------------------------------------------------------------------- #
# Modèle
# --------------------------------------------------------------------------- #
def build_model(num_classes: int = 2) -> nn.Module:
    """Doit correspondre exactement à l'architecture utilisée à l'entraînement."""
    model = models.resnet50(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(model.fc.in_features, num_classes),
    )
    return model


def load_model() -> nn.Module:
    model = build_model(len(CLASS_NAMES))
    if os.path.exists(CHECKPOINT_PATH):
        state_dict = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        model.load_state_dict(state_dict)
        log.info(f"Poids chargés depuis {CHECKPOINT_PATH}")
    else:
        log.warning(
            f"Checkpoint '{CHECKPOINT_PATH}' introuvable — le modèle utilise des poids "
            "aléatoires. Copiez best_model.pth (généré par le notebook d'entraînement) "
            "à la racine du projet Flask, ou définissez la variable d'environnement "
            "CHECKPOINT_PATH."
        )
    model.to(DEVICE)
    model.eval()
    return model


model = load_model()
cam_extractor = SmoothGradCAMpp(model, target_layer="layer4")

eval_transform = transforms.Compose([
    transforms.Lambda(lambda img: img.convert("RGB")),  # force RGB (cf. bug historique)
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# --------------------------------------------------------------------------- #
# Inférence + Grad-CAM
# --------------------------------------------------------------------------- #
def predict_and_explain(image: Image.Image):
    input_tensor = eval_transform(image).unsqueeze(0).to(DEVICE)
    input_tensor.requires_grad_()

    outputs = model(input_tensor)
    probs = torch.softmax(outputs, dim=1)[0]
    malignant_prob = probs[1].item()
    predicted_idx = 1 if malignant_prob >= DECISION_THRESHOLD else 0
    predicted_class = CLASS_NAMES[predicted_idx]
    confidence = malignant_prob if predicted_idx == 1 else 1 - malignant_prob

    # Grad-CAM par rapport à la classe prédite
    activation_map = cam_extractor(predicted_idx, outputs)[0]
    heatmap = TF.resize(activation_map.unsqueeze(0), input_tensor.shape[-2:])[0]
    heatmap = heatmap.squeeze().cpu().detach().numpy()

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image_np = input_tensor.squeeze().cpu().detach().numpy().transpose(1, 2, 0)
    image_np = np.clip(std * image_np + mean, 0, 1)

    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.imshow(image_np)
    ax.imshow(heatmap, cmap="jet", alpha=0.45)
    ax.axis("off")
    fig.tight_layout(pad=0)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    heatmap_b64 = base64.b64encode(buf.read()).decode("utf-8")

    return {
        "predicted_class": predicted_class,
        "confidence": round(confidence * 100, 1),
        "malignant_probability": round(malignant_prob * 100, 1),
        "benign_probability": round((1 - malignant_prob) * 100, 1),
        "threshold_used": DECISION_THRESHOLD,
        "heatmap_image": f"data:image/png;base64,{heatmap_b64}",
    }


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "Aucune image reçue."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Aucun fichier sélectionné."}), 400

    try:
        image = Image.open(file.stream)
    except Exception:
        return jsonify({"error": "Fichier image invalide."}), 400

    try:
        result = predict_and_explain(image)
    except Exception as exc:
        log.exception("Erreur pendant l'inférence")
        return jsonify({"error": f"Erreur pendant l'analyse : {exc}"}), 500

    return jsonify(result)


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "device": str(DEVICE),
        "checkpoint_loaded": os.path.exists(CHECKPOINT_PATH),
    })


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
