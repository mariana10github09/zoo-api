from flask import Flask, request, jsonify
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

# Columnas del dataset
colnames = [
    "name", "hair", "feathers", "eggs", "milk", "airborne", "aquatic", "predator",
    "toothed", "backbone", "breathes", "venomous", "fins", "legs", "tail",
    "domestic", "catsize", "class_type"
]

# Cargar dataset
data = pd.read_csv("zoo.data", names=colnames)

# Variables (X) y etiqueta (y)
X = data.drop(columns=["name", "class_type"])
y = data["class_type"]

# Diccionario class_type → nombre animal
class_to_name = data.groupby("class_type")["name"].first().to_dict()

# Entrenar modelo
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X, y)

@app.route("/predict", methods=["POST"])
def predict():
    body = request.json

    # Validación
    required = list(X.columns)
    for field in required:
        if field not in body:
            return jsonify({"error": f"Missing field: {field}"}), 400

    # Crear vector de entrada
    values = [body[col] for col in X.columns]

    # Predicción
    predicted_class = int(model.predict([values])[0])
    animal_name = class_to_name.get(predicted_class, "Unknown")

    return jsonify({
        "predicted_class": predicted_class,
        "animal": animal_name,
        "input_data": body
    })

# Configuración para Render (puerto dinámico)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
