import os
import pandas as pd
import numpy as np
import uvicorn
import tensorflow as tf
from keras.models import load_model
from flask import Flask, request, jsonify

app = Flask(__name__)

model = load_model('Model_fix.h5')



labels = {
    "beras": {
        "description": "Plant rice seeds in waterlogged soil. Ensure that the rice plants get enough water throughout the growing season. Control weeds around the crop to prevent competition for nutrients.",
        "image_url": "https://storage.googleapis.com/capstonec23-ps118.appspot.com/predict/beras.jpg"
    },
    "jagung": {
        "description": "Plant corn seeds in fertile, nutrient-rich soil. Apply water regularly, especially during cob formation. Apply nitrogen-containing fertilizer to improve plant growth.",
        "image_url": "https://storage.googleapis.com/capstonec23-ps118.appspot.com/predict/jagung.webp"
    },
    "buncis": {
        "description": "Plant chickpea seeds in an area that is exposed to full sunlight. Keep the soil moist with regular watering. Support the bean plants with tugal or wire to help vertical growth.",
        "image_url": "https://storage.googleapis.com/capstonec23-ps118.appspot.com/predict/buncis.jpg"
    },
    "kacangmerah": {
        "description": "Plant red bean seeds in moist, fertile soil. Keep the soil moist with regular watering. Provide vertical support such as tugal or wire as the plants grow.",
        "image_url": "https://storage.googleapis.com/capstonec23-ps118.appspot.com/predict/kacangmerah.jpg"
    },
    "kacanggude": {
        "description": "Plant the mung bean seeds in a place that is exposed to full sunlight. Make sure the soil is moist, but not waterlogged. Protect the plants from pests such as caterpillars or aphids.",
        "image_url": "https://storage.googleapis.com/capstonec23-ps118.appspot.com/predict/kacanggude.jpg"
    },
    "kacangngengat": {
        "description": "Caring for moth beans is the same as caring for green beans. Please refer to the previous answer.",
        "image_url": "https://storage.googleapis.com/capstonec23-ps118.appspot.com/predict/kacangngengat.jpg"
    },
    "kacanghijau": {
        "description": "Plant mung bean seeds in a place with full sun. Make sure the soil is moist with regular watering. Support the plants with a tugal or wire as they grow.",
        "image_url": "https://storage.googleapis.com/capstonec23-ps118.appspot.com/predict/kacanghijau.webp"
    },
    "gramhitam": {
        "description": "Plant black gram seeds in moist, fertile soil. Make sure the plants get enough sunlight. Provide vertical support such as a tugal or wire as the plants grow.",
        "image_url": "https://storage.googleapis.com/capstonec23-ps118.appspot.com/predict/gramhitam.jpeg"
    },
    "lentil": {
        "description": "Plant lentil seeds in loose, humus-rich soil. Keep the soil moist with regular watering. Keep the area around the plants clean to avoid weeds.",
        "image_url": "https://storage.googleapis.com/capstonec23-ps118.appspot.com/predict/lentil.jpg"
    },
    "delima": {
        "description": "Plant pomegranate seeds in a place that is exposed to full sunlight. Ensure that the soil remains moist, especially during fruit formation. Apply fertilizer rich in potassium to promote fruit growth.",
        "image_url": "https://storage.googleapis.com/capstonec23-ps118.appspot.com/predict/delima.jpg"
    },
    "pisang": {
        "description": "Plant banana seeds in fertile and moist soil. Ensure the soil is kept moist with regular watering.Apply fertilizers containing nitrogen and potassium for good growth.",
        "image_url": "https://storage.googleapis.com/capstonec23-ps118.appspot.com/predict/pisang.jpg"
    },
    "mangga": {
        "description": "Plant mango seedlings in a place that is exposed to full sunlight. Keep the soil moist with regular watering. Apply fertilizer containing complete nutrients for good growth and flowering.",
        "image_url": "https://storage.googleapis.com/capstonec23-ps118.appspot.com/predict/mangga.jpeg"
    },
    "anggur": {
        "description": "Plant grape seedlings in a place with full sun. Ensure moist soil with regular watering. Support the grape vines with a series or fence for vine growth.",
        "image_url": "https://storage.googleapis.com/capstonec23-ps118.appspot.com/predict/anggur.jpg"
    },
    "semangka": {
        "description": "Plant watermelon seeds in loose, fertile soil. Ensure the soil is moist with regular watering. Apply fertilizers containing nitrogen and potassium for good plant growth and fruit formation.",
        "image_url": "https://storage.googleapis.com/capstonec23-ps118.appspot.com/predict/semangka.jpg"
    },
    "melon musk": {
        "description": "Plant musk melon seeds in a place that is exposed to full sunlight. Keep the soil moist with regular watering. Support the melon plants with a tugal or wire when growing.",
        "image_url": "https://storage.googleapis.com/capstonec23-ps118.appspot.com/predict/melon_musk.jpg"
    },
    "apel": {
        "description": "Plant apple seeds in a spot that is exposed to full sunlight. Keep the soil moist with regular watering. Protect the plants from pests and diseases common to apples.",
        "image_url": "https://storage.googleapis.com/capstonec23-ps118.appspot.com/predict/apel.webp"
    },
    "jeruk": {
        "description": "Plant orange seeds in a place with full sun. Ensure the soil is moist with regular watering. Apply fertilizer rich in nutrients for good growth and flowering.",
        "image_url": "https://storage.googleapis.com/capstonec23-ps118.appspot.com/predict/jeruk.jpg"
    },
    "pepaya": {
        "description": "Plant papaya seedlings in a place that is exposed to full sunlight. Keep the soil moist with regular watering. Keep the area around the plant clean to prevent weed growth.",
        "image_url": "https://storage.googleapis.com/capstonec23-ps118.appspot.com/predict/pepaya.jpg"
    },
    "kelapa": {
        "description": "Plant coconut seeds in a place that is exposed to full sunlight. Ensure moist soil with regular watering. Support the coconut plant with supports as it grows.",
        "image_url": "https://storage.googleapis.com/capstonec23-ps118.appspot.com/predict/kelapa.jpg"
    },
    "katun": {
        "description": "Plant cotton seeds in fertile, well-drained soil. Ensure the soil is kept moist with regular watering. Protect the plants from pests such as aphids and caterpillars.",
        "image_url": "https://storage.googleapis.com/capstonec23-ps118.appspot.com/predict/katun.jpg"
    },
    "rammi": {
        "description": "Plant jute seeds in loose, nutrient-rich soil. Keep the soil moist with regular watering. Keep the area around the plant clean to prevent weeds.",
        "image_url": "https://storage.googleapis.com/capstonec23-ps118.appspot.com/predict/rami.jpg"
    },
    "kopi": {
        "description": "Plant coffee seeds in a place with partial to full sun exposure. Ensure the soil is moist with regular watering. Apply special fertilizer for coffee plants as per package instructions.",
        "image_url": "https://storage.googleapis.com/capstonec23-ps118.appspot.com/predict/kopi.jpg"
    }
}

@app.route("/predict", methods=["POST"])
def predict():
    data = [
        request.form.get("N"),
        request.form.get("P"),
        request.form.get("K"),
        request.form.get("temperature"),
        request.form.get("humidity"),
        request.form.get("ph"),
        request.form.get("rainfall")
    ]

    # Jika data tidak ada
    if None in data:
        return jsonify({"message": "Maaf, harap isi semua field yang diperlukan."})
    
    try:
        prediksi = np.array([eval(val) for val in data])
        if not isinstance(prediksi, np.ndarray):
            raise ValueError
    except (ValueError, SyntaxError):
        return jsonify({"message": "Maaf, format data gejala tidak valid."})
    
    # Konversi data menjadi format yang dapat digunakan oleh model
    prediksi_df = pd.DataFrame([prediksi], columns=["N", "P", "K", "temperature", "humidity", "ph", "rainfall"])

    # Prediksi
    predictions = model.predict(prediksi_df)

    # Menentukan label berdasarkan nilai probabilitas
    predicted_class = np.argmax(predictions[0])
    predicted_label = list(labels.keys())[predicted_class]

    predictions_result = {
        "label": predicted_label,
        "description": labels[predicted_label]["description"],
        "image_url": labels[predicted_label]["image_url"]
    }
    return jsonify(predictions_result)


    # return jsonify({"message": "Prediksi: " + predicted_label})

port = os.environ.get("PORT", 8080)
print(f"Listening to http://0.0.0.0:{port}")
app.run(host='0.0.0.0',port=port)