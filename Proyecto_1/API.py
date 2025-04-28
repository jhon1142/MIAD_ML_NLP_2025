#!/usr/bin/python

from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os

# Cargar el modelo y los objetos necesarios
model = joblib.load('model_proyecto1.pkl')
artist_popularity_dict = joblib.load('artist_popularity_dict.pkl')
mean_popularity = joblib.load('mean_popularity.pkl')
encoder = joblib.load('track_genre_encoder.pkl')

# Definir las features esperadas
features = [
    'danceability', 'energy', 'loudness', 'speechiness', 'acousticness',
    'instrumentalness', 'valence', 'tempo', 'duration_ms', 'artist_popularity', 'track_genre'
]

app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    return """
<h1>Music Popularity Prediction API</h1>
<p>Use <code>POST /predict</code> with JSON body containing the audio features, artist name, and track genre.</p>
<pre>{
  "danceability": 0.8,
  "energy": 0.7,
  "loudness": -5.0,
  "speechiness": 0.04,
  "acousticness": 0.2,
  "instrumentalness": 0.0,
  "valence": 0.6,
  "tempo": 120.0,
  "duration_ms": 210000,
  "track_genre": "pop",
  "artists": "Dua Lipa"
}</pre>
"""

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        host = request.host
        return f"""
<h3>Send a POST request with JSON to this endpoint.</h3>
<p>Example:</p>
<pre>curl -X POST http://{host}/predict -H 'Content-Type: application/json' -d '{{"danceability":0.8,...,"artists":"Dua Lipa"}}'</pre>
"""

    try:
        # Recibir datos del JSON
        input_data = request.get_json()

        # Convertir a DataFrame
        df = pd.DataFrame([input_data])

        # --- Preprocesamiento igual que en entrenamiento ---
        if 'artists' in df.columns:
            df['artist_popularity'] = df['artists'].map(artist_popularity_dict)
            df['artist_popularity'].fillna(mean_popularity, inplace=True)
        if 'track_genre' in df.columns:
            df['track_genre'] = encoder.transform(df[['track_genre']])
            df['track_genre'] = df['track_genre'].astype(float)

        # Asegurar que están todas las columnas que el modelo espera
        df = df[features]

        # Predecir
        prediction = model.predict(df)[0]
        return jsonify({'predicted_popularity': float(prediction)})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)


