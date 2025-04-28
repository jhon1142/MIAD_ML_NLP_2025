#!/usr/bin/python

from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
import xgboost as xgb
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

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Recibir datos del JSON
        input_data = request.get_json()

        # Convertir a DataFrame
        df = pd.DataFrame([input_data])

        # --- Preprocesamiento igual que en entrenamiento ---
        
        # Mapear artist_popularity si envían el nombre del artista
        if 'artists' in df.columns:
            df['artist_popularity'] = df['artists'].map(artist_popularity_dict)
            df['artist_popularity'].fillna(mean_popularity, inplace=True)

        # Codificar track_genre
        if 'track_genre' in df.columns:
            df['track_genre'] = encoder.transform(df[['track_genre']])
            df['track_genre'] = df['track_genre'].astype(float)

        # Asegurar que están todas las columnas que el modelo espera
        df = df[features]

        # Predecir
        prediction = model.predict(df)[0]

        return jsonify({
            'predicted_popularity': float(prediction)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)

