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

# Suponemos que tienes un archivo CSV con las observaciones de validación
validation_set = pd.read_csv('validation_set.csv')

# Preprocesar las observaciones del set de validación de la misma manera que los datos de entrada
def preprocess_validation_data(df):
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

    return df

app = Flask(__name__)

@app.route('/')
def home():
    return """
    <h1>Music Popularity Prediction API</h1>
    <p>Use POST /predict with JSON body containing the audio features, artist name, and track genre.</p>
    <pre>
    {
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
    }
    </pre>
    """

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


@app.route('/validate', methods=['GET'])
def validate():
    try:
        # Tomar las primeras dos observaciones del set de validación
        validation_subset = validation_set.head(2)

        # Preprocesar los datos de validación
        preprocessed_data = preprocess_validation_data(validation_subset)

        # Predecir la popularidad para las dos observaciones
        predictions = model.predict(preprocessed_data)

        # Devolver las predicciones
        return jsonify({
            'predictions': [{
                'observation': validation_subset.iloc[i].to_dict(),
                'predicted_popularity': float(predictions[i])
            } for i in range(len(predictions))]
        })

    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)

