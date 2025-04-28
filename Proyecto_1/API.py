import pickle
from flask import Flask, request, jsonify

# Carga tu modelo previamente entrenado
with open('modelo_entrenado.pkl', 'rb') as model_file:
    modelo = pickle.load(model_file)

# Inicializa la aplicación Flask
app = Flask(__name__)

# Definir la ruta para hacer predicciones
@app.route('/predict', methods=['POST'])
def predict():
    # Obtén los datos enviados en el cuerpo de la solicitud
    datos = request.get_json()

    # Aquí puedes adaptar el código para procesar los datos según el formato esperado por tu modelo
    entrada = datos['entrada']  # Suponiendo que la entrada esté en 'entrada'
    
    # Realiza la predicción
    prediccion = modelo.predict([entrada])

    # Retorna la predicción como JSON
    return jsonify({'prediccion': prediccion.tolist()})

# Inicia el servidor
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
